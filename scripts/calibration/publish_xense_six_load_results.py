#!/usr/bin/env python3
"""Replay the repository TacSL baseline, then export auditable website assets.

Run --render-tacsl --headless in Isaac Lab first; run without flags in a
NumPy/OpenCV environment afterwards. Source logs/calibration remain local.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ('G010', 'G030', 'G060', 'G110', 'G160', 'G210')
OLD = ROOT / 'logs/vitacsim_tacsl_compare_20260914/normal_force'
OURS = ROOT / 'logs/xense_batched_markers_20260915/input/normal_force'
OFFICIAL = ROOT / 'logs/xense_batched_markers_official_20260915'
REPLAY = ROOT / 'logs/xense_tacsl_current_optics_20260915'
OUT = ROOT / 'docs/media/xense-six-load'


def read(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def replay(args):
    import torch
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender
    cfg = validation_gelsight_render_cfg(profile='advisor', flat_contact=True,
                                        enable_marker=True, marker_pattern='xense')
    renderer = GelsightRender(cfg, args.device)
    marker = renderer._marker_sim
    renderer._marker_sim = None
    for case in ('no_contact',) + CASES:
        folder = OLD / case / 'tacsl'
        height = (np.zeros((700, 400), np.float32) if case == 'no_contact' else
                  np.load(folder / 'tactile_height_depth_projected.npy'))
        rgb = renderer.render(torch.as_tensor(height, device=args.device)[None])[0]
        displacement = (np.zeros((marker.num_markers, 2), np.float32) if case == 'no_contact' else
                        np.load(folder / 'tactile_marker_displacement.npy'))
        rgb = marker.draw_markers_on_image(
            rgb, marker.rest_xy + torch.as_tensor(displacement, device=args.device))
        save(REPLAY / case / 'rgb.png', rgb.cpu().numpy())
        print(case, 'replayed uncorrected depth', float(height.max()), flush=True)
    app.close()


def error(a, b):
    diff = a.astype(np.float64) - b.astype(np.float64)
    return dict(mae=float(np.abs(diff).mean()), rmse=float(np.sqrt((diff**2).mean())))


def label(image, title):
    bar = np.zeros((42, image.shape[1], 3), np.uint8)
    scale = min(0.65, (image.shape[1]-16) / max(cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0], 1))
    cv2.putText(bar, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def verify():
    """Recompute published metrics using only committed RGB images."""
    report = json.loads((OUT/'metrics.json').read_text())
    for case in CASES:
        real = read(OUT/case/'real.png').astype(np.float64)
        real_delta = real - read(OUT/'backgrounds/real.png')
        for method in ('vitacsim', 'tacsl', 'official'):
            image = read(OUT/case/(method+'.png')).astype(np.float64)
            delta = image - read(OUT/'backgrounds'/(method+'.png'))
            pairs = dict(full_rgb=(real, image), full_delta_rgb=(real_delta, delta),
                         roi_delta_rgb=(real_delta[240:460,90:310], delta[240:460,90:310]))
            for metric, (a, b) in pairs.items():
                for stat, value in error(a, b).items():
                    assert np.isclose(value, report['cases'][case][method][metric][stat],
                                      rtol=0, atol=1e-10), (case, method, metric, stat)
    for method, metrics in report['mean_per_case'].items():
        for metric, stats in metrics.items():
            for stat, value in stats.items():
                expected = np.mean([report['cases'][c][method][metric][stat] for c in CASES])
                assert np.isclose(value, expected, rtol=0, atol=1e-10)
    print('PASS: all published per-case and aggregate MAE/RMSE values recomputed from committed RGB.')


def publish():
    backgrounds = dict(real=read(ROOT/'data/calibration/tactile/advisor_processed/bg.jpg'),
                       vitacsim=read(OURS/'no_contact/vitacsim/tactile_rgb.png'),
                       tacsl=read(REPLAY/'no_contact/rgb.png'),
                       official=read(OFFICIAL/'official_no_contact.png'))
    names = dict(real='Real', vitacsim='ViTacSim', tacsl='TacSL depth-only replay', official='Official Xensim')
    report = dict(protocol='RGB 0-255, markers included; no registration or per-image color fitting. '
                  'Delta = contact minus each method own no-contact image. '
                  'Fixed ROI x=[90,310), y=[240,460) shared by all cases/methods. '
                  'TacSL is repository depth-only implementation replayed with current optics; '
                  'official receives ViTacSim corrected depth. Six in-sample loads, not independent validation.',
                  k_ref=1350, depth_gain=3.2, cases={})
    full_rows, crop_rows = [], []
    roi = np.s_[240:460, 90:310, :]
    for case in CASES:
        images = dict(real=read(OFFICIAL/case/'real.png'),
                      vitacsim=read(OFFICIAL/case/'vitacsim.png'),
                      tacsl=read(REPLAY/case/'rgb.png'),
                      official=read(OFFICIAL/case/'official_xensim.png'))
        assert all(im.shape == (700, 400, 3) for im in (*images.values(), *backgrounds.values()))
        delta = {key: im.astype(np.float32)-backgrounds[key].astype(np.float32)
                 for key, im in images.items()}
        report['cases'][case] = {key: dict(full_rgb=error(images['real'], images[key]),
            full_delta_rgb=error(delta['real'], delta[key]),
            roi_delta_rgb=error(delta['real'][roi], delta[key][roi]))
            for key in ('vitacsim', 'tacsl', 'official')}
        full, crop = [], []
        for key, im in images.items():
            save(OUT/case/(key+'.png'), im)
            full.append(label(im, case+' | '+names[key]))
            crop.append(label(im[roi], case+' | '+names[key]))
        full_rows.append(np.hstack(full))
        crop_rows.append(np.hstack(crop))
    for key, im in backgrounds.items():
        save(OUT/'backgrounds'/(key+'.png'), im)
    save(OUT/'all-loads-full.png', np.vstack(full_rows))
    save(OUT/'all-loads-crops.png', np.vstack(crop_rows))
    report['mean_per_case'] = {key: {metric: {stat: float(np.mean([
        report['cases'][case][key][metric][stat] for case in CASES]))
        for stat in ('mae', 'rmse')} for metric in ('full_rgb','full_delta_rgb','roi_delta_rgb')}
        for key in ('vitacsim','tacsl','official')}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-tacsl', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Check metrics using committed images only.')
    import sys
    if '--render-tacsl' in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.render_tacsl:
        replay(args)
    elif args.verify:
        verify()
    else:
        publish()
