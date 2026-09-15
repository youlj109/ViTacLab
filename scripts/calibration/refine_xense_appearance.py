#!/usr/bin/env python3
"""Fit resting marker shape and validate optical tuning on alternating nut loads.

Run with Isaac Lab's Python. Uses saved physical depths without modifying them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

from sweep_xense_force_depth_gain import CASES, ROI_XYXY, _clean_markers, _align_frame_background, _load_rgb, _label


def write_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('logs/xense_pooled_final_physical'))
    parser.add_argument('--out-dir', type=Path, default=Path('logs/xense_appearance_refined_20260915'))
    parser.add_argument('--flat-reference', action='store_true', help='Evaluate local zero-normal optical reference.')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import advisor_xense_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_marker import MarkerSimulator

    root = Path('data/calibration/tactile/advisor_processed')
    bg = _load_rgb(root / 'bg_clean.jpg')
    raw = _load_rgb(root / 'bg.jpg')
    rest = np.load(root / 'marker_rest.npy')
    marker_cfg = dict(marker_shape='measured', marker_reference_path='bg.jpg', marker_blend_alpha=1.0)

    heights = np.stack([np.load(args.input_root/'normal_force'/c/'vitacsim/tactile_height_corrected.npy') for c in CASES])
    height_tensor = torch.from_numpy(heights).to(args.device)
    real_images, real_clean, masks = [], [], []
    x0,y0,x1,y1 = ROI_XYXY
    for case in CASES:
        real = _load_rgb(Path('data/calibration/tactile/real/normal_force')/case/'rgb.png')
        clean, mask = _clean_markers(real, rest)
        aligned, _ = _align_frame_background(clean, bg, marker_mask=mask,
                                             center_xy=(200.,350.), contact_radius_px=85.)
        real_images.append(real)
        real_clean.append(aligned)
        masks.append(mask[y0:y1,x0:x1] == 0)
    real_delta = np.array(real_clean, np.float32)[:,y0:y1,x0:x1] - bg[y0:y1,x0:x1]
    train, heldout = [0,2,4], [1,3,5]
    base_cfg = advisor_xense_render_cfg(enable_marker_simulation=False, marker_pattern='none')
    renderer = GelsightRender(base_cfg, args.device)
    def score(rgb):
        delta = rgb[:,y0:y1,x0:x1].astype(np.float32)-bg[y0:y1,x0:x1]
        rmse = [float(np.sqrt(np.mean((delta[i]-real_delta[i])[masks[i]]**2))) for i in range(6)]
        norm = [rmse[i]/max(float(np.sqrt(np.mean(real_delta[i][masks[i]]**2))),1e-6) for i in range(6)]
        return dict(rmse=rmse, train=float(np.mean(np.array(norm)[train])),
                    heldout=float(np.mean(np.array(norm)[heldout])))
    baseline_rgb = renderer.render(height_tensor).cpu().numpy()
    baseline_score = score(baseline_rgb)
    best_rgb, best_score, best_overrides = baseline_rgb, baseline_score, {}
    rows = []
    for iterations, gain, tint in itertools.product([6,10,14], [0.8,1.,1.2], [6.,12.,18.]):
        overrides = dict(taxim_response_mesh_smooth_iterations=iterations,
                         taxim_rgb_response_gain=gain, taxim_contact_red_tilt_additive=tint)
        if args.flat_reference:
            overrides['taxim_zero_normal_reference'] = True
        renderer.cfg = base_cfg.replace(**overrides)
        rgb = renderer.render(height_tensor).cpu().numpy()
        result = score(rgb)
        rows.append(dict(parameters=overrides, **result))
        if result['train'] < best_score['train']:
            best_rgb, best_score, best_overrides = rgb, result, overrides
        print('OPTICS', iterations, gain, tint, result, flush=True)
    # A faint or blank contact can reduce raw image error. Require the optical
    # candidate to also outperform the no-contact prediction (normalized RMSE=1).
    optical_accepted = best_score['heldout'] < min(baseline_score['heldout'], 1.0)
    if not optical_accepted:
        best_rgb, best_score, best_overrides = baseline_rgb, baseline_score, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    def marker_renderer(measured):
        return MarkerSimulator(pattern='xense', image_height=bg.shape[0], image_width=bg.shape[1],
                               device=args.device, rest_xy_override=rest,
                               marker_shape='measured' if measured else 'gaussian',
                               gaussian_sigma_x_px=2.6, gaussian_sigma_y_px=2.9,
                               blend_alpha=1.0 if measured else 0.6,
                               reference_rgb=raw, clean_background_rgb=bg)
    old_draw, new_draw = marker_renderer(False), marker_renderer(True)
    def composite(draw, rgb, displacement):
        return draw.draw_markers_on_image(torch.from_numpy(rgb).to(args.device),
                    draw.rest_xy + torch.as_tensor(displacement, device=args.device)).cpu().numpy()
    full_rows, crop_rows, clean_rows, marker_errors = [], [], [], []
    for i, case in enumerate(CASES):
        src = args.input_root/'normal_force'/case/'vitacsim'
        dst = args.out_dir/'normal_force'/case/'vitacsim'
        displacement = np.load(src/'tactile_marker_displacement.npy')
        new = composite(new_draw, best_rgb[i], displacement)
        old = composite(old_draw, baseline_rgb[i], displacement)
        marker_errors.append(dict(case=case, **{
            name: float(np.sqrt(np.mean((im[y0:y1,x0:x1].astype(np.float32)-
                 real_images[i][y0:y1,x0:x1])[~masks[i]]**2)))
            for name, im in [('before',old),('after',new)]}))
        write_rgb(dst/'tactile_rgb_corrected.png', new)
        write_rgb(dst/'tactile_rgb_marker_free.png', best_rgb[i])
        for filename in ['tactile_height_corrected.npy','tactile_marker_displacement.npy']:
            shutil.copyfile(src/filename, dst/filename)
        images = [real_images[i], old, new]
        names = [case+' real', 'before', 'refined']
        full_rows.append(np.hstack([_label(im,n) for im,n in zip(images,names)]))
        crop_rows.append(np.hstack([_label(im[y0:y1,x0:x1],n) for im,n in zip(images,names)]))
        clean_rows.append(np.hstack([_label(im[y0:y1,x0:x1],n) for im,n in
                                    zip([real_clean[i],baseline_rgb[i],best_rgb[i]],names)]))
    renderer.cfg = base_cfg.replace(**best_overrides)
    resting_rgb = renderer.render(torch.zeros_like(height_tensor[:1])).cpu().numpy()[0]
    write_rgb(args.out_dir/'normal_force/no_contact/vitacsim/tactile_rgb.png',
              composite(new_draw, resting_rgb, np.zeros_like(rest)))
    write_rgb(args.out_dir/'all_masses_full.png', np.vstack(full_rows))
    write_rgb(args.out_dir/'all_masses_crops.png', np.vstack(crop_rows))
    write_rgb(args.out_dir/'all_masses_marker_free.png', np.vstack(clean_rows))
    report = dict(train_cases=[CASES[i] for i in train], heldout_cases=[CASES[i] for i in heldout],
                  optical_accepted=optical_accepted, baseline=baseline_score, refined=best_score,
                  optical_parameters=best_overrides, marker_parameters=marker_cfg,
                  marker_contact_patch_rmse=marker_errors, candidates=rows,
                  note='Optics scored on unpainted pixels outside marker masks. Heldout loads share object and session; not an independent object benchmark. Marker displacements copied unchanged.')
    (args.out_dir/'fit_report.json').write_text(json.dumps(report,indent=2))
    print('RESULT', json.dumps({k:v for k,v in report.items() if k!='candidates'}), flush=True)
    app.close()


if __name__ == '__main__':
    main()
