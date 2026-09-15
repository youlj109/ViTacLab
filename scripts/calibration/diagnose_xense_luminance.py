#!/usr/bin/env python3
"""Measure signed contact luminance at successive optical rendering stages."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher
from sweep_xense_force_depth_gain import CASES, _load_rgb, _clean_markers, _label
from evaluate_xense_ball_polycalib import _pure_polycalib_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('logs/xense_luminance_audit_20260915'))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import advisor_xense_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender
    bg = _load_rgb(Path('data/calibration/tactile/advisor_processed/bg_clean.jpg'))
    rest = np.load('data/calibration/tactile/advisor_processed/marker_rest.npy')
    root = Path('logs/xense_pooled_final_physical/normal_force')
    heights = np.stack([np.load(root/c/'vitacsim/tactile_height_corrected.npy') for c in CASES])
    tensors = torch.as_tensor(heights, device=args.device)
    base = advisor_xense_render_cfg(enable_marker_simulation=False, marker_pattern='none')
    pure = _pure_polycalib_cfg(base)
    configs = {'pure': pure, 'pure_negative_depth': pure,
               'pure_scale045': pure.replace(taxim_height_scale=.45),
               'without_tint': base.replace(taxim_contact_red_tilt_strength=0.,taxim_contact_red_tilt_additive=0.),
               'production':base,
               'flat_reference':base.replace(taxim_zero_normal_reference=True)}
    outputs = {}
    for name,cfg in configs.items():
        renderer = GelsightRender(cfg, args.device)
        outputs[name] = renderer.render(-tensors if name=='pure_negative_depth' else tensors).cpu().numpy()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    rows, panels = [], []
    weights = np.array([.2126,.7152,.0722])
    for i,case in enumerate(CASES):
        real = _load_rgb(Path('data/calibration/tactile/real/normal_force')/case/'rgb.png')
        clean, marker_mask = _clean_markers(real,rest)
        contact = heights[i] > .1*heights[i].max()
        roi = np.zeros(contact.shape,bool);roi[260:440,110:290]=True
        mask = contact & (marker_mask==0)
        imgs = {'real':clean, **{k:v[i] for k,v in outputs.items()}}
        stats = {}
        for name,im in imgs.items():
            delta = im.astype(np.float32)-bg
            lum = delta@weights
            stats[name] = dict(contact_mean_luminance=float(lum[mask].mean()),
                               roi_mean_luminance=float(lum[roi & (marker_mask==0)].mean()),
                               contact_rgb_mean=delta[mask].mean(0).tolist(),
                               positive_fraction=float((lum[mask]>0).mean()))
            cv2.imwrite(str(args.out_dir/f'{case}_{name}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        panels.append(np.hstack([_label(im[240:460,90:310],f'{case} {name}') for name,im in imgs.items()]))
        rows.append(dict(case=case,stages=stats))
    cv2.imwrite(str(args.out_dir/'stages.png'),cv2.cvtColor(np.vstack(panels),cv2.COLOR_RGB2BGR))
    (args.out_dir/'metrics.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows),flush=True)
    app.close()


if __name__=='__main__':
    main()
