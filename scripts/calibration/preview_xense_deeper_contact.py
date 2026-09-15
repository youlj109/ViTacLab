#!/usr/bin/env python3
"""Compare deeper contact at fixed k_ref=1350 using the gain=1 saved fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from isaaclab.app import AppLauncher
from replay_xense_depth_gain import save
from sweep_xense_force_depth_gain import CASES, _load_rgb, _label_above


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('logs/xense_fixed1350_deeper_preview_20260915'))
    parser.add_argument('--gains',type=float,nargs='+',default=[1.,1.1,1.2,1.3,1.4])
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if any(not np.isfinite(g) or g<=0 for g in args.gains):
        parser.error('All gains must be finite and positive')
    app=AppLauncher(args).app
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender
    root=Path('logs/xense_kref_real_fit_20260915/normal_force')
    gains=args.gains
    cfg=validation_gelsight_render_cfg(profile='advisor',flat_contact=True,enable_marker=True,marker_pattern='xense')
    renderer=GelsightRender(cfg,args.device)
    marker=renderer._marker_sim
    renderer._marker_sim=None
    heights=np.stack([np.load(root/c/'vitacsim/tactile_height_corrected.npy') for c in CASES])
    images={};clean_images={}
    for gain in gains:
        h=heights*gain
        rgb=renderer.render(torch.as_tensor(h,device=args.device))
        clean_images[gain]=rgb.cpu().numpy()
        rendered=[]
        for i,c in enumerate(CASES):
            disp=np.load(root/c/'vitacsim/tactile_marker_displacement.npy')
            image=marker.draw_markers_on_image(rgb[i],marker.rest_xy+torch.as_tensor(disp,device=args.device)).cpu().numpy()
            dst=args.output/f'gain_{gain:.1f}'/c
            save(dst/'tactile_rgb_corrected.png',image)
            save(dst/'tactile_rgb_marker_free.png',clean_images[gain][i])
            np.save(dst/'tactile_height_corrected.npy',h[i])
            rendered.append(image)
        images[gain]=rendered
    full,crops,clean_rows=[],[],[]
    for i,c in enumerate(CASES):
        real=_load_rgb(Path('data/calibration/tactile/real/normal_force')/c/'rgb.png')
        row=[_label_above(real,c+' real')]
        crop=[_label_above(real[240:460,90:310],c+' real')]
        clean_row=[]
        for gain in gains:
            label=f'gain={gain:.1f}, peak={heights[i].max()*gain*1000:.3f} mm'
            row.append(_label_above(images[gain][i],label))
            crop.append(_label_above(images[gain][i][240:460,90:310],label))
            clean_row.append(_label_above(clean_images[gain][i][240:460,90:310],label))
        full.append(np.hstack(row));crops.append(np.hstack(crop));clean_rows.append(np.hstack(clean_row))
    save(args.output/'all_masses_full.png',np.vstack(full))
    save(args.output/'all_masses_crops.png',np.vstack(crops))
    save(args.output/'all_masses_marker_free.png',np.vstack(clean_rows))
    report=dict(k_ref=1350.,source_depth_gain=1.,depth_gains=gains,
                peaks_mm={c:[float(heights[i].max()*g*1000) for g in gains] for i,c in enumerate(CASES)},
                note='Visual depth trial, no error-based rejection. Optics and saved marker motion unchanged; PhysX not rerun.')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    app.close()


if __name__=='__main__':
    main()
