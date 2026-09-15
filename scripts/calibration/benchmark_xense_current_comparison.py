#!/usr/bin/env python3
"""Prepare selected gain=3.2 replay and benchmark current renderer per load."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from isaaclab.app import AppLauncher
from sweep_xense_force_depth_gain import CASES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('logs/xense_gain320_comparison_20260915/input'))
    parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--compare-serial',action='store_true',help='Also benchmark the original ordered marker path in the same process.')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    app=AppLauncher(args).app
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender
    source=Path('logs/xense_fixed1350_gain200_preview_20260915/gain_3.2')
    motion=Path('logs/xense_kref_real_fit_20260915/normal_force')
    cfg=validation_gelsight_render_cfg(profile='advisor',flat_contact=True,enable_marker=True,marker_pattern='xense')
    renderer=GelsightRender(cfg,args.device)
    results={}
    serial_results={}
    for case in CASES:
        dst=args.output/'normal_force'/case/'vitacsim'
        dst.mkdir(parents=True,exist_ok=True)
        for filename in ['tactile_rgb_corrected.png','tactile_rgb_marker_free.png','tactile_height_corrected.npy']:
            shutil.copyfile(source/case/filename,dst/filename)
        shutil.copyfile(motion/case/'vitacsim/tactile_marker_displacement.npy',dst/'tactile_marker_displacement.npy')
        height=torch.as_tensor(np.load(dst/'tactile_height_corrected.npy'),device=args.device)[None]
        if args.compare_serial:
            marker=renderer._marker_sim
            optimized_draw=marker.draw_markers_on_image
            marker.draw_markers_on_image=marker._draw_measured_markers_serial
            serial_image=renderer.render(height).clone()
            marker.draw_markers_on_image=optimized_draw
            assert torch.equal(serial_image,renderer.render(height)), f'{case}: full renderer output changed'
        for _ in range(args.warmup):
            renderer.render(height)
        torch.cuda.synchronize()
        elapsed=[]
        for _ in range(args.iterations):
            start=time.perf_counter()
            renderer.render(height)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-start)
        results[case]=dict(fps=1/float(np.mean(elapsed)),mean_ms=float(np.mean(elapsed))*1000,
                           median_ms=float(np.median(elapsed))*1000,p95_ms=float(np.percentile(elapsed,95))*1000)
        print(case,results[case],flush=True)
        if args.compare_serial:
            marker.draw_markers_on_image=marker._draw_measured_markers_serial
            for _ in range(args.warmup):
                renderer.render(height)
            torch.cuda.synchronize()
            serial_elapsed=[]
            for _ in range(args.iterations):
                start=time.perf_counter()
                renderer.render(height)
                torch.cuda.synchronize()
                serial_elapsed.append(time.perf_counter()-start)
            serial_results[case]=dict(fps=1/float(np.mean(serial_elapsed)),mean_ms=float(np.mean(serial_elapsed))*1000,
                                     median_ms=float(np.median(serial_elapsed))*1000,p95_ms=float(np.percentile(serial_elapsed,95))*1000)
            marker.draw_markers_on_image=optimized_draw
            print(case,'serial',serial_results[case],flush=True)
    dst=args.output/'normal_force/no_contact/vitacsim'
    dst.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path('logs/xense_flat_reference_refined_20260915/normal_force/no_contact/vitacsim/tactile_rgb.png'),dst/'tactile_rgb.png')
    report=dict(cases=results,iterations=args.iterations,warmup=args.warmup,
                aggregate_fps=1000/float(np.mean([r['mean_ms'] for r in results.values()])),
                scope='Batch=1, 400x700, Taxim + recomputed FOTS + measured marker compositing, CUDA synchronized. GPU inputs resident; no PhysX, camera, force reconstruction, disk I/O or host RGB transfer. Timing FOTS uses corrected height with no external shear; comparison images retain previously saved marker displacement.',
                gpu=torch.cuda.get_device_name(),k_ref=1350,depth_gain=3.2)
    if args.compare_serial:
        report['serial_cases']=serial_results
        report['serial_aggregate_fps']=1000/float(np.mean([r['mean_ms'] for r in serial_results.values()]))
        report['full_renderer_images_bit_identical']=True
    (args.output/'vitacsim_fps.json').write_text(json.dumps(report,indent=2))
    app.close()


if __name__=='__main__':
    main()
