#!/usr/bin/env python3
"""Read-only stage microbenchmarks for the current G110 renderer workload."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from isaaclab.app import AppLauncher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('logs/xense_gain320_stage_profile_20260915.json'))
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    app=AppLauncher(args).app
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender
    cfg=validation_gelsight_render_cfg(profile='advisor',flat_contact=True,enable_marker=True,marker_pattern='xense')
    renderer=GelsightRender(cfg,args.device)
    marker=renderer._marker_sim
    path=Path('logs/xense_gain320_comparison_20260915/input/normal_force/G110/vitacsim/tactile_height_corrected.npy')
    height=torch.as_tensor(np.load(path),device=args.device)[None]
    # Capture the exact intermediate inputs once, then restore original methods.
    captured={}
    displacement_fn=marker.displacements_from_height_mm
    drawing_fn=marker.draw_markers_on_image
    def capture_displacement(h,**kw):
        captured['height']=h
        return displacement_fn(h,**kw)
    def capture_draw(rgb,pos):
        captured['rgb']=rgb;captured['pos']=pos
        return drawing_fn(rgb,pos)
    marker.displacements_from_height_mm=capture_displacement
    marker.draw_markers_on_image=capture_draw
    renderer.render(height)
    marker.displacements_from_height_mm=displacement_fn
    marker.draw_markers_on_image=drawing_fn
    def bench(fn):
        for _ in range(100):fn()
        torch.cuda.synchronize()
        times=[]
        for _ in range(100):
            t=time.perf_counter();fn();torch.cuda.synchronize();times.append(time.perf_counter()-t)
        return dict(mean_ms=float(np.mean(times))*1000,median_ms=float(np.median(times))*1000,
                    p95_ms=float(np.percentile(times,95))*1000)
    results={'full_renderer':bench(lambda:renderer.render(height))}
    results['fots_displacement']=bench(lambda:displacement_fn(captured['height']))
    results['marker_compositing']=bench(lambda:drawing_fn(captured['rgb'],captured['pos']))
    renderer._marker_sim=None
    results['optics_without_markers']=bench(lambda:renderer.render(height))
    renderer._marker_sim=marker
    report=dict(case='G110',gpu=torch.cuda.get_device_name(),marker_count=marker.num_markers,
                iterations=100,warmup=100,stages=results,
                scope='Separate CUDA-synchronized wall-time microbenchmarks on captured intermediate inputs; stage times need not sum exactly. No production algorithm changes.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    app.close()


if __name__=='__main__':
    main()
