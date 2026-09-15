#!/usr/bin/env python3
"""Exercise measured marker compositing without launching an Isaac scene."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

path = Path(__file__).resolve().parents[2] / 'source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/visuotactile_marker.py'
spec = importlib.util.spec_from_file_location('marker_under_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

bg = np.full((40,40,3), 160, np.uint8)
reference = bg.copy()
reference[18:23,18:23] = (40,50,70)
sim = module.MarkerSimulator(pattern='xense', image_height=40, image_width=40,
    device='cpu', marker_shape='measured', blend_alpha=1., rest_xy_override=np.array([[20.,20.]]),
    reference_rgb=reference, clean_background_rgb=bg)
rgb = torch.from_numpy(bg)
restored = sim.draw_markers_on_image(rgb, sim.rest_xy)
assert torch.equal(restored, torch.from_numpy(reference)), 'Resting printed profile must reproduce reference'
assert torch.equal(rgb, torch.from_numpy(bg)), 'Compositing must not mutate input'
shifted = sim.draw_markers_on_image(rgb, sim.rest_xy+torch.tensor([3.,-2.]))
expected = bg.copy(); expected[16:21,21:26] = (40,50,70)
assert torch.equal(shifted, torch.from_numpy(expected)), 'Marker translation must follow displacement'
fractional = sim.draw_markers_on_image(rgb, sim.rest_xy+.5)
assert fractional.dtype == torch.uint8 and torch.isfinite(fractional).all()
assert torch.equal(sim.draw_markers_on_image(rgb, sim.rest_xy+100), rgb)
lit = torch.full_like(rgb,200)
out = sim.draw_markers_on_image(lit,sim.rest_xy)
assert out[20,20,0] == 50, 'Measured marker must transmit changed contact illumination'
print('PASS: rest reconstruction, translation, subpixel motion, offscreen clipping, illumination, input immutability')

# Preserve ordered rounding for overlaps; exercise disjoint and clipped patches
# on CPU and CUDA with non-uniform backgrounds and fractional displacement.
import cv2

torch.set_num_threads(4)
repo=path.parents[6]
data=repo/'data/calibration/tactile/advisor_processed'
raw=cv2.cvtColor(cv2.imread(str(data/'bg.jpg')),cv2.COLOR_BGR2RGB)
clean=cv2.cvtColor(cv2.imread(str(data/'bg_clean.jpg')),cv2.COLOR_BGR2RGB)
rest=np.load(data/'marker_rest.npy')
devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
for device in devices:
    measured=module.MarkerSimulator(pattern='xense',image_height=700,image_width=400,
        device=device,marker_shape='measured',blend_alpha=1.,rest_xy_override=rest,
        reference_rgb=raw,clean_background_rgb=clean)
    cases=['G010','G030','G060','G110','G160','G210']
    for case in cases:
        folder=repo/'logs/xense_gain320_comparison_20260915/input/normal_force'/case/'vitacsim'
        image=cv2.cvtColor(cv2.imread(str(folder/'tactile_rgb_marker_free.png')),cv2.COLOR_BGR2RGB)
        image=torch.as_tensor(image,device=device)
        positions=measured.rest_xy+torch.as_tensor(np.load(folder/'tactile_marker_displacement.npy'),device=device)
        a=measured._draw_measured_markers_serial(image,positions)
        b=measured.draw_markers_on_image(image,positions)
        assert torch.equal(a,b), f'{device} {case}: optimized pixels differ from serial reference'
    image=torch.as_tensor(clean,device=device)
    for shift in [(0.,0.),(.37,-.61),(-30.2,18.9),(800.,-900.)]:
        positions=measured.rest_xy+torch.tensor(shift,device=device)
        assert torch.equal(measured._draw_measured_markers_serial(image,positions),
                           measured.draw_markers_on_image(image,positions)), (device,shift)
    positions=measured.rest_xy.clone()
    positions[1]=positions[0]+.5
    assert torch.equal(measured._draw_measured_markers_serial(image,positions),
                       measured.draw_markers_on_image(image,positions)), 'overlap semantics changed'
    noncontiguous=image.transpose(0,1).contiguous().transpose(0,1)
    assert torch.equal(measured._draw_measured_markers_serial(noncontiguous,measured.rest_xy),
                       measured.draw_markers_on_image(noncontiguous,measured.rest_xy))
    print(f'PASS {device}: six real-load images bit-identical; subpixel/clipped/offscreen/overlap/noncontiguous inputs')
