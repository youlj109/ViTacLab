#!/usr/bin/env python3
"""Re-render a saved Xense height map without rerunning Isaac Sim physics.

This is intentionally a small calibration utility: the normal-force demo saves
the physical depth maps in the workspace, and this script lets Taxim optical
parameters be evaluated independently of PhysX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--height-map", type=Path, required=True, help="Corrected height-map .npy (H,W), in meters."
    )
    parser.add_argument("--marker-height", type=Path, default=None, help="Optional marker-driving height-map .npy.")
    parser.add_argument("--out", type=Path, required=True, help="Output RGB PNG.")
    parser.add_argument("--no-marker", action="store_true")
    parser.add_argument("--taxim-height-scale", type=float, default=None)
    parser.add_argument("--red-tilt-strength", type=float, default=None)
    parser.add_argument("--red-tilt-additive", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app

    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import advisor_xense_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender

    cfg = advisor_xense_render_cfg(
        enable_marker_simulation=not args.no_marker,
        marker_pattern="xense",
    )
    overrides = {}
    if args.taxim_height_scale is not None:
        overrides["taxim_height_scale"] = float(args.taxim_height_scale)
    if args.red_tilt_strength is not None:
        overrides["taxim_contact_red_tilt_strength"] = float(args.red_tilt_strength)
    if args.red_tilt_additive is not None:
        overrides["taxim_contact_red_tilt_additive"] = float(args.red_tilt_additive)
    if overrides:
        cfg = cfg.replace(**overrides)

    height = torch.from_numpy(np.load(args.height_map).astype(np.float32)).unsqueeze(0).to(args.device)
    marker_height = None
    if args.marker_height is not None:
        marker_height = torch.from_numpy(np.load(args.marker_height).astype(np.float32)).unsqueeze(0).to(args.device)

    renderer = GelsightRender(cfg, args.device)
    image = renderer.render(height, marker_height_map=marker_height)[0].detach().cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[OK] saved {args.out}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
