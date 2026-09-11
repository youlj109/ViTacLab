#!/usr/bin/env python3
"""Smoke-test official XenseSim inside an already-started Isaac Sim process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-site", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/xensim_official_comparison_best/isaac_inprocess_smoke.png"),
    )
    args = parser.parse_args()

    app = SimulationApp({"headless": True})
    try:
        # Append after Isaac startup so its tested NumPy/PyTorch stack keeps
        # precedence; only missing XenseSim dependencies come from this path.
        sys.path.append(str(args.vendor_site))
        import cv2
        import numpy as np
        from xensim import PROJ_DIR
        from xensim.core import FemSensor

        sensor = FemSensor(
            PROJ_DIR / "assets/fem/g1-ws_table.npz",
            PROJ_DIR / "assets/fem/g1-ws.npz",
            depth_size=(100, 175),
            render_size=(400, 700),
            visible=False,
        )
        sensor.set_sim(show_marker=True, smooth_norm=8, rgb_gain=1.3)
        depth = np.full((175, 100), 0.2, np.float32)
        depth[64:112, 28:72] = -0.22
        sensor.step(depth, nstep=3)
        rgb = sensor.get_image()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Failed to write {args.output}")
        print(f"official XenseSim rendered inside Isaac Sim: {args.output.resolve()}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
