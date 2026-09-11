#!/usr/bin/env python3
"""Benchmark the ViTacSim Taxim/FOTS renderer on a saved corrected height map.

This script must run with Isaac Sim Python because the renderer configuration
uses Isaac Lab utilities. It deliberately measures rendering only; camera,
PhysX contact, and force-to-depth reconstruction are outside this timing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--height",
        type=Path,
        default=Path("logs/xense_marker_crisper/normal_force/G110/vitacsim/tactile_height_corrected.npy"),
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes, for example 1,8,32.",
    )
    parser.add_argument("--output", type=Path, default=Path("logs/xensim_official_comparison_best/vitacsim_render_benchmark.json"))
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": True})
    try:
        import numpy as np
        import torch

        from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import advisor_xense_render_cfg
        from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender

        height_one = torch.from_numpy(np.load(args.height).astype(np.float32)).cuda().unsqueeze(0)
        batch_sizes = tuple(int(value) for value in args.batch_sizes.split(",") if value.strip())
        results = {}
        for batch_size in batch_sizes:
            height = height_one.repeat(batch_size, 1, 1)
            batch_result = {}
            for markers in (False, True):
                cfg = advisor_xense_render_cfg(
                    enable_marker_simulation=markers,
                    marker_pattern="xense" if markers else "none",
                )
                renderer = GelsightRender(cfg, "cuda:0")
                for _ in range(args.warmup):
                    renderer.render(height)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.iterations):
                    renderer.render(height)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                batch_result["with_markers" if markers else "without_markers"] = {
                    "iterations": args.iterations,
                    "batch_size": batch_size,
                    "total_seconds": elapsed,
                    "milliseconds_per_batch": elapsed * 1000.0 / args.iterations,
                    "sensor_frames_per_second": args.iterations * batch_size / elapsed,
                }
            results[f"batch_{batch_size}"] = batch_result

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
