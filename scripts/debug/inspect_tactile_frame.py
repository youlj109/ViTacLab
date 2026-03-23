#!/usr/bin/env python3
"""
Inspect a saved tactile force frame (.npz).

Usage (from repo root):

    python scripts/debug/inspect_tactile_frame.py \
        --path tactile_force_frame_env0_sensor0.npz

It will:
- print basic statistics (shape, min, max, mean) for:
  - normal_force: (nrows, ncols)
  - shear_force:  (nrows, ncols, 2)
  - ff_image:     rendered force-field image
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a saved tactile force frame npz.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="tactile_force_frame_env0_sensor0.npz",
        help="Path to the .npz file (default: tactile_force_frame_env0_sensor0.npz).",
    )
    args = parser.parse_args()

    data = np.load(args.path)
    print(f"# Loaded: {args.path}")
    print("# Keys:", data.files)

    def _stats(name: str) -> None:
        if name not in data.files:
            print(f"- {name}: <missing>")
            return
        arr = data[name]
        print(
            f"- {name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
        )

    _stats("normal_force")
    _stats("shear_force")
    _stats("ff_image")

    # Print full arrays (may be large).
    if "normal_force" in data.files:
        nf = data["normal_force"]
        print("\n=== normal_force (full array) ===")
        print(nf)

    if "shear_force" in data.files:
        sf = data["shear_force"]
        print("\n=== shear_force (full array) ===")
        print(sf)

    if "ff_image" in data.files:
        ff = data["ff_image"]
        print("\n=== ff_image (full array) ===")
        print(ff)


if __name__ == "__main__":
    main()

