#!/usr/bin/env python3
"""Create real-robot tactile calibration folder layout for RGB + marker joint fitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    LATERAL_W100_FX,
    WEIGHT_CASES,
    _fx_tag,
    default_real_root,
    write_manifest,
)


def _touch_placeholder(path: Path, text: str) -> None:
    if path.is_file():
        return
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare real tactile calibration directory tree.")
    parser.add_argument("--real-root", type=str, default=str(default_real_root()))
    args = parser.parse_args()

    root = Path(args.real_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    nf = root / "normal_force"
    (nf / "no_contact").mkdir(parents=True, exist_ok=True)
    _touch_placeholder(
        nf / "no_contact" / "README.txt",
        "Place no_contact/rgb.png and optional marker_displacement.npy (zeros).\n",
    )
    for wid in WEIGHT_CASES:
        d = nf / wid
        d.mkdir(parents=True, exist_ok=True)
        _touch_placeholder(
            d / "README.txt",
            f"Place {wid}/rgb.png and marker_displacement.npy (dx,dy vs no_contact rest).\n",
        )

    lat = root / "lateral_force" / "W100"
    for fx in LATERAL_W100_FX:
        d = lat / _fx_tag(fx)
        d.mkdir(parents=True, exist_ok=True)
        _touch_placeholder(
            d / "README.txt",
            f"W100 lateral Fx={fx} N: rgb.png + marker_displacement.npy\n",
        )

    write_manifest(root / "manifest.json")
    print(f"[OK] real calibration template -> {root}")
    print("[NEXT] Copy real captures into each case folder, then run fit_vitacsim_rgb_marker.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
