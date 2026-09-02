#!/usr/bin/env python3
"""Copy sim sweep RGB/marker into a flat reference tree for real-robot capture guidance."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    LATERAL_W100_FX,
    WEIGHT_CASES,
    _fx_tag,
    default_sim_root,
    load_lateral_cases,
    load_nf_cases,
    repo_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export sim calibration captures as reference.")
    parser.add_argument("--sim-root", type=str, default=str(default_sim_root()))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root() / "data" / "calibration" / "tactile" / "sim_reference"),
    )
    parser.add_argument("--sensor-mode", type=str, default="vitacsim")
    args = parser.parse_args()

    sim_root = Path(args.sim_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nf = load_nf_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode)
    lat = load_lateral_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode)

    manifest: dict = {"cases": []}

    def _export(case_id: str, src_dir: Path, rel: str) -> None:
        dst = out_dir / rel
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("tactile_rgb.png", "tactile_marker_displacement.npy", "summary.json"):
            sp = src_dir / name
            if sp.is_file():
                shutil.copy2(sp, dst / name)
        manifest["cases"].append({"case_id": case_id, "path": rel})

    for cid in ["no_contact", *WEIGHT_CASES]:
        s = nf.get(cid)
        if s is None or s.source_dir is None:
            continue
        _export(cid, s.source_dir, f"normal_force/{cid}")

    for fx in LATERAL_W100_FX:
        cid = f"W100_{_fx_tag(fx)}"
        s = lat.get(cid)
        if s is None or s.source_dir is None:
            continue
        _export(cid, s.source_dir, f"lateral_force/W100/{_fx_tag(fx)}")

    index = sim_root / "sim_index.json"
    if index.is_file():
        shutil.copy2(index, out_dir / "sim_index.json")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] sim reference -> {out_dir}")
    print("[INFO] Use these as visual reference when capturing real rgb.png into data/calibration/tactile/real/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
