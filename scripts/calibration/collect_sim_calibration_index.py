#!/usr/bin/env python3
"""Scan sim calibration sweep outputs and write index.json for fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    ADVISOR_WEIGHT_CASES,
    CALIB_SCHEMA,
    CALIB_SCHEMA_ADVISOR,
    LATERAL_W100_FX,
    WEIGHT_CASES,
    case_completeness,
    default_sim_root,
    load_advisor_sim_nf_cases,
    load_lateral_cases,
    load_nf_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Index sim calibration sweep for RGB+marker fit.")
    parser.add_argument("--sim-root", type=str, default=str(default_sim_root()))
    parser.add_argument("--sensor-mode", type=str, default="vitacsim")
    parser.add_argument("--profile", type=str, default="advisor", choices=("advisor", "cylinder"))
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    sim_root = Path(args.sim_root).expanduser().resolve()
    if args.profile == "advisor":
        nf = load_advisor_sim_nf_cases(sim_root, sensor_mode=args.sensor_mode)
        expected_nf = ["no_contact", *ADVISOR_WEIGHT_CASES]
        schema = CALIB_SCHEMA_ADVISOR
    else:
        nf = load_nf_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode)
        expected_nf = ["no_contact", *WEIGHT_CASES]
        schema = CALIB_SCHEMA
    lat = load_lateral_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode)

    index = {
        "schema": schema,
        "profile": args.profile,
        "sim_root": str(sim_root),
        "sensor_mode": args.sensor_mode,
        "normal_force": {
            "completeness": case_completeness(nf),
            "marker_disp_max_px": {k: v.marker_disp_max_px for k, v in nf.items()},
            "summary_fn_peak": {
                k: float(v.summary.get("fn_peak_max", 0.0)) for k, v in nf.items() if v.summary
            },
        },
        "lateral_force_W100": {
            "completeness": case_completeness(lat),
            "marker_disp_max_px": {k: v.marker_disp_max_px for k, v in lat.items()},
        },
        "expected_nf_cases": expected_nf,
        "expected_lateral_fx": list(LATERAL_W100_FX) if args.profile == "cylinder" else [],
    }

    out_path = Path(args.out).expanduser().resolve() if args.out else sim_root / "sim_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[OK] sim index -> {out_path}")

    nf_ok = sum(1 for c in index["normal_force"]["completeness"].values() if c["rgb"])
    lat_ok = sum(1 for c in index["lateral_force_W100"]["completeness"].values() if c["rgb"])
    print(f"[INFO] NF rgb cases: {nf_ok}/{len(index['expected_nf_cases'])}")
    print(f"[INFO] Lateral rgb cases: {lat_ok}/{len(LATERAL_W100_FX)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
