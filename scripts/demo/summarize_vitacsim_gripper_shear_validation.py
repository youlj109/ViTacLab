#!/usr/bin/env python3
"""Aggregate gripper shear validation sweep outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUTPUT_SCHEMA = "sf_gripper_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="logs/vitacsim_validation/shear_force/gripper")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    rows: list[dict] = []
    for p in sorted(root.glob("**/summary.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("output_schema") != OUTPUT_SCHEMA:
            continue
        rows.append(data)

    lines = [
        "# Gripper Shear Validation Sweep",
        "",
        f"Schema: `{OUTPUT_SCHEMA}`",
        "",
        "| tag | weight | mode | shear | ft_peak | fn_peak L/R |",
        "|-----|--------|------|-------|---------|-------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('shear_action_tag', '?')} | {r.get('weight_id', '?')} | {r.get('sensor_mode', '?')} | "
            f"{r.get('shear_action_x', 0):.2f} | {r.get('ft_peak_max', 0):.3f} | "
            f"{r.get('fn_peak_max_left', 0):.2f}/{r.get('fn_peak_max_right', 0):.2f} |"
        )
    report = "\n".join(lines) + "\n"
    out = root / "SWEEP_REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[INFO] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
