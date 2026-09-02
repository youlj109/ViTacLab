#!/usr/bin/env python3
"""Aggregate lateral-force validation sweep outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from validation_beta_config import CONTACT_VALID_FN_RATIO, SF_SCHEMA

LEGACY_SCHEMA = "sf_lateral_v1"


def _collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(root.glob("**/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if data.get("output_schema") not in (SF_SCHEMA, LEGACY_SCHEMA):
            continue
        data["_path"] = str(summary_path.parent.relative_to(root))
        rows.append(data)
    return rows


def _is_valid(r: dict) -> bool:
    if "contact_valid" in r:
        return bool(r["contact_valid"])
    nom = float(r.get("nominal_fn_n", 0))
    fn = float(r.get("physx_fn_total_mean", 0))
    cnt = float(r.get("physx_contact_count_mean", r.get("physx_contact_count_last", 0)))
    return fn >= CONTACT_VALID_FN_RATIO * nom and cnt > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="logs/vitacsim_validation/shear_force/lateral")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = _collect(root)

    lines = [
        "# Lateral Force Validation Sweep",
        "",
        f"Schema: `{SF_SCHEMA}` (legacy `{LEGACY_SCHEMA}` still listed)",
        "",
        "| tag | weight | mode | Fx | physx_fn | ratio | valid | physx_ft | ft_peak | fn_peak |",
        "|-----|--------|------|----|----------|-------|-------|----------|---------|---------|",
    ]
    for r in rows:
        nom = float(r.get("nominal_fn_n", 0))
        fn = float(r.get("physx_fn_total_mean", 0))
        ratio = float(r.get("physx_fn_ratio_nominal", fn / nom if nom > 1e-9 else 0))
        valid = _is_valid(r)
        lines.append(
            f"| {r.get('lateral_force_tag', '?')} | {r.get('weight_id', '?')} | {r.get('sensor_mode', '?')} | "
            f"{r.get('lateral_force_x_n', 0):.3f} | {fn:.3f} | {ratio:.2f} | {valid} | "
            f"{r.get('physx_ft_total_mean', 0):.3f} | {r.get('ft_peak_max', 0):.3f} | "
            f"{r.get('fn_peak_max', 0):.3f} |"
        )

    lines += [
        "",
        "Notes:",
        "- **valid=false**: weight lost pad contact (often light mass + high Fx); excluded from rebattle curves.",
        "- Per-weight Fx sweeps are capped in `validation_beta_config.WEIGHT_FX_SWEEP`.",
    ]

    report = "\n".join(lines) + "\n"
    out = root / "SWEEP_REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[INFO] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
