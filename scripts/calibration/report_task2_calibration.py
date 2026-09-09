#!/usr/bin/env python3
"""Generate Task 2 calibration report (advisor M2 + mass sweep)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    ADVISOR_MARKER_PATTERN,
    ADVISOR_MASS_G,
    ADVISOR_REAL_TO_SIM_NF,
    ADVISOR_WEIGHT_CASES,
    default_real_root,
    default_sim_root,
    load_advisor_real_nf_cases,
    load_mapped_sim_nf_for_advisor,
    repo_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 2 calibration report.")
    parser.add_argument("--real-root", type=str, default=str(default_real_root()))
    parser.add_argument("--sim-root", type=str, default=str(default_sim_root()))
    parser.add_argument("--fitted", type=str, default=str(repo_root() / "data/calibration/tactile/fitted_params.json"))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root() / "logs/vitacsim_calibration/task2"),
    )
    args = parser.parse_args()

    real_root = Path(args.real_root).expanduser().resolve()
    sim_root = Path(args.sim_root).expanduser().resolve()
    fitted_path = Path(args.fitted).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    real_nf = load_advisor_real_nf_cases(real_root)
    sim_nf = load_mapped_sim_nf_for_advisor(sim_root)

    fitted = json.loads(fitted_path.read_text(encoding="utf-8")) if fitted_path.is_file() else {}
    best = fitted.get("best_metrics", {})
    conv = fitted.get("convergence", {})

    lines = [
        "# ViTacSim Task 2 Calibration Report (Advisor Data)",
        "",
        "## Data source",
        "- Real sensor: **lab Xense** (`marker_pattern=xense`, per 共谋大业)",
        "- Real: `file-000.mp4` (no_contact bg) + `correct.zip` (10–210 g mass sweep)",
        "- Sim proxy: cylinder NF sweep mapped by mass (object differs — see table)",
        "- Sim marker in existing sweep may still be `gelsight`; re-run sweep with `MARKER_PATTERN=xense` for full parity",
        "",
        "## Real → sim NF proxy",
        "",
        "| Real case | Mass (g) | Sim proxy |",
        "|-----------|----------|-----------|",
    ]
    for cid in ADVISOR_WEIGHT_CASES:
        lines.append(f"| {cid} | {ADVISOR_MASS_G[cid]} | {ADVISOR_REAL_TO_SIM_NF[cid]} |")

    lines.extend(
        [
            "",
            "## Marker displacement (real, max px)",
            "",
            "| Case | mass (g) | real max px | sim proxy max px |",
            "|------|----------|-------------|------------------|",
        ]
    )
    for cid in ADVISOR_WEIGHT_CASES:
        r = real_nf.get(cid)
        s = sim_nf.get(cid)
        rm = r.marker_disp_max_px if r else 0.0
        sm = s.marker_disp_max_px if s else 0.0
        lines.append(f"| {cid} | {ADVISOR_MASS_G[cid]} | {rm:.3f} | {sm:.3f} |")

    lines.extend(
        [
            "",
            "## Joint fit (grid search)",
            "",
            f"- Best `marker_displacement_gain`: **{best.get('marker_displacement_gain', 'n/a')}**",
            f"- Best `rgb_diff_scale`: **{best.get('rgb_diff_scale', 'n/a')}**",
            f"- Best loss_total: **{best.get('loss_total', 'n/a')}**",
            f"- RGB cases used: **{best.get('rgb_cases_used', 'n/a')}**",
            f"- Loss improved vs defaults: **{conv.get('loss_improved', 'n/a')}** "
            f"(default {conv.get('loss_default', 'n/a')} → best {conv.get('loss_best', 'n/a')})",
            "",
            "## Apply fitted params",
            "",
            "```python",
            'render_cfg = validation_gelsight_render_cfg(fitted_params_path="data/calibration/tactile/fitted_params.json")',
            "```",
            "",
            "## Limitations (documented)",
            f"- Real marker tracking: `{ADVISOR_MARKER_PATTERN}` rest grid (14×14 staggered).",
            "- Real object: mass sweep indent; sim proxy: composite cylinder (mass-only mapping).",
            "- No lateral real cases in advisor zip; lateral term omitted from fit.",
            "- Semi-transparent marker appearance not fully modeled; fit uses displacement + RGB diff.",
            "- `polycalib.npz` unchanged; only `bg.jpg` updated from file-000.",
            "",
        ]
    )

    report = out_dir / "TASK2_CALIBRATION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
