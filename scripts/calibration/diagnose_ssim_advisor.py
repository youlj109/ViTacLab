#!/usr/bin/env python3
"""Advisor Task3 SSIM breakdown: RGB vs bg-diff, contact area, polycalib status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibration_io import (
    ADVISOR_MASS_G,
    ADVISOR_WEIGHT_CASES,
    load_advisor_real_nf_cases,
    load_bg_from_path,
    load_mapped_sim_nf_for_advisor,
    repo_root,
    rgb_diff_magnitude,
    sim_nf_dir,
)
from report_task3_validation import _diff_vis, _ssim_gray


def _load_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Advisor SSIM diagnosis.")
    parser.add_argument("--sim-root", type=str, default=str(repo_root() / "logs/vitacsim_calibration/sweep"))
    parser.add_argument("--real-root", type=str, default=str(repo_root() / "data/calibration/tactile/real"))
    parser.add_argument(
        "--out",
        type=str,
        default=str(repo_root() / "logs/vitacsim_validation/task3/DIAGNOSIS_SSIM_ADVISOR.md"),
    )
    args = parser.parse_args()

    sim_root = Path(args.sim_root).expanduser().resolve()
    real_nf = load_advisor_real_nf_cases(Path(args.real_root).expanduser().resolve())
    tacsl_nf = load_mapped_sim_nf_for_advisor(sim_root, sensor_mode="tacsl")
    vit_nf = load_mapped_sim_nf_for_advisor(sim_root, sensor_mode="vitacsim")

    bg = load_bg_from_path(repo_root() / "data/calibration/tactile/advisor_processed/bg_clean.jpg")
    pc = repo_root() / "source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data/polycalib.npz"

    lines = [
        "# Advisor SSIM diagnosis\n",
        "## Polycalib\n",
        f"- Path: `{pc}`",
        f"- Installed: **{pc.is_file()}** ({pc.stat().st_size // 1024} KiB)\n",
        "## Per-case metrics (Real vs ViTacSim corrected)\n",
        "| Case | g | SSIM rgb | SSIM diff | diff MAD | marker sim max | Fn sim | Fn κ |",
        "|------|---|----------|-----------|----------|----------------|--------|------|",
    ]

    ssim_rgb_vals: list[float] = []
    ssim_diff_vals: list[float] = []

    for cid in ADVISOR_WEIGHT_CASES:
        r = real_nf.get(cid)
        v = vit_nf.get(cid)
        if r is None or v is None or r.rgb is None or v.rgb is None:
            continue
        ssim_rgb = _ssim_gray(r.rgb, v.rgb)
        ssim_diff = _ssim_gray(_diff_vis(r.rgb, bg), _diff_vis(v.rgb, bg))
        mad = float(np.abs(r.rgb.astype(np.float32) - v.rgb.astype(np.float32)).mean())
        ssim_rgb_vals.append(ssim_rgb)
        ssim_diff_vals.append(ssim_diff)

        summary = _load_summary(sim_nf_dir(sim_root, cid, sensor_mode="vitacsim") / "summary.json")
        fn_sim = float(summary.get("physx_fn_total_mean", float("nan")))
        fn_nom = ADVISOR_MASS_G[cid] / 1000.0 * 9.81
        kappa = fn_sim / fn_nom if fn_nom > 0 else float("nan")
        marker_max = float(summary.get("marker_disp_max_px", float("nan")))

        lines.append(
            f"| {cid} | {ADVISOR_MASS_G[cid]} | {ssim_rgb:.3f} | {ssim_diff:.3f} | {mad:.1f} | "
            f"{marker_max:.2f} | {fn_sim:.3f} | {kappa:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            f"- Mean SSIM (raw RGB): **{np.mean(ssim_rgb_vals):.3f}**",
            f"- Mean SSIM (bg-diff): **{np.mean(ssim_diff_vals):.3f}**",
            "- Low SSIM usually means: (1) contact patch shape/position mismatch, (2) Taxim polycalib residual, "
            "(3) marker overlay difference, (4) lighting/bg not matched.",
            "",
            "## Advisor asks (only if SSIM stays < 0.50 after marker fix + re-sweep)",
            "1. Raw **no-marker** Xense frames for G010/G110 (PNG) to separate marker vs Taxim error.",
            "2. Confirm **M2 nut pose** on gel (center offset, tilt) or share one CAD screenshot with contact ring highlighted.",
            "3. Optional: lab **Fn readout** at G110 only (one number) to validate κ≈0.67 vs geometry.",
        ]
    )

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] SSIM diagnosis -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
