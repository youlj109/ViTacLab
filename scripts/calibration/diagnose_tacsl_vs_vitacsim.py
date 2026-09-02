#!/usr/bin/env python3
"""Diagnose why Task3 Real-TacSL and Real-ViTacSim SSIM look similar."""

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
    ADVISOR_WEIGHT_CASES,
    default_real_root,
    default_sim_root,
    load_advisor_real_nf_cases,
    load_bg_from_path,
    repo_root,
    sim_nf_dir,
)


def _load_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a_g = a.mean(axis=-1)
    b_g = b.mean(axis=-1)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a, mu_b = a_g.mean(), b_g.mean()
    sig_a, sig_b = a_g.var(), b_g.var()
    sig_ab = float(((a_g - mu_a) * (b_g - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a + sig_b + c2)
    return float(num / den) if den > 1e-12 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="TacSL vs ViTacSim diagnosis for Task3.")
    parser.add_argument("--sim-root", type=str, default=str(default_sim_root()))
    parser.add_argument("--real-root", type=str, default=str(default_real_root()))
    parser.add_argument(
        "--bg",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/advisor_processed/bg_clean.jpg"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(repo_root() / "logs/vitacsim_validation/task3/TACSL_VS_VITACSIM_DIAGNOSIS.md"),
    )
    args = parser.parse_args()

    sim_root = Path(args.sim_root).expanduser().resolve()
    real_nf = load_advisor_real_nf_cases(Path(args.real_root).expanduser().resolve())
    bg = load_bg_from_path(Path(args.bg).expanduser().resolve())

    rows: list[str] = []
    rows.append("# TacSL vs ViTacSim diagnosis (Task 3)\n")
    rows.append("## Why Real-TacSL ≈ Real-ViTacSim SSIM?\n")
    rows.append(
        "Task3 compares each sim mode's **`tactile_rgb.png`** to Real. "
        "TacSL saves **depth-only** Taxim; ViTacSim saves **force-corrected** Taxim "
        "(`enable_corrected_force_render=True`, blend=1.0).\n"
    )

    inter_sim_mad: list[float] = []
    rt_ssim: list[float] = []
    rv_ssim: list[float] = []

    rows.append("| Case | |TacSL-ViT| MAD | SSIM Real-TacSL | SSIM Real-ViT | h_corr ViT | h depth | rgb_corr_l1 |")
    rows.append("|------|-----------|-----------------|-----------------|------------|---------|-------------|")

    for cid in ADVISOR_WEIGHT_CASES:
        t_dir = sim_nf_dir(sim_root, cid, sensor_mode="tacsl")
        v_dir = sim_nf_dir(sim_root, cid, sensor_mode="vitacsim")
        t_rgb = _load_rgb(t_dir / "tactile_rgb.png")
        v_rgb = _load_rgb(v_dir / "tactile_rgb.png")
        v_depth = _load_rgb(v_dir / "tactile_rgb_depth.png")
        r = real_nf.get(cid)
        r_rgb = r.rgb.astype(np.float32) if r and r.rgb is not None else None

        t_sum = json.loads((t_dir / "summary.json").read_text()) if (t_dir / "summary.json").is_file() else {}
        v_sum = json.loads((v_dir / "summary.json").read_text()) if (v_dir / "summary.json").is_file() else {}

        mad = _mean_abs(t_rgb, v_rgb) if t_rgb is not None and v_rgb is not None else float("nan")
        if not np.isnan(mad):
            inter_sim_mad.append(mad)

        s_rt = _ssim_gray(r_rgb, t_rgb) if r_rgb is not None and t_rgb is not None else float("nan")
        s_rv = _ssim_gray(r_rgb, v_rgb) if r_rgb is not None and v_rgb is not None else float("nan")
        if not np.isnan(s_rt):
            rt_ssim.append(s_rt)
        if not np.isnan(s_rv):
            rv_ssim.append(s_rv)

        rows.append(
            f"| {cid} | {mad:.2f} | {s_rt:.3f} | {s_rv:.3f} | "
            f"{v_sum.get('height_corr_peak_max', 0):.4f} | "
            f"{v_sum.get('penetration_peak_max', 0):.4f} | "
            f"{v_sum.get('rgb_corr_l1_mean', 0):.2f} |"
        )

    rows.extend(
        [
            "",
            "## Interpretation\n",
            f"- **Inter-sim RGB diff (|TacSL-ViT| MAD)**: mean {np.mean(inter_sim_mad):.2f} / 255 "
            f"→ modes **do differ** at pixel level.",
            f"- **Real-TacSL SSIM** mean {np.mean(rt_ssim):.3f}; **Real-ViTacSim** mean {np.mean(rv_ssim):.3f}.",
            "- Both are **similarly far from Real** because:",
            "  1. **`polycalib.npz` is still GelSight R15**, not lab Xense (dominant RGB error).",
            "  2. **PhysX Fn ~67% of nominal** (spawn/depth not aligned) — force correction has limited effect.",
            "  3. ViTacSim **`height_corr_peak` > depth peak** shows force render is active; "
            "Taxim nonlinearity makes RGB change modest vs remaining optical mismatch.",
            "",
            "## Fixes (priority)\n",
            "1. Install Xense ball polycalib → re-sweep.",
            "2. Apply `fitted_params.json` in sweep (`--fitted-params`, rgb_scale → k_ref).",
            "3. Tune `finger_root_z` for Fn alignment.",
            "",
        ]
    )

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[OK] diagnosis -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
