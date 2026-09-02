#!/usr/bin/env python3
"""Joint fit of ViTacSim Taxim RGB + marker params against real tactile captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    ADVISOR_MARKER_PATTERN,
    ADVISOR_REAL_TO_SIM_NF,
    ADVISOR_WEIGHT_CASES,
    CALIB_SCHEMA,
    CALIB_SCHEMA_ADVISOR,
    WEIGHT_CASES,
    default_real_root,
    default_sim_root,
    load_advisor_real_nf_cases,
    load_lateral_cases,
    load_mapped_sim_nf_for_advisor,
    load_nf_cases,
    load_bg_from_path,
    marker_curve_loss,
    rgb_loss_l1,
    repo_root,
)


def _eval_params(
    *,
    sim_nf: dict,
    real_nf: dict,
    sim_lat: dict,
    real_lat: dict,
    nf_ids: tuple[str, ...],
    bg: np.ndarray | None,
    displacement_gain: float,
    rgb_scale: float,
    alpha_rgb: float,
    beta_marker: float,
    use_lateral: bool,
) -> dict[str, Any]:
    rgb_losses: list[float] = []
    per_case_rgb: dict[str, float] = {}
    for cid in ["no_contact", *nf_ids]:
        s = sim_nf.get(cid)
        r = real_nf.get(cid)
        if s is None or r is None or s.rgb is None or r.rgb is None:
            continue
        loss = rgb_loss_l1(s.rgb, r.rgb, bg, rgb_scale=rgb_scale)
        rgb_losses.append(loss)
        per_case_rgb[cid] = loss

    nf_marker = marker_curve_loss(sim_nf, real_nf, nf_ids, displacement_gain=displacement_gain)
    lat_marker = float("nan")
    if use_lateral:
        lat_ids = [k for k in sim_lat if k.startswith("W100")]
        lat_marker = marker_curve_loss(sim_lat, real_lat, lat_ids, displacement_gain=displacement_gain)

    rgb_mean = float(np.mean(rgb_losses)) if rgb_losses else float("nan")
    marker_terms = [x for x in (nf_marker, lat_marker if use_lateral else float("nan")) if not np.isnan(x)]
    marker_mean = float(np.mean(marker_terms)) if marker_terms else float("nan")

    if np.isnan(rgb_mean):
        total = marker_mean
    elif np.isnan(marker_mean):
        total = rgb_mean
    else:
        total = alpha_rgb * rgb_mean + beta_marker * marker_mean

    return {
        "loss_total": float(total),
        "loss_rgb_mean": rgb_mean,
        "loss_rgb_per_case": per_case_rgb,
        "loss_marker_nf": nf_marker,
        "loss_marker_lateral": lat_marker if use_lateral else None,
        "rgb_cases_used": len(rgb_losses),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit ViTacSim RGB + marker calibration params.")
    parser.add_argument("--sim-root", type=str, default=str(default_sim_root()))
    parser.add_argument("--real-root", type=str, default=str(default_real_root()))
    parser.add_argument("--sensor-mode", type=str, default="vitacsim")
    parser.add_argument("--bg-path", type=str, default="", help="bg.jpg for RGB diff loss (optional).")
    parser.add_argument("--sim-only", action="store_true", help="Use sim as pseudo-real (pipeline test).")
    parser.add_argument(
        "--profile",
        type=str,
        default="advisor",
        choices=("cylinder", "advisor"),
        help="advisor=G010..G210 vs mapped sim; cylinder=W200..W010 + lateral",
    )
    parser.add_argument("--alpha-rgb", type=float, default=1.0)
    parser.add_argument("--beta-marker", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/fitted_params.json"),
    )
    args = parser.parse_args()

    sim_root = Path(args.sim_root).expanduser().resolve()
    real_root = Path(args.real_root).expanduser().resolve()
    use_lateral = args.profile == "cylinder"
    nf_ids = WEIGHT_CASES if args.profile == "cylinder" else ADVISOR_WEIGHT_CASES

    if args.profile == "advisor":
        sim_nf = load_mapped_sim_nf_for_advisor(sim_root, sensor_mode=args.sensor_mode)
        real_nf = load_advisor_real_nf_cases(real_root) if not args.sim_only else sim_nf
    else:
        sim_nf = load_nf_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode)
        real_nf = load_nf_cases(real_root, prefix="real") if not args.sim_only else sim_nf

    sim_lat = load_lateral_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode) if use_lateral else {}
    real_lat = load_lateral_cases(real_root, prefix="real") if use_lateral and not args.sim_only else {}

    if args.sim_only:
        print("[WARN] --sim-only: using sim as pseudo-real.")

    bg = None
    if args.bg_path:
        bg = load_bg_from_path(Path(args.bg_path).expanduser().resolve())
    if bg is None:
        for candidate in (
            repo_root() / "data/calibration/tactile/advisor_processed/bg_clean.jpg",
            repo_root() / "data/calibration/tactile/advisor_processed/bg.jpg",
            repo_root()
            / "source"
            / "ViTacLab"
            / "ViTacLab"
            / "assets"
            / "sensor"
            / "tacsl_sensor"
            / "gelsight_r15_data"
            / "bg.jpg",
        ):
            bg = load_bg_from_path(candidate)
            if bg is not None:
                break

    gain_grid = [
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
    ]
    rgb_scale_grid = [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for g in gain_grid:
        for rs in rgb_scale_grid:
            metrics = _eval_params(
                sim_nf=sim_nf,
                real_nf=real_nf,
                sim_lat=sim_lat,
                real_lat=real_lat,
                nf_ids=nf_ids,
                bg=bg,
                displacement_gain=g,
                rgb_scale=rs,
                alpha_rgb=float(args.alpha_rgb),
                beta_marker=float(args.beta_marker),
                use_lateral=use_lateral,
            )
            if np.isnan(metrics["loss_total"]):
                continue
            row = {"marker_displacement_gain": g, "rgb_diff_scale": rs, **metrics}
            grid_rows.append(row)
            if best is None or row["loss_total"] < best["loss_total"]:
                best = row

    if best is None:
        print("[ERROR] No valid sim/real pairs found.", file=sys.stderr)
        return 1

    default_gain = 0.35
    default_scale = 1.0
    default_metrics = _eval_params(
        sim_nf=sim_nf,
        real_nf=real_nf,
        sim_lat=sim_lat,
        real_lat=real_lat,
        nf_ids=nf_ids,
        bg=bg,
        displacement_gain=default_gain,
        rgb_scale=default_scale,
        alpha_rgb=float(args.alpha_rgb),
        beta_marker=float(args.beta_marker),
        use_lateral=use_lateral,
    )

    schema = CALIB_SCHEMA if args.profile == "cylinder" else CALIB_SCHEMA_ADVISOR
    real_marker_pattern = "gelsight" if args.profile == "cylinder" else ADVISOR_MARKER_PATTERN
    result = {
        "schema": schema,
        "profile": args.profile,
        "sim_root": str(sim_root),
        "real_root": str(real_root),
        "sim_only": bool(args.sim_only),
        "real_marker_pattern": real_marker_pattern,
        "real_to_sim_nf": ADVISOR_REAL_TO_SIM_NF if args.profile == "advisor" else None,
        "recommended_gelsight_render_cfg": {
            "enable_marker_simulation": True,
            "marker_pattern": real_marker_pattern,
            "marker_displacement_gain": best["marker_displacement_gain"],
        },
        "sim_marker_pattern_note": (
            "Advisor sweep uses PROFILE=advisor, MARKER_PATTERN=xense, M2 nut, 400x700."
            if args.profile == "advisor"
            else None
        ),
        "recommended_force_render_k_ref_scale": best["rgb_diff_scale"],
        "best_metrics": best,
        "default_metrics": {
            "marker_displacement_gain": default_gain,
            "rgb_diff_scale": default_scale,
            **default_metrics,
        },
        "convergence": {
            "loss_improved": bool(best["loss_total"] < default_metrics["loss_total"]),
            "loss_default": default_metrics["loss_total"],
            "loss_best": best["loss_total"],
            "relative_reduction": float(
                (default_metrics["loss_total"] - best["loss_total"]) / max(default_metrics["loss_total"], 1e-9)
            ),
        },
        "grid_search": grid_rows,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] fitted params -> {out_path}")
    print(
        f"[BEST] gain={best['marker_displacement_gain']:.3f} rgb_scale={best['rgb_diff_scale']:.3f} "
        f"loss={best['loss_total']:.6f} (rgb_cases={best['rgb_cases_used']}) "
        f"improved={result['convergence']['loss_improved']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
