#!/usr/bin/env python3
"""Sweep Taxim optical softness without changing force-corrected depth."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sweep_xense_force_depth_gain import (  # noqa: E402
    CASES,
    ROI_XYXY,
    _align_frame_background,
    _clean_markers,
    _label,
    _load_rgb,
    _ssim_gray,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("logs/xense_marker_crisper"))
    parser.add_argument("--real-root", type=Path, default=Path("data/calibration/tactile/real/normal_force"))
    parser.add_argument(
        "--background",
        type=Path,
        default=Path("data/calibration/tactile/advisor_processed/bg_clean.jpg"),
    )
    parser.add_argument(
        "--marker-rest",
        type=Path,
        default=Path("data/calibration/tactile/advisor_processed/marker_rest.npy"),
    )
    parser.add_argument("--smoothing-kernels", type=int, nargs="+", default=(7, 15, 23, 31))
    parser.add_argument("--psf-kernels", type=int, nargs="+", default=(11, 21, 31))
    parser.add_argument("--normal-kernels", type=int, nargs="+", default=(1, 31, 61))
    parser.add_argument("--response-mesh-scale", type=int, default=1)
    parser.add_argument("--response-smooth-iterations", type=int, default=0)
    parser.add_argument("--response-mesh-blends", type=float, nargs="+", default=(1.0,))
    parser.add_argument(
        "--chroma-gain",
        type=float,
        default=-1.0,
        help="Override contact chroma gain; negative keeps the advisor default.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/xense_optical_softness_sweep"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app

    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import advisor_xense_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    background = _load_rgb(args.background.expanduser().resolve())
    marker_rest = np.load(args.marker_rest.expanduser().resolve()).astype(np.float32)
    x0, y0, x1, y1 = ROI_XYXY

    heights = []
    real_crops = []
    real_deltas = []
    for case in CASES:
        heights.append(
            np.load(
                args.input_root / "normal_force" / case / "vitacsim" / "tactile_height_corrected.npy"
            ).astype(np.float32)
        )
        real = _load_rgb(args.real_root / case / "rgb.png")
        real_clean, marker_mask = _clean_markers(real, marker_rest)
        real_aligned, _ = _align_frame_background(
            real_clean,
            background,
            marker_mask=marker_mask,
            center_xy=(200.0, 350.0),
            contact_radius_px=85.0,
        )
        crop = real_aligned[y0:y1, x0:x1]
        real_crops.append(crop)
        real_deltas.append(crop.astype(np.float32) - background[y0:y1, x0:x1].astype(np.float32))

    height_batch = torch.from_numpy(np.stack(heights)).to(args.device)
    base_cfg = advisor_xense_render_cfg(enable_marker_simulation=False, marker_pattern="none")
    rows = []
    predictions: dict[tuple[int, int, int, float], np.ndarray] = {}
    for smoothing in args.smoothing_kernels:
        for psf in args.psf_kernels:
            for normal in args.normal_kernels:
                for mesh_blend in args.response_mesh_blends:
                    cfg = base_cfg.replace(
                        taxim_smoothing_kernel_size=int(smoothing),
                        taxim_contact_psf_kernel_size=int(psf),
                        taxim_normal_smoothing_kernel_size=int(normal),
                        taxim_response_mesh_scale=int(args.response_mesh_scale),
                        taxim_response_mesh_smooth_iterations=int(args.response_smooth_iterations),
                        taxim_response_mesh_blend=float(mesh_blend),
                        taxim_contact_chroma_gain=(
                            float(args.chroma_gain)
                            if args.chroma_gain >= 0.0
                            else float(base_cfg.taxim_contact_chroma_gain)
                        ),
                    )
                    renderer = GelsightRender(cfg, args.device)
                    rendered = renderer.render(height_batch).detach().cpu().numpy()
                    per_case = []
                    aligned_predictions = []
                    for index, case in enumerate(CASES):
                        aligned, _ = _align_frame_background(
                            rendered[index],
                            background,
                            marker_mask=np.zeros(background.shape[:2], np.uint8),
                            center_xy=(200.0, 350.0),
                            contact_radius_px=85.0,
                        )
                        crop = aligned[y0:y1, x0:x1]
                        delta = (
                            crop.astype(np.float32)
                            - background[y0:y1, x0:x1].astype(np.float32)
                        )
                        real_delta = real_deltas[index]
                        per_case.append(
                            {
                                "case": case,
                                "rgb_mad": float(
                                    np.abs(real_crops[index].astype(np.float32) - crop).mean()
                                ),
                                "response_mae": float(np.abs(real_delta - delta).mean()),
                                "response_rmse": float(np.sqrt(np.mean((real_delta - delta) ** 2))),
                                "rgb_ssim": _ssim_gray(real_crops[index], crop),
                            }
                        )
                        aligned_predictions.append(crop)
                    key = (int(smoothing), int(psf), int(normal), float(mesh_blend))
                    predictions[key] = np.stack(aligned_predictions)
                    rows.append(
                        {
                            "smoothing_kernel": int(smoothing),
                            "psf_kernel": int(psf),
                            "normal_kernel": int(normal),
                            "response_mesh_scale": int(args.response_mesh_scale),
                            "response_smooth_iterations": int(args.response_smooth_iterations),
                            "response_mesh_blend": float(mesh_blend),
                            "chroma_gain": (
                                float(args.chroma_gain)
                                if args.chroma_gain >= 0.0
                                else float(base_cfg.taxim_contact_chroma_gain)
                            ),
                            "mean_rgb_mad": float(np.mean([x["rgb_mad"] for x in per_case])),
                            "mean_response_mae": float(
                                np.mean([x["response_mae"] for x in per_case])
                            ),
                            "mean_response_rmse": float(
                                np.mean([x["response_rmse"] for x in per_case])
                            ),
                            "mean_rgb_ssim": float(np.mean([x["rgb_ssim"] for x in per_case])),
                            "per_case": per_case,
                        }
                    )
                    del renderer
                    gc.collect()
                    torch.cuda.empty_cache()

    rows.sort(key=lambda item: (item["mean_response_mae"], item["mean_rgb_mad"]))
    best = rows[0]
    baseline = next(
        row
        for row in rows
        if row["smoothing_kernel"] == 7
        and row["psf_kernel"] == 11
        and row["normal_kernel"] == 1
        and row["response_mesh_blend"] == float(args.response_mesh_blends[0])
    )
    panel_candidates = rows[: min(4, len(rows))]
    panels = []
    for index, case in enumerate(CASES):
        columns = [_label(real_crops[index], f"{case} real marker-free")]
        for candidate in panel_candidates:
            key = (
                candidate["smoothing_kernel"],
                candidate["psf_kernel"],
                candidate["normal_kernel"],
                candidate["response_mesh_blend"],
            )
            columns.append(
                _label(
                    predictions[key][index],
                    f"b{key[3]:.2f} MAE={candidate['mean_response_mae']:.3f}",
                )
            )
        panels.append(np.concatenate(columns, axis=1))
    panel = np.concatenate(panels, axis=0)
    cv2.imwrite(str(out_dir / "best_contact_crops.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    payload = {"best": best, "baseline": baseline, "all": rows}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "baseline": baseline}, indent=2))
    print(f"[OK] {out_dir / 'best_contact_crops.png'}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
