#!/usr/bin/env python3
"""Sweep a global gain on saved force-corrected Xense height maps.

The gain is applied uniformly to the complete dense height map, so all spatial
and relative-depth relationships are preserved.  This isolates force-to-depth
amplitude from PhysX, circle detection, and Taxim polynomial calibration.
"""

from __future__ import annotations

import argparse
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

from advisor_image_utils import build_marker_inpaint_mask, detect_printed_markers  # noqa: E402
from build_xense_polycalib import _align_frame_background, _inpaint_with_mask  # noqa: E402

CASES = ("G010", "G030", "G060", "G110", "G160", "G210")
ROI_XYXY = (90, 240, 310, 460)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _clean_markers(rgb: np.ndarray, marker_rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tracked = build_marker_inpaint_mask(rgb, marker_rest, radius_px=2.5)
    direct, direct_radius = detect_printed_markers(rgb)
    if direct.shape[0] >= 80:
        direct_mask = build_marker_inpaint_mask(
            rgb,
            direct,
            radius_px=direct_radius,
            refine_centers=False,
        )
        core = cv2.bitwise_or(tracked, direct_mask)
    else:
        core = tracked
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(core, repair_kernel, iterations=1)
    return _inpaint_with_mask(rgb, mask), mask


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a_gray = a.astype(np.float64).mean(axis=-1)
    b_gray = b.astype(np.float64).mean(axis=-1)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a, mu_b = float(a_gray.mean()), float(b_gray.mean())
    var_a, var_b = float(a_gray.var()), float(b_gray.var())
    cov = float(((a_gray - mu_a) * (b_gray - mu_b)).mean())
    return float(
        ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2))
        / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2) + 1.0e-12)
    )


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 28), (18, 18, 18), thickness=-1)
    cv2.putText(
        bgr,
        text,
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _label_above(rgb: np.ndarray, text: str) -> np.ndarray:
    """Add a title strip without covering any full-frame sensor pixels."""
    header = np.full((28, rgb.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        header,
        text,
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((header, rgb), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("logs/xense_kref66_ballfixed_all"),
        help="Root containing normal_force/Gxxx/vitacsim saved outputs.",
    )
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path("data/calibration/tactile/real/normal_force"),
    )
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
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=(0.25, 0.35, 0.45, 0.55, 0.70, 1.0),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/xense_kref66_ballfixed_all/depth_gain_sweep"),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app

    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import advisor_xense_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender

    sweep_root = args.sweep_root.expanduser().resolve()
    real_root = args.real_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    background = _load_rgb(args.background.expanduser().resolve())
    marker_rest = np.load(args.marker_rest.expanduser().resolve()).astype(np.float32)
    x0, y0, x1, y1 = ROI_XYXY

    real_frames: list[np.ndarray] = []
    real_crops: list[np.ndarray] = []
    real_deltas: list[np.ndarray] = []
    height_maps: list[np.ndarray] = []
    original_peaks_mm: list[float] = []
    for case_id in CASES:
        real = _load_rgb(real_root / case_id / "rgb.png")
        real_clean, marker_mask = _clean_markers(real, marker_rest)
        real_aligned, _ = _align_frame_background(
            real_clean,
            background,
            marker_mask=marker_mask,
            center_xy=(200.0, 350.0),
            contact_radius_px=85.0,
        )
        real_frames.append(real_aligned)
        real_crops.append(real_aligned[y0:y1, x0:x1])
        real_deltas.append(
            real_aligned[y0:y1, x0:x1].astype(np.float32)
            - background[y0:y1, x0:x1].astype(np.float32)
        )
        height = np.load(
            sweep_root / "normal_force" / case_id / "vitacsim" / "tactile_height_corrected.npy"
        ).astype(np.float32)
        height_maps.append(height)
        original_peaks_mm.append(float(height.max() * 1000.0))

    cfg = advisor_xense_render_cfg(enable_marker_simulation=False, marker_pattern="none")
    renderer = GelsightRender(cfg, args.device)
    height_batch = torch.from_numpy(np.stack(height_maps, axis=0)).to(args.device)
    gain_rows: list[dict[str, object]] = []
    predictions: dict[float, list[np.ndarray]] = {}
    full_predictions: dict[float, list[np.ndarray]] = {}

    for gain in args.gains:
        rendered = renderer.render(height_batch * float(gain)).detach().cpu().numpy()
        gain_metrics: list[dict[str, float | str]] = []
        gain_predictions: list[np.ndarray] = []
        gain_full_predictions: list[np.ndarray] = []
        for index, case_id in enumerate(CASES):
            sim_aligned, _ = _align_frame_background(
                rendered[index],
                background,
                marker_mask=np.zeros(background.shape[:2], dtype=np.uint8),
                center_xy=(200.0, 350.0),
                contact_radius_px=85.0,
            )
            sim_crop = sim_aligned[y0:y1, x0:x1]
            sim_delta = sim_crop.astype(np.float32) - background[y0:y1, x0:x1].astype(np.float32)
            real_delta = real_deltas[index]
            gain_predictions.append(sim_crop)
            gain_full_predictions.append(sim_aligned)
            gain_metrics.append(
                {
                    "case_id": case_id,
                    "corrected_peak_mm": original_peaks_mm[index] * float(gain),
                    "rgb_ssim": _ssim_gray(real_crops[index], sim_crop),
                    "rgb_mad": float(
                        np.abs(real_crops[index].astype(np.float32) - sim_crop.astype(np.float32)).mean()
                    ),
                    "response_rmse": float(np.sqrt(np.mean((real_delta - sim_delta) ** 2))),
                    "response_mae": float(np.abs(real_delta - sim_delta).mean()),
                    "response_correlation": float(
                        np.corrcoef(real_delta.reshape(-1), sim_delta.reshape(-1))[0, 1]
                    ),
                }
            )
        predictions[float(gain)] = gain_predictions
        full_predictions[float(gain)] = gain_full_predictions
        gain_rows.append(
            {
                "gain": float(gain),
                "mean_rgb_ssim": float(np.mean([float(row["rgb_ssim"]) for row in gain_metrics])),
                "mean_rgb_mad": float(np.mean([float(row["rgb_mad"]) for row in gain_metrics])),
                "mean_response_rmse": float(
                    np.mean([float(row["response_rmse"]) for row in gain_metrics])
                ),
                "mean_response_correlation": float(
                    np.mean([float(row["response_correlation"]) for row in gain_metrics])
                ),
                "per_case": gain_metrics,
            }
        )

    panel_rows: list[np.ndarray] = []
    for case_index, case_id in enumerate(CASES):
        cells = [_label(real_crops[case_index], f"{case_id} real marker-free")]
        for gain in args.gains:
            peak = original_peaks_mm[case_index] * float(gain)
            cells.append(_label(predictions[float(gain)][case_index], f"gain={gain:.2f}, peak={peak:.2f}mm"))
        panel_rows.append(np.concatenate(cells, axis=1))
    panel = np.concatenate(panel_rows, axis=0)
    cv2.imwrite(str(out_dir / "depth_gain_contact_crops.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    full_panel_rows: list[np.ndarray] = []
    for case_index, case_id in enumerate(CASES):
        cells = [_label_above(real_frames[case_index], f"{case_id} real marker-free")]
        for gain in args.gains:
            peak = original_peaks_mm[case_index] * float(gain)
            cells.append(
                _label_above(
                    full_predictions[float(gain)][case_index],
                    f"gain={gain:.2f}, peak={peak:.2f}mm",
                )
            )
        full_panel_rows.append(np.concatenate(cells, axis=1))
    full_panel = np.concatenate(full_panel_rows, axis=0)
    cv2.imwrite(str(out_dir / "depth_gain_full_frames.png"), cv2.cvtColor(full_panel, cv2.COLOR_RGB2BGR))

    payload = {
        "cases": list(CASES),
        "roi_xyxy": list(ROI_XYXY),
        "original_peak_mm": dict(zip(CASES, original_peaks_mm)),
        "gains": gain_rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"gains": [{k: v for k, v in row.items() if k != "per_case"} for row in gain_rows]}, indent=2))
    print(f"[OK] {out_dir / 'depth_gain_contact_crops.png'}")
    print(f"[OK] {out_dir / 'depth_gain_full_frames.png'}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
