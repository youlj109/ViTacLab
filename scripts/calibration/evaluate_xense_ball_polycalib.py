#!/usr/bin/env python3
"""Compare Xense ball-calibration frames with images replayed through Taxim polycalib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher


def _sphere_height_map(
    *,
    image_hw: tuple[int, int],
    center_xy: tuple[float, float],
    contact_radius_px: float,
    ball_radius_px: float,
    mm_per_pixel: float,
) -> np.ndarray:
    """Spherical-cap indentation whose zero level is the annotated contact rim."""
    height, width = image_hw
    yy, xx = np.mgrid[:height, :width]
    cx, cy = center_xy
    radial_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    cap_radius = min(max(float(contact_radius_px), 1.0), float(ball_radius_px) * 0.999)
    inside = radial_sq <= cap_radius * cap_radius
    sphere_z = np.sqrt(np.clip(ball_radius_px * ball_radius_px - radial_sq, 0.0, None))
    rim_z = np.sqrt(max(ball_radius_px * ball_radius_px - cap_radius * cap_radius, 0.0))
    height_px = np.where(inside, np.maximum(sphere_z - rim_z, 0.0), 0.0)
    return (height_px * float(mm_per_pixel) / 1000.0).astype(np.float32)


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


def _crop_about_center(image: np.ndarray, center_xy: tuple[float, float], radius: float) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy = center_xy
    pad = int(np.ceil(radius + 12.0))
    x0, x1 = max(0, int(round(cx)) - pad), min(width, int(round(cx)) + pad + 1)
    y0, y1 = max(0, int(round(cy)) - pad), min(height, int(round(cy)) + pad + 1)
    return image[y0:y1, x0:x1]


def _label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (18, 18, 18), thickness=-1)
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _pure_polycalib_cfg(cfg):
    """Disable production-only effects so this isolates the fitted polynomial table."""
    return cfg.replace(
        taxim_height_scale=1.0,
        taxim_rgb_response_gain=1.0,
        taxim_smoothing_kernel_size=5,
        taxim_contact_edge_denoise_blend=0.0,
        taxim_gradient_edge_suppress=0.0,
        taxim_contact_chroma_gain=1.0,
        taxim_contact_soften_blend=0.0,
        taxim_edge_soften_strength=0.0,
        taxim_contact_psf_blend=0.0,
        taxim_contact_red_tilt_strength=0.0,
        taxim_contact_red_tilt_additive=0.0,
        taxim_response_mesh_scale=1,
        taxim_response_mesh_smooth_iterations=0,
        taxim_response_mesh_blend=0.0,
        taxim_response_load_gain_min=1.0,
        taxim_response_load_gain_max=1.0,
        taxim_final_response_psf_blend=0.0,
        taxim_illumination_blend=0.0,
        taxim_illumination_bias_blend=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/calibration/tactile/ball_calib_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/xense_ball_polycalib_evaluation"))
    parser.add_argument(
        "--polycalib",
        type=Path,
        default=None,
        help="Optional calibration table to evaluate instead of the installed Xense table.",
    )
    parser.add_argument(
        "--response-gains",
        type=float,
        nargs="+",
        default=(1.0,),
        help="Evaluate scalar gains on the marker-free Taxim response inside annotated contact disks.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app

    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import advisor_xense_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = np.load(data_dir / "dataPack.npz", allow_pickle=True)
    color_order = str(np.asarray(pack["color_order"]).item()) if "color_order" in pack.files else "BGR"
    real_clean = np.asarray(pack["imgs"], dtype=np.uint8)
    background = np.asarray(pack["f0"], dtype=np.uint8)
    if color_order.upper() == "BGR":
        real_clean = real_clean[..., ::-1].copy()
        background = background[..., ::-1].copy()
    elif color_order.upper() != "RGB":
        raise ValueError(f"Unsupported dataPack color_order: {color_order}")
    centers = np.asarray(pack["touch_center"], dtype=np.float64)
    radii = np.asarray(pack["touch_radius"], dtype=np.float64)
    marker_masks = np.asarray(pack["marker_masks"], dtype=bool)
    names = [str(x) for x in pack["names"]]

    mm_per_pixel = 17.5 / 400.0
    ball_radius_mm = 3.0
    ball_radius_px = ball_radius_mm / mm_per_pixel
    cfg = _pure_polycalib_cfg(
        advisor_xense_render_cfg(enable_marker_simulation=False, marker_pattern="none")
    )
    if args.polycalib is not None:
        polycalib = args.polycalib.expanduser().resolve()
        if not polycalib.is_file():
            raise FileNotFoundError(polycalib)
        cfg = cfg.replace(
            base_data_path="/",
            sensor_data_dir_name="",
            calib_path=str(polycalib),
            background_path=str((data_dir / "bg_clean.png").resolve()),
        )
    renderer = GelsightRender(cfg, args.device)

    predictions: list[np.ndarray] = []
    inverted_depth_predictions: list[np.ndarray] = []
    response_samples: list[tuple[np.ndarray, np.ndarray]] = []
    heights: list[np.ndarray] = []
    rows: list[dict[str, float | int | str | bool]] = []
    height, width = background.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    for index, (name, center, radius) in enumerate(zip(names, centers, radii)):
        height_map = _sphere_height_map(
            image_hw=(height, width),
            center_xy=(float(center[0]), float(center[1])),
            contact_radius_px=float(radius),
            ball_radius_px=ball_radius_px,
            mm_per_pixel=mm_per_pixel,
        )
        prediction = (
            renderer.render(torch.from_numpy(height_map).unsqueeze(0).to(args.device))[0]
            .detach()
            .cpu()
            .numpy()
        )
        # Diagnostic ablation: an inverted height preserves gradient magnitude
        # but rotates its direction by pi. The corrected calibration should make
        # this variant worse than the documented positive-penetration input.
        inverted_depth_prediction = (
            renderer.render(torch.from_numpy(-height_map).unsqueeze(0).to(args.device))[0]
            .detach()
            .cpu()
            .numpy()
        )
        predictions.append(prediction)
        inverted_depth_predictions.append(inverted_depth_prediction)
        heights.append(height_map)

        cx, cy = float(center[0]), float(center[1])
        valid_radius = min(float(radius), ball_radius_px * 0.999)
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= valid_radius * valid_radius) & ~marker_masks[index]
        real_delta = real_clean[index].astype(np.float64) - background.astype(np.float64)
        sim_delta = prediction.astype(np.float64) - background.astype(np.float64)
        response_samples.append((real_delta[disk], sim_delta[disk]))
        inverted_depth_delta = inverted_depth_prediction.astype(np.float64) - background.astype(np.float64)
        response_rmse = float(np.sqrt(np.mean((real_delta[disk] - sim_delta[disk]) ** 2)))
        response_mae = float(np.mean(np.abs(real_delta[disk] - sim_delta[disk])))
        inverted_depth_rmse = float(
            np.sqrt(np.mean((real_delta[disk] - inverted_depth_delta[disk]) ** 2))
        )
        inverted_depth_mae = float(np.mean(np.abs(real_delta[disk] - inverted_depth_delta[disk])))
        if int(disk.sum()) >= 2:
            corr = float(np.corrcoef(real_delta[disk].reshape(-1), sim_delta[disk].reshape(-1))[0, 1])
            inverted_depth_corr = float(
                np.corrcoef(
                    real_delta[disk].reshape(-1),
                    inverted_depth_delta[disk].reshape(-1),
                )[0, 1]
            )
        else:
            corr = float("nan")
            inverted_depth_corr = float("nan")
        real_crop = _crop_about_center(real_clean[index], (cx, cy), ball_radius_px)
        sim_crop = _crop_about_center(prediction, (cx, cy), ball_radius_px)
        inverted_depth_crop = _crop_about_center(
            inverted_depth_prediction, (cx, cy), ball_radius_px
        )
        rows.append(
            {
                "frame_index": index,
                "file": name,
                "center_x": cx,
                "center_y": cy,
                "annotated_radius_px": float(radius),
                "physical_ball_radius_px": ball_radius_px,
                "radius_was_clipped": bool(radius > ball_radius_px * 0.999),
                "valid_fit_pixels": int(disk.sum()),
                "response_rmse": response_rmse,
                "response_mae": response_mae,
                "response_correlation": corr,
                "crop_ssim": _ssim_gray(real_crop, sim_crop),
                "inverted_depth_response_rmse": inverted_depth_rmse,
                "inverted_depth_response_mae": inverted_depth_mae,
                "inverted_depth_response_correlation": inverted_depth_corr,
                "inverted_depth_crop_ssim": _ssim_gray(real_crop, inverted_depth_crop),
            }
        )

    gain_sweep = []
    for gain in args.response_gains:
        real_values = np.concatenate([sample[0].reshape(-1, 3) for sample in response_samples])
        sim_values = np.concatenate([sample[1].reshape(-1, 3) for sample in response_samples])
        prediction_values = sim_values * float(gain)
        error = prediction_values - real_values
        gain_sweep.append(
            {
                "response_gain": float(gain),
                "response_mae": float(np.mean(np.abs(error))),
                "response_rmse": float(np.sqrt(np.mean(error * error))),
                "response_correlation": float(
                    np.corrcoef(real_values.reshape(-1), prediction_values.reshape(-1))[0, 1]
                ),
                "real_signal_energy": float(np.mean(np.abs(real_values))),
                "sim_signal_energy": float(np.mean(np.abs(prediction_values))),
            }
        )
    gain_sweep.sort(key=lambda item: (item["response_rmse"], item["response_mae"]))
    best_response_gain = float(gain_sweep[0]["response_gain"])

    rmse = np.asarray([float(row["response_rmse"]) for row in rows])
    order = np.argsort(rmse)
    representative = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    if 0 not in representative:
        representative.insert(0, 0)

    panel_rows: list[np.ndarray] = []
    for index in representative:
        raw_bgr = cv2.imread(str(data_dir / "ball" / names[index]), cv2.IMREAD_COLOR)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        real_rgb = real_clean[index]
        sim_rgb = np.clip(
            background.astype(np.float32)
            + (predictions[index].astype(np.float32) - background.astype(np.float32))
            * best_response_gain,
            0,
            255,
        ).astype(np.uint8)
        inverted_depth_rgb = inverted_depth_predictions[index]
        error = np.clip(np.abs(real_rgb.astype(np.int16) - sim_rgb.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        inverted_depth_error = np.clip(
            np.abs(real_rgb.astype(np.int16) - inverted_depth_rgb.astype(np.int16)) * 4,
            0,
            255,
        ).astype(np.uint8)
        height_norm = np.clip(heights[index] / max(float(heights[index].max()), 1.0e-9), 0.0, 1.0)
        height_rgb = cv2.applyColorMap((height_norm * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        height_rgb = cv2.cvtColor(height_rgb, cv2.COLOR_BGR2RGB)
        row = np.concatenate(
            [
                _label(raw_rgb, f"{names[index]} raw real"),
                _label(real_rgb, "marker-clean + bg-aligned real"),
                _label(sim_rgb, f"+depth replay x{best_response_gain:.2f}"),
                _label(inverted_depth_rgb, "inverted-depth replay (diagnostic)"),
                _label(error, "|real-current| x4"),
                _label(inverted_depth_error, "|real-inverted-depth| x4"),
                _label(height_rgb, "sphere height"),
            ],
            axis=1,
        )
        panel_rows.append(row)
    panel = np.concatenate(panel_rows, axis=0)
    cv2.imwrite(str(out_dir / "ball_real_vs_polycalib.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    inverted_depth_rmse = np.asarray([float(row["inverted_depth_response_rmse"]) for row in rows])
    summary = {
        "frame_count": len(rows),
        "data_pack_color_order": color_order,
        "ball_radius_mm": ball_radius_mm,
        "mm_per_pixel": mm_per_pixel,
        "physical_ball_radius_px": ball_radius_px,
        "annotated_radius_min_px": float(radii.min()),
        "annotated_radius_median_px": float(np.median(radii)),
        "annotated_radius_max_px": float(radii.max()),
        "radius_clipped_frame_count": int(np.sum(radii > ball_radius_px * 0.999)),
        "annotation_crosses_image_frame_count": int(
            np.sum(
                (centers[:, 0] - radii < 0)
                | (centers[:, 0] + radii >= width)
                | (centers[:, 1] - radii < 0)
                | (centers[:, 1] + radii >= height)
            )
        ),
        "response_rmse_mean": float(rmse.mean()),
        "response_rmse_median": float(np.median(rmse)),
        "response_rmse_max": float(rmse.max()),
        "response_correlation_mean": float(
            np.nanmean([float(row["response_correlation"]) for row in rows])
        ),
        "crop_ssim_mean": float(np.mean([float(row["crop_ssim"]) for row in rows])),
        "best_response_gain": best_response_gain,
        "response_gain_sweep": gain_sweep,
        "inverted_depth_response_rmse_mean": float(inverted_depth_rmse.mean()),
        "inverted_depth_response_rmse_median": float(np.median(inverted_depth_rmse)),
        "inverted_depth_response_rmse_max": float(inverted_depth_rmse.max()),
        "inverted_depth_response_correlation_mean": float(
            np.nanmean([float(row["inverted_depth_response_correlation"]) for row in rows])
        ),
        "inverted_depth_crop_ssim_mean": float(
            np.mean([float(row["inverted_depth_crop_ssim"]) for row in rows])
        ),
        "representative_frame_indices": representative,
        "per_frame": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_frame"}, indent=2))
    print(f"[OK] comparison: {out_dir / 'ball_real_vs_polycalib.png'}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
