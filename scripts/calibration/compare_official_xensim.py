#!/usr/bin/env python3
"""Compare official XenseSim FemSensor with ViTacSim on saved nut depths.

Run in the environment containing the official wheel::

    conda run --no-capture-output -n xensim-official-1.0 \
      python scripts/calibration/compare_official_xensim.py

The vendor wheel and its non-commercial assets are deliberately kept outside
Git under ``third_party/xensim_official_v1.0.0``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from xensim import PROJ_DIR
from xensim.core import FemSensor


CASES = ("G010", "G030", "G060", "G110", "G160", "G210")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {path}")


def fit_size(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if image.shape[:2] == shape:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float32)
    pred = prediction.astype(np.float32)
    error = pred - ref
    mse = float(np.mean(error * error))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "psnr_db": float(20.0 * np.log10(255.0 / max(np.sqrt(mse), 1e-12))),
    }


def signal_energy(image: np.ndarray) -> float:
    return float(np.mean(np.abs(image.astype(np.float32))))


def marker_stats(displacement_px: np.ndarray) -> dict[str, float]:
    magnitude = np.linalg.norm(displacement_px.reshape(-1, 2), axis=1)
    top_count = min(10, magnitude.size)
    return {
        "mean_px": float(magnitude.mean()),
        "top10_mean_px": float(np.sort(magnitude)[-top_count:].mean()),
        "max_px": float(magnitude.max()),
    }


def label(image: np.ndarray, text: str) -> np.ndarray:
    bar_height = 34
    canvas = np.zeros((image.shape[0] + bar_height, image.shape[1], 3), np.uint8)
    canvas[bar_height:] = image
    cv2.putText(
        canvas,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("logs/xense_marker_crisper"))
    parser.add_argument("--output", type=Path, default=Path("logs/xensim_official_comparison"))
    parser.add_argument("--nstep", type=int, default=3)
    parser.add_argument("--smooth-norm", type=int, default=5)
    parser.add_argument("--rgb-gain", type=float, default=1.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    real_root = Path("data/calibration/tactile/real/normal_force")
    real_bg = read_rgb(Path("data/calibration/tactile/advisor_processed/bg.jpg"))
    vitac_bg = read_rgb(
        args.input_root / "normal_force/no_contact/vitacsim/tactile_rgb.png"
    )

    sensor = FemSensor(
        calibrate_file=PROJ_DIR / "assets/fem/g1-ws_table.npz",
        fem_file=PROJ_DIR / "assets/fem/g1-ws.npz",
        depth_size=(100, 175),
        render_size=(400, 700),
        visible=False,
    )
    sensor.set_sim(
        show_marker=True,
        smooth_norm=args.smooth_norm,
        rgb_gain=args.rgb_gain,
    )

    no_contact = np.full((700, 400), 0.2, np.float32)
    sensor.step(no_contact, nstep=args.nstep)
    official_bg = fit_size(sensor.get_image(), real_bg.shape[:2])
    official_marker_rest = sensor.get_marker().copy()
    write_rgb(args.output / "official_no_contact.png", official_bg)

    results: dict[str, object] = {
        "official_version": "1.0.0",
        "nstep": args.nstep,
        "smooth_norm": args.smooth_norm,
        "rgb_gain": args.rgb_gain,
        "depth_input": "ViTacSim force-corrected height, converted m -> negative mm",
        "cases": {},
    }
    rows = []
    crop_rows = []
    for case in CASES:
        case_dir = args.input_root / "normal_force" / case / "vitacsim"
        height_m = np.load(case_dir / "tactile_height_corrected.npy").astype(np.float32)
        depth_mm = np.full(height_m.shape, 0.2, np.float32)
        contact = height_m > 0
        depth_mm[contact] = -height_m[contact] * 1000.0

        started = time.perf_counter()
        sensor.step(depth_mm, nstep=args.nstep)
        official = fit_size(sensor.get_image(), real_bg.shape[:2])
        official_marker = sensor.get_marker().copy()
        elapsed = time.perf_counter() - started

        real = read_rgb(real_root / case / "rgb.png")
        vitac = fit_size(read_rgb(case_dir / "tactile_rgb_corrected.png"), real.shape[:2])
        official = fit_size(official, real.shape[:2])
        real_bg_fit = fit_size(real_bg, real.shape[:2])
        vitac_bg_fit = fit_size(vitac_bg, real.shape[:2])
        official_bg_fit = fit_size(official_bg, real.shape[:2])

        # Background-subtracted signed RGB preserves the contact colour lobes
        # while removing each renderer's unrelated resting illumination.
        real_delta = real.astype(np.float32) - real_bg_fit.astype(np.float32)
        vitac_delta = vitac.astype(np.float32) - vitac_bg_fit.astype(np.float32)
        official_delta = official.astype(np.float32) - official_bg_fit.astype(np.float32)

        ys, xs = np.nonzero(contact)
        center_x = int((xs.min() + xs.max()) / 2)
        center_y = int((ys.min() + ys.max()) / 2)
        radius = 100
        roi = np.s_[
            max(0, center_y - radius) : min(real.shape[0], center_y + radius),
            max(0, center_x - radius) : min(real.shape[1], center_x + radius),
            :,
        ]

        # Official marker coordinates are millimetres. Convert their XY
        # displacement to pixels using the documented G1 gel dimensions.
        official_marker_delta_mm = official_marker - official_marker_rest
        official_marker_delta_px = np.stack(
            (
                official_marker_delta_mm[..., 0] * real.shape[1] / 17.3,
                official_marker_delta_mm[..., 1] * real.shape[0] / 29.14,
            ),
            axis=-1,
        ).reshape(-1, 2)
        real_marker_delta_px = np.load(real_root / case / "marker_displacement.npy")
        vitac_marker_delta_px = np.load(case_dir / "tactile_marker_displacement.npy")

        case_metrics = {
            "peak_depth_mm": float(height_m.max() * 1000.0),
            "official_seconds": elapsed,
            "full_rgb": {
                "vitacsim": metrics(real, vitac),
                "official_xensim": metrics(real, official),
            },
            "background_subtracted_rgb": {
                "vitacsim": metrics(real_delta, vitac_delta),
                "official_xensim": metrics(real_delta, official_delta),
            },
            "contact_roi_background_subtracted_rgb": {
                "vitacsim": metrics(real_delta[roi], vitac_delta[roi]),
                "official_xensim": metrics(real_delta[roi], official_delta[roi]),
            },
            "contact_roi_signal_energy": {
                "real": signal_energy(real_delta[roi]),
                "vitacsim": signal_energy(vitac_delta[roi]),
                "official_xensim": signal_energy(official_delta[roi]),
            },
            "marker_displacement": {
                "real": marker_stats(real_marker_delta_px),
                "vitacsim": marker_stats(vitac_marker_delta_px),
                "official_xensim": marker_stats(official_marker_delta_px),
            },
        }
        results["cases"][case] = case_metrics
        write_rgb(args.output / case / "official_xensim.png", official)
        write_rgb(args.output / case / "real.png", real)
        write_rgb(args.output / case / "vitacsim.png", vitac)
        np.save(args.output / case / "official_marker_displacement_px.npy", official_marker_delta_px)
        row = np.hstack(
            [
                label(real, f"{case} real"),
                label(vitac, "ViTacSim (ours)"),
                label(official, "official XenseSim FEM"),
            ]
        )
        write_rgb(args.output / case / "comparison.png", row)
        rows.append(row)
        crop_row = np.hstack(
            [
                label(real[roi], f"{case} real crop"),
                label(vitac[roi], "ViTacSim crop"),
                label(official[roi], "official FEM crop"),
            ]
        )
        write_rgb(args.output / case / "contact_crop_comparison.png", crop_row)
        crop_rows.append(crop_row)
        print(
            f"{case}: peak={case_metrics['peak_depth_mm']:.3f} mm, "
            f"official={elapsed:.3f}s, "
            f"delta MAE ours={case_metrics['background_subtracted_rgb']['vitacsim']['mae']:.3f}, "
            f"official={case_metrics['background_subtracted_rgb']['official_xensim']['mae']:.3f}"
        )

    write_rgb(args.output / "all_masses_real_ours_official.png", np.vstack(rows))
    write_rgb(args.output / "all_masses_contact_crops.png", np.vstack(crop_rows))
    with (args.output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(f"Wrote comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
