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
    lines = text.split('\n')
    bar_height = 34 * len(lines)
    canvas = np.zeros((image.shape[0] + bar_height, image.shape[1], 3), np.uint8)
    canvas[bar_height:] = image
    for i,line in enumerate(lines):
        width=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,0.56,1)[0][0]
        scale=0.56*min(1.,(image.shape[1]-16)/max(width,1))
        cv2.putText(canvas,line,(8,23+i*34),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("logs/xense_marker_crisper"))
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Optional previous ViTacSim run to include as a fourth-column comparison.",
    )
    parser.add_argument("--output", type=Path, default=Path("logs/xensim_official_comparison"))
    parser.add_argument("--nstep", type=int, default=3)
    parser.add_argument("--smooth-norm", type=int, default=5)
    parser.add_argument("--rgb-gain", type=float, default=1.0)
    parser.add_argument('--vitacsim-fps',type=Path,default=None)
    parser.add_argument('--benchmark-iterations',type=int,default=0)
    parser.add_argument('--benchmark-warmup',type=int,default=10)
    args = parser.parse_args()
    vitac_fps=json.loads(args.vitacsim_fps.read_text()) if args.vitacsim_fps else None

    args.output.mkdir(parents=True, exist_ok=True)
    real_root = Path("data/calibration/tactile/real/normal_force")
    real_bg = read_rgb(Path("data/calibration/tactile/advisor_processed/bg.jpg"))
    vitac_bg = read_rgb(
        args.input_root / "normal_force/no_contact/vitacsim/tactile_rgb.png"
    )
    baseline_bg = None
    if args.baseline_root is not None:
        baseline_bg = read_rgb(
            args.baseline_root / "normal_force/no_contact/vitacsim/tactile_rgb.png"
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
    if args.benchmark_iterations:
        for _ in range(args.benchmark_warmup):
            sensor.step(no_contact,nstep=args.nstep)
            sensor.get_image()
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
        "vitacsim_benchmark":vitac_fps,
        "official_benchmark_scope":"Batch=1, step(nstep=3)+get_image, no Isaac/PhysX/camera/disk. Host depth input, SDK RGB output; CPU wall time, no explicit vendor GPU synchronization API. Per-load warmup followed by repeated stationary input.",
    }
    rows = []
    crop_rows = []
    for case in CASES:
        case_dir = args.input_root / "normal_force" / case / "vitacsim"
        height_m = np.load(case_dir / "tactile_height_corrected.npy").astype(np.float32)
        depth_mm = np.full(height_m.shape, 0.2, np.float32)
        contact = height_m > 0
        depth_mm[contact] = -height_m[contact] * 1000.0
        if args.benchmark_iterations:
            for _ in range(args.benchmark_warmup):
                sensor.step(no_contact,nstep=args.nstep)

        started = time.perf_counter()
        sensor.step(depth_mm, nstep=args.nstep)
        official = fit_size(sensor.get_image(), real_bg.shape[:2])
        official_depth_mm = sensor.get_depth().copy()
        official_marker = sensor.get_marker().copy()
        elapsed = time.perf_counter() - started
        benchmark=None
        if args.benchmark_iterations:
            for _ in range(args.benchmark_warmup):
                sensor.step(depth_mm,nstep=args.nstep)
                sensor.get_image()
            samples=[]
            for _ in range(args.benchmark_iterations):
                start=time.perf_counter()
                sensor.step(depth_mm,nstep=args.nstep)
                sensor.get_image()
                samples.append(time.perf_counter()-start)
            benchmark=dict(fps=1/float(np.mean(samples)),mean_ms=float(np.mean(samples))*1000,
                           median_ms=float(np.median(samples))*1000,p95_ms=float(np.percentile(samples,95))*1000,
                           iterations=args.benchmark_iterations,warmup=args.benchmark_warmup)
            official=fit_size(sensor.get_image(),real_bg.shape[:2])
            official_depth_mm=sensor.get_depth().copy()
            official_marker=sensor.get_marker().copy()

        real = read_rgb(real_root / case / "rgb.png")
        vitac = fit_size(read_rgb(case_dir / "tactile_rgb_corrected.png"), real.shape[:2])
        baseline = None
        baseline_delta = None
        if args.baseline_root is not None:
            baseline = fit_size(
                read_rgb(
                    args.baseline_root
                    / "normal_force"
                    / case
                    / "vitacsim"
                    / "tactile_rgb_corrected.png"
                ),
                real.shape[:2],
            )
        official = fit_size(official, real.shape[:2])
        real_bg_fit = fit_size(real_bg, real.shape[:2])
        vitac_bg_fit = fit_size(vitac_bg, real.shape[:2])
        baseline_bg_fit = (
            fit_size(baseline_bg, real.shape[:2]) if baseline_bg is not None else None
        )
        official_bg_fit = fit_size(official_bg, real.shape[:2])

        # Background-subtracted signed RGB preserves the contact colour lobes
        # while removing each renderer's unrelated resting illumination.
        real_delta = real.astype(np.float32) - real_bg_fit.astype(np.float32)
        vitac_delta = vitac.astype(np.float32) - vitac_bg_fit.astype(np.float32)
        if baseline is not None and baseline_bg_fit is not None:
            baseline_delta = baseline.astype(np.float32) - baseline_bg_fit.astype(np.float32)
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
            "official_benchmark":benchmark,
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
        if baseline is not None and baseline_delta is not None:
            case_metrics["full_rgb"]["baseline_vitacsim"] = metrics(real, baseline)
            case_metrics["background_subtracted_rgb"]["baseline_vitacsim"] = metrics(
                real_delta, baseline_delta
            )
            case_metrics["contact_roi_background_subtracted_rgb"]["baseline_vitacsim"] = metrics(
                real_delta[roi], baseline_delta[roi]
            )
            case_metrics["contact_roi_signal_energy"]["baseline_vitacsim"] = signal_energy(
                baseline_delta[roi]
            )
        results["cases"][case] = case_metrics
        write_rgb(args.output / case / "official_xensim.png", official)
        write_rgb(args.output / case / "real.png", real)
        write_rgb(args.output / case / "vitacsim.png", vitac)
        np.save(args.output / case / "official_marker_displacement_px.npy", official_marker_delta_px)
        np.save(args.output / case / "official_deformed_depth_mm.npy", official_depth_mm)
        real_label=f'{case} Real'
        vitac_label='ViTacSim'
        official_label='Official XenseSim'
        if vitac_fps is not None and benchmark is not None:
            real_label+='\nFPS: N/A (still image)'
            vitac_label+=f" | {vitac_fps['cases'][case]['fps']:.1f} FPS\nTaxim+FOTS; batch=1"
            official_label+=f" | {benchmark['fps']:.1f} FPS\nFEM+RGB; nstep={args.nstep}"
        row_images = [label(real, real_label)]
        if baseline is not None:
            row_images.append(label(baseline, "previous ViTacSim"))
        row_images.extend(
            [label(vitac, vitac_label), label(official, official_label)]
        )
        row = np.hstack(row_images)
        write_rgb(args.output / case / "comparison.png", row)
        rows.append(row)
        crop_images = [label(real[roi], real_label)]
        if baseline is not None:
            crop_images.append(label(baseline[roi], "previous ViTacSim crop"))
        crop_images.extend(
            [label(vitac[roi], vitac_label), label(official[roi], official_label)]
        )
        crop_row = np.hstack(crop_images)
        write_rgb(args.output / case / "contact_crop_comparison.png", crop_row)
        crop_rows.append(crop_row)
        print(
            f"{case}: peak={case_metrics['peak_depth_mm']:.3f} mm, "
            f"official={elapsed:.3f}s, "
            f"delta MAE ours={case_metrics['background_subtracted_rgb']['vitacsim']['mae']:.3f}, "
            f"official={case_metrics['background_subtracted_rgb']['official_xensim']['mae']:.3f}"
        )

    montage_name = (
        "all_masses_real_previous_hybrid_official.png"
        if args.baseline_root is not None
        else "all_masses_real_ours_official.png"
    )
    write_rgb(args.output / montage_name, np.vstack(rows))
    write_rgb(args.output / "all_masses_contact_crops.png", np.vstack(crop_rows))
    with (args.output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(f"Wrote comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
