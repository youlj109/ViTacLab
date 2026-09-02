#!/usr/bin/env python3
"""Score sim vs real contact centroid alignment for contact_xy_probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from calibration_io import load_bg_from_path, rgb_diff_magnitude, repo_root
from report_task3_validation import _diff_vis, _ssim_gray


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _contact_centroid(rgb: np.ndarray, bg: np.ndarray) -> tuple[float, float]:
    mag = rgb_diff_magnitude(rgb, bg)
    thr = float(np.percentile(mag, 95))
    mask = mag > thr
    if int(mask.sum()) < 10:
        mask = mag > float(np.percentile(mag, 90))
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def score_run(*, sim_rgb: Path, real_rgb: Path, bg: Path) -> dict[str, float]:
    bg_arr = load_bg_from_path(bg) if bg.is_file() else _load_rgb(bg)
    real = _load_rgb(real_rgb)
    sim = _load_rgb(sim_rgb)
    rc = _contact_centroid(real, bg_arr)
    sc = _contact_centroid(sim, bg_arr)
    return {
        "centroid_dx": sc[0] - rc[0],
        "centroid_dy": sc[1] - rc[1],
        "ssim_diff": _ssim_gray(_diff_vis(real, bg_arr), _diff_vis(sim, bg_arr)),
    }


if __name__ == "__main__":
    root = repo_root()
    row = score_run(
        sim_rgb=root / "logs/vitacsim_calibration/sweep/normal_force/G110/vitacsim/tactile_rgb_corrected.png",
        real_rgb=root / "data/calibration/tactile/real/normal_force/G110/rgb.png",
        bg=root / "data/calibration/tactile/advisor_processed/bg_clean.jpg",
    )
    print(row)
