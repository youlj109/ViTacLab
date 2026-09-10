#!/usr/bin/env python3
"""Build Xense polycalib.npz from ball_calib_raw/ (auto-annotate + Taxim fit + install).

Pipeline:
  1. Optional: import_ball_calib_video.py
  2. Auto-generate dataPack.npz (circle detection; Taxim GUI not required)
  3. Detect/inpaint printed markers and exclude their pixels from the optical fit
  4. Run Taxim polyTableCalib.py with Xense sensor params
  5. Install polycalib (+ the paired clean background) into xense_lab_data/

Usage::

    python3 scripts/calibration/build_xense_polycalib.py
    python3 scripts/calibration/build_xense_polycalib.py --skip-import
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import repo_root  # noqa: E402
from advisor_image_utils import (  # noqa: E402
    build_advisor_marker_rest,
    build_marker_inpaint_mask,
    detect_printed_markers,
    measure_marker_residual,
)
from import_ball_calib_video import (  # noqa: E402
    XENSE_BALL_RADIUS_MM,
    XENSE_SENSING_MM,
    _contact_blob,
    _load_rgb,
    _save_rgb,
)

TAXIM_NUM_BINS = 125


def _default_taxim_repo() -> Path:
    env = __import__("os").environ.get("TAXIM_REPO", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "third_party" / "Taxim"


def _default_background_paths() -> tuple[Path, Path]:
    advisor_dir = repo_root() / "data" / "calibration" / "tactile" / "advisor_processed"
    return advisor_dir / "bg.jpg", advisor_dir / "bg_clean.jpg"


def _default_marker_rest_path() -> Path:
    return repo_root() / "data" / "calibration" / "tactile" / "advisor_processed" / "marker_rest.npy"


def _list_ball_images(data_dir: Path) -> list[Path]:
    ball_dir = data_dir / "ball"
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(p for p in ball_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No ball images under {ball_dir}")
    return files


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _inpaint_with_mask(rgb: np.ndarray, mask: np.ndarray, *, radius: int = 10) -> np.ndarray:
    """Inpaint a marker mask while preserving RGB channel order."""
    import cv2

    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cleaned_bgr = cv2.inpaint(bgr, mask.astype(np.uint8), int(radius), cv2.INPAINT_NS)
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask > 0).astype(np.uint8)[:, :, None] * 255, 3, axis=2)


def _align_frame_background(
    frame_rgb: np.ndarray,
    background_rgb: np.ndarray,
    *,
    marker_mask: np.ndarray,
    center_xy: tuple[float, float],
    contact_radius_px: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Remove frame-wise low-frequency exposure drift using non-contact pixels."""
    height, width = frame_rgb.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    cx, cy = center_xy
    contact_exclusion = (xx - cx) ** 2 + (yy - cy) ** 2 <= (1.45 * contact_radius_px) ** 2
    valid = ~contact_exclusion & ~marker_mask.astype(bool)
    # A stride keeps the robust fit inexpensive while retaining broad image coverage.
    sample = valid & ((xx % 3) == 0) & ((yy % 3) == 0)
    xn = (xx.astype(np.float64) - 0.5 * (width - 1)) / max(0.5 * width, 1.0)
    yn = (yy.astype(np.float64) - 0.5 * (height - 1)) / max(0.5 * height, 1.0)
    design = np.stack((xn * xn, yn * yn, xn * yn, xn, yn, np.ones_like(xn)), axis=-1)
    delta = frame_rgb.astype(np.float64) - background_rgb.astype(np.float64)
    bias = np.zeros_like(delta)
    fit_rmse: list[float] = []

    for channel in range(3):
        a = design[sample]
        b = delta[:, :, channel][sample]
        keep = np.ones(b.shape, dtype=bool)
        coeff = np.zeros(6, dtype=np.float64)
        for _ in range(3):
            coeff, *_ = np.linalg.lstsq(a[keep], b[keep], rcond=None)
            residual = b - a @ coeff
            median = float(np.median(residual[keep]))
            mad = float(np.median(np.abs(residual[keep] - median)))
            robust_sigma = max(1.4826 * mad, 0.5)
            updated = np.abs(residual - median) <= 3.5 * robust_sigma
            if int(updated.sum()) < 100 or np.array_equal(updated, keep):
                break
            keep = updated
        bias[:, :, channel] = design @ coeff
        fit_rmse.append(float(np.sqrt(np.mean((b[keep] - (a[keep] @ coeff)) ** 2))))

    aligned = np.clip(frame_rgb.astype(np.float64) - bias, 0.0, 255.0).astype(np.uint8)
    return aligned, {
        "background_bias_abs_mean": float(np.mean(np.abs(bias))),
        "background_fit_rmse_mean": float(np.mean(fit_rmse)),
    }


def _build_datapack(
    data_dir: Path,
    *,
    pixmm: float,
    bg_raw_path: Path | None = None,
    bg_clean_path: Path | None = None,
    marker_rest_path: Path | None = None,
) -> Path:
    import cv2

    default_bg_raw, default_bg_clean = _default_background_paths()
    bg_raw_path = (bg_raw_path or default_bg_raw).expanduser().resolve()
    bg_clean_path = (bg_clean_path or default_bg_clean).expanduser().resolve()
    marker_rest_path = (marker_rest_path or _default_marker_rest_path()).expanduser().resolve()
    if not bg_raw_path.is_file():
        raise FileNotFoundError(f"No-contact background with markers not found: {bg_raw_path}")
    if not bg_clean_path.is_file():
        raise FileNotFoundError(f"No-contact clean background not found: {bg_clean_path}")

    ball_paths = _list_ball_images(data_dir)

    bg_rgb = _load_rgb(bg_raw_path)
    bg_clean_rgb = _load_rgb(bg_clean_path)
    if bg_clean_rgb.shape != bg_rgb.shape:
        raise ValueError(
            f"Clean background shape {bg_clean_rgb.shape} does not match raw background {bg_rgb.shape}"
        )
    h, w = bg_rgb.shape[:2]
    if marker_rest_path.is_file():
        marker_rest = np.asarray(np.load(marker_rest_path), dtype=np.float32)
        if marker_rest.ndim != 2 or marker_rest.shape[1] != 2 or not np.isfinite(marker_rest).all():
            raise ValueError(f"Invalid marker rest array in {marker_rest_path}: {marker_rest.shape}")
        marker_radius_px = 2.5
    else:
        marker_rest, marker_radius_px = build_advisor_marker_rest(
            bg_rgb,
            pattern="xense",
            image_height=h,
            image_width=w,
        )
    bg_marker_mask = build_marker_inpaint_mask(
        bg_rgb,
        marker_rest,
        radius_px=marker_radius_px,
    )
    # The ViTacLab renderer interprets grad_r/g/b and its background as RGB.
    # Keep the custom data pack explicitly RGB end-to-end; OpenCV's original
    # Taxim GUI stored BGR while still naming channel 0 "r", which swaps red and
    # blue when the resulting table is consumed by this renderer.
    fit_background = bg_clean_rgb.copy()

    imgs: list[np.ndarray] = []
    imgs_unaligned: list[np.ndarray] = []
    marker_masks: list[np.ndarray] = []
    centers: list[list[float]] = []
    radii: list[float] = []
    names: list[str] = []
    records: list[dict] = []
    annotation_tiles: list[np.ndarray] = []
    diagnostics_dir = data_dir / "diagnostics" / "polycalib_marker_masking"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(bg_rgb, diagnostics_dir / "background_raw.png")
    _save_rgb(_mask_rgb(bg_marker_mask), diagnostics_dir / "background_marker_mask.png")
    _save_rgb(bg_clean_rgb, diagnostics_dir / "background_clean.png")
    _save_rgb(bg_clean_rgb, data_dir / "bg_clean.png")

    for frame_index, p in enumerate(ball_paths):
        frame_rgb = _load_rgb(p)
        if frame_rgb.shape != bg_rgb.shape:
            raise ValueError(
                f"Calibration frame {p} has shape {frame_rgb.shape}, expected {bg_rgb.shape}"
            )
        # Remove the printed pattern first. Detecting a circle on the raw frame
        # lets moving markers bias both the fitted center and radius.
        tracked_marker_mask = build_marker_inpaint_mask(
            frame_rgb,
            marker_rest,
            radius_px=marker_radius_px,
        )
        direct_markers, direct_marker_radius = detect_printed_markers(frame_rgb)
        if direct_markers.shape[0] >= 80:
            direct_marker_mask = build_marker_inpaint_mask(
                frame_rgb,
                direct_markers,
                radius_px=direct_marker_radius,
                refine_centers=False,
            )
            marker_core_mask = cv2.bitwise_or(tracked_marker_mask, direct_marker_mask)
        else:
            marker_core_mask = tracked_marker_mask
        # Marker detection uses a tight core mask. Expand the final repair/fit
        # exclusion once, after merging tracked and directly detected centers,
        # so the gray optical fringe is removed without double-padding two masks.
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        marker_mask = cv2.dilate(marker_core_mask, repair_kernel, iterations=1)
        frame_clean_rgb = _inpaint_with_mask(frame_rgb, marker_mask)
        marker_residual = measure_marker_residual(frame_clean_rgb, marker_rest)

        ball_radius_px = XENSE_BALL_RADIUS_MM / float(pixmm)
        (cy, cx), radius_px, diff_mean = _contact_blob(
            bg_clean_rgb,
            frame_clean_rgb,
            max_radius_px=ball_radius_px,
        )
        if radius_px < 3.0:
            h, w = frame_rgb.shape[:2]
            cy, cx, radius_px = h / 2.0, w / 2.0, max(radius_px, 8.0)

        frame_aligned_rgb, background_alignment = _align_frame_background(
            frame_clean_rgb,
            bg_clean_rgb,
            marker_mask=marker_mask,
            center_xy=(float(cx), float(cy)),
            contact_radius_px=float(radius_px),
        )

        # Official Taxim stores touch_center as OpenCV coordinates [x, y]. Both
        # the circle detector and the optical fit now use marker-free images.
        imgs.append(frame_aligned_rgb)
        imgs_unaligned.append(frame_clean_rgb)
        marker_masks.append(marker_mask > 0)
        centers.append([float(cx), float(cy)])
        radii.append(float(radius_px))
        names.append(p.name)
        records.append(
            {
                "file": p.name,
                "touch_center_yx": [cy, cx],
                "touch_radius_px": radius_px,
                "diff_mean": diff_mean,
                "ball_radius_mm": XENSE_BALL_RADIUS_MM,
                "pixmm": pixmm,
                "direct_marker_count": int(direct_markers.shape[0]),
                "marker_mask_fraction": float(np.mean(marker_mask > 0)),
                "post_inpaint_residual_markers": int(marker_residual["residual_markers"]),
                "post_inpaint_residual_ratio": float(marker_residual["residual_ratio"]),
                **background_alignment,
            }
        )

        # Montage the exact marker-free, background-aligned image entering the
        # optical fit so annotation review cannot be confused with raw inputs.
        overlay = frame_aligned_rgb.copy()
        cv2.circle(
            overlay,
            (int(round(cx)), int(round(cy))),
            int(round(radius_px)),
            (40, 255, 40),
            thickness=2,
        )
        cv2.drawMarker(
            overlay,
            (int(round(cx)), int(round(cy))),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        cv2.putText(
            overlay,
            f"{p.name} r={radius_px:.1f}px",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotation_tiles.append(cv2.resize(overlay, (140, 245), interpolation=cv2.INTER_AREA))

        if frame_index == 0:
            _save_rgb(frame_rgb, diagnostics_dir / "ball_000_raw.png")
            _save_rgb(_mask_rgb(marker_mask), diagnostics_dir / "ball_000_marker_mask.png")
            _save_rgb(frame_clean_rgb, diagnostics_dir / "ball_000_clean.png")
            _save_rgb(frame_aligned_rgb, diagnostics_dir / "ball_000_background_aligned.png")

    out = data_dir / "dataPack.npz"
    np.savez(
        out,
        f0=fit_background,
        f0_raw=bg_rgb,
        imgs=np.stack(imgs, axis=0),
        imgs_unaligned=np.stack(imgs_unaligned, axis=0),
        marker_masks=np.stack(marker_masks, axis=0),
        marker_rest_xy=marker_rest.astype(np.float32),
        marker_radius_px=np.asarray(marker_radius_px, dtype=np.float32),
        marker_mask_version=np.asarray("xense-v1"),
        touch_center=np.asarray(centers, dtype=np.float32),
        touch_radius=np.asarray(radii, dtype=np.float32),
        names=np.asarray(names),
        img_size=np.asarray(fit_background.shape),
        color_order=np.asarray("RGB"),
    )
    if annotation_tiles:
        columns = 10
        rows = []
        for start in range(0, len(annotation_tiles), columns):
            row = annotation_tiles[start : start + columns]
            if len(row) < columns:
                row.extend([np.zeros_like(annotation_tiles[0])] * (columns - len(row)))
            rows.append(np.concatenate(row, axis=1))
        _save_rgb(
            np.concatenate(rows, axis=0),
            diagnostics_dir / "contact_annotation_montage.png",
        )

    ann_path = data_dir / "auto_annotation.json"
    ann_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    frame_mask_fractions = np.asarray([r["marker_mask_fraction"] for r in records], dtype=np.float64)
    frame_residual_counts = np.asarray(
        [r["post_inpaint_residual_markers"] for r in records], dtype=np.int64
    )
    mask_report = {
        "marker_count": int(marker_rest.shape[0]),
        "marker_radius_px": float(marker_radius_px),
        "background_with_markers": str(bg_raw_path),
        "background_clean": str(bg_clean_path),
        "marker_rest": str(marker_rest_path) if marker_rest_path.is_file() else "detected_from_background",
        "background_mask_fraction": float(np.mean(bg_marker_mask > 0)),
        "frame_mask_fraction_min": float(np.min(frame_mask_fractions)),
        "frame_mask_fraction_mean": float(np.mean(frame_mask_fractions)),
        "frame_mask_fraction_max": float(np.max(frame_mask_fractions)),
        "post_inpaint_residual_markers_mean": float(np.mean(frame_residual_counts)),
        "post_inpaint_residual_markers_max": int(np.max(frame_residual_counts)),
        "post_inpaint_zero_residual_frame_count": int(np.count_nonzero(frame_residual_counts == 0)),
        "background_residual": measure_marker_residual(bg_clean_rgb, marker_rest),
        "fit_policy": "inpaint marker pixels, then exclude the same per-frame mask from polynomial fitting",
        "diagnostics_dir": str(diagnostics_dir),
    }
    mask_report_path = data_dir / "marker_mask_report.json"
    mask_report_path.write_text(json.dumps(mask_report, indent=2), encoding="utf-8")
    print(f"[OK] dataPack -> {out} ({len(imgs)} frames)")
    print(f"[OK] auto_annotation -> {ann_path}")
    print(
        "[OK] marker masking -> "
        f"{mask_report_path} ({marker_rest.shape[0]} markers, "
        f"mean frame coverage={np.mean(frame_mask_fractions):.2%})"
    )
    return out


def _run_poly_table_calib(
    data_dir: Path,
    taxim_repo: Path,
    *,
    pixmm: float,
    num_bins: int,
    deep_weight_gamma: float = 1.0,
    edge_weight_min: float = 0.20,
    edge_weight_power: float = 2.0,
    grad_table_smooth_sigma: float = 0.0,
) -> Path:
    """Run Taxim polyTableCalib with Xense params and NaN-safe polynomial fit."""
    import scipy.ndimage
    from scipy import interpolate
    from scipy.linalg import lstsq

    sys.path.insert(0, str(taxim_repo))
    import Basics.sensorParams as psp  # noqa: WPS433
    from Basics.Geometry import Circle  # noqa: WPS433

    # Patch sensor params for this run only.
    psp.ball_radius = float(XENSE_BALL_RADIUS_MM)
    psp.pixmm = float(pixmm)
    psp.numBins = int(num_bins)
    psp.h = 700
    psp.w = 400

    data_file = np.load(data_dir / "dataPack.npz", allow_pickle=True)
    f0 = data_file["f0"]
    imgs = data_file["imgs"]
    radius_record = data_file["touch_radius"]
    touch_center_record = data_file["touch_center"]
    marker_masks = (
        np.asarray(data_file["marker_masks"], dtype=bool)
        if "marker_masks" in data_file.files
        else np.zeros(imgs.shape[:3], dtype=bool)
    )
    if marker_masks.shape != imgs.shape[:3]:
        raise ValueError(
            f"marker_masks shape {marker_masks.shape} does not match calibration frames {imgs.shape[:3]}"
        )
    marker_mask_version = (
        str(np.asarray(data_file["marker_mask_version"]).item())
        if "marker_mask_version" in data_file.files
        else "none"
    )
    if "marker_masks" not in data_file.files:
        print("[WARN] dataPack has no marker_masks; fitting without marker exclusion")

    # dataPack.f0 is the verified marker-free no-contact image. The original
    # Taxim preprocessing was intended to suppress markers in its background;
    # blurring this already-clean reference creates a synthetic full-frame color
    # residual that the renderer can never reproduce consistently.
    bg_proc = f0.astype(np.float64)

    def _interpolate(img: np.ndarray) -> np.ndarray:
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]
        if newarr.size == 0:
            return np.zeros_like(img, dtype=np.float64)
        gd1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="nearest", fill_value=0)
        return np.nan_to_num(gd1, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_poly_params(xf: np.ndarray, yf: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        xf = np.asarray(xf, dtype=np.float64).ravel()
        yf = np.asarray(yf, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        mask = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(b)
        if int(mask.sum()) < 6:
            return np.zeros(6, dtype=np.float64)
        xf, yf, b = xf[mask], yf[mask], b[mask]
        a = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(xf.shape)]).T
        if w is not None:
            ww = np.asarray(w, dtype=np.float64).ravel()[mask]
            ww = np.nan_to_num(ww, nan=1.0, posinf=1.0, neginf=1.0)
            ww = np.clip(ww, 1.0e-6, None)
            sw = np.sqrt(ww)
            a = a * sw[:, None]
            b = b * sw
        params, *_ = lstsq(a, b)
        return np.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)

    value_list: list[np.ndarray] = []
    locx_list: list[np.ndarray] = []
    locy_list: list[np.ndarray] = []
    bins = int(psp.numBins)
    ball_radius_pix = float(psp.ball_radius) / float(psp.pixmm)

    radius_max = float(np.max(radius_record)) if np.size(radius_record) > 0 else 1.0
    frame_weights: list[float] = []
    gamma = max(float(deep_weight_gamma), 0.0)
    edge_min = float(np.clip(edge_weight_min, 0.0, 1.0))
    edge_pow = max(float(edge_weight_power), 0.0)
    fit_mask_records: list[dict[str, int | float]] = []

    for idx_i in range(int(np.shape(imgs)[0])):
        print(f"# iter {idx_i}")
        frame = imgs[idx_i, :, :, :]
        dI = frame.astype("float") - bg_proc
        circle = Circle(
            int(touch_center_record[idx_i, 0]),
            int(touch_center_record[idx_i, 1]),
            int(radius_record[idx_i]),
        )
        center = circle.center
        radius = circle.radius
        sizey, sizex = dI.shape[:2]
        xqq, yqq = np.meshgrid(range(sizex), range(sizey))
        xq = xqq - center[0]
        yq = yqq - center[1]
        rsqcoord = xq * xq + yq * yq
        rad_sq = radius * radius
        valid_rad = min(rad_sq, int(ball_radius_pix * ball_radius_pix))
        circle_mask = rsqcoord < valid_rad
        frame_marker_mask = marker_masks[idx_i]
        valid_mask = circle_mask & ~frame_marker_mask
        circle_pixels = int(np.count_nonzero(circle_mask))
        excluded_marker_pixels = int(np.count_nonzero(circle_mask & frame_marker_mask))
        valid_pixels = int(np.count_nonzero(valid_mask))
        if valid_pixels < 6:
            raise RuntimeError(
                f"Frame {idx_i} has only {valid_pixels} usable contact pixels after marker masking"
            )
        fit_mask_records.append(
            {
                "frame_index": idx_i,
                "circle_pixels": circle_pixels,
                "excluded_marker_pixels": excluded_marker_pixels,
                "valid_pixels": valid_pixels,
                "excluded_fraction": float(excluded_marker_pixels / max(circle_pixels, 1)),
            }
        )
        valid_id = np.nonzero(valid_mask)
        xvalid = xq[valid_id]
        yvalid = yq[valid_id]
        rvalid = np.sqrt(xvalid * xvalid + yvalid * yvalid)
        gradxseq = np.arcsin(np.clip(rvalid / ball_radius_pix, 0.0, 1.0))
        # Runtime converts documented positive penetration to a negative Taxim
        # surface before taking its gradient, so its gradient points outward.
        # Use the same convention here (the previous inward direction differed
        # by pi and swapped the opposed cyan/red illumination response).
        gradyseq = np.arctan2(yvalid, xvalid)
        binm = bins - 1
        x_binr = 0.5 * np.pi / binm
        y_binr = 2 * np.pi / binm
        idx_x = np.floor(gradxseq / x_binr).astype("int")
        idx_y = np.floor((gradyseq + np.pi) / y_binr).astype("int")

        value_map = np.zeros((bins, bins, 3))
        loc_x_map = np.zeros((bins, bins))
        loc_y_map = np.zeros((bins, bins))
        bin_counts = np.zeros((bins, bins), dtype=np.int32)
        edge_ratio = np.clip(rvalid / max(float(radius), 1.0e-6), 0.0, 1.0)
        edge_weight = edge_min + (1.0 - edge_min) * ((1.0 - edge_ratio) ** edge_pow)
        valid_r = dI[:, :, 0][valid_id] * edge_weight
        valid_g = dI[:, :, 1][valid_id] * edge_weight
        valid_b = dI[:, :, 2][valid_id] * edge_weight
        valid_x = xqq[valid_id]
        valid_y = yqq[valid_id]
        # Repeated NumPy advanced-index writes do not accumulate. Use add.at and
        # average each occupied bin so one arbitrary pixel cannot define a bin.
        np.add.at(value_map[:, :, 0], (idx_x, idx_y), valid_r)
        np.add.at(value_map[:, :, 1], (idx_x, idx_y), valid_g)
        np.add.at(value_map[:, :, 2], (idx_x, idx_y), valid_b)
        np.add.at(loc_x_map, (idx_x, idx_y), valid_x)
        np.add.at(loc_y_map, (idx_x, idx_y), valid_y)
        np.add.at(bin_counts, (idx_x, idx_y), 1)
        occupied = bin_counts > 0
        value_map[occupied] /= bin_counts[occupied, None]
        loc_x_map[occupied] /= bin_counts[occupied]
        loc_y_map[occupied] /= bin_counts[occupied]
        loc_x_map = _interpolate(loc_x_map)
        loc_y_map = _interpolate(loc_y_map)
        value_map[:, :, 0] = _interpolate(value_map[:, :, 0])
        value_map[:, :, 1] = _interpolate(value_map[:, :, 1])
        value_map[:, :, 2] = _interpolate(value_map[:, :, 2])
        value_list.append(value_map)
        locx_list.append(loc_x_map)
        locy_list.append(loc_y_map)
        r_norm = float(radius_record[idx_i]) / max(radius_max, 1.0e-6)
        w_i = float(np.clip(r_norm, 1.0e-6, 1.0) ** gamma)
        frame_weights.append(max(w_i, 1.0e-6))

    table_v = np.array(value_list)
    table_x = np.array(locx_list)
    table_y = np.array(locy_list)
    table_w = np.asarray(frame_weights, dtype=np.float64)
    grad_r = np.zeros((bins, bins, 6))
    grad_g = np.zeros((bins, bins, 6))
    grad_b = np.zeros((bins, bins, 6))
    for i in range(table_v.shape[1]):
        for j in range(table_v.shape[2]):
            grad_r[i, j, :] = _fit_poly_params(table_x[:, i, j], table_y[:, i, j], table_v[:, i, j, 0], table_w)
            grad_g[i, j, :] = _fit_poly_params(table_x[:, i, j], table_y[:, i, j], table_v[:, i, j, 1], table_w)
            grad_b[i, j, :] = _fit_poly_params(table_x[:, i, j], table_y[:, i, j], table_v[:, i, j, 2], table_w)

    sigma = max(float(grad_table_smooth_sigma), 0.0)
    if sigma > 1.0e-6:
        # Smooth over (grad_mag_bin, grad_dir_bin) only; keep polynomial coefficient axis intact.
        for k in range(6):
            grad_r[:, :, k] = scipy.ndimage.gaussian_filter(grad_r[:, :, k], sigma=sigma, mode="nearest")
            grad_g[:, :, k] = scipy.ndimage.gaussian_filter(grad_g[:, :, k], sigma=sigma, mode="nearest")
            grad_b[:, :, k] = scipy.ndimage.gaussian_filter(grad_b[:, :, k], sigma=sigma, mode="nearest")

    out = data_dir / "polycalib.npz"
    np.savez(
        out,
        bins=bins,
        grad_r=grad_r,
        grad_g=grad_g,
        grad_b=grad_b,
        deep_weight_gamma=gamma,
        edge_weight_min=edge_min,
        edge_weight_power=edge_pow,
        grad_table_smooth_sigma=sigma,
        marker_masked_fit=np.asarray("marker_masks" in data_file.files),
        marker_mask_version=np.asarray(marker_mask_version),
        marker_mask_fraction_mean=np.asarray(float(np.mean(marker_masks)), dtype=np.float64),
    )
    fit_mask_report_path = data_dir / "fit_marker_exclusion.json"
    fit_mask_report_path.write_text(json.dumps(fit_mask_records, indent=2), encoding="utf-8")
    print(f"[OK] polycalib -> {out}")
    print(f"[OK] fit marker exclusion -> {fit_mask_report_path}")
    return out


def main() -> int:
    default_bg_raw, default_bg_clean = _default_background_paths()
    default_marker_rest = _default_marker_rest_path()
    parser = argparse.ArgumentParser(description="Build and install Xense polycalib from ball video.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/ball_calib_raw"),
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(repo_root() / "data/calibration/file-000.mp4"),
    )
    parser.add_argument("--num-ball", type=int, default=50)
    parser.add_argument("--num-validation", type=int, default=50)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--taxim-repo", type=str, default="")
    parser.add_argument(
        "--bg-raw",
        type=str,
        default=str(default_bg_raw),
        help="True no-contact Xense background with printed markers.",
    )
    parser.add_argument(
        "--bg-clean",
        type=str,
        default=str(default_bg_clean),
        help="Marker-free version of the same true no-contact background.",
    )
    parser.add_argument(
        "--marker-rest",
        type=str,
        default=str(default_marker_rest),
        help="Rest marker coordinates paired with --bg-raw (falls back to image detection if absent).",
    )
    parser.add_argument(
        "--pixmm",
        type=float,
        default=0.0,
        help="mm/px for Taxim (default: Xense sensing width / 400).",
    )
    parser.add_argument("--bg-install", type=str, default="", help="Optional bg_clean.jpg for xense_lab_data.")
    parser.add_argument(
        "--deep-weight-gamma",
        type=float,
        default=0.0,
        help="Weight larger contact-radius samples higher during poly fit (0=uniform, 1=linear).",
    )
    parser.add_argument(
        "--edge-weight-min",
        type=float,
        default=1.0,
        help="Minimum fitted response at the contact rim (1 preserves measured calibration amplitude).",
    )
    parser.add_argument(
        "--edge-weight-power",
        type=float,
        default=2.0,
        help="Power on (1-r/radius) for edge weighting. Higher suppresses rim samples more strongly.",
    )
    parser.add_argument(
        "--grad-table-smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma on fitted grad tables across bins (0 disables).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    taxim_repo = Path(args.taxim_repo or _default_taxim_repo()).expanduser().resolve()
    bg_raw_path = Path(args.bg_raw).expanduser().resolve()
    bg_clean_path = Path(args.bg_clean).expanduser().resolve()
    marker_rest_path = Path(args.marker_rest).expanduser().resolve()
    out_w = 400
    pixmm = float(args.pixmm) if args.pixmm > 0 else XENSE_SENSING_MM[0] / out_w

    if not args.skip_import:
        import_cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "import_ball_calib_video.py"),
            "--video",
            str(Path(args.video).expanduser().resolve()),
            "--out-dir",
            str(data_dir),
            "--num-ball",
            str(int(args.num_ball)),
            "--num-validation",
            str(int(args.num_validation)),
            "--reference-bg",
            str(bg_raw_path),
        ]
        print("[RUN]", " ".join(import_cmd))
        subprocess.run(import_cmd, check=True)

    _build_datapack(
        data_dir,
        pixmm=pixmm,
        bg_raw_path=bg_raw_path,
        bg_clean_path=bg_clean_path,
        marker_rest_path=marker_rest_path,
    )
    validation_dir = data_dir.parent / f"{data_dir.name}_validation"
    if (validation_dir / "ball").is_dir():
        _build_datapack(
            validation_dir,
            pixmm=pixmm,
            bg_raw_path=bg_raw_path,
            bg_clean_path=bg_clean_path,
            marker_rest_path=marker_rest_path,
        )
    polycalib_path = _run_poly_table_calib(
        data_dir,
        taxim_repo,
        pixmm=pixmm,
        num_bins=TAXIM_NUM_BINS,
        deep_weight_gamma=float(args.deep_weight_gamma),
        edge_weight_min=float(args.edge_weight_min),
        edge_weight_power=float(args.edge_weight_power),
        grad_table_smooth_sigma=float(args.grad_table_smooth_sigma),
    )

    install_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "install_taxim_polycalib.py"),
        "--polycalib",
        str(polycalib_path),
    ]
    bg_install = args.bg_install.strip()
    if not bg_install:
        paired_bg = data_dir / "bg_clean.png"
        advisor_bg = repo_root() / "data/calibration/tactile/advisor_processed/bg_clean.jpg"
        # Prefer the user-verified marker-free no-contact reference directly.
        # The generated PNG is only a paired fallback for custom datasets.
        if advisor_bg.is_file():
            bg_install = str(advisor_bg)
        elif paired_bg.is_file():
            bg_install = str(paired_bg)
    if bg_install:
        install_cmd.extend(["--bg", bg_install])

    print("[RUN]", " ".join(install_cmd))
    subprocess.run(install_cmd, check=True)

    print("")
    print("[DONE] Xense polycalib installed.")
    print("[NEXT] SKIP_EXISTING=0 bash bash_command/run_vitacsim_calibration_sweep_dual.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
