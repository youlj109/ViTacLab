# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Advisor Xense image helpers: marker rest detection and clean background (no printed dots)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def detect_markers_local(
    gray: np.ndarray,
    rest_xy: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Refine marker centers by searching darkest pixel near each rest grid point."""
    try:
        import cv2
    except ImportError:
        return rest_xy.copy()

    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    h, w = gray.shape
    pts: list[np.ndarray] = []
    r = max(3, int(radius))
    for i in range(rest_xy.shape[0]):
        cx, cy = int(round(rest_xy[i, 0])), int(round(rest_xy[i, 1]))
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        patch = blur[y0:y1, x0:x1]
        if patch.size == 0:
            pts.append(rest_xy[i].astype(np.float32))
            continue
        _, _, min_loc, _ = cv2.minMaxLoc(patch)
        pts.append(np.array([float(min_loc[0] + x0), float(min_loc[1] + y0)], dtype=np.float32))
    return np.stack(pts, axis=0)


def _refine_centers_to_darkest(
    gray: np.ndarray,
    centers_xy: np.ndarray,
    *,
    search_radius: int = 8,
) -> np.ndarray:
    """Snap each marker center to the darkest pixel in a local window."""
    try:
        import cv2
    except ImportError:
        return centers_xy.copy()

    h, w = gray.shape
    out = centers_xy.astype(np.float32).copy()
    sr = max(4, int(search_radius))
    for i in range(out.shape[0]):
        cx, cy = int(round(float(out[i, 0]))), int(round(float(out[i, 1])))
        x0, x1 = max(0, cx - sr), min(w, cx + sr + 1)
        y0, y1 = max(0, cy - sr), min(h, cy + sr + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        _, _, min_loc, _ = cv2.minMaxLoc(patch)
        out[i, 0] = float(min_loc[0] + x0)
        out[i, 1] = float(min_loc[1] + y0)
    return out


def detect_printed_markers(
    rgb: np.ndarray,
    *,
    blackhat_thresh: int = 6,
    min_area: int = 2,
    max_area: int = 120,
    min_local_contrast: float = 4.0,
    merge_radius_px: float = 4.0,
) -> tuple[np.ndarray, float]:
    """Detect all printed marker dots directly from the image (not a fixed grid)."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required for printed marker detection") from e

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = gray.astype(np.uint8)
    candidates: list[tuple[float, float, float]] = []

    for k in (7, 9, 11, 13):
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
        _, blob = cv2.threshold(blackhat, blackhat_thresh, 255, cv2.THRESH_BINARY)
        blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        num, labels, stats, _centroids = cv2.connectedComponentsWithStats(blob, connectivity=8)
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            comp = labels == lab
            ys, xs = np.where(comp)
            vals = gray[ys, xs]
            j = int(vals.argmin())
            ax, ay = float(xs[j]), float(ys[j])
            ix, iy = int(round(ax)), int(round(ay))
            h, w = gray.shape
            if not (2 <= ix < w - 2 and 2 <= iy < h - 2):
                continue
            ring = gray[max(0, iy - 10) : min(h, iy + 11), max(0, ix - 10) : min(w, ix + 11)].copy()
            ry = iy - max(0, iy - 10)
            rx = ix - max(0, ix - 10)
            ring[max(0, ry - 2) : ry + 3, max(0, rx - 2) : rx + 3] = np.nan
            local_bg = float(np.nanmedian(ring))
            contrast = local_bg - float(gray[iy, ix])
            if contrast < min_local_contrast:
                continue
            candidates.append((ax, ay, contrast))

    if not candidates:
        return np.zeros((0, 2), dtype=np.float32), 2.5

    # Keep the strongest response when multiple kernels hit the same dot.
    candidates.sort(key=lambda t: -t[2])
    merged: list[tuple[float, float]] = []
    merge_r = float(merge_radius_px)
    for ax, ay, _ in candidates:
        if any((ax - mx) ** 2 + (ay - my) ** 2 <= merge_r**2 for mx, my in merged):
            continue
        merged.append((ax, ay))

    centers = np.array(merged, dtype=np.float32)
    # Typical Xense printed dot radius at 400x700.
    return centers, 2.5


def build_marker_inpaint_mask(
    rgb: np.ndarray,
    centers_xy: np.ndarray,
    *,
    radius_px: float,
    refine_centers: bool = True,
) -> np.ndarray:
    """Build a binary mask covering printed marker dots (grid + dark-blob detection)."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required for marker inpaint mask") from e

    gray_for_refine = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    centers = (
        _refine_centers_to_darkest(gray_for_refine, centers_xy, search_radius=8)
        if refine_centers
        else centers_xy
    )

    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # One notch above prior settings; centers are refined to actual dot cores first.
    r_disk = int(max(8, round(float(radius_px) * 3.4)))
    r_prox = r_disk + 7
    for i in range(centers.shape[0]):
        cx = int(round(float(centers[i, 0])))
        cy = int(round(float(centers[i, 1])))
        cv2.circle(mask, (cx, cy), r_disk, 255, thickness=-1)

    prox = np.zeros((h, w), dtype=np.uint8)
    for i in range(centers.shape[0]):
        cx = int(round(float(centers[i, 0])))
        cy = int(round(float(centers[i, 1])))
        cv2.circle(prox, (cx, cy), r_prox, 255, thickness=-1)

    gray = gray_for_refine.astype(np.uint8)
    # Dark printed dots: black-hat picks small dark features on bright gel.
    k = max(11, r_disk * 2 + 1)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, blob = cv2.threshold(blackhat, 6, 255, cv2.THRESH_BINARY)
    blob = cv2.bitwise_and(blob, prox)
    mask = cv2.bitwise_or(mask, blob)

    # Xense markers can be greenish; remove low-saturation dark cores + green blobs.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(green, blob))

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, dilate_k, iterations=1)
    return mask


def inpaint_markers(
    rgb: np.ndarray,
    centers_xy: np.ndarray,
    *,
    radius_px: float,
    inpaint_radius: int | None = None,
) -> np.ndarray:
    """Remove printed marker dots via OpenCV inpaint (gel-only background for Taxim)."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required for marker inpaint") from e

    mask = build_marker_inpaint_mask(rgb, centers_xy, radius_px=radius_px)
    ir = int(inpaint_radius if inpaint_radius is not None else 8)
    # NS inpaint gives smoother gel fill than TELEA for dense dot grids.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(bgr, mask, ir, cv2.INPAINT_NS)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)


def measure_marker_residual(
    rgb: np.ndarray,
    centers_xy: np.ndarray,
    *,
    patch_radius: int = 10,
    contrast_threshold: float = 8.0,
) -> dict[str, float | int]:
    """Count marker sites that still look darker than local gel (quality check)."""
    try:
        import cv2
    except ImportError:
        return {"residual_markers": -1, "residual_ratio": -1.0}

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    h, w = gray.shape
    residual = 0
    pr = int(max(4, patch_radius))
    sr = max(pr, 7)
    for i in range(centers_xy.shape[0]):
        cx = int(round(float(centers_xy[i, 0])))
        cy = int(round(float(centers_xy[i, 1])))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        x0, x1 = max(0, cx - sr), min(w, cx + sr + 1)
        y0, y1 = max(0, cy - sr), min(h, cy + sr + 1)
        patch = gray[y0:y1, x0:x1]
        _, _, min_loc, _ = cv2.minMaxLoc(patch)
        ax = int(min_loc[0] + x0)
        ay = int(min_loc[1] + y0)
        x0, x1 = max(0, ax - pr), min(w, ax + pr + 1)
        y0, y1 = max(0, ay - pr), min(h, ay + pr + 1)
        ring = gray[y0:y1, x0:x1]
        ring_mean = float(ring.mean())
        center = float(gray[ay, ax])
        local_bg = float(blur[ay, ax])
        if center < local_bg - contrast_threshold or center < ring_mean - contrast_threshold:
            residual += 1
    total = int(centers_xy.shape[0])
    return {
        "residual_markers": residual,
        "residual_ratio": float(residual / max(total, 1)),
        "marker_count": total,
    }


def _approx_xense_rest_xy(image_height: int, image_width: int) -> tuple[np.ndarray, float]:
    """14x14 staggered grid (same layout as visuotactile_marker PATTERN_SPECS['xense'])."""
    rows, cols = 14, 14
    margin_frac = 0.10
    radius_px = 2.0
    margin_x = margin_frac * image_width
    margin_y = margin_frac * image_height
    xs = np.linspace(margin_x, image_width - margin_x, cols, dtype=np.float32)
    ys = np.linspace(margin_y, image_height - margin_y, rows, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pos = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    stagger = (np.arange(rows) % 2) * (0.5 * (xs[1] - xs[0]) if cols > 1 else 0.0)
    row_idx = np.arange(rows).repeat(cols)
    pos[:, 0] += stagger[row_idx]
    return pos.astype(np.float32), float(radius_px)


def build_advisor_marker_rest(
    no_contact_rgb: np.ndarray,
    *,
    pattern: str,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, float]:
    """Detect lab marker rest coordinates on native-resolution no_contact frame."""
    if pattern != "xense":
        raise ValueError(f"Unsupported advisor pattern={pattern!r} (expected xense)")
    try:
        detected, radius = detect_printed_markers(no_contact_rgb)
        if detected.shape[0] >= 80:
            return detected.astype(np.float32), radius
    except ImportError:
        pass
    # Fallback for unusual frames: approximate grid + local refinement.
    rest_xy, radius = _approx_xense_rest_xy(image_height, image_width)
    gray = np.mean(no_contact_rgb.astype(np.float32), axis=-1)
    detected = detect_markers_local(gray, rest_xy, radius=8)
    return detected.astype(np.float32), radius


def make_clean_background(
    no_contact_rgb: np.ndarray,
    marker_rest: np.ndarray,
    *,
    radius_px: float,
) -> np.ndarray:
    """Return gel-only background (all printed markers inpainted out)."""
    return inpaint_markers(no_contact_rgb, marker_rest, radius_px=radius_px, inpaint_radius=10)
