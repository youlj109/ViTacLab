# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Schematic 2D layout (palmar view) for compositing Wuji per-link tactile grids.

Finger numbering matches the URDF / cfg: **finger1 = thumb**, **finger5 = pinky**;
fingers 2–4 are index, middle, ring in order.

Layout is normalized ``(x0, y0, x1, y1)`` in image coordinates (origin top-left, *y* down),
each in ``[0, 1]``. Regions outside all boxes stay black in composite renders.

**Palmar view (掌心朝向画图外的观察者):** for ``right_hand_schematic_layout``, the thumb chain sits on
the **left** side of the figure and the pinky on the **right** — that is the usual anatomical palmar
layout for a **right** hand, not a mirror mistake. If your 3D camera shows the thumb on the opposite
side of the screen, enable ``HandTactilePlotCfg.schematic_mirror_x`` in
:mod:`~ViTacLab.assets.sensor.wuji_hand_grid_tactile.plot_hand_tactile` to flip the composite horizontally.
"""

from __future__ import annotations

from typing import Literal

from .wuji_hand_grid_tactile_cfg import WUJI_LEFT_LINK_NAMES, WUJI_RIGHT_LINK_NAMES


def _mirror_x_norm_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (1.0 - x1, y0, 1.0 - x0, y1)


def _thumb_segment_boxes() -> dict[str, tuple[float, float, float, float]]:
    """Thumb (finger1): short stack from palm toward upper-left (gentle drift, not a long diagonal)."""
    out: dict[str, tuple[float, float, float, float]] = {}
    w, h = 0.048, 0.036
    # link1 sits flush with palm left edge (~0.38); small |dx| keeps the chain compact so the
    # bounding box (and fill_canvas zoom) does not leave a large empty wedge beside the thumb.
    x0_base, y0_base = 0.332, 0.50
    dx, dy = -0.021, -0.031
    parts = ("link1", "link2", "link3", "link4", "tip_link")
    for i, part in enumerate(parts):
        key = f"right_finger1_{part}"
        xa = x0_base + i * dx
        ya = y0_base + i * dy
        out[key] = (xa, ya, xa + w, ya + h)
    return out


def _finger_column_boxes(finger: int, xc: float) -> dict[str, tuple[float, float, float, float]]:
    """One finger (2–5): vertical stack from palm toward top of image (decreasing y)."""
    out: dict[str, tuple[float, float, float, float]] = {}
    w, h = 0.042, 0.036
    # link1 bottom touches palm top band; later links go up (smaller y).
    y_link1_top = 0.48
    dy = 0.034
    parts = ("link1", "link2", "link3", "link4", "tip_link")
    for i, part in enumerate(parts):
        key = f"right_finger{finger}_{part}"
        y_top = y_link1_top - i * dy
        xa = xc - w / 2.0
        out[key] = (xa, y_top, xa + w, y_top + h)
    return out


def right_hand_schematic_layout() -> dict[str, tuple[float, float, float, float]]:
    """Normalized boxes for **right** hand, palmar view (thumb left, pinky right among digits)."""
    out: dict[str, tuple[float, float, float, float]] = {}
    # Palm (large patch).
    out["right_palm_link"] = (0.38, 0.52, 0.62, 0.74)
    out.update(_thumb_segment_boxes())
    # Index → pinky: increasing x (pinky most to the right in image).
    xc = {2: 0.44, 3: 0.50, 4: 0.56, 5: 0.62}
    for f in (2, 3, 4, 5):
        out.update(_finger_column_boxes(f, xc[f]))
    # Sanity: all cfg link names present.
    missing = [n for n in WUJI_RIGHT_LINK_NAMES if n not in out]
    if missing:
        raise RuntimeError(f"Layout missing keys: {missing}")
    return out


def left_hand_schematic_layout() -> dict[str, tuple[float, float, float, float]]:
    """Same geometry as right, mirrored horizontally; keys use ``left_`` prefix."""
    right = right_hand_schematic_layout()
    out: dict[str, tuple[float, float, float, float]] = {}
    for name, box in right.items():
        left_name = name.replace("right_", "left_", 1)
        out[left_name] = _mirror_x_norm_box(box)
    missing = [n for n in WUJI_LEFT_LINK_NAMES if n not in out]
    if missing:
        raise RuntimeError(f"Left layout missing keys: {missing}")
    return out


def schematic_layout_for_side(side: Literal["right", "left"]) -> dict[str, tuple[float, float, float, float]]:
    if side == "right":
        return right_hand_schematic_layout()
    return left_hand_schematic_layout()


def layout_union_bounds(layout: dict[str, tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of all normalized schematic regions."""
    xs0 = min(b[0] for b in layout.values())
    ys0 = min(b[1] for b in layout.values())
    xs1 = max(b[2] for b in layout.values())
    ys1 = max(b[3] for b in layout.values())
    return xs0, ys0, xs1, ys1


def norm_boxes_to_pixels(
    layout: dict[str, tuple[float, float, float, float]],
    width: int,
    height: int,
    *,
    fill_canvas: bool = True,
    margin_frac: float = 0.03,
) -> dict[str, tuple[int, int, int, int]]:
    """Convert normalized boxes to integer pixel ``(x0, y0, x1, y1)`` (x1/y1 exclusive).

    If ``fill_canvas`` is True (default), the union of all regions is **uniformly scaled** so the
    hand uses as much of the canvas as possible (with a small margin). Otherwise boxes are mapped
    from the full ``[0,1]×[0,1]`` square as before (large black borders).
    """
    if width <= 0 or height <= 0:
        return {}

    def _clamp_box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        return x0, y0, x1, y1

    if not fill_canvas:
        pix: dict[str, tuple[int, int, int, int]] = {}
        for k, (nx0, ny0, nx1, ny1) in layout.items():
            x0 = int(round(nx0 * width))
            y0 = int(round(ny0 * height))
            x1 = int(round(nx1 * width))
            y1 = int(round(ny1 * height))
            pix[k] = _clamp_box(x0, y0, x1, y1)
        return pix

    bx0, by0, bx1, by1 = layout_union_bounds(layout)
    span_x = float(bx1 - bx0)
    span_y = float(by1 - by0)
    if span_x <= 0.0:
        span_x = 1e-6
    if span_y <= 0.0:
        span_y = 1e-6

    mf = max(0.0, float(margin_frac))
    mx = mf * float(width)
    my = mf * float(height)
    usable_w = max(1.0, float(width) - 2.0 * mx)
    usable_h = max(1.0, float(height) - 2.0 * my)
    s = min(usable_w / span_x, usable_h / span_y)
    content_w = span_x * s
    content_h = span_y * s
    ox = mx + 0.5 * (usable_w - content_w)
    oy = my + 0.5 * (usable_h - content_h)

    pix = {}
    for k, (nx0, ny0, nx1, ny1) in layout.items():
        px0 = ox + (nx0 - bx0) * s
        py0 = oy + (ny0 - by0) * s
        px1 = ox + (nx1 - bx0) * s
        py1 = oy + (ny1 - by0) * s
        x0 = int(round(px0))
        y0 = int(round(py0))
        x1 = int(round(px1))
        y1 = int(round(py1))
        pix[k] = _clamp_box(x0, y0, x1, y1)
    return pix
