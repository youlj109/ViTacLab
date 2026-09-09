# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Palmar schematic layout for UR10e + Shadow Hand (ViTacLab USD body names).

Normalized boxes ``(x0, y0, x1, y1)`` use image coordinates (origin top-left, *y* down), each in ``[0, 1]``.
Keys match :data:`~ViTacLab.assets.sensor.shadow_hand_full_tactile.UR10E_SHADOW_HAND_TACTILE_BODY_NAMES`
(excluding ``forearm`` / ``wrist`` mount links, which sit outside the palmar patch).

Finger codes on the Shadow Hand: **ff** index, **mf** middle, **rf** ring, **lf** little, **th** thumb.
Layout targets the **left** Shadow Hand on UR10e (palmar view: thumb toward the **right** of the figure).
"""

from __future__ import annotations

from .shadow_hand_full_tactile_sensor_cfg import UR10E_SHADOW_HAND_TACTILE_BODY_NAMES

# Links drawn on the palmar schematic (hand only; wrist/forearm omitted).
SHADOW_HAND_SCHEMATIC_BODY_NAMES: tuple[str, ...] = tuple(
    n for n in UR10E_SHADOW_HAND_TACTILE_BODY_NAMES if n not in ("forearm", "wrist")
)


def _seg_stack(
    prefix: str,
    parts: tuple[str, ...],
    *,
    xc: float,
    y_top: float,
    w: float,
    h: float,
    dy: float,
) -> dict[str, tuple[float, float, float, float]]:
    out: dict[str, tuple[float, float, float, float]] = {}
    xa = xc - w / 2.0
    for i, part in enumerate(parts):
        ya = y_top + i * dy
        out[f"{prefix}{part}"] = (xa, ya, xa + w, ya + h)
    return out


def shadow_hand_schematic_layout() -> dict[str, tuple[float, float, float, float]]:
    """Normalized palmar boxes for ViTacLab UR10e + left Shadow Hand tactile links."""
    out: dict[str, tuple[float, float, float, float]] = {}
    w_seg, h_seg = 0.040, 0.034
    dy = 0.032

    # Palm (DexCube contact band sits roughly over the central palm).
    out["palm"] = (0.36, 0.54, 0.64, 0.76)

    # Index → little: increasing x (little finger toward +x in schematic).
    out.update(
        _seg_stack(
            "ff",
            ("knuckle", "proximal", "middle", "distal", "tip"),
            xc=0.40,
            y_top=0.46,
            w=w_seg,
            h=h_seg,
            dy=-dy,
        )
    )
    out.update(
        _seg_stack(
            "mf",
            ("knuckle", "proximal", "middle", "distal", "tip"),
            xc=0.46,
            y_top=0.44,
            w=w_seg,
            h=h_seg,
            dy=-dy,
        )
    )
    out.update(
        _seg_stack(
            "rf",
            ("knuckle", "proximal", "middle", "distal", "tip"),
            xc=0.52,
            y_top=0.44,
            w=w_seg,
            h=h_seg,
            dy=-dy,
        )
    )
    # Little finger includes metacarpal segment at the palm base.
    out["lfmetacarpal"] = (0.575, 0.52, 0.575 + w_seg, 0.52 + h_seg)
    out.update(
        _seg_stack(
            "lf",
            ("knuckle", "proximal", "middle", "distal", "tip"),
            xc=0.58,
            y_top=0.46,
            w=w_seg,
            h=h_seg,
            dy=-dy,
        )
    )

    # Thumb chain: toward +x (anatomical left hand, palmar view).
    th_w, th_h = 0.038, 0.032
    th_dx, th_dy = 0.024, -0.028
    x0, y0 = 0.62, 0.56
    for i, part in enumerate(("base", "proximal", "hub", "middle", "distal", "tip")):
        key = f"th{part}" if part != "base" else "thbase"
        xa = x0 + i * th_dx
        ya = y0 + i * th_dy
        out[key] = (xa, ya, xa + th_w, ya + th_h)

    missing = [n for n in SHADOW_HAND_SCHEMATIC_BODY_NAMES if n not in out]
    if missing:
        raise RuntimeError(f"Shadow hand schematic layout missing keys: {missing}")
    return out


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
    """Convert normalized boxes to integer pixel ``(x0, y0, x1, y1)`` (x1/y1 exclusive)."""
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
    span_x = max(float(bx1 - bx0), 1e-6)
    span_y = max(float(by1 - by0), 1e-6)

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

    pix: dict[str, tuple[int, int, int, int]] = {}
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
