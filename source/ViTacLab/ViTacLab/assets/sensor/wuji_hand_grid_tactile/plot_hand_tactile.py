# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composite Wuji hand tactile grids into two schematic images: normal heatmap and tangential arrows."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .hand_tactile_layout import norm_boxes_to_pixels, schematic_layout_for_side


def _mirror_pixel_boxes_x(
    pix: dict[str, tuple[int, int, int, int]], canvas_width: int
) -> dict[str, tuple[int, int, int, int]]:
    """Flip schematic regions horizontally (image x → canvas_width - x)."""
    w = int(canvas_width)
    if w <= 0:
        return pix
    out: dict[str, tuple[int, int, int, int]] = {}
    for k, (x0, y0, x1, y1) in pix.items():
        out[k] = (w - x1, y0, w - x0, y1)
    return out


def _to_numpy(x, dtype=np.float32) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=dtype)
    return a


def _orient_scalar_patch_for_schematic(
    patch: np.ndarray,
    *,
    swap_uv: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> np.ndarray:
    """Apply transpose / flips so 2D tactile data matches palmar schematic (sensor index vs screen)."""
    disp = np.asarray(patch, dtype=np.float32)
    if disp.ndim != 2:
        return disp
    if swap_uv:
        disp = np.ascontiguousarray(disp.T)
    if flip_rows:
        disp = np.ascontiguousarray(np.flipud(disp))
    if flip_cols:
        disp = np.ascontiguousarray(np.fliplr(disp))
    return disp


def _orient_friction_uv_for_schematic(
    fu: np.ndarray,
    fv: np.ndarray,
    *,
    swap_uv: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Same spatial reordering as :func:`_orient_scalar_patch_for_schematic` for both components."""
    u = np.asarray(fu, dtype=np.float32)
    v = np.asarray(fv, dtype=np.float32)
    if swap_uv:
        u = np.ascontiguousarray(u.T)
        v = np.ascontiguousarray(v.T)
    if flip_rows:
        u = np.ascontiguousarray(np.flipud(u))
        v = np.ascontiguousarray(np.flipud(v))
    if flip_cols:
        u = np.ascontiguousarray(np.fliplr(u))
        v = np.ascontiguousarray(np.fliplr(v))
    return u, v


def _aggregate_filters(arr: np.ndarray, mode: Literal["sum", "max", "mean"]) -> np.ndarray:
    """Reduce leading filter/object dimension if present: (F, H, W) -> (H, W)."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if mode == "sum":
            return np.sum(arr, axis=0)
        if mode == "max":
            return np.max(arr, axis=0)
        if mode == "mean":
            return np.mean(arr, axis=0)
    raise ValueError(f"Expected (H,W) or (F,H,W), got shape {arr.shape}")


def _resize_patch(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize 2D patch to (out_h, out_w)."""
    if patch.shape == (out_h, out_w):
        return patch
    in_h, in_w = patch.shape
    if in_h <= 0 or in_w <= 0 or out_h <= 0 or out_w <= 0:
        return np.zeros((out_h, out_w), dtype=patch.dtype)
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    x0 = np.clip(x0, 0, in_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    Ia = patch[y0[:, None], x0[None, :]]
    Ib = patch[y0[:, None], x1[None, :]]
    Ic = patch[y1[:, None], x0[None, :]]
    Id = patch[y1[:, None], x1[None, :]]
    wa = (1.0 - wx)[None, :] * (1.0 - wy)[:, None]
    wb = wx[None, :] * (1.0 - wy)[:, None]
    wc = (1.0 - wx)[None, :] * wy[:, None]
    wd = wx[None, :] * wy[:, None]
    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).astype(np.float32)


def _resize_patch_nearest(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize 2D patch to ``(out_h, out_w)`` (one source cell per destination pixel)."""
    if patch.shape == (out_h, out_w):
        return patch
    in_h, in_w = patch.shape
    if in_h <= 0 or in_w <= 0 or out_h <= 0 or out_w <= 0:
        return np.zeros((out_h, out_w), dtype=patch.dtype)
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    yi = np.clip(np.round(y).astype(np.int64), 0, in_h - 1)
    xi = np.clip(np.round(x).astype(np.int64), 0, in_w - 1)
    return patch[yi[:, None], xi[None, :]].astype(np.float32)


def _box_area_pixel(box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def _fair_pixel_edges(start: int, length: int, ncells: int) -> np.ndarray:
    """``ncells+1`` monotonic edge indices partitioning ``[start, start+length)`` into fair row/column heights."""
    if ncells <= 0:
        return np.array([start, start + length], dtype=np.int64)
    q, r = divmod(int(length), int(ncells))
    edges = [start]
    pos = start
    for i in range(ncells):
        w = q + (1 if i < r else 0)
        pos = min(start + length, pos + max(1, w))
        edges.append(pos)
    edges[-1] = start + length
    return np.asarray(edges, dtype=np.int64)


def _draw_discrete_grid_mpl(
    ax,
    box: tuple[int, int, int, int],
    in_h: int,
    in_w: int,
    *,
    color: str = "#d2d2d2",
    linewidth: float = 0.65,
    alpha: float = 0.9,
    zorder: float = 1.0,
) -> None:
    """Draw cell borders for a tactile patch (matplotlib axes, image y-down matching :func:`render_friction_arrows_image`)."""
    x0, y0, x1, y1 = box
    bh, bw = y1 - y0, x1 - x0
    if bh <= 0 or bw <= 0 or in_h <= 0 or in_w <= 0:
        return
    row_e = _fair_pixel_edges(y0, bh, in_h)
    col_e = _fair_pixel_edges(x0, bw, in_w)
    for i in range(row_e.size):
        yi = float(row_e[i])
        ax.plot([x0, x1], [yi, yi], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, clip_on=True, solid_capstyle="butt")
    for j in range(col_e.size):
        xj = float(col_e[j])
        ax.plot([xj, xj], [y0, y1], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, clip_on=True, solid_capstyle="butt")


def _draw_discrete_normal_patch(
    canvas: np.ndarray,
    disp: np.ndarray,
    box: tuple[int, int, int, int],
    vmin: float,
    vmax: float,
    *,
    colormap: str,
    draw_grid: bool,
    draw_values: bool,
    min_cell_px_for_text: int,
    value_fmt: str,
) -> None:
    """Paint one schematic region as sharp grid cells; optional borders and force text (mutates ``canvas``)."""
    from PIL import Image, ImageDraw, ImageFont

    x0, y0, x1, y1 = box
    bh, bw = y1 - y0, x1 - x0
    disp = np.asarray(disp, dtype=np.float32)
    in_h, in_w = int(disp.shape[0]), int(disp.shape[1])
    if in_h <= 0 or in_w <= 0 or bh <= 0 or bw <= 0:
        return

    # If the schematic box has fewer pixels than tactile rows/cols, downsample (avoids zero-height strips).
    if in_h > bh or in_w > bw:
        disp = _resize_patch_nearest(disp, bh, bw)
        in_h, in_w = bh, bw

    row_e = _fair_pixel_edges(y0, bh, in_h)
    col_e = _fair_pixel_edges(x0, bw, in_w)

    for i in range(in_h):
        for j in range(in_w):
            ya0, ya1 = int(row_e[i]), int(row_e[i + 1])
            xa0, xa1 = int(col_e[j]), int(col_e[j + 1])
            if ya1 <= ya0:
                ya1 = ya0 + 1
            if xa1 <= xa0:
                xa1 = xa0 + 1
            ya1 = min(y1, ya1)
            xa1 = min(x1, xa1)
            cell_rgb = _scalar_to_rgb(disp[i : i + 1, j : j + 1], vmin, vmax, colormap)[0, 0]
            canvas[ya0:ya1, xa0:xa1, :] = cell_rgb

    if not (draw_grid or draw_values):
        return

    sub = canvas[y0:y1, x0:x1].copy()
    img = Image.fromarray(sub)
    draw = ImageDraw.Draw(img)

    font_cache: dict[int, Any] = {}

    def _font(sz: int) -> Any:
        if sz in font_cache:
            return font_cache[sz]
        font = None
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ):
            try:
                font = ImageFont.truetype(path, sz)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", sz)
            except Exception:
                font = ImageFont.load_default()
        font_cache[sz] = font
        return font

    for i in range(in_h):
        for j in range(in_w):
            ya0 = int(row_e[i]) - y0
            ya1 = int(row_e[i + 1]) - y0
            xa0 = int(col_e[j]) - x0
            xa1 = int(col_e[j + 1]) - x0
            if ya1 <= ya0:
                ya1 = ya0 + 1
            if xa1 <= xa0:
                xa1 = xa0 + 1
            ch, cw = ya1 - ya0, xa1 - xa0
            if draw_grid and ch > 1 and cw > 1:
                draw.rectangle([xa0, ya0, xa1 - 1, ya1 - 1], outline=(210, 210, 210), width=1)
            if draw_values and ch >= min_cell_px_for_text and cw >= min_cell_px_for_text:
                val = float(disp[i, j])
                if not np.isfinite(val):
                    continue
                txt = value_fmt.format(val)
                cy_g = (int(row_e[i]) + int(row_e[i + 1])) // 2
                cx_g = (int(col_e[j]) + int(col_e[j + 1])) // 2
                r, g, b = (int(x) for x in canvas[cy_g, cx_g, :])
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                fill = (16, 16, 16) if lum > 160 else (248, 248, 248)
                cx = (xa0 + xa1) // 2
                cy = (ya0 + ya1) // 2
                fs = int(max(7, min(ch, cw) // 4))
                fs = min(fs, 14)
                font = _font(fs)
                draw.text((cx, cy), txt, fill=fill, font=font, anchor="mm")

    canvas[y0:y1, x0:x1] = np.asarray(img)


_CMAP_CACHE: dict[str, Any] = {}


def _bry_heatmap_rgb_float(t: np.ndarray) -> np.ndarray:
    """Map ``t in [0,1]`` to RGB in ``[0,1]`` by **component-wise lerp in sRGB** (no colormap).

    Piecewise linear, **C0-continuous** at ``t=0.5``:

    * ``t in [0, 1/2]``: (0,0,1) → (1,0,0)   (蓝 → 红)
    * ``t in (1/2, 1]``: (1,0,0) → (1,1,0)   (红 → 黄)

    ``t`` can be any shape; return shape ``(*t.shape, 3)`` float32.
    """
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    out = np.empty(t.shape + (3,), dtype=np.float32)
    m = t <= 0.5
    u = np.where(m, t * 2.0, (t - 0.5) * 2.0)
    # First half: R=u, G=0, B=1-u
    r = np.where(m, u, 1.0)
    g = np.where(m, 0.0, u)
    b = np.where(m, 1.0 - u, 0.0)
    out[..., 0] = r
    out[..., 1] = g
    out[..., 2] = b
    return out


def _get_matplotlib_colormap(name: str):
    """Return a matplotlib ``Colormap``; ``heatmap``/``bry`` use :func:`_bry_heatmap_rgb_float` in :func:`_scalar_to_rgb` instead."""
    import matplotlib.colors as mcolors

    if name in _CMAP_CACHE:
        return _CMAP_CACHE[name]
    if name == "bry":
        cmap = mcolors.LinearSegmentedColormap.from_list("bry", [(0, 0, 1), (1, 0, 0), (1, 1, 0)])
    elif name == "wuji_heat":
        # Deep blue → purple/magenta → orange → yellow (reference-style tactile heatmap).
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "wuji_heat",
            ["#050528", "#201070", "#7A2088", "#E040A0", "#F87020", "#FFEA60"],
        )
    else:
        try:
            from matplotlib import colormaps

            cmap = colormaps[name]
        except (KeyError, ImportError):
            import matplotlib.cm as cm

            cmap = cm.get_cmap(name)
    _CMAP_CACHE[name] = cmap
    return cmap


def _scalar_to_rgb(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    colormap: str = "heatmap",
) -> np.ndarray:
    """Map scalar field to uint8 RGB. Default ``heatmap``/``bry``/``heatmap_bry`` = direct B→R→Y in RGB lerp; else matplotlib."""
    v = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(v)
    if not np.any(finite):
        return np.zeros((*v.shape, 3), dtype=np.uint8)
    if vmin is None:
        vmin = float(np.min(v[finite]))
    if vmax is None:
        vmax = float(np.max(v[finite]))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    t = (v - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    key = str(colormap).strip().lower()
    if key in ("heatmap", "heatmap_bry", "bry"):
        rgb = _bry_heatmap_rgb_float(t)
    else:
        cmap = _get_matplotlib_colormap(colormap)
        rgba = np.asarray(cmap(t))
        if rgba.ndim == 3 and rgba.shape[-1] >= 3:
            rgb = rgba[..., :3]
        else:
            rgb = rgba
    return (np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def strip_wuji_sensor_key_prefix(
    grids: dict[str, np.ndarray],
    prefix: str = "wuji_grid_tactile_",
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in grids.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


@dataclass
class HandTactilePlotCfg:
    """Rendering options for composite hand tactile figures."""

    canvas_width: int = 900
    canvas_height: int = 900
    aggregate_filters: Literal["sum", "max", "mean"] = "sum"
    """How to collapse filter/object dimension on grids shaped (F, H, W)."""
    normal_display: Literal["abs", "signed"] = "abs"
    """``abs``: colormap on |normal|; ``signed``: map raw signed n·f with the same heat colormap."""
    normal_vmin: float | None = None
    normal_vmax: float | None = None
    """If None, use min/max of composited non-background samples (per image)."""
    friction_arrow_subsample: int = 4
    """Approximate pixel step on resized patch for one arrow per cell block."""
    friction_scale: float = 11.0
    """Quiver scale (smaller → longer arrows). Matplotlib ``scale`` parameter."""
    friction_min_mag: float = 1e-9
    """Skip arrows with magnitude below this (after aggregation)."""
    friction_draw_cell_grid: bool = True
    """Draw light grid lines per tactile cell (same topology as the normal-force discrete layout)."""
    normal_discrete_cells: bool = True
    """If True, one solid color per tactile cell. If False, bilinear upsample + colormap (smooth between cells)."""
    normal_draw_cell_grid: bool = True
    """Draw light rectangle around each cell (discrete mode only)."""
    normal_draw_cell_values: bool = True
    """Overlay force scalar in each cell when large enough (discrete mode only)."""
    normal_min_cell_px_for_text: int = 10
    """Minimum cell height and width (px) to draw a numeric label."""
    normal_value_fmt: str = "{:.3g}"
    """Format for cell text (e.g. ``\"{:.2f}\"`` or ``\"{:.3g}\"``)."""
    schematic_fill_canvas: bool = True
    """Scale the hand schematic so its bounding box fills the canvas (minus margin)."""
    schematic_margin_frac: float = 0.03
    """Fractional margin on each side when ``schematic_fill_canvas`` is True."""
    schematic_mirror_x: bool = False
    """If True, flip the schematic left–right after layout (match camera / habit without swapping link data)."""
    normal_colormap: str = "heatmap"
    """``heatmap``/``bry``: RGB component lerp (蓝→红→黄). Others: matplotlib colormaps (e.g. ``wuji_heat``, ``turbo``)."""
    normal_scale: Literal["minmax", "percentile"] = "percentile"
    """How to set vmin/vmax when both are None: full range vs robust percentiles (better contrast)."""
    normal_percentile_low: float = 2.0
    normal_percentile_high: float = 98.0
    """Used when ``normal_scale == \"percentile\"`` and vmin/vmax are not set."""
    normal_plot_swap_uv_axes: bool = True
    """If True, transpose each link grid so body **v** maps to image **rows** and **u** to **columns**.

    Default **True** for Wuji-style layouts where **v** follows the finger toward the fingertip: the
    composite schematic runs digits **upward**, so **v** should be vertical on screen (matches viewport /
    pad orientation). **False** keeps raw sensor storage layout: ``force_grid[..., iu, iv]`` with **iu**
    → rows and **iv** → columns (older behavior; can look 90° rotated vs the palm mesh).
    """

    normal_plot_flip_rows: bool = True
    normal_plot_flip_cols: bool = True
    """After :attr:`normal_plot_swap_uv_axes`, flip row / column order (``flipud`` / ``fliplr``).

    Transpose alone can leave the patch **180°** relative to the schematic (index ``0`` at the wrong
    anatomical edge). Defaults **True** / **True** restore palmar alignment for Wuji; set both **False**
    if you only want a pure transpose. When :attr:`normal_plot_swap_uv_axes` is **False**, set these
    **False** as well unless you intend to mirror raw sensor indices.
    """


def render_normal_force_image(
    normal_grids: dict[str, np.ndarray],
    *,
    side: Literal["right", "left"] = "right",
    cfg: HandTactilePlotCfg | None = None,
) -> np.ndarray:
    """Return ``(H, W, 3)`` uint8 RGB image; background black.

    ``normal_grids`` maps **link names** (e.g. ``right_palm_link``) to arrays ``(H, W)`` or ``(F, H, W)``.
    """
    cfg = cfg or HandTactilePlotCfg()
    layout = schematic_layout_for_side(side)
    pix = norm_boxes_to_pixels(
        layout,
        cfg.canvas_width,
        cfg.canvas_height,
        fill_canvas=cfg.schematic_fill_canvas,
        margin_frac=cfg.schematic_margin_frac,
    )
    if cfg.schematic_mirror_x:
        pix = _mirror_pixel_boxes_x(pix, cfg.canvas_width)
    canvas = np.zeros((cfg.canvas_height, cfg.canvas_width, 3), dtype=np.uint8)

    collected: list[float] = []
    patches_disp: dict[str, tuple[np.ndarray, tuple[int, int, int, int]]] = {}

    for name, box in pix.items():
        if name not in normal_grids:
            continue
        arr = _to_numpy(normal_grids[name])
        arr = _aggregate_filters(arr, cfg.aggregate_filters)
        if cfg.normal_display == "abs":
            disp = np.abs(arr)
        else:
            disp = arr
        finite = np.isfinite(disp)
        if np.any(finite):
            collected.extend(disp[finite].ravel().tolist())
        patches_disp[name] = (disp.astype(np.float32), box)

    if cfg.normal_vmin is not None and cfg.normal_vmax is not None:
        vmin, vmax = cfg.normal_vmin, cfg.normal_vmax
    elif len(collected) == 0:
        vmin, vmax = 0.0, 1.0
    elif cfg.normal_scale == "percentile" and cfg.normal_vmin is None and cfg.normal_vmax is None:
        arr = np.asarray(collected, dtype=np.float64)
        pl = float(np.clip(cfg.normal_percentile_low, 0.0, 100.0))
        ph = float(np.clip(cfg.normal_percentile_high, 0.0, 100.0))
        if ph <= pl:
            ph = min(100.0, pl + 1.0)
        vmin = float(np.percentile(arr, pl))
        vmax = float(np.percentile(arr, ph))
        if vmax <= vmin:
            vmax = vmin + 1e-8
    else:
        vmin = float(np.min(collected)) if cfg.normal_vmin is None else cfg.normal_vmin
        vmax = float(np.max(collected)) if cfg.normal_vmax is None else cfg.normal_vmax
        if vmax <= vmin:
            vmax = vmin + 1e-8

    # Draw large regions first (palm) so smaller digit patches stay visible if boxes overlap slightly.
    for name, (disp, box) in sorted(patches_disp.items(), key=lambda it: -_box_area_pixel(it[1][1])):
        if disp.ndim == 2:
            disp = _orient_scalar_patch_for_schematic(
                disp,
                swap_uv=cfg.normal_plot_swap_uv_axes,
                flip_rows=cfg.normal_plot_flip_rows,
                flip_cols=cfg.normal_plot_flip_cols,
            )
        if cfg.normal_discrete_cells:
            _draw_discrete_normal_patch(
                canvas,
                disp,
                box,
                vmin,
                vmax,
                colormap=str(cfg.normal_colormap),
                draw_grid=cfg.normal_draw_cell_grid,
                draw_values=cfg.normal_draw_cell_values,
                min_cell_px_for_text=int(cfg.normal_min_cell_px_for_text),
                value_fmt=str(cfg.normal_value_fmt),
            )
        else:
            x0, y0, x1, y1 = box
            bh, bw = y1 - y0, x1 - x0
            small = _resize_patch(disp, bh, bw)
            rgb = _scalar_to_rgb(small, vmin, vmax, str(cfg.normal_colormap))
            canvas[y0:y1, x0:x1, :] = rgb

    return canvas


def render_friction_arrows_image(
    friction_uv_grids: dict[str, np.ndarray],
    *,
    side: Literal["right", "left"] = "right",
    cfg: HandTactilePlotCfg | None = None,
) -> np.ndarray:
    """Return ``(H, W, 3)`` uint8 RGB with tangential arrows (white on black).

    ``friction_uv_grids`` maps link names to ``(2, H, W)`` or ``(F, 2, H, W)`` with channel 0 = tangent
    **u**, 1 = **v** (same as :class:`~ViTacLab.assets.sensor.grid_tactile.GridTactileSensor`).

    **Frames:** ``fu``, ``fv`` are **body tangent-plane** scalars (same as the sensor: world vectors
    dotted with **world** ``t_u,t_v`` equals ``F_body·e_u``, ``F_body·e_v``). They are **not** world
    ``X/Y/Z`` and the schematic is **not** a camera projection of world axes.

    **Schematic pixel axes:** applies :attr:`~HandTactilePlotCfg.normal_plot_swap_uv_axes` and the
    row/column flips (:attr:`~HandTactilePlotCfg.normal_plot_flip_rows` / ``flip_cols``) to match
    :func:`render_normal_force_image`. Quiver ``U`` / ``V`` use ``U = fu``, ``V = fv`` after those
    transforms when swap is **True**, else ``U = fv``, ``V = fu``.
    """
    cfg = cfg or HandTactilePlotCfg()
    layout = schematic_layout_for_side(side)
    pix = norm_boxes_to_pixels(
        layout,
        cfg.canvas_width,
        cfg.canvas_height,
        fill_canvas=cfg.schematic_fill_canvas,
        margin_frac=cfg.schematic_margin_frac,
    )
    if cfg.schematic_mirror_x:
        pix = _mirror_pixel_boxes_x(pix, cfg.canvas_width)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(cfg.canvas_width / 100.0, cfg.canvas_height / 100.0),
        dpi=100,
    )
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_xlim(0, cfg.canvas_width)
    ax.set_ylim(cfg.canvas_height, 0)
    ax.set_aspect("equal")

    friction_items = sorted(
        ((n, pix[n]) for n in friction_uv_grids if n in pix),
        key=lambda it: -_box_area_pixel(it[1]),
    )

    for name, box in friction_items:
        arr = _to_numpy(friction_uv_grids[name])
        if arr.ndim == 4:
            # (F, 2, H, W) -> sum over F
            arr = np.sum(arr, axis=0)
        if arr.ndim != 3 or arr.shape[0] != 2:
            raise ValueError(f"{name}: expected (2,H,W) or (F,2,H,W), got {arr.shape}")
        fu, fv = arr[0], arr[1]
        fu = _aggregate_filters(fu[np.newaxis, ...], cfg.aggregate_filters)
        fv = _aggregate_filters(fv[np.newaxis, ...], cfg.aggregate_filters)
        fu, fv = _orient_friction_uv_for_schematic(
            fu,
            fv,
            swap_uv=cfg.normal_plot_swap_uv_axes,
            flip_rows=cfg.normal_plot_flip_rows,
            flip_cols=cfg.normal_plot_flip_cols,
        )

        x0, y0, x1, y1 = box
        bh, bw = y1 - y0, x1 - x0
        in_h, in_w = int(fu.shape[0]), int(fu.shape[1])
        if cfg.friction_draw_cell_grid and in_h > 0 and in_w > 0 and bh > 0 and bw > 0:
            _draw_discrete_grid_mpl(ax, box, in_h, in_w, zorder=1.0)

        fu_r = _resize_patch(fu.astype(np.float32), bh, bw)
        fv_r = _resize_patch(fv.astype(np.float32), bh, bw)

        step = max(1, int(cfg.friction_arrow_subsample))
        ys = np.arange(step // 2, bh, step, dtype=np.float32)
        xs = np.arange(step // 2, bw, step, dtype=np.float32)
        if ys.size == 0 or xs.size == 0:
            continue
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        # Map body (fu,fv) to quiver (U horizontal, V down): depends on whether rows = u or v (swap_uv).
        if cfg.normal_plot_swap_uv_axes:
            U = fu_r[Y.astype(int), X.astype(int)]
            Vimg = fv_r[Y.astype(int), X.astype(int)]
        else:
            U = fv_r[Y.astype(int), X.astype(int)]
            Vimg = fu_r[Y.astype(int), X.astype(int)]
        mag = np.sqrt(U * U + Vimg * Vimg)
        mask = mag >= cfg.friction_min_mag
        if not np.any(mask):
            continue
        Px = x0 + X[mask]
        Py = y0 + Y[mask]
        Qu = U[mask]
        Qv = Vimg[mask]
        ax.quiver(
            Px,
            Py,
            Qu,
            Qv,
            angles="xy",
            scale_units="xy",
            scale=cfg.friction_scale,
            color="white",
            width=0.0022,
            headwidth=3.0,
            headlength=4.0,
            headaxislength=3.5,
            zorder=2.0,
        )

    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    buf.seek(0)
    try:
        import imageio.v2 as imageio

        img = imageio.imread(buf)
    except Exception:
        from PIL import Image

        img = np.asarray(Image.open(buf).convert("RGB"))
    if img.shape[0] != cfg.canvas_height or img.shape[1] != cfg.canvas_width:
        from PIL import Image

        img = np.asarray(
            Image.fromarray(img).resize((cfg.canvas_width, cfg.canvas_height), Image.Resampling.BILINEAR)
        )
    return img.astype(np.uint8)


def render_hand_tactile_pair(
    normal_grids: dict[str, np.ndarray],
    friction_uv_grids: dict[str, np.ndarray],
    *,
    side: Literal["right", "left"] = "right",
    cfg: HandTactilePlotCfg | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: normal heatmap image and tangential arrow image."""
    return (
        render_normal_force_image(normal_grids, side=side, cfg=cfg),
        render_friction_arrows_image(friction_uv_grids, side=side, cfg=cfg),
    )
