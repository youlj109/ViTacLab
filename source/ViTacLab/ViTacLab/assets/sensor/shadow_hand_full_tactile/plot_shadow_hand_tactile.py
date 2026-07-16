# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live matplotlib views for :class:`~ViTacLab.assets.sensor.shadow_hand_full_tactile.ShadowHandFullTactileSensor`.

* **Schematic** — palm-frame voxel data cropped into :func:`shadow_hand_schematic_layout` regions
  (same idea as Wuji :func:`~ViTacLab.assets.sensor.wuji_hand_grid_tactile.plot_hand_tactile.render_hand_tactile_pair`).
* **2D** — optional raw max-projection of the voxel grid onto the palm +X/+Y plane.
* **3D** — occupied voxels in palm coordinates, colored by ``|fn| + ‖ft‖``.
"""

from __future__ import annotations

import inspect
import io
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from ..wuji_hand_grid_tactile.plot_hand_tactile import _resize_patch, _scalar_to_rgb
from .shadow_hand_tactile_layout import (
    layout_union_bounds,
    norm_boxes_to_pixels,
    shadow_hand_schematic_layout,
)


def _to_numpy(x: torch.Tensor | np.ndarray, *, dtype=np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def voxel_contact_intensity(voxel: torch.Tensor | np.ndarray) -> np.ndarray:
    """Scalar contact strength per cell: ``|fn| + ‖ft‖`` from ``(..., 3)`` voxel channels."""
    v = _to_numpy(voxel)
    if v.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3 [fn, ft1, ft2], got shape {v.shape}")
    fn = np.abs(v[..., 0])
    f1, f2 = v[..., 1], v[..., 2]
    ft = np.sqrt(np.maximum(f1 * f1 + f2 * f2, 0.0))
    return (fn + ft).astype(np.float32)


def max_project_voxel_intensity(
    voxel: torch.Tensor | np.ndarray,
    *,
    axis: Literal["x", "y", "z"] = "z",
    reduce: Literal["max", "sum"] = "max",
) -> np.ndarray:
    """Collapse one palm axis; default ``z`` yields a palm **+X/+Y** heatmap (looking along +Z)."""
    intensity = voxel_contact_intensity(voxel)
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    if reduce == "max":
        return np.max(intensity, axis=ax).astype(np.float32)
    return np.sum(intensity, axis=ax).astype(np.float32)


def _intensity_rgba(
    intensity: np.ndarray,
    *,
    cmap_name: str,
    vmax: float | None,
    alpha: float,
) -> np.ndarray:
    import matplotlib

    norm_vmax = float(intensity.max()) if vmax is None else float(vmax)
    norm_vmax = max(norm_vmax, 1e-9)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(intensity / norm_vmax)
    rgba[..., 3] = alpha
    return rgba


def _filled_mask(int_np: np.ndarray) -> np.ndarray:
    imax = float(np.max(int_np)) if int_np.size else 0.0
    if imax <= 0.0:
        return np.zeros(int_np.shape, dtype=bool)
    eps = max(imax * 1e-9, np.finfo(np.float64).tiny * 1e3)
    return int_np > eps


def _mirror_pixel_boxes_x(
    pix: dict[str, tuple[int, int, int, int]], canvas_width: int
) -> dict[str, tuple[int, int, int, int]]:
    w = int(canvas_width)
    if w <= 0:
        return pix
    out: dict[str, tuple[int, int, int, int]] = {}
    for k, (x0, y0, x1, y1) in pix.items():
        out[k] = (w - x1, y0, w - x0, y1)
    return out


def _box_area_pixel(box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def _schematic_box_to_palm_roi(
    box_norm: tuple[float, float, float, float],
    layout_bounds: tuple[float, float, float, float],
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Map normalized schematic box to palm-frame ``(x0, y0, x1, y1)`` metres (+Y up)."""
    lx0, ly0, lx1, ly1 = layout_bounds
    nx0, ny0, nx1, ny1 = box_norm
    span_x = max(float(lx1 - lx0), 1e-6)
    span_y = max(float(ly1 - ly0), 1e-6)
    px0 = bmin[0] + (nx0 - lx0) / span_x * (bmax[0] - bmin[0])
    px1 = bmin[0] + (nx1 - lx0) / span_x * (bmax[0] - bmin[0])
    # Schematic image y grows downward; palm +Y grows upward.
    py1 = bmax[1] - (ny0 - ly0) / span_y * (bmax[1] - bmin[1])
    py0 = bmax[1] - (ny1 - ly0) / span_y * (bmax[1] - bmin[1])
    return (min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1))


def _palm_roi_to_voxel_slices(
    roi: tuple[float, float, float, float],
    nx: int,
    ny: int,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
) -> tuple[int, int, int, int]:
    """Integer index ranges ``ix0:ix1``, ``iy0:iy1`` for ``voxel[ix, iy, ...]``."""
    x0, y0, x1, y1 = roi
    bx0, by0, bx1, by1 = bmin[0], bmin[1], bmax[0], bmax[1]
    sx = max(bx1 - bx0, 1e-9)
    sy = max(by1 - by0, 1e-9)
    ix0 = int(np.floor((x0 - bx0) / sx * nx))
    ix1 = int(np.ceil((x1 - bx0) / sx * nx))
    iy0 = int(np.floor((y0 - by0) / sy * ny))
    iy1 = int(np.ceil((y1 - by0) / sy * ny))
    ix0 = int(np.clip(ix0, 0, nx))
    ix1 = int(np.clip(ix1, 0, nx))
    iy0 = int(np.clip(iy0, 0, ny))
    iy1 = int(np.clip(iy1, 0, ny))
    if ix1 <= ix0:
        ix1 = min(nx, ix0 + 1)
    if iy1 <= iy0:
        iy1 = min(ny, iy0 + 1)
    return ix0, ix1, iy0, iy1


def _extract_palm_patches_from_voxel(
    voxel: np.ndarray,
    *,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    layout_bounds: tuple[float, float, float, float],
    layout: dict[str, tuple[float, float, float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Per-link ``|fn|`` patches and ``(ft1, ft2)`` pairs from one palm-frame voxel volume."""
    if voxel.ndim != 4 or voxel.shape[-1] != 3:
        raise ValueError(f"Expected voxel (nx, ny, nz, 3), got {voxel.shape}")
    nx, ny, _nz, _ = voxel.shape
    normal: dict[str, np.ndarray] = {}
    friction: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fn_abs = np.abs(voxel[..., 0])
    ft1 = voxel[..., 1]
    ft2 = voxel[..., 2]
    ft_mag = np.sqrt(np.maximum(ft1 * ft1 + ft2 * ft2, 0.0))
    intensity = fn_abs + ft_mag

    for name, box_norm in layout.items():
        roi = _schematic_box_to_palm_roi(box_norm, layout_bounds, bmin, bmax)
        ix0, ix1, iy0, iy1 = _palm_roi_to_voxel_slices(roi, nx, ny, bmin, bmax)
        if ix1 <= ix0 or iy1 <= iy0:
            normal[name] = np.zeros((1, 1), dtype=np.float32)
            friction[name] = (
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            )
            continue
        sub_i = intensity[ix0:ix1, iy0:iy1, :]
        if sub_i.size == 0:
            normal[name] = np.zeros((1, 1), dtype=np.float32)
            friction[name] = (
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            )
            continue
        iz = np.argmax(sub_i, axis=2)
        fn_patch = np.take_along_axis(fn_abs[ix0:ix1, iy0:iy1, :], iz[..., None], axis=2).squeeze(-1)
        f1_patch = np.take_along_axis(ft1[ix0:ix1, iy0:iy1, :], iz[..., None], axis=2).squeeze(-1)
        f2_patch = np.take_along_axis(ft2[ix0:ix1, iy0:iy1, :], iz[..., None], axis=2).squeeze(-1)
        normal[name] = fn_patch.astype(np.float32)
        friction[name] = (f1_patch.astype(np.float32), f2_patch.astype(np.float32))
    return normal, friction


def render_shadow_hand_schematic_normal(
    voxel: torch.Tensor | np.ndarray,
    *,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    cfg: "ShadowHandTactilePlotCfg | None" = None,
) -> np.ndarray:
    """Return ``(H, W, 3)`` uint8 RGB schematic normal-force heatmap."""
    cfg = cfg or ShadowHandTactilePlotCfg()
    v = _to_numpy(voxel)
    layout = shadow_hand_schematic_layout()
    layout_bounds = layout_union_bounds(layout)
    normal, _ = _extract_palm_patches_from_voxel(
        v, bmin=bmin, bmax=bmax, layout_bounds=layout_bounds, layout=layout
    )
    pix = norm_boxes_to_pixels(
        layout,
        cfg.schematic_canvas_width,
        cfg.schematic_canvas_height,
        fill_canvas=cfg.schematic_fill_canvas,
        margin_frac=cfg.schematic_margin_frac,
    )
    if cfg.schematic_mirror_x:
        pix = _mirror_pixel_boxes_x(pix, cfg.schematic_canvas_width)

    collected: list[float] = []
    for name, patch in normal.items():
        finite = np.isfinite(patch)
        if np.any(finite):
            collected.extend(patch[finite].ravel().tolist())

    if cfg.schematic_normal_vmax is not None:
        vmax = float(cfg.schematic_normal_vmax)
        vmin = 0.0
    elif len(collected) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = 0.0
        vmax = float(np.max(collected))
        if vmax <= vmin:
            vmax = vmin + 1e-8

    canvas = np.zeros((cfg.schematic_canvas_height, cfg.schematic_canvas_width, 3), dtype=np.uint8)
    for name, box in sorted(pix.items(), key=lambda it: -_box_area_pixel(it[1])):
        if name not in normal:
            continue
        patch = normal[name]
        x0, y0, x1, y1 = box
        bh, bw = y1 - y0, x1 - x0
        if bh <= 0 or bw <= 0:
            continue
        small = _resize_patch(patch, bh, bw)
        canvas[y0:y1, x0:x1, :] = _scalar_to_rgb(small, vmin, vmax, cfg.schematic_colormap)
    return canvas


def render_shadow_hand_schematic_friction(
    voxel: torch.Tensor | np.ndarray,
    *,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    cfg: "ShadowHandTactilePlotCfg | None" = None,
) -> np.ndarray:
    """Return ``(H, W, 3)`` uint8 RGB schematic tangential arrows (white on black)."""
    cfg = cfg or ShadowHandTactilePlotCfg()
    v = _to_numpy(voxel)
    layout = shadow_hand_schematic_layout()
    layout_bounds = layout_union_bounds(layout)
    _, friction = _extract_palm_patches_from_voxel(
        v, bmin=bmin, bmax=bmax, layout_bounds=layout_bounds, layout=layout
    )
    pix = norm_boxes_to_pixels(
        layout,
        cfg.schematic_canvas_width,
        cfg.schematic_canvas_height,
        fill_canvas=cfg.schematic_fill_canvas,
        margin_frac=cfg.schematic_margin_frac,
    )
    if cfg.schematic_mirror_x:
        pix = _mirror_pixel_boxes_x(pix, cfg.schematic_canvas_width)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(cfg.schematic_canvas_width / 100.0, cfg.schematic_canvas_height / 100.0),
        dpi=100,
    )
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_xlim(0, cfg.schematic_canvas_width)
    ax.set_ylim(cfg.schematic_canvas_height, 0)
    ax.set_aspect("equal")

    for name, box in sorted(
        ((n, pix[n]) for n in friction if n in pix),
        key=lambda it: -_box_area_pixel(it[1]),
    ):
        fu, fv = friction[name]
        x0, y0, x1, y1 = box
        bh, bw = y1 - y0, x1 - x0
        if bh <= 0 or bw <= 0:
            continue
        fu_r = _resize_patch(fu.astype(np.float32), bh, bw)
        fv_r = _resize_patch(fv.astype(np.float32), bh, bw)
        step = max(1, int(cfg.friction_arrow_subsample))
        ys = np.arange(step // 2, bh, step, dtype=np.float32)
        xs = np.arange(step // 2, bw, step, dtype=np.float32)
        if ys.size == 0 or xs.size == 0:
            continue
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        U = fu_r[Y.astype(int), X.astype(int)]
        Vimg = fv_r[Y.astype(int), X.astype(int)]
        mag = np.sqrt(U * U + Vimg * Vimg)
        mask = mag >= cfg.friction_min_mag
        if not np.any(mask):
            continue
        Px = x0 + X[mask]
        Py = y0 + Y[mask]
        ax.quiver(
            Px,
            Py,
            U[mask],
            Vimg[mask],
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
    fig.savefig(buf, format="png", facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    try:
        import imageio.v2 as imageio

        img = imageio.imread(buf)
    except Exception:
        from PIL import Image

        img = np.asarray(Image.open(buf).convert("RGB"))
    if img.shape[0] != cfg.schematic_canvas_height or img.shape[1] != cfg.schematic_canvas_width:
        from PIL import Image

        img = np.asarray(
            Image.fromarray(img).resize(
                (cfg.schematic_canvas_width, cfg.schematic_canvas_height),
                Image.Resampling.BILINEAR,
            )
        )
    return img.astype(np.uint8)


def render_shadow_hand_tactile_pair(
    voxel: torch.Tensor | np.ndarray,
    *,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    cfg: "ShadowHandTactilePlotCfg | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Schematic normal heatmap + tangential arrow image (Wuji ``render_hand_tactile_pair`` analogue)."""
    cfg = cfg or ShadowHandTactilePlotCfg()
    return (
        render_shadow_hand_schematic_normal(voxel, bmin=bmin, bmax=bmax, cfg=cfg),
        render_shadow_hand_schematic_friction(voxel, bmin=bmin, bmax=bmax, cfg=cfg),
    )


@dataclass
class ShadowHandTactilePlotCfg:
    """Rendering options for live shadow-hand voxel plots."""

    cmap: str = "magma"
    voxel_alpha: float = 0.85
    show_3d: bool = True
    show_2d: bool = False
    """Raw palm-frame max-projection (off by default; use schematic composite instead)."""
    show_schematic: bool = True
    """Hand-shaped schematic composite from :mod:`shadow_hand_tactile_layout`."""
    project_axis: Literal["x", "y", "z"] = "z"
    project_reduce: Literal["max", "sum"] = "max"
    pause_s: float = 0.001
    schematic_canvas_width: int = 540
    schematic_canvas_height: int = 540
    schematic_fill_canvas: bool = True
    schematic_margin_frac: float = 0.03
    schematic_mirror_x: bool = False
    schematic_colormap: str = "heatmap"
    schematic_normal_vmax: float | None = None
    friction_arrow_subsample: int = 4
    friction_scale: float = 11.0
    friction_min_mag: float = 1e-9


@dataclass
class ShadowHandTactileLivePlot:
    """Handles returned by :func:`open_shadow_hand_tactile_live_plot`."""

    fig: object
    ax_2d: object | None = None
    ax_3d: object | None = None
    ax_schematic_n: object | None = None
    ax_schematic_f: object | None = None
    im_2d: object | None = field(default=None, repr=False)
    im_schematic_n: object | None = field(default=None, repr=False)
    im_schematic_f: object | None = field(default=None, repr=False)
    scatter_2d: object | None = field(default=None, repr=False)
    scatter_3d: object | None = field(default=None, repr=False)
    cfg: ShadowHandTactilePlotCfg = field(default_factory=ShadowHandTactilePlotCfg)


def _apply_palm_voxel_view_limits(ax, bmin: tuple[float, float, float], bmax: tuple[float, float, float]) -> None:
    xm, ym, zm = bmin
    xM, yM, zM = bmax
    cx, cy, cz = 0.5 * (xm + xM), 0.5 * (ym + yM), 0.5 * (zm + zM)
    sx, sy, sz = xM - xm, yM - ym, zM - zm
    ax.set_xlim(cx - sx, cx + sx)
    ax.set_ylim(cy - sy, cy + sy)
    ax.set_zlim(cz - sz, cz + sz)
    ax.set_box_aspect((sx, sy, sz))


def open_shadow_hand_tactile_live_plot(
    *,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    cfg: ShadowHandTactilePlotCfg | None = None,
) -> ShadowHandTactileLivePlot:
    """Open a non-blocking matplotlib window (schematic composite, optional 2D/3D voxel views)."""
    import matplotlib.pyplot as plt

    cfg = cfg or ShadowHandTactilePlotCfg()
    plt.ion()

    ncols = int(cfg.show_schematic) * 2 + int(cfg.show_2d) + int(cfg.show_3d)
    if ncols == 0:
        raise ValueError("At least one of show_schematic / show_2d / show_3d must be True.")

    fig_w = 4.0 * ncols + 0.8
    fig = plt.figure(num="Shadow Hand full tactile — palm frame", figsize=(fig_w, 4.8))
    session = ShadowHandTactileLivePlot(fig=fig, cfg=cfg)
    col = 1
    ch, cw = cfg.schematic_canvas_height, cfg.schematic_canvas_width
    z = np.zeros((ch, cw, 3), dtype=np.uint8)

    if cfg.show_schematic:
        ax_n = fig.add_subplot(1, ncols, col)
        col += 1
        ax_f = fig.add_subplot(1, ncols, col)
        col += 1
        im_n = ax_n.imshow(z, origin="upper", interpolation="nearest")
        im_f = ax_f.imshow(z, origin="upper", interpolation="nearest")
        ax_n.set_title("|fn| (schematic hand)")
        ax_f.set_title("Tangential friction (arrows)")
        for ax in (ax_n, ax_f):
            ax.set_xticks([])
            ax.set_yticks([])
        session.ax_schematic_n = ax_n
        session.ax_schematic_f = ax_f
        session.im_schematic_n = im_n
        session.im_schematic_f = im_f

    if cfg.show_2d:
        ax2 = fig.add_subplot(1, ncols, col)
        col += 1
        ax2.set_title(f"max projection (palm +{cfg.project_axis.upper()})")
        ax2.set_xlabel("palm +X (m)")
        ax2.set_ylabel("palm +Y (m)")
        ax2.set_aspect("equal")
        dummy = np.zeros((2, 2), dtype=np.float32)
        im = ax2.imshow(
            dummy,
            origin="lower",
            extent=(bmin[0], bmax[0], bmin[1], bmax[1]),
            cmap=cfg.cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="|fn| + ‖ft‖")
        session.ax_2d = ax2
        session.im_2d = im

    if cfg.show_3d:
        ax3 = fig.add_subplot(1, ncols, col, projection="3d")
        _apply_palm_voxel_view_limits(ax3, bmin, bmax)
        ax3.set_xlabel("palm +X (m)")
        ax3.set_ylabel("palm +Y (m)")
        ax3.set_zlabel("palm +Z (m)")
        ax3.set_title("3D voxels (|fn| + ‖ft‖)")
        session.ax_3d = ax3

    fig.tight_layout()
    fig.show()
    return session


def update_shadow_hand_tactile_live_plot(
    session: ShadowHandTactileLivePlot,
    voxel: torch.Tensor | np.ndarray,
    *,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    mean_palm: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Refresh 2D/3D views from a single ``(nx, ny, nz, 3)`` voxel slice."""
    import matplotlib.pyplot as plt

    cfg = session.cfg
    intensity = voxel_contact_intensity(voxel)
    int_np = intensity
    vmax = float(int_np.max()) if int_np.size else 1.0
    vmax = max(vmax, 1e-9)

    if session.im_schematic_n is not None and session.im_schematic_f is not None:
        img_n, img_f = render_shadow_hand_tactile_pair(voxel, bmin=bmin, bmax=bmax, cfg=cfg)
        session.im_schematic_n.set_data(img_n)
        session.im_schematic_f.set_data(img_f)

    if session.ax_2d is not None and session.im_2d is not None:
        proj = max_project_voxel_intensity(voxel, axis=cfg.project_axis, reduce=cfg.project_reduce)
        session.im_2d.set_data(proj)
        session.im_2d.set_clim(0.0, vmax)
        if session.scatter_2d is not None:
            session.scatter_2d.remove()
            session.scatter_2d = None
        if mean_palm is not None:
            mp = _to_numpy(mean_palm).reshape(3)
            if np.isfinite(mp).all():
                session.scatter_2d = session.ax_2d.scatter(
                    [mp[0]],
                    [mp[1]],
                    c="lime",
                    s=80,
                    edgecolors="k",
                    linewidths=0.6,
                    zorder=5,
                )

    if session.ax_3d is not None:
        ax = session.ax_3d
        filled = _filled_mask(int_np)
        for c in list(ax.collections):
            c.remove()
        if session.scatter_3d is not None:
            try:
                session.scatter_3d.remove()
            except Exception:
                pass
            session.scatter_3d = None

        if bool(filled.any()):
            nx, ny, nz = int_np.shape
            rgba = _intensity_rgba(int_np, cmap_name=cfg.cmap, vmax=vmax, alpha=cfg.voxel_alpha)
            x_e = np.linspace(bmin[0], bmax[0], nx + 1, dtype=np.float64)
            y_e = np.linspace(bmin[1], bmax[1], ny + 1, dtype=np.float64)
            z_e = np.linspace(bmin[2], bmax[2], nz + 1, dtype=np.float64)
            kw = dict(facecolors=rgba, edgecolor="k", linewidth=0.06, shade=False)
            if "x" in inspect.signature(ax.voxels).parameters:
                ax.voxels(filled, **kw, x=x_e, y=y_e, z=z_e)
            else:
                ax.voxels(filled, **kw)

        if mean_palm is not None:
            mp = _to_numpy(mean_palm).reshape(3)
            if np.isfinite(mp).all():
                session.scatter_3d = ax.scatter(
                    [mp[0]],
                    [mp[1]],
                    [mp[2]],
                    c="lime",
                    s=120,
                    depthshade=True,
                    edgecolors="k",
                    linewidths=0.8,
                    zorder=10,
                )
        _apply_palm_voxel_view_limits(ax, bmin, bmax)

    session.fig.canvas.draw_idle()
    session.fig.canvas.flush_events()
    plt.pause(cfg.pause_s)
