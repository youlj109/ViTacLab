# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Marker motion simulation for optical tactile sensors (FOTS-style, TacEx-compatible).

TacEx [2411.04776] couples Taxim (RGB) with FOTS [2404.19217] (marker displacement field).
ViTacSim applies marker overlay **after** Taxim rendering using height-map-driven displacements.

Supported patterns:
- ``gelsight``: sparse black dots (GelSight / GelSight Mini style).
- ``xense``: denser staggered black dots (lab Xense-style layout, black small circles).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import torch

PatternName = Literal["gelsight", "xense", "none"]


class MarkerPattern(str, Enum):
    NONE = "none"
    GELSIGHT = "gelsight"
    XENSE = "xense"


@dataclass(frozen=True)
class MarkerPatternSpec:
    """Rest marker layout and appearance."""

    name: str
    grid_rows: int
    grid_cols: int
    radius_px: float
    color_rgb: tuple[int, int, int]
    margin_frac: float = 0.12
    stagger_odd_rows: bool = False


PATTERN_SPECS: dict[str, MarkerPatternSpec] = {
    MarkerPattern.GELSIGHT.value: MarkerPatternSpec(
        name="gelsight",
        grid_rows=10,
        grid_cols=10,
        radius_px=4.0,
        color_rgb=(0, 0, 0),
        margin_frac=0.14,
        stagger_odd_rows=False,
    ),
    MarkerPattern.XENSE.value: MarkerPatternSpec(
        name="xense",
        grid_rows=14,
        grid_cols=14,
        radius_px=2.0,
        color_rgb=(0, 0, 0),
        margin_frac=0.10,
        stagger_odd_rows=True,
    ),
}


def _rest_marker_positions(spec: MarkerPatternSpec, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Return rest marker centers in pixel coordinates (x, y), shape (M, 2)."""
    margin_x = spec.margin_frac * width
    margin_y = spec.margin_frac * height
    xs = torch.linspace(margin_x, width - margin_x, spec.grid_cols, device=device)
    ys = torch.linspace(margin_y, height - margin_y, spec.grid_rows, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    if spec.stagger_odd_rows:
        row_idx = torch.arange(spec.grid_rows, device=device).repeat_interleave(spec.grid_cols)
        stagger = (row_idx % 2) * (0.5 * (xs[1] - xs[0]) if spec.grid_cols > 1 else 0.0)
        pos[:, 0] = pos[:, 0] + stagger
    return pos


def _contact_points_from_height(
    height_mm: torch.Tensor,
    *,
    deadband_mm: float,
    max_contacts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample contact cells from a single height map (H, W) in Taxim mm units.

    Returns:
        positions (K, 2) pixel (x, y), heights (K,) mm penetration.
    """
    h, w = height_mm.shape
    mask = height_mm > deadband_mm
    if not bool(mask.any()):
        return (
            torch.zeros(0, 2, device=height_mm.device, dtype=torch.float32),
            torch.zeros(0, device=height_mm.device, dtype=torch.float32),
        )
    ys, xs = torch.where(mask)
    vals = height_mm[ys, xs]
    if vals.numel() > max_contacts:
        idx = torch.linspace(0, vals.numel() - 1, max_contacts, device=height_mm.device).long()
        ys, xs, vals = ys[idx], xs[idx], vals[idx]
    pos = torch.stack((xs.float(), ys.float()), dim=-1)
    return pos, vals


def compute_marker_displacements_fots_dilate(
    height_mm: torch.Tensor,
    marker_rest_xy: torch.Tensor,
    *,
    lambda_d: float = 0.0025,
    max_contacts: int = 256,
    deadband_mm: float = 0.02,
    displacement_gain: float = 0.35,
) -> torch.Tensor:
    """FOTS-style dilate displacement from normal indentation (Eq. 11 simplified).

    Args:
        height_mm: (H, W) penetration height in Taxim mm units.
        marker_rest_xy: (M, 2) rest positions in pixels.
        lambda_d: Gaussian falloff from contact cell to marker (1/pixel^2 scale).
        displacement_gain: Scales summed displacement vectors.

    Returns:
        (M, 2) displacement in pixels (dx, dy).
    """
    m_count = marker_rest_xy.shape[0]
    device = height_mm.device
    if m_count == 0:
        return torch.zeros(0, 2, device=device)

    contacts, heights = _contact_points_from_height(height_mm, deadband_mm=deadband_mm, max_contacts=max_contacts)
    if contacts.shape[0] == 0:
        return torch.zeros(m_count, 2, device=device)

    # (M, K)
    diff = marker_rest_xy.unsqueeze(1) - contacts.unsqueeze(0)
    dist2 = (diff**2).sum(dim=-1)
    weights = torch.exp(-float(lambda_d) * dist2) * heights.unsqueeze(0)
    disp = (weights.unsqueeze(-1) * diff).sum(dim=1) * float(displacement_gain)
    return disp


def compute_marker_displacements_shear_proxy(
    height_mm: torch.Tensor,
    marker_rest_xy: torch.Tensor,
    *,
    shear_gain: float = 8.0,
) -> torch.Tensor:
    """Proxy shear marker motion from height-map gradient (used when object twist is unavailable)."""
    if marker_rest_xy.shape[0] == 0:
        return torch.zeros(0, 2, device=height_mm.device)
    gy, gx = torch.gradient(height_mm, dim=(0, 1))
    xi = marker_rest_xy[:, 0].long().clamp(0, height_mm.shape[1] - 1)
    yi = marker_rest_xy[:, 1].long().clamp(0, height_mm.shape[0] - 1)
    dx = -float(shear_gain) * gx[yi, xi]
    dy = -float(shear_gain) * gy[yi, xi]
    return torch.stack((dx, dy), dim=-1)


def compute_marker_displacements_shear_field(
    shear_disp_px: torch.Tensor,
    marker_rest_xy: torch.Tensor,
) -> torch.Tensor:
    """Sample a dense shear displacement field (H, W, 2) in pixels at marker rest positions."""
    if marker_rest_xy.shape[0] == 0:
        return torch.zeros(0, 2, device=shear_disp_px.device)
    if shear_disp_px.ndim != 3 or shear_disp_px.shape[-1] != 2:
        raise ValueError(f"shear_disp_px must be (H, W, 2), got {tuple(shear_disp_px.shape)}")

    h, w = shear_disp_px.shape[0], shear_disp_px.shape[1]
    grid_x = 2.0 * marker_rest_xy[:, 0] / max(float(w - 1), 1.0) - 1.0
    grid_y = 2.0 * marker_rest_xy[:, 1] / max(float(h - 1), 1.0) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, -1, 2)
    field = shear_disp_px.permute(2, 0, 1).unsqueeze(0)
    sampled = torch.nn.functional.grid_sample(
        field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(1).permute(1, 0)


class MarkerSimulator:
    """Draw displaced markers on Taxim RGB images."""

    def __init__(
        self,
        *,
        pattern: PatternName = "gelsight",
        image_height: int,
        image_width: int,
        device: str | torch.device,
        lambda_d: float = 0.0025,
        displacement_gain: float = 0.35,
        shear_gain: float = 8.0,
        deadband_mm: float = 0.02,
        max_contacts: int = 256,
        blend_alpha: float = 0.92,
        max_displacement_px: float = 25.0,
        rest_xy_override: np.ndarray | torch.Tensor | None = None,
    ):
        self.device = torch.device(device)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.lambda_d = float(lambda_d)
        self.displacement_gain = float(displacement_gain)
        self.shear_gain = float(shear_gain)
        self.deadband_mm = float(deadband_mm)
        self.max_contacts = int(max_contacts)
        self.blend_alpha = float(blend_alpha)
        self.max_displacement_px = float(max_displacement_px)

        if pattern == "none":
            self.enabled = False
            self.spec: MarkerPatternSpec | None = None
            self.rest_xy = torch.zeros(0, 2, device=self.device)
        else:
            self.enabled = True
            self.spec = PATTERN_SPECS[pattern]
            if rest_xy_override is not None:
                rest = torch.as_tensor(rest_xy_override, dtype=torch.float32, device=self.device)
                if rest.ndim != 2 or rest.shape[-1] != 2:
                    raise ValueError(f"rest_xy_override must be (M,2), got {tuple(rest.shape)}")
                self.rest_xy = rest
            else:
                self.rest_xy = _rest_marker_positions(self.spec, image_height, image_width, self.device)

    @property
    def num_markers(self) -> int:
        return int(self.rest_xy.shape[0])

    def displacements_from_height_mm(
        self,
        height_mm: torch.Tensor,
        *,
        shear_disp_px: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute (M, 2) displacements for one height map (H, W)."""
        if not self.enabled:
            return torch.zeros(0, 2, device=self.device)
        d_dilate = compute_marker_displacements_fots_dilate(
            height_mm,
            self.rest_xy,
            lambda_d=self.lambda_d,
            max_contacts=self.max_contacts,
            deadband_mm=self.deadband_mm,
            displacement_gain=self.displacement_gain,
        )
        d_shear = compute_marker_displacements_shear_proxy(height_mm, self.rest_xy, shear_gain=self.shear_gain)
        d_force = (
            compute_marker_displacements_shear_field(shear_disp_px, self.rest_xy)
            if shear_disp_px is not None
            else torch.zeros_like(d_shear)
        )
        disp = d_dilate + d_shear + d_force
        if self.max_displacement_px > 0.0:
            mag = torch.linalg.norm(disp, dim=-1, keepdim=True).clamp(min=1e-9)
            scale = torch.clamp(self.max_displacement_px / mag, max=1.0)
            disp = disp * scale
        return disp

    def draw_markers_on_image(self, rgb: torch.Tensor, displaced_xy: torch.Tensor) -> torch.Tensor:
        """Composite markers onto RGB image (H, W, 3) uint8."""
        if not self.enabled or self.spec is None or displaced_xy.shape[0] == 0:
            return rgb
        out = rgb.clone()
        color = torch.tensor(self.spec.color_rgb, device=out.device, dtype=out.dtype)
        r = int(max(1, round(self.spec.radius_px)))
        h, w = out.shape[0], out.shape[1]
        for i in range(displaced_xy.shape[0]):
            cx = int(torch.round(displaced_xy[i, 0]).item())
            cy = int(torch.round(displaced_xy[i, 1]).item())
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)
            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy = torch.arange(y0, y1, device=out.device).view(-1, 1).float()
            xx = torch.arange(x0, x1, device=out.device).view(1, -1).float()
            mask = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= float(r * r)
            patch = out[y0:y1, x0:x1]
            patch[mask] = (self.blend_alpha * color + (1.0 - self.blend_alpha) * patch[mask]).to(out.dtype)
            out[y0:y1, x0:x1] = patch
        return out

    def composite_batch(
        self,
        rgb_batch: torch.Tensor,
        height_mm_batch: torch.Tensor,
        *,
        shear_disp_px_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply markers to batch (N,H,W,3) using height maps already in Taxim mm (N,H,W).

        Args:
            rgb_batch: Taxim RGB batch (N, H, W, 3).
            height_mm_batch: FOTS height maps in Taxim mm (N, H, W).
            shear_disp_px_batch: Optional ViT shear displacement field in pixels (N, H, W, 2).

        Returns:
            rgb_with_markers, marker_displacements (N, M, 2)
        """
        if not self.enabled:
            z = torch.zeros(rgb_batch.shape[0], 0, 2, device=rgb_batch.device)
            return rgb_batch, z

        n = rgb_batch.shape[0]
        m = self.rest_xy.shape[0]
        all_disp = torch.zeros(n, m, 2, device=rgb_batch.device)
        out = rgb_batch.clone()
        for i in range(n):
            shear_i = None if shear_disp_px_batch is None else shear_disp_px_batch[i]
            disp = self.displacements_from_height_mm(height_mm_batch[i], shear_disp_px=shear_i)
            all_disp[i] = disp
            pos = self.rest_xy + disp
            out[i] = self.draw_markers_on_image(out[i], pos)
        return out, all_disp
