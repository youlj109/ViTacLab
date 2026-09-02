# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VisuoTactileSensorData:
    """Data container for the visuo-tactile sensor.

    This class contains the tactile sensor data that includes:

    - Camera-based tactile sensing (RGB and depth images)
    - Force field tactile sensing (normal and shear forces)
    - Tactile point positions and contact information

    """

    # Camera-based tactile data
    tactile_depth_image: torch.Tensor | None = None
    """Tactile depth images. Shape is (num_instances, height, width, 1)."""

    tactile_rgb_image: torch.Tensor | None = None
    """Tactile RGB images rendered using the Taxim approach from :cite:t:`si2022taxim`.
    Shape is (num_instances, height, width, 3).
    """

    tactile_rgb_image_corrected: torch.Tensor | None = None
    """Stage-C force-corrected tactile RGB image. Shape is (num_instances, height, width, 3)."""

    tactile_height_map_corrected: torch.Tensor | None = None
    """Stage-C force-corrected render height map. Shape is (num_instances, height, width)."""

    tactile_marker_displacement: torch.Tensor | None = None
    """FOTS-style marker displacements in pixels (dx, dy). Shape is (num_instances, num_markers, 2)."""

    # Force field tactile data
    tactile_points_pos_w: torch.Tensor | None = None
    """Positions of tactile points in world frame. Shape is (num_instances, num_tactile_points, 3)."""

    tactile_points_quat_w: torch.Tensor | None = None
    """Orientations of tactile points in world frame. Shape is (num_instances, num_tactile_points, 4)."""

    penetration_depth: torch.Tensor | None = None
    """Penetration depth at each tactile point. Shape is (num_instances, num_tactile_points)."""

    tactile_normal_force: torch.Tensor | None = None
    """Normal forces at each tactile point in sensor frame. Shape is (num_instances, num_tactile_points)."""

    tactile_shear_force: torch.Tensor | None = None
    """Shear forces at each tactile point in sensor frame. Shape is (num_instances, num_tactile_points, 2)."""

    tactile_shear_magnitude: torch.Tensor | None = None
    """Shear force magnitude at each tactile point. Shape is (num_instances, num_tactile_points)."""

    tactile_tangential_speed: torch.Tensor | None = None
    """Tangential relative speed at each tactile point. Shape is (num_instances, num_tactile_points)."""

    contact_mask: torch.Tensor | None = None
    """Boolean contact mask per tactile point. Shape is (num_instances, num_tactile_points)."""

    slip_mask: torch.Tensor | None = None
    """Boolean slip mask per tactile point. Shape is (num_instances, num_tactile_points)."""

    tri_modal: dict[str, Any] | None = None
    """Unified tri-modal output dictionary for policy consumption.

    Keys include: ``vision_rgb``, ``vision_depth``, ``force_normal``, ``force_shear``,
    ``force_shear_magnitude``, ``tangential_speed``, ``contact_mask``, ``slip_mask``.
    """

    policy_obs: dict[str, torch.Tensor] | None = None
    """Policy-ready unified observations.

    Keys include:
    - ``force_token``: per-point feature token, shape ``(N, P, 6)``
      where channels are ``[fn, fsx, fsy, tangential_speed, contact_mask, slip_mask]``.
    - ``force_flat``: flattened force token, shape ``(N, P*6)``.
    - ``depth_flat``: flattened depth image, shape ``(N, H*W)`` (if depth exists).
    - ``rgb_flat``: flattened RGB image, shape ``(N, H*W*3)`` (if RGB exists).
    """
