# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data container for Shadow Hand full-hand voxel tactile sensor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.sensors.contact_sensor import ContactSensorData


@dataclass
class ShadowHandFullTactileData(ContactSensorData):
    """Unified palm-frame voxel grid covering all hand contact links."""

    voxel_grid: torch.Tensor | None = None
    """Aggregated tactile volume in **palm link body frame** (axes +X,+Y,+Z fixed to the palm prim).

    Shape ``(N, F, nx, ny, nz, 3)`` where ``N`` is envs, ``F`` filter prims, ``nx,ny,nz`` from config.

    Last dimension: ``[f_n, f_t1, f_t2]`` — signed normal force scalar from ``get_contact_data`` (along contact
    normal), and friction force projected onto palm +X / +Y in world (two tangential components).

    Multiple contacts landing in the same cell are summed.
    """

    contact_normal_points_mean_palm: torch.Tensor | None = None
    """Mean position of **normal-force** contact samples (``get_contact_data`` ``points``) mapped into the **palm**
    body frame — same ``quat_apply_inverse`` transform as used for voxel binning.

    Shape ``(N, F, 3)`` in metres. NaN when there were no contacts for that env/filter in the last buffer update.
    """

    contact_normal_point_count: torch.Tensor | None = None
    """Sample counts aggregated into :attr:`contact_normal_points_mean_palm` (one row per contact point). Shape ``(N, F)``."""
