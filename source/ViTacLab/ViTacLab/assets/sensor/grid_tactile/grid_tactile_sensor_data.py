# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data container for the grid tactile sensor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.sensors.contact_sensor import ContactSensorData


@dataclass
class GridTactileSensorData(ContactSensorData):
    """Contact sensor data plus binned maps on the tactile patch."""

    force_grid: torch.Tensor | None = None
    """Signed **normal** force map: sums of :math:`f_{normal} \\cdot n_{pad}` per cell (from ``get_contact_data``).

    Shape ``(N, B, F, H, W)``. Same units as force (N).

    Contributions can be positive or negative. Cells with no contact remain zero.
    """

    friction_grid_uv: torch.Tensor | None = None
    """**Tangential (friction) force** components on the patch tangent axes (from ``get_friction_data``).

    Shape ``(N, B, F, 2, H, W)``: channel ``0`` is projection on body tangent axis ``u``, channel ``1`` on ``v``
    (same ordering as :attr:`GridTactileSensorCfg.pad_normal_axis` tangent pair). Units: N.

    Cells with no friction contribution remain zero.
    """
