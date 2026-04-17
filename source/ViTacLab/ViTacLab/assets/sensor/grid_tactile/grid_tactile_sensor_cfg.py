# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`GridTactileSensor`."""

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .grid_tactile_sensor import GridTactileSensor


@configclass
class GridTactileSensorCfg(ContactSensorCfg):
    """Configuration for a contact-based grid tactile sensor.

    This wraps :class:`~isaaclab.sensors.contact_sensor.ContactSensor` with PhysX
    :meth:`~omni.physics.tensors.impl.api.RigidContactView.get_contact_data` (normal)
    and optionally :meth:`~omni.physics.tensors.impl.api.RigidContactView.get_friction_data`
    (tangential friction), then bins both into 2D grids in the body tangent plane.

    .. note::

        ``filter_prim_paths_expr`` must be non-empty: ``get_contact_data`` requires filter prims.

    """

    class_type: type = GridTactileSensor

    grid_resolution: tuple[int, int] = (16, 16)
    """Grid size ``(height, width)`` in cells."""

    patch_extent: tuple[float, float] = (0.08, 0.08)
    """Physical span ``(extent_u, extent_v)`` in meters along the two tangent body axes."""

    pad_normal_axis: int = 2
    """Body-frame axis index (0=x, 1=y, 2=z) of the outward patch normal. Tangent axes are the other two."""

    track_friction: bool = True
    """If True, bin :meth:`RigidContactView.get_friction_data` into :attr:`GridTactileSensorData.friction_grid_uv`."""
