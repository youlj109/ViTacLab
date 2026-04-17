# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`ShadowHandFullTactileSensor`."""

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .shadow_hand_full_tactile_sensor import ShadowHandFullTactileSensor


@configclass
class ShadowHandFullTactileSensorCfg(ContactSensorCfg):
    """Full-hand unified voxel tactile for Shadow Hand (ViTacLab UR10e + Shadow USD).

    Requires ``activate_contact_sensors=True`` on the hand articulation spawn, non-empty
    ``filter_prim_paths_expr``, and a valid ``palm_link_name_substr`` matching one rigid body
    name under ``body_names`` (used as the voxel reference frame).

    Override ``prim_path`` to match **all** hand links that report contacts (regex). The default
    is broad—narrow it to the hand subtree to exclude arm links when needed.
    """

    class_type: type = ShadowHandFullTactileSensor

    prim_path: str = "{ENV_REGEX_NS}/Robot/.*"
    """Regex matching rigid bodies with contact reporters (ViTacLab single-arm uses ``Robot``; dual-arm may use ``LeftRobot``)."""

    max_contact_data_count_per_prim: int = 512
    """Raise this for contact-rich multi-link hands (bounds PhysX contact buffers)."""

    palm_link_name_substr: str = "palm"
    """Substring matched against :attr:`ContactSensor.body_names` to pick the palm rigid body."""

    voxel_resolution: tuple[int, int, int] = (64, 48, 10)
    """Grid size ``(nx, ny, nz)`` along palm frame +X, +Y, +Z."""

    voxel_min_bounds_palm: tuple[float, float, float] = (-0.11, -0.10, -0.06)
    """Volume minimum ``(x,y,z)`` in **palm body frame** (m)."""

    voxel_max_bounds_palm: tuple[float, float, float] = (0.11, 0.10, 0.10)
    """Volume maximum ``(x,y,z)`` in **palm body frame** (m)."""

    track_friction: bool = True
    """If True, fill tangential channels from :meth:`~omni.physics.tensors.impl.api.RigidContactView.get_friction_data`."""
