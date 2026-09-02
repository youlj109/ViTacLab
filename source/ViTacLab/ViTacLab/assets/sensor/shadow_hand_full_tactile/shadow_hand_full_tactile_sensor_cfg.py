# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`ShadowHandFullTactileSensor`."""

from __future__ import annotations

from typing import Any

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .shadow_hand_full_tactile_sensor import ShadowHandFullTactileSensor

# Rigid-body names under ``{ENV_REGEX_NS}/Robot`` for ViTacLab UR10e + Shadow Hand USD
# (``ur10e_shadow_*_hand_glb_withtac*.usd``). Verified via ``--diag_only`` on 2026-06-11.
UR10E_SHADOW_HAND_TACTILE_BODY_NAMES: tuple[str, ...] = (
    "forearm",
    "wrist",
    "palm",
    "ffknuckle",
    "ffproximal",
    "ffmiddle",
    "ffdistal",
    "fftip",
    "lfmetacarpal",
    "lfknuckle",
    "lfproximal",
    "lfmiddle",
    "lfdistal",
    "lftip",
    "mfknuckle",
    "mfproximal",
    "mfmiddle",
    "mfdistal",
    "mftip",
    "rfknuckle",
    "rfproximal",
    "rfmiddle",
    "rfdistal",
    "rftip",
    "thbase",
    "thproximal",
    "thhub",
    "thmiddle",
    "thdistal",
    "thtip",
)

UR10E_ARM_BODY_NAMES: tuple[str, ...] = (
    "base_link",
    "base",
    "base_link_inertia",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)


def shadow_hand_tactile_prim_path_expr(robot_root_prim_path_expr: str = "{ENV_REGEX_NS}/Robot") -> str:
    """Contact-sensor ``prim_path`` regex covering Shadow Hand links only (excludes UR10e arm + ``imu``/``ee_link``)."""
    root = robot_root_prim_path_expr.rstrip("/")
    alts = "|".join(UR10E_SHADOW_HAND_TACTILE_BODY_NAMES)
    return f"{root}/({alts})"


def build_shadow_hand_full_tactile_sensor_cfg(
    *,
    robot_root_prim_path_expr: str = "{ENV_REGEX_NS}/Robot",
    filter_prim_paths_expr: list[str] | None = None,
    palm_link_name_substr: str = "palm",
    voxel_resolution: tuple[int, int, int] = (64, 48, 10),
    voxel_min_bounds_palm: tuple[float, float, float] = (-0.11, -0.10, -0.06),
    voxel_max_bounds_palm: tuple[float, float, float] = (0.11, 0.10, 0.10),
    max_contact_data_count_per_prim: int = 512,
    **kwargs: Any,
) -> ShadowHandFullTactileSensorCfg:
    """Factory for :class:`ShadowHandFullTactileSensorCfg` with hand-only ``prim_path``."""
    if filter_prim_paths_expr is None:
        filter_prim_paths_expr = ["{ENV_REGEX_NS}/object"]
    return ShadowHandFullTactileSensorCfg(
        prim_path=shadow_hand_tactile_prim_path_expr(robot_root_prim_path_expr),
        filter_prim_paths_expr=list(filter_prim_paths_expr),
        palm_link_name_substr=palm_link_name_substr,
        voxel_resolution=voxel_resolution,
        voxel_min_bounds_palm=voxel_min_bounds_palm,
        voxel_max_bounds_palm=voxel_max_bounds_palm,
        max_contact_data_count_per_prim=max_contact_data_count_per_prim,
        **kwargs,
    )


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

    prim_path: str = shadow_hand_tactile_prim_path_expr()
    """Hand-only regex (see :data:`UR10E_SHADOW_HAND_TACTILE_BODY_NAMES`). Use ``build_shadow_hand_full_tactile_sensor_cfg`` for dual-arm roots."""

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
