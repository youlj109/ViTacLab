# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand: per-link :class:`~ViTacLab.assets.sensor.grid_tactile.GridTactileSensorCfg`."""

from .shadow_hand_grid_tactile_cfg import (
    SHADOW_DEFAULT_GRID_RESOLUTION_PER_LINK,
    SHADOW_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK,
    SHADOW_DEFAULT_PATCH_EXTENT_PER_LINK,
    SHADOW_HAND_LINK_NAMES,
    SHADOW_HAND_LINK_NAMES_ALL,
    SHADOW_HAND_MOUNT_LINK_NAMES,
    build_shadow_hand_grid_tactile_sensor_cfgs,
    default_shadow_hand_grid_resolution_per_link,
    default_shadow_hand_max_contact_per_link,
    default_shadow_hand_patch_extent_per_link,
    shadow_hand_default_grid_cell_count_total,
    shadow_hand_link_names,
)

__all__ = [
    "SHADOW_DEFAULT_GRID_RESOLUTION_PER_LINK",
    "SHADOW_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK",
    "SHADOW_DEFAULT_PATCH_EXTENT_PER_LINK",
    "SHADOW_HAND_LINK_NAMES",
    "SHADOW_HAND_LINK_NAMES_ALL",
    "SHADOW_HAND_MOUNT_LINK_NAMES",
    "build_shadow_hand_grid_tactile_sensor_cfgs",
    "default_shadow_hand_grid_resolution_per_link",
    "default_shadow_hand_max_contact_per_link",
    "default_shadow_hand_patch_extent_per_link",
    "shadow_hand_default_grid_cell_count_total",
    "shadow_hand_link_names",
]
