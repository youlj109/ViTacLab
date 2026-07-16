# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wuji hand: per-link :class:`~ViTacLab.assets.sensor.grid_tactile.GridTactileSensorCfg` (pad frame configurable)."""

from .hand_tactile_layout import (
    left_hand_schematic_layout,
    norm_boxes_to_pixels,
    right_hand_schematic_layout,
    schematic_layout_for_side,
)
from .plot_hand_tactile import (
    HandTactilePlotCfg,
    render_friction_arrows_image,
    render_hand_tactile_pair,
    render_normal_force_image,
    strip_wuji_sensor_key_prefix,
)
from .wuji_hand_grid_tactile_cfg import (
    WUJI_DEFAULT_GRID_RESOLUTION_PER_LINK_LEFT,
    WUJI_DEFAULT_GRID_RESOLUTION_PER_LINK_RIGHT,
    WUJI_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK_LEFT,
    WUJI_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK_RIGHT,
    WUJI_DEFAULT_PATCH_EXTENT_PER_LINK_LEFT,
    WUJI_DEFAULT_PATCH_EXTENT_PER_LINK_RIGHT,
    WUJI_LEFT_LINK_NAMES,
    WUJI_RIGHT_LINK_NAMES,
    build_wuji_hand_grid_tactile_sensor_cfgs,
    default_wuji_grid_resolution_per_link,
    default_wuji_max_contact_per_link,
    default_wuji_patch_extent_per_link,
    wuji_default_grid_cell_count_total,
    wuji_link_names,
)

__all__ = [
    "HandTactilePlotCfg",
    "left_hand_schematic_layout",
    "norm_boxes_to_pixels",
    "render_friction_arrows_image",
    "render_hand_tactile_pair",
    "render_normal_force_image",
    "right_hand_schematic_layout",
    "schematic_layout_for_side",
    "strip_wuji_sensor_key_prefix",
    "WUJI_DEFAULT_GRID_RESOLUTION_PER_LINK_LEFT",
    "WUJI_DEFAULT_GRID_RESOLUTION_PER_LINK_RIGHT",
    "WUJI_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK_LEFT",
    "WUJI_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK_RIGHT",
    "WUJI_DEFAULT_PATCH_EXTENT_PER_LINK_LEFT",
    "WUJI_DEFAULT_PATCH_EXTENT_PER_LINK_RIGHT",
    "WUJI_LEFT_LINK_NAMES",
    "WUJI_RIGHT_LINK_NAMES",
    "build_wuji_hand_grid_tactile_sensor_cfgs",
    "default_wuji_grid_resolution_per_link",
    "default_wuji_max_contact_per_link",
    "default_wuji_patch_extent_per_link",
    "wuji_default_grid_cell_count_total",
    "wuji_link_names",
]
