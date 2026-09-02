# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand: one :class:`~ViTacLab.assets.sensor.grid_tactile.GridTactileSensorCfg` per rigid link.

This mirrors the Wuji per-link grid tactile builder style, but uses ViTacLab UR10e + Shadow Hand
body names from :mod:`ViTacLab.assets.sensor.shadow_hand_full_tactile`.
"""

from __future__ import annotations

from typing import TypeVar

from ..grid_tactile import GridTactileSensorCfg
from ..shadow_hand_full_tactile import UR10E_SHADOW_HAND_TACTILE_BODY_NAMES

_T = TypeVar("_T")

SHADOW_HAND_MOUNT_LINK_NAMES: tuple[str, ...] = ("forearm", "wrist")
"""Shadow links near the wrist mount that are often excluded from tactile learning inputs."""

SHADOW_HAND_LINK_NAMES_ALL: tuple[str, ...] = UR10E_SHADOW_HAND_TACTILE_BODY_NAMES
"""All Shadow Hand tactile body names from ViTacLab UR10e + Shadow USD."""

SHADOW_HAND_LINK_NAMES: tuple[str, ...] = tuple(
    n for n in SHADOW_HAND_LINK_NAMES_ALL if n not in SHADOW_HAND_MOUNT_LINK_NAMES
)
"""Default Shadow Hand links used for per-link grid tactile (mount links excluded)."""


def shadow_hand_link_names(*, include_mount_links: bool = False) -> tuple[str, ...]:
    """Return Shadow Hand link names for grid tactile configuration."""
    return SHADOW_HAND_LINK_NAMES_ALL if include_mount_links else SHADOW_HAND_LINK_NAMES


def _tactile_tier_for_link(link_name: str) -> str:
    if link_name == "palm":
        return "large"
    if link_name in SHADOW_HAND_MOUNT_LINK_NAMES:
        return "small"
    if link_name.endswith("tip"):
        return "medium"
    return "small"


def default_shadow_hand_max_contact_per_link(*, include_mount_links: bool = False) -> dict[str, int]:
    """Per-link defaults for ``max_contact_data_count_per_prim``."""
    out: dict[str, int] = {}
    for name in shadow_hand_link_names(include_mount_links=include_mount_links):
        tier = _tactile_tier_for_link(name)
        if tier == "large":
            out[name] = 128
        elif tier == "medium":
            out[name] = 56
        else:
            out[name] = 32
    return out


def default_shadow_hand_grid_resolution_per_link(
    *, include_mount_links: bool = False
) -> dict[str, tuple[int, int]]:
    """Per-link defaults for ``grid_resolution`` (H, W)."""
    out: dict[str, tuple[int, int]] = {}
    for name in shadow_hand_link_names(include_mount_links=include_mount_links):
        tier = _tactile_tier_for_link(name)
        if tier == "large":
            out[name] = (12, 12)  # palm
        elif tier == "medium":
            out[name] = (6, 6)  # fingertip links
        else:
            out[name] = (4, 4)  # other segments / mount
    return out


def default_shadow_hand_patch_extent_per_link(
    *, include_mount_links: bool = False
) -> dict[str, tuple[float, float]]:
    """Per-link defaults for ``patch_extent`` (extent_u, extent_v) in meters."""
    out: dict[str, tuple[float, float]] = {}
    for name in shadow_hand_link_names(include_mount_links=include_mount_links):
        tier = _tactile_tier_for_link(name)
        if tier == "large":
            out[name] = (0.10, 0.10)
        elif tier == "medium":
            out[name] = (0.05, 0.05)
        else:
            out[name] = (0.04, 0.04)
    return out


def shadow_hand_default_grid_cell_count_total(*, include_mount_links: bool = False) -> int:
    """Total H×W cells implied by :func:`default_shadow_hand_grid_resolution_per_link`."""
    d = default_shadow_hand_grid_resolution_per_link(include_mount_links=include_mount_links)
    return int(sum(int(h) * int(w) for h, w in d.values()))


def _merge_link_dict(defaults: dict[str, _T], overrides: dict[str, _T] | None, *, label: str) -> dict[str, _T]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    for k, v in overrides.items():
        if k not in defaults:
            raise KeyError(f"{label}: unknown Shadow Hand link {k!r}. Valid keys: {sorted(defaults.keys())}")
        out[k] = v
    return out


def build_shadow_hand_grid_tactile_sensor_cfgs(
    *,
    hand_root_prim_path_expr: str = "{ENV_REGEX_NS}/Robot",
    filter_prim_paths_expr: list[str],
    include_mount_links: bool = False,
    max_contact_per_link: dict[str, int] | None = None,
    grid_resolution_per_link: dict[str, tuple[int, int]] | None = None,
    patch_extent_per_link: dict[str, tuple[float, float]] | None = None,
    patch_center_offset_per_link: dict[str, tuple[float, float]] | None = None,
    update_period: float = 0.0,
    history_length: int = 0,
    debug_vis: bool = False,
    track_friction: bool = True,
    track_pose: bool = True,
    pad_normal_axis: int = 0,
    pad_normal_sign: int = 1,
    swap_tangent_axes: bool = False,
) -> dict[str, GridTactileSensorCfg]:
    """Build one :class:`GridTactileSensorCfg` per Shadow Hand link.

    Returns:
        Map ``sensor_cfg_key -> GridTactileSensorCfg`` with keys ``shadow_grid_tactile_<link_name>``.
    """
    root = hand_root_prim_path_expr.rstrip("/")
    names = shadow_hand_link_names(include_mount_links=include_mount_links)

    d_contact = _merge_link_dict(
        default_shadow_hand_max_contact_per_link(include_mount_links=include_mount_links),
        max_contact_per_link,
        label="max_contact_per_link",
    )
    d_res = _merge_link_dict(
        default_shadow_hand_grid_resolution_per_link(include_mount_links=include_mount_links),
        grid_resolution_per_link,
        label="grid_resolution_per_link",
    )
    d_patch = _merge_link_dict(
        default_shadow_hand_patch_extent_per_link(include_mount_links=include_mount_links),
        patch_extent_per_link,
        label="patch_extent_per_link",
    )
    d_off = _merge_link_dict(
        {ln: (0.0, 0.5 * float(d_patch[ln][1])) for ln in names},
        patch_center_offset_per_link,
        label="patch_center_offset_per_link",
    )

    out: dict[str, GridTactileSensorCfg] = {}
    for link in names:
        key = f"shadow_grid_tactile_{link}"
        out[key] = GridTactileSensorCfg(
            prim_path=f"{root}/{link}",
            update_period=update_period,
            history_length=history_length,
            debug_vis=debug_vis,
            filter_prim_paths_expr=list(filter_prim_paths_expr),
            max_contact_data_count_per_prim=int(d_contact[link]),
            grid_resolution=(int(d_res[link][0]), int(d_res[link][1])),
            patch_extent=(float(d_patch[link][0]), float(d_patch[link][1])),
            patch_center_offset=(float(d_off[link][0]), float(d_off[link][1])),
            pad_normal_axis=int(pad_normal_axis),
            pad_normal_sign=int(pad_normal_sign),
            swap_tangent_axes=bool(swap_tangent_axes),
            track_friction=track_friction,
            track_pose=track_pose,
        )
    return out


SHADOW_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK: dict[str, int] = default_shadow_hand_max_contact_per_link()
SHADOW_DEFAULT_GRID_RESOLUTION_PER_LINK: dict[str, tuple[int, int]] = default_shadow_hand_grid_resolution_per_link()
SHADOW_DEFAULT_PATCH_EXTENT_PER_LINK: dict[str, tuple[float, float]] = default_shadow_hand_patch_extent_per_link()
