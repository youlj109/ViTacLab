# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wuji hand: one :class:`~ViTacLab.assets.sensor.grid_tactile.GridTactileSensorCfg` per rigid link.

Link names match ``wuji-hand-description-main/urdf/right.urdf`` / ``left.urdf`` (no ``*_tip_fixed`` links —
those are fixed joints, not separate links in the 26-link list).

**Link body frame (Wuji URDF convention):** **+Z points along the finger toward the fingertip** (distal).
The tactile patch lies in a plane whose **outward normal** is some **±X / ±Y** (or another axis), never
along **Z** if Z is the bone direction — so ``pad_normal_axis`` should be ``0`` or ``1`` for typical
side/palm pads, not ``2`` (that would make **Z** the surface normal and contradict “Z = fingertip”).

:func:`build_wuji_hand_grid_tactile_sensor_cfgs` aligns grid **tangent v** (second index, second extent,
``patch_center_offset[1]``) with **body Z** by default (``fingertip_body_axis=2``) so “along the grid
width / second dimension” matches **along the finger**, not world coordinates.

**Outward normal:** set ``pad_normal_axis`` and ``pad_normal_sign`` in the **link body** frame. Wrong axis
or sign breaks ``pressure = f·n`` and friction binning.

Pad origin is taken at **bottom-center**; ``patch_center_offset = (0, extent_v/2)`` shifts the bin window
along **tangent v** (body **Z** when aligned) so ``z=0`` at the proximal edge of the pad maps to the
start of the **v** grid axis.

**Default grid tiers** (link names from URDF; total cells ≈ **600**):

- **Large** (``*_palm_link`` only): 1 link × **15×10** cells = 150.
- **Medium** (``*_tip_link``, ``*_link2``, ``*_link3``): 15 links × **5×5** cells = 375.
- **Small** (``*_link1``, ``*_link4``): 10 links × **2×4** cells = 80.

Sum: 150 + 375 + 80 = **605** cells.
"""

from __future__ import annotations

from typing import Literal, TypeVar

from ..grid_tactile import GridTactileSensorCfg

_T = TypeVar("_T")

# -----------------------------------------------------------------------------
# Link inventory (right hand; left is the same pattern with ``left_`` prefix)
# -----------------------------------------------------------------------------

WUJI_RIGHT_LINK_NAMES: tuple[str, ...] = (
    "right_palm_link",
    "right_finger1_link1",
    "right_finger1_link2",
    "right_finger1_link3",
    "right_finger1_link4",
    "right_finger1_tip_link",
    "right_finger2_link1",
    "right_finger2_link2",
    "right_finger2_link3",
    "right_finger2_link4",
    "right_finger2_tip_link",
    "right_finger3_link1",
    "right_finger3_link2",
    "right_finger3_link3",
    "right_finger3_link4",
    "right_finger3_tip_link",
    "right_finger4_link1",
    "right_finger4_link2",
    "right_finger4_link3",
    "right_finger4_link4",
    "right_finger4_tip_link",
    "right_finger5_link1",
    "right_finger5_link2",
    "right_finger5_link3",
    "right_finger5_link4",
    "right_finger5_tip_link",
)

WUJI_LEFT_LINK_NAMES: tuple[str, ...] = tuple(
    n.replace("right_", "left_", 1) for n in WUJI_RIGHT_LINK_NAMES
)


def wuji_link_names(side: Literal["right", "left"]) -> tuple[str, ...]:
    """Rigid link names for one Wuji hand side (matches URDF ``<link name=...>``)."""
    return WUJI_RIGHT_LINK_NAMES if side == "right" else WUJI_LEFT_LINK_NAMES


def _tactile_tier_for_link(link_name: str) -> Literal["large", "medium", "small"]:
    """Classify link for default grid / contact / patch (see module docstring)."""
    if "palm" in link_name:
        return "large"
    if "tip_link" in link_name or "_link2" in link_name or "_link3" in link_name:
        return "medium"
    return "small"


def default_wuji_max_contact_per_link(side: Literal["right", "left"]) -> dict[str, int]:
    """Per-link ``max_contact_data_count_per_prim`` defaults (keyed by link name).

    Scales with tier: palm (large) carries the most points; fingertips share the medium tier with link2/3.
    """
    names = wuji_link_names(side)
    out: dict[str, int] = {}
    for name in names:
        tier = _tactile_tier_for_link(name)
        if tier == "large":
            out[name] = 128
        elif tier == "medium":
            out[name] = 56
        else:
            out[name] = 32
    return out


def default_wuji_grid_resolution_per_link(side: Literal["right", "left"]) -> dict[str, tuple[int, int]]:
    """Per-link ``grid_resolution`` (H, W) defaults; product H×W sums to ≈600 over 26 links."""
    names = wuji_link_names(side)
    out: dict[str, tuple[int, int]] = {}
    for name in names:
        tier = _tactile_tier_for_link(name)
        if tier == "large":
            out[name] = (15, 10)  # 150 cells, palm only
        elif tier == "medium":
            out[name] = (5, 5)  # 25 cells × 15 links (tips + link2/3)
        else:
            out[name] = (2, 4)  # 8 cells × 10 links
    return out


def default_wuji_patch_extent_per_link(side: Literal["right", "left"]) -> dict[str, tuple[float, float]]:
    """Per-link ``patch_extent`` (extent_u, extent_v) in meters (body Y / Z tangent plane)."""
    names = wuji_link_names(side)
    out: dict[str, tuple[float, float]] = {}
    for name in names:
        tier = _tactile_tier_for_link(name)
        if tier == "large":
            out[name] = (0.12, 0.12)
        elif tier == "medium":
            out[name] = (0.06, 0.06)
        else:
            out[name] = (0.042, 0.042)
    return out


def wuji_default_grid_cell_count_total(side: Literal["right", "left"]) -> int:
    """Total H×W cells implied by :func:`default_wuji_grid_resolution_per_link` (sanity ≈ 600)."""
    d = default_wuji_grid_resolution_per_link(side)
    return int(sum(int(h) * int(w) for h, w in d.values()))


def _tangent_pair_excluding_normal(normal_axis: int) -> tuple[int, int]:
    """The two body axes spanning the tangent plane when the outward normal is along ``normal_axis``."""
    if int(normal_axis) not in (0, 1, 2):
        raise ValueError(f"pad_normal_axis must be 0, 1, or 2, got {normal_axis}")
    axes = [0, 1, 2]
    axes.remove(int(normal_axis))
    return axes[0], axes[1]


def _wuji_swap_tangent_so_v_is_fingertip(pad_normal_axis: int, fingertip_body_axis: int) -> bool:
    """Return ``swap_tangent_axes`` so tangent **v** (2nd grid dim) aligns with ``fingertip_body_axis``.

    Canonical order in the plane is ``(lower body index, higher)``. **v** is the second; if the fingertip
    axis is the **first** index in that pair, swap so **v** = fingertip (pairs with ``extent[1]`` /
    ``patch_center_offset[1]``).
    """
    pna = int(pad_normal_axis)
    fta = int(fingertip_body_axis)
    if fta == pna:
        raise ValueError(
            f"Wuji fingertip_body_axis={fta} equals pad_normal_axis={pna}: the fingertip direction must lie "
            "**in** the patch plane. With +Z along the finger, use ``pad_normal_axis`` 0 or 1 (outward ±X "
            "or ±Y), not 2.",
        )
    b0, b1 = _tangent_pair_excluding_normal(pna)
    if fta not in (b0, b1):
        raise ValueError(
            f"fingertip_body_axis={fta} is not in the tangent plane for pad_normal_axis={pna} "
            f"(in-plane axes are {b0} and {b1}).",
        )
    return fta == b0


def _merge_link_dict(defaults: dict[str, _T], overrides: dict[str, _T] | None, *, label: str) -> dict[str, _T]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    for k, v in overrides.items():
        if k not in defaults:
            raise KeyError(f"{label}: unknown Wuji link {k!r}. Valid keys: {sorted(defaults.keys())}")
        out[k] = v
    return out


def build_wuji_hand_grid_tactile_sensor_cfgs(
    *,
    hand_root_prim_path_expr: str,
    filter_prim_paths_expr: list[str],
    side: Literal["right", "left"] = "right",
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
    pad_normal_sign: Literal[-1, 1] = 1,
    swap_tangent_axes: bool = False,
    fingertip_body_axis: int | None = 2,
) -> dict[str, GridTactileSensorCfg]:
    """Build one :class:`GridTactileSensorCfg` per Wuji link.

    Args:
        hand_root_prim_path_expr: Articulation root pattern, e.g. ``"{ENV_REGEX_NS}/WujiPad"``.
        filter_prim_paths_expr: Contact filter prims (same semantics as :class:`GridTactileSensorCfg`).
        side: ``"right"`` or ``"left"`` — selects link name prefix and default dict keys.
        max_contact_per_link: Optional overrides for :attr:`GridTactileSensorCfg.max_contact_data_count_per_prim`.
        grid_resolution_per_link: Optional overrides for grid H×W per link.
        patch_extent_per_link: Optional overrides for tangent patch size per link.
        patch_center_offset_per_link: Optional overrides for :attr:`GridTactileSensorCfg.patch_center_offset`.
        pad_normal_axis: Link-body axis (0/1/2) of the outward normal (see :class:`GridTactileSensorCfg`).
        pad_normal_sign: ``+1`` or ``-1`` along that axis (outward from skin).
        swap_tangent_axes: Used only when ``fingertip_body_axis is None`` (manual layout). Otherwise ignored.
        fingertip_body_axis: Link axis toward fingertip (default ``2`` = **+Z**). When set, computes
            ``swap_tangent_axes`` so grid **v** = this axis (``patch_center_offset[1]`` along the finger).

    Returns:
        Map ``sensor_cfg_key -> GridTactileSensorCfg`` with keys ``wuji_grid_tactile_<link_name>``.
    """
    root = hand_root_prim_path_expr.rstrip("/")
    names = wuji_link_names(side)
    if fingertip_body_axis is None:
        effective_swap = bool(swap_tangent_axes)
    else:
        effective_swap = _wuji_swap_tangent_so_v_is_fingertip(int(pad_normal_axis), int(fingertip_body_axis))

    d_contact = _merge_link_dict(
        default_wuji_max_contact_per_link(side), max_contact_per_link, label="max_contact_per_link"
    )
    d_res = _merge_link_dict(
        default_wuji_grid_resolution_per_link(side),
        grid_resolution_per_link,
        label="grid_resolution_per_link",
    )
    d_patch = _merge_link_dict(
        default_wuji_patch_extent_per_link(side),
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
        key = f"wuji_grid_tactile_{link}"
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
            pad_normal_sign=pad_normal_sign,
            swap_tangent_axes=effective_swap,
            track_friction=track_friction,
            track_pose=track_pose,
        )
    return out


# 模块导入时生成，便于在 IDE / 文档中直接查看各 link 的默认表
WUJI_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK_RIGHT: dict[str, int] = default_wuji_max_contact_per_link("right")
WUJI_DEFAULT_MAX_CONTACT_DATA_COUNT_PER_LINK_LEFT: dict[str, int] = default_wuji_max_contact_per_link("left")
WUJI_DEFAULT_GRID_RESOLUTION_PER_LINK_RIGHT: dict[str, tuple[int, int]] = default_wuji_grid_resolution_per_link("right")
WUJI_DEFAULT_GRID_RESOLUTION_PER_LINK_LEFT: dict[str, tuple[int, int]] = default_wuji_grid_resolution_per_link("left")
WUJI_DEFAULT_PATCH_EXTENT_PER_LINK_RIGHT: dict[str, tuple[float, float]] = default_wuji_patch_extent_per_link("right")
WUJI_DEFAULT_PATCH_EXTENT_PER_LINK_LEFT: dict[str, tuple[float, float]] = default_wuji_patch_extent_per_link("left")
