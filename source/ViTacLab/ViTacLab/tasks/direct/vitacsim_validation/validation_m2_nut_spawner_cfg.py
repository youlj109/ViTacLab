# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""M2 hex nut contact object for advisor normal-force validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
import trimesh
from isaaclab.sim import schemas
from isaaclab.sim.spawners.meshes import meshes, meshes_cfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from .m2_nut_spec import ADVISOR_CASE_MASS_G, M2_GEOMETRY

if TYPE_CHECKING:
    from pxr import Usd


def _hex_ring_mesh(outer_r: float, inner_r: float, height: float) -> trimesh.Trimesh:
    """Hexagonal nut ring (6-sided outer, cylindrical hole)."""
    # Construct the annular prism explicitly. ``Trimesh.difference`` requires an
    # optional boolean backend which is not included in the Isaac Lab image; the
    # old fallback silently produced a solid hexagon and erased the nut hole.
    segments_per_edge = 4
    count = 6 * segments_per_edge
    corner_angles = np.pi / 6.0 + np.arange(6, dtype=np.float64) * (np.pi / 3.0)
    corners = np.stack((np.cos(corner_angles), np.sin(corner_angles)), axis=-1) * float(outer_r)

    outer_xy: list[np.ndarray] = []
    for edge in range(6):
        start = corners[edge]
        end = corners[(edge + 1) % 6]
        for step in range(segments_per_edge):
            t = float(step) / float(segments_per_edge)
            outer_xy.append((1.0 - t) * start + t * end)
    outer_xy_arr = np.asarray(outer_xy, dtype=np.float64)
    inner_angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float64)
    inner_xy_arr = np.stack((np.cos(inner_angles), np.sin(inner_angles)), axis=-1) * float(inner_r)

    def _layer(xy: np.ndarray, z: float) -> np.ndarray:
        return np.column_stack((xy, np.full(xy.shape[0], z, dtype=np.float64)))

    # Vertex blocks: outer-bottom, inner-bottom, outer-top, inner-top.
    vertices = np.vstack(
        (
            _layer(outer_xy_arr, 0.0),
            _layer(inner_xy_arr, 0.0),
            _layer(outer_xy_arr, float(height)),
            _layer(inner_xy_arr, float(height)),
        )
    )
    faces: list[tuple[int, int, int]] = []
    for i in range(count):
        j = (i + 1) % count
        ob_i, ob_j = i, j
        ib_i, ib_j = count + i, count + j
        ot_i, ot_j = 2 * count + i, 2 * count + j
        it_i, it_j = 3 * count + i, 3 * count + j

        # Top and bottom annuli.
        faces.extend(((ot_i, ot_j, it_j), (ot_i, it_j, it_i)))
        faces.extend(((ob_i, ib_j, ob_j), (ob_i, ib_i, ib_j)))
        # Outer wall and inward-facing cylindrical hole wall.
        faces.extend(((ob_i, ob_j, ot_j), (ob_i, ot_j, ot_i)))
        faces.extend(((ib_i, it_j, ib_j), (ib_i, it_i, it_j)))

    ring = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=True)
    ring.fix_normals()
    return ring


def _spawn_mesh_part(
    prim_path: str,
    mesh: trimesh.Trimesh,
    translation: tuple[float, float, float],
    shape_common: dict,
) -> None:
    mesh_cfg = meshes_cfg.MeshCfg(
        visual_material=shape_common["visual_material"],
        physics_material=shape_common["physics_material"],
        collision_props=shape_common["collision_props"],
        rigid_props=None,
        mass_props=None,
    )
    meshes._spawn_mesh_geom_from_mesh(prim_path, mesh_cfg, mesh, translation=translation)


@clone
def spawn_m2_nut(
    prim_path: str,
    cfg: "ValidationM2NutSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn M2 hex nut; rigid root at bottom-face center (z=0 contact ring)."""

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    shape_common = dict(
        visual_material=cfg.visual_material,
        physics_material=cfg.physics_material,
        collision_props=cfg.collision_props,
        rigid_props=None,
        mass_props=None,
    )

    vs = float(getattr(cfg, "visual_scale", 1.0))
    outer_r = cfg.outer_radius * vs
    inner_r = cfg.hole_radius * vs
    height = cfg.nut_height * vs
    ring = _hex_ring_mesh(outer_r, inner_r, height)
    _spawn_mesh_part(f"{prim_path}/m2_nut_ring", ring, (0.0, 0.0, 0.0), shape_common)

    return stage.GetPrimAtPath(prim_path)


@configclass
class ValidationM2NutSpawnerCfg(RigidObjectSpawnerCfg):
    """Standard M2 nut; vary only suspended mass per advisor case."""

    func: Callable = spawn_m2_nut

    mass_props: sim_utils.MassPropertiesCfg | None = sim_utils.MassPropertiesCfg(mass=0.110)
    rigid_props: sim_utils.RigidBodyPropertiesCfg | None = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        enable_gyroscopic_forces=False,
    )
    collision_props: sim_utils.CollisionPropertiesCfg | None = sim_utils.CollisionPropertiesCfg()

    visual_material: sim_utils.PreviewSurfaceCfg | None = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.55, 0.58, 0.62)
    )
    physics_material: sim_utils.RigidBodyMaterialCfg | None = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.9,
        dynamic_friction=0.85,
    )

    outer_radius: float = M2_GEOMETRY.circumradius
    hole_radius: float = M2_GEOMETRY.hole_radius
    nut_height: float = M2_GEOMETRY.height
    visual_scale: float = 1.0


def validation_m2_nut_spawner_cfg(
    case_id: str,
    *,
    visual_scale: float = 1.0,
    width_across_flats: float = M2_GEOMETRY.width_across_flats,
    hole_diameter: float = M2_GEOMETRY.hole_diameter,
) -> ValidationM2NutSpawnerCfg:
    if case_id not in ADVISOR_CASE_MASS_G:
        raise KeyError(f"Unknown case_id={case_id!r}; expected one of {sorted(ADVISOR_CASE_MASS_G)}")
    width_across_flats = float(width_across_flats)
    hole_diameter = float(hole_diameter)
    if width_across_flats <= 0.0:
        raise ValueError("width_across_flats must be positive")
    if not 0.0 < hole_diameter < width_across_flats:
        raise ValueError("hole_diameter must be positive and smaller than width_across_flats")
    mass = float(ADVISOR_CASE_MASS_G[case_id]) / 1000.0
    return ValidationM2NutSpawnerCfg(
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        outer_radius=width_across_flats / (3.0**0.5),
        hole_radius=0.5 * hole_diameter,
        visual_scale=float(visual_scale),
    )
