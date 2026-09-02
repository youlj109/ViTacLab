# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Validation weight: chamfered large cylinder + stem + sphere (coaxial, compound colliders)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
import trimesh
from isaaclab.sim import schemas
from isaaclab.sim.spawners.meshes import meshes, meshes_cfg
from isaaclab.sim.spawners.shapes.shapes import spawn_cylinder, spawn_sphere
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from .weight_spec import GEOMETRY, LAYOUT, WEIGHT_MASS_KG

if TYPE_CHECKING:
    from pxr import Usd

_FLAT_DISK_H = 0.0


def _frustum_mesh(r_bottom: float, r_top: float, height: float, sections: int = 48) -> trimesh.Trimesh:
    """Frustum along +z from z=0 (radius r_bottom) to z=height (radius r_top)."""

    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    bottom = np.column_stack([r_bottom * np.cos(angles), r_bottom * np.sin(angles), np.zeros(sections)])
    top = np.column_stack([r_top * np.cos(angles), r_top * np.sin(angles), np.full(sections, height)])
    vertices = np.vstack([bottom, top])
    faces: list[list[int]] = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([i, j, sections + j])
        faces.append([i, sections + j, sections + i])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _disk_mesh(radius: float, sections: int = 48) -> trimesh.Trimesh:
    """Flat disk in z=0 plane, radius ``radius``."""

    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(sections)])
    vertices = np.vstack([ring, [[0.0, 0.0, 0.0]]])
    center = len(vertices) - 1
    faces = [[center, (i + 1) % sections, i] for i in range(sections)]
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _frustum_with_caps(
    r_bottom: float,
    r_top: float,
    height: float,
    *,
    cap_bottom: bool = False,
    cap_top: bool = False,
    sections: int = 48,
) -> trimesh.Trimesh:
    parts = [_frustum_mesh(r_bottom, r_top, height, sections=sections)]
    if cap_bottom and r_bottom > 0.0:
        parts.append(_disk_mesh(r_bottom, sections=sections))
    if cap_top and r_top > 0.0:
        top = _disk_mesh(r_top, sections=sections)
        top.apply_translation([0.0, 0.0, height])
        parts.append(top)
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


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
def spawn_validation_weight(
    prim_path: str,
    cfg: "ValidationWeightSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn compound validation weight; rigid-body root at bottom-face center (z=0).

    All parts share the z axis (x=y=0): large cylinder flat centers, stem centers, sphere center.
    Large cylinder has 1 mm × 45° chamfers on top/bottom outer rims.
    """

    vs = float(getattr(cfg, "visual_scale", 1.0))
    if vs <= 0.0:
        raise ValueError(f"visual_scale must be > 0, got {vs}")

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

    c = cfg.chamfer * vs
    r = cfg.main_radius * vs
    rf = cfg.flat_radius * vs
    h_main = cfg.main_cylinder_height * vs
    h_tot = cfg.large_height * vs
    stem_radius = cfg.stem_radius * vs
    stem_height = cfg.stem_height * vs
    stem_center_z = cfg.stem_center_z * vs
    sphere_radius = cfg.top_sphere_radius * vs
    sphere_center_z = cfg.sphere_center_z * vs

    # Bottom chamfer (flat cap at z=0) + main wall + top chamfer (flat cap at z=h_tot).
    _spawn_mesh_part(
        f"{prim_path}/weight_bottom_chamfer",
        _frustum_with_caps(rf, r, c, cap_bottom=True),
        (0.0, 0.0, 0.0),
        shape_common,
    )

    main = sim_utils.CylinderCfg(radius=r, height=h_main, axis="Z", **shape_common)
    spawn_cylinder(f"{prim_path}/weight_body", main, translation=(0.0, 0.0, c + 0.5 * h_main))

    _spawn_mesh_part(
        f"{prim_path}/weight_top_chamfer",
        _frustum_with_caps(r, rf, c, cap_top=True),
        (0.0, 0.0, h_tot - c),
        shape_common,
    )

    stem = sim_utils.CylinderCfg(radius=stem_radius, height=stem_height, axis="Z", **shape_common)
    spawn_cylinder(f"{prim_path}/weight_stem", stem, translation=(0.0, 0.0, stem_center_z))

    top = sim_utils.SphereCfg(radius=sphere_radius, **shape_common)
    spawn_sphere(f"{prim_path}/weight_top", top, translation=(0.0, 0.0, sphere_center_z))

    return stage.GetPrimAtPath(prim_path)


@configclass
class ValidationWeightSpawnerCfg(RigidObjectSpawnerCfg):
    """Fixed validation-weight geometry; vary only ``mass_props.mass`` per preset."""

    func: Callable = spawn_validation_weight

    mass_props: sim_utils.MassPropertiesCfg | None = sim_utils.MassPropertiesCfg(mass=0.100)
    rigid_props: sim_utils.RigidBodyPropertiesCfg | None = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        enable_gyroscopic_forces=False,
    )
    collision_props: sim_utils.CollisionPropertiesCfg | None = sim_utils.CollisionPropertiesCfg()

    visual_material: sim_utils.PreviewSurfaceCfg | None = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.72, 0.72, 0.78)
    )
    physics_material: sim_utils.RigidBodyMaterialCfg | None = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.9,
        dynamic_friction=0.85,
    )

    chamfer: float = LAYOUT.chamfer
    main_radius: float = LAYOUT.large_radius
    flat_radius: float = LAYOUT.flat_radius
    large_height: float = LAYOUT.large_height
    main_cylinder_height: float = LAYOUT.main_cylinder_height

    stem_radius: float = GEOMETRY.stem_diameter * 0.5
    stem_height: float = GEOMETRY.stem_height
    stem_center_z: float = LAYOUT.stem_center_z
    top_sphere_radius: float = GEOMETRY.sphere_radius
    sphere_center_z: float = LAYOUT.sphere_center_z
    visual_scale: float = 1.0


def validation_weight_spawner_cfg(weight_id: str, *, visual_scale: float = 1.0) -> ValidationWeightSpawnerCfg:
    """Return spawner cfg for ``W200`` … ``W010`` presets."""

    if weight_id not in WEIGHT_MASS_KG:
        raise KeyError(f"Unknown weight_id={weight_id!r}; expected one of {sorted(WEIGHT_MASS_KG)}")
    mass = float(WEIGHT_MASS_KG[weight_id])
    return ValidationWeightSpawnerCfg(
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        visual_scale=float(visual_scale),
    )
