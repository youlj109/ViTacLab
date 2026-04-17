# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compound knob-style weight: one rigid root + cylinder body + thin stem + sphere top (PhysX colliders)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.shapes.shapes import spawn_cylinder, spawn_sphere
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from pxr import Usd


@clone
def spawn_knob_weight(
    prim_path: str,
    cfg: "KnobWeightSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a single rigid body with three collision primitives (main cylinder, stem, top sphere)."""

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    # Child parts: collision + visuals only (mass / rigid body live on ``prim_path``).
    _shape_common = dict(
        visual_material=cfg.visual_material,
        physics_material=cfg.physics_material,
        collision_props=cfg.collision_props,
        rigid_props=None,
        mass_props=None,
    )
    main = sim_utils.CylinderCfg(
        radius=cfg.main_radius,
        height=cfg.main_height,
        axis="Z",
        **_shape_common,
    )
    stem = sim_utils.CylinderCfg(
        radius=cfg.stem_radius,
        height=cfg.stem_height,
        axis="Z",
        **_shape_common,
    )
    top = sim_utils.SphereCfg(
        radius=cfg.top_sphere_radius,
        **_shape_common,
    )

    spawn_cylinder(f"{prim_path}/weight_body", main, translation=cfg.main_translation)
    spawn_cylinder(f"{prim_path}/weight_stem", stem, translation=cfg.stem_translation)
    spawn_sphere(f"{prim_path}/weight_top", top, translation=cfg.top_translation)

    return stage.GetPrimAtPath(prim_path)


@configclass
class KnobWeightSpawnerCfg(RigidObjectSpawnerCfg):
    """Stacked cylinder + cylinder + sphere; matches a small calibration-weight silhouette."""

    func: Callable = spawn_knob_weight

    mass_props: sim_utils.MassPropertiesCfg | None = sim_utils.MassPropertiesCfg(mass=0.05)
    rigid_props: sim_utils.RigidBodyPropertiesCfg | None = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        enable_gyroscopic_forces=False,
    )
    collision_props: sim_utils.CollisionPropertiesCfg | None = sim_utils.CollisionPropertiesCfg()

    visual_material: sim_utils.PreviewSurfaceCfg | None = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.75, 0.35, 0.15)
    )
    physics_material: sim_utils.RigidBodyMaterialCfg | None = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.9, dynamic_friction=0.8
    )
    scale: float = 0.3
    main_radius: float = 0.02 * scale
    main_height: float = 0.03 * scale
    main_translation: tuple[float, float, float] = (0.0, 0.0, 0.015 * scale)

    stem_radius: float = 0.006 * scale
    stem_height: float = 0.02 * scale
    stem_translation: tuple[float, float, float] = (0.0, 0.0, 0.04 * scale)

    top_sphere_radius: float = 0.01 * scale
    top_translation: tuple[float, float, float] = (0.0, 0.0, 0.055 * scale)
