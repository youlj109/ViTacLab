# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Forge parallel-gripper shear validation cfg (W100 in gripper, lateral EE motion)."""

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ViTacLab.assets.sensor import VisuoTactileSensorV2Cfg
from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
from ViTacLab.tasks.direct.simple_gripper.forge_env_cfg import ForgeSceneCfg, ForgeTaskPegInsertCfg
from ViTacLab.tasks.direct.simple_gripper.forge_tasks_cfg import ForgePegInsert
from ViTacLab.tasks.direct.vitacsim_validation.validation_weight_spawner_cfg import validation_weight_spawner_cfg
from ViTacLab.tasks.direct.vitacsim_validation.weight_spec import WEIGHT_MASS_KG

from isaaclab_tasks.direct.factory.factory_tasks_cfg import HeldAssetCfg


@configclass
class ValidationWeightHeldAssetCfg(HeldAssetCfg):
    """Held-asset geometry for Factory grasp math (W100 large cylinder)."""

    diameter: float = 0.025
    height: float = 0.025
    mass: float = 0.100
    friction: float = 0.75
    usd_path: str = ""  # unused; procedural W100 spawned separately


@configclass
class GripperShearForgeTask(ForgePegInsert):
    name: str = "gripper_shear"
    held_asset_cfg: ValidationWeightHeldAssetCfg = ValidationWeightHeldAssetCfg()
    # Reduce in-gripper randomization for repeatable validation.
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]


def _sensor_cfg(*, side: str, mode: str, render_cfg) -> VisuoTactileSensorV2Cfg:
    use_normal = mode == "vitacsim"
    use_slip = mode == "vitacsim"
    k_ref = 1.0e4 if mode == "tacsl" else 66.0
    finger = "R15_leftfinger" if side == "left" else "R15_rightfinger"
    return VisuoTactileSensorV2Cfg(
        prim_path=f"/World/envs/env_.*/Robot/{finger}/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=render_cfg,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="/World/envs/env_.*/contact_object",
        contact_object_is_deformable=False,
        depth_penetration_deadband=0.0001,
        normal_contact_stiffness=1.0,
        normal_correction_k_ref=k_ref,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
        enable_normal_correction=use_normal,
        enable_slip_stick_reconstruction=use_slip,
        enable_corrected_force_render=(mode == "vitacsim"),
        corrected_force_render_blend=1.0,
        require_physx_sparse_anchors=(mode == "vitacsim"),
        strict_target_contact_attribution=True,
        camera_cfg=TiledCameraCfg(
            prim_path=f"/World/envs/env_.*/Robot/{finger}/elastomer_tip/cam",
            height=render_cfg.image_height,
            width=render_cfg.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def make_weight_rigid_cfg(weight_id: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/contact_object",
        spawn=validation_weight_spawner_cfg(weight_id),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


@configclass
class GripperShearValidationEnvCfg(ForgeTaskPegInsertCfg):
    """Forge peg-insert env cfg adapted for W100 gripper shear validation."""

    task_name: str = "gripper_shear"
    enable_cameras: bool = True
    obs_mode: str = "full"
    episode_length_s: float = 30.0
    validation_weight_id: str = "W100"
    sensor_mode: str = "vitacsim"


def build_gripper_shear_env_cfg(
    *,
    weight_id: str = "W100",
    sensor_mode: str = "vitacsim",
    enable_marker: bool = True,
    marker_pattern: str = "gelsight",
) -> GripperShearValidationEnvCfg:
    """Return Forge cfg for gripper shear validation (1 env, cameras on)."""

    render_cfg = validation_gelsight_render_cfg(enable_marker=enable_marker, marker_pattern=marker_pattern)
    scene = ForgeSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    scene.tactile_sensor_left = _sensor_cfg(side="left", mode=sensor_mode, render_cfg=render_cfg)
    scene.tactile_sensor_right = _sensor_cfg(side="right", mode=sensor_mode, render_cfg=render_cfg)
    scene.third_person_camera = None

    cfg = GripperShearValidationEnvCfg(
        scene=scene,
        task=GripperShearForgeTask(),
        validation_weight_id=weight_id,
        sensor_mode=sensor_mode,
    )
    cfg.robot.spawn.usd_path = "source/ViTacLab/ViTacLab/assets/data/Robots/Franka/Franka_R15/franka_mimic_edit.usd"
    cfg.robot.init_state.joint_pos["panda_finger_joint2"] = 0.04
    cfg._weight_rigid_cfg = make_weight_rigid_cfg(weight_id)  # type: ignore[attr-defined]
    return cfg
