# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pose pretraining: same rigid knob, random orientation; infer pose from GelSight TacSL."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ..gelsight_finger_pretrain_base_cfg import (
    GelsightFingerPretrainSceneCfg,
    build_gelsight_finger_robot_cfg,
)
from ..gelsight_pretrain_obs import pose_pretrain_obs_dim
from ..mass_pretrain.knob_weight_spawner_cfg import KnobWeightSpawnerCfg


@configclass
class GelsightFingerPosePretrainEnvCfg(DirectRLEnvCfg):
    """Fixed mass/geometry; randomize orientation at reset. GT pose in ``extras`` / ``gt_*``."""

    decimation: int = 2
    episode_length_s: float = 3.0
    action_space: int = 1
    observation_space: int = pose_pretrain_obs_dim(False, False)
    state_space: int = 0
    asymmetric_obs: bool = False
    enable_cameras: bool = True

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        use_fabric=True,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
        ),
    )

    scene: GelsightFingerPretrainSceneCfg = GelsightFingerPretrainSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = build_gelsight_finger_robot_cfg()

    contact_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/contact_object",
        spawn=KnobWeightSpawnerCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.46)),
    )

    use_full_tactile_obs: bool = False
    # If True, prepend object XYZ in env frame (like mass_pretrain); default is tactile-only.
    include_object_position_in_obs: bool = False

    object_reset_xy_range: tuple[float, float] = (-0.002, 0.002)

    # Intrinsic Tait–Bryan angles (rad), Isaac ``quat_from_euler_xyz`` order roll→pitch→yaw.
    euler_roll_range_rad: tuple[float, float] = (-0.35, 0.35)
    euler_pitch_range_rad: tuple[float, float] = (-0.35, 0.35)
    euler_yaw_range_rad: tuple[float, float] = (-3.14159, 3.14159)

    print_pose_mean_interval: int = 0
