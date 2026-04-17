# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pressure / mass pretraining: random object mass, infer mass from GelSight TacSL."""

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
from ..gelsight_pretrain_obs import pretrain_obs_dim
from .knob_weight_spawner_cfg import KnobWeightSpawnerCfg


@configclass
class GelsightFingerMassPretrainEnvCfg(DirectRLEnvCfg):
    """Randomize rigid-body mass; supervision targets in ``extras`` / ``gt_mass_kg``."""

    decimation: int = 2
    episode_length_s: float = 3.0
    action_space: int = 1
    observation_space: int = pretrain_obs_dim(False)
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

    # Knob-shaped weight on the pad (compound colliders; mass overridden at reset).
    contact_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/contact_object",
        spawn=KnobWeightSpawnerCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.46)),
    )

    use_full_tactile_obs: bool = False

    mass_range_kg: tuple[float, float] = (0.02, 0.2)
    object_reset_xy_range: tuple[float, float] = (-0.002, 0.002)
    gravity_m_s2: float = 9.81

    # 调试：TacSL patch 均值曲线 / 打印（与 friction_pretrain 同名 cfg，便于 run_gelsight_finger_pretrain 共用入参）。
    print_xyz_force_mean_interval: int = 0
    plot_xyz_force_live: bool = False
    plot_xyz_force_live_source: str = "tactile"  # 仅兼容 CLI；本任务只画 TacSL
    plot_xyz_force_live_max_points: int = 4000
    plot_xyz_force_live_update_interval: int = 1
