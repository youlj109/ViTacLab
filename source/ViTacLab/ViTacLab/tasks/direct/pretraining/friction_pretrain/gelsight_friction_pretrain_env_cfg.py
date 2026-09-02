# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""切向（剪切）触觉与滑动标签对齐：固定法向载荷，随机摩擦系数，可选单一平面切向激励。"""

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
from ..mass_pretrain.knob_weight_spawner_cfg import KnobWeightSpawnerCfg


@configclass
class GelsightFingerFrictionPretrainEnvCfg(DirectRLEnvCfg):
    """预训练目标：TacSL **切向**（剪切）与 **是否滑动** ``gt_is_sliding`` 对齐；法向载荷固定以便摩擦标签可解释。"""

    decimation: int = 2
    episode_length_s: float = 4.0
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

    contact_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/contact_object",
        spawn=KnobWeightSpawnerCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.46)),
    )

    use_full_tactile_obs: bool = False

    # 固定质量 → 固定重力法向载荷 F_n ≈ m g（不随机质量，避免与摩擦学习纠缠）。
    object_mass_kg: float = 0.08

    # 重置时极小 xy 抖动（毫米以下），避免物体偏离指尖垫中心。
    object_reset_xy_range: tuple[float, float] = (-0.0002, 0.0002)

    # 物体 env 系 z 低于「名义放置 z0」超过该高度则视为掉落，terminated 并 reset。
    terminate_on_object_drop: bool = True
    object_drop_height_m: float = 0.035

    # env 系 xy 虚拟墙：越界时夹回、法向速度按恢复系数反射；若 flip_lateral_push_at_xy_bounds 则同时翻转侧向力符号。
    enforce_object_xy_walls: bool = True
    boundary_wall_restitution: float = 0.2
    # 无虚拟墙时：仅按位置阈值翻转侧向力符号（不夹紧、不改速度）。
    flip_lateral_push_at_xy_bounds: bool = True
    # 相对 init_state 平面中心 (0,0) 的半宽（米）；虚拟墙位于 ±(half_extent - margin)。
    object_xy_bounds_half_extent: float = 0.006
    # >0 时墙面向内缩，减少物体越界深度。
    boundary_flip_margin_m: float = 0.0

    static_friction_range: tuple[float, float] = (0.01, 1.05)
    dynamic_friction_scale: float = 0.88

    # 可选：世界系 XY 平面恒定力（符号由触边翻转），用于诱发静摩擦→滑动（关则仅靠接触动力学）。
    enable_lateral_push: bool = True
    lateral_push_force_n: float = 0.2
    lateral_push_force_y_n: float = 0.4
    push_start_env_steps: int = 20

    # 调试：每 N 个 env 步打印外加力 / TacSL patch 均值（0=关闭）。
    print_xyz_force_mean_interval: int = 0
    # 实时 matplotlib：三条曲线 ~ t；applied=外加恒力；tactile=TacSL 与 FF 同源。
    plot_xyz_force_live: bool = False
    plot_xyz_force_live_source: str = "tactile"  # "applied" | "tactile"
    plot_xyz_force_live_max_points: int = 4000
    plot_xyz_force_live_update_interval: int = 1

    slip_speed_threshold_m_s: float = 0.02
    contact_normal_mean_threshold: float = 1.0e-5
    gravity_m_s2: float = 9.81
