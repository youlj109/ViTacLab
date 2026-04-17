# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi-Stab (Bimanual Stabilization): fixed UR10e arms; Shadow Hand wrist + fingers move a plate + ball."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_cfg import (
    UR10E_DUAL_SHADOWHAND_LEFT_CFG,
    UR10E_DUAL_SHADOWHAND_RIGHT_CFG,
    UR10eDualShadowHandTacSLSceneCfg,
)


def _ur10e_bi_stab_right_arm_joint_pos() -> dict[str, float]:
    """Same analytic fixed pose as :func:`ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env_cfg._ur10e_hand_over_right_arm_joint_pos`."""

    return {
        "shoulder_pan_joint": -0.2381820505749035,
        "shoulder_lift_joint": -0.7457377055913127,
        "elbow_joint": 0.7105199793481264,
        "wrist_1_joint": -1.5355786007566066,
        "wrist_2_joint": -1.5707963265900005,
        "wrist_3_joint": -1.8089821043693861,
    }


def _ur10e_bi_stab_left_arm_joint_pos() -> dict[str, float]:
    """Same analytic fixed pose as :func:`ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env_cfg._ur10e_hand_over_left_arm_joint_pos`."""

    return {
        "shoulder_pan_joint": -0.0647053798972387,
        "shoulder_lift_joint": -0.6931192918684786,
        "elbow_joint": 0.9688935001938827,
        "wrist_1_joint": -1.8465705353261026,
        "wrist_2_joint": -1.5707963265904117,
        "wrist_3_joint": 1.5060872198977679,
    }


UR10E_DUAL_SHADOWHAND_BISTAB_NUM_HAND_DOFS: int = 24


def _bi_stab_obs_dim(num_hand_dofs: int) -> int:
    """Per-agent policy observation size (same layout as hand-over)."""

    return 3 * num_hand_dofs + 89


def _bi_stab_state_dim(num_hand_dofs: int) -> int:
    """Central critic state dimension (same layout as hand-over)."""

    return 6 * num_hand_dofs + 154


PLATE_USD_PATH: str = "source/ViTacLab/ViTacLab/assets/data/Objects/RealCups/plate/plate.usd"


@configclass
class UR10eDualShadowHandBiStabEnvCfg(DirectMARLEnvCfg):
    """Dual Shadow Hands stabilize a ball on a plate (arms fixed at a teleop-like pose)."""

    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {
        "right_hand": UR10E_DUAL_SHADOWHAND_BISTAB_NUM_HAND_DOFS,
        "left_hand": UR10E_DUAL_SHADOWHAND_BISTAB_NUM_HAND_DOFS,
    }
    observation_spaces = {
        "right_hand": _bi_stab_obs_dim(UR10E_DUAL_SHADOWHAND_BISTAB_NUM_HAND_DOFS),
        "left_hand": _bi_stab_obs_dim(UR10E_DUAL_SHADOWHAND_BISTAB_NUM_HAND_DOFS),
    }
    state_space = _bi_stab_state_dim(UR10E_DUAL_SHADOWHAND_BISTAB_NUM_HAND_DOFS)

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    right_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_RIGHT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_RIGHT_CFG.init_state.replace(
            joint_pos=_ur10e_bi_stab_right_arm_joint_pos(),
        ),
    )
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_bi_stab_left_arm_joint_pos(),
        ),
    )

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    fingertip_body_names: tuple[str, ...] = (
        "ffdistal",
        "mfdistal",
        "rfdistal",
        "lfdistal",
        "thdistal",
    )

    enable_cameras: bool = False

    # Uniform scale applied to ``plate.usd`` spawn (tune if the asset is authored in meters).
    plate_scale: tuple[float, float, float] = (2.0, 2.0, 2.0)

    # --- Plate (USD) between the two arms, on the shared table workspace ---
    plate_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLATE_USD_PATH,
            scale=plate_scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=300.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.45),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # --- Ball (sphere) resting on the plate; policy steers it toward the goal marker ---
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.85, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.47), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/bistab_goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 1.0)),
            ),
        },
    )

    # TacSL marker class so :meth:`UR10eDualShadowHandDirectMARLBaseEnv._setup_scene` spawns
    # ``left_*`` / ``right_*`` ``tactile_sensor_*`` when ``enable_cameras`` (play --show_rgb / training).
    scene: UR10eDualShadowHandTacSLSceneCfg = UR10eDualShadowHandTacSLSceneCfg(
        num_envs=2048,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    reset_position_noise = 0.008
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    arm_reset_dof_pos_noise_scale: float = 0.0

    fall_dist = 0.22
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    dist_reward_scale = 20.0

    # Stability: reward low ball linear speed when close to the goal (tune in sim).
    stab_reward_scale: float = 0.5
    stab_vel_scale: float = 8.0
    stab_goal_dist_thresh: float = 0.04

    # Goal marker: uniform random XY in a disk of this radius (m) around plate center; fixed Z (env-local).
    goal_xy_radius: float = 0.1
    goal_z: float = 0.4

    # Ball reset offset above plate default (env-local); used after plate pose is sampled.
    ball_z_offset_plate: float = 0.035
