# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual blind in-hand: fixed arms; each hand rotates its own object toward an independent target orientation."""

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


@configclass
class UR10eDualShadowHandBiBlindInhandSceneCfg(UR10eDualShadowHandTacSLSceneCfg):
    """TacSL force field contacts either manipuland: ``object_left`` / ``object_right``."""

    @classmethod
    def _tactile_params(cls) -> dict:
        p = super()._tactile_params()
        p["contact_object_prim_path_expr"] = "/World/envs/env_.*/object_(left|right)"
        return p


def _ur10e_bi_blind_inhand_right_arm_joint_pos() -> dict[str, float]:
    return {
        "shoulder_pan_joint": -0.2381820505749035,
        "shoulder_lift_joint": -0.7457377055913127,
        "elbow_joint": 0.7105199793481264,
        "wrist_1_joint": -1.5355786007566066,
        "wrist_2_joint": -1.5707963265900005,
        "wrist_3_joint": -1.8089821043693861,
    }


def _ur10e_bi_blind_inhand_left_arm_joint_pos() -> dict[str, float]:
    return {
        "shoulder_pan_joint": -0.0647053798972387,
        "shoulder_lift_joint": -0.6931192918684786,
        "elbow_joint": 0.9688935001938827,
        "wrist_1_joint": -1.8465705353261026,
        "wrist_2_joint": -1.5707963265904117,
        "wrist_3_joint": 1.5060872198977679,
    }


UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS: int = 24


def _bi_blind_inhand_obs_dim(num_hand_dofs: int) -> int:
    """Per-agent policy observation: hand (same as bi-stab) + one object + goal quat + quat error (no goal position)."""

    # 3 * num_hand_dofs + (fingertip 65) + object 13 + goal_rot 4 + quat_err 4 = 3*24 + 86 = 158
    return 3 * num_hand_dofs + 86


def _bi_blind_inhand_state_dim(num_hand_dofs: int) -> int:
    """Central critic: concatenate per-agent observations (no shared object tensor between agents)."""

    return 2 * _bi_blind_inhand_obs_dim(num_hand_dofs)


@configclass
class UR10eDualShadowHandBiBlindInhandEnvCfg(DirectMARLEnvCfg):
    """Dual Shadow Hands in-hand rotation matching: each side aligns its own object to a sampled target orientation."""

    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {
        "right_hand": UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS,
        "left_hand": UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS,
    }
    observation_spaces = {
        "right_hand": _bi_blind_inhand_obs_dim(UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS),
        "left_hand": _bi_blind_inhand_obs_dim(UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS),
    }
    state_space = _bi_blind_inhand_state_dim(UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS)

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
            joint_pos=_ur10e_bi_blind_inhand_right_arm_joint_pos(),
        ),
    )
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_bi_blind_inhand_left_arm_joint_pos(),
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

    # Object in each hand's workspace (env-local); yellow (left), cyan (right).
    object_left_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_left",
        spawn=sim_utils.SphereCfg(
            radius=0.028,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.65),
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
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, 0.17, 0.47), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    object_right_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_right",
        spawn=sim_utils.SphereCfg(
            radius=0.028,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 0.95)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.65),
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
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, -0.17, 0.47), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal_left_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/bi_blind_inhand_goal_left",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.028,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 1.0)),
            ),
        },
    )
    goal_right_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/bi_blind_inhand_goal_right",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.028,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.35)),
            ),
        },
    )

    scene: UR10eDualShadowHandBiBlindInhandSceneCfg = UR10eDualShadowHandBiBlindInhandSceneCfg(
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

    # Rotation reward (same spirit as single in-hand ``compute_rewards``).
    rot_reward_scale: float = 0.35
    rot_eps: float = 0.05
    success_tolerance: float = 0.18
    reach_goal_bonus: float = 12.0
    action_penalty_scale: float = -0.0002
