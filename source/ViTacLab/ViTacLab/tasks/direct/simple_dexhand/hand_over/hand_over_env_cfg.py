# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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
    UR10eDualShadowHandDirectSceneCfg,
)


def _ur10e_hand_over_right_arm_joint_pos() -> dict[str, float]:
    """Fixed UR10e pose for hand-over (same analytic pose as single-arm in-hand task).

    See :func:`ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg._ur10e_left_inhand_arm_joint_pos`.
    """

    return {
        "shoulder_pan_joint": -0.2381820505749035,
        "shoulder_lift_joint": -0.7457377055913127,
        "elbow_joint": 0.7105199793481264,
        "wrist_1_joint": -1.5355786007566066,
        "wrist_2_joint": -1.5707963265900005,
        "wrist_3_joint": -1.8089821043693861,
    }


def _ur10e_hand_over_left_arm_joint_pos() -> dict[str, float]:
    """Left arm: mirror shoulder pan so the elbow reaches toward the shared workspace (opposing base)."""

    return {
        "shoulder_pan_joint": -0.0647053798972387,
        "shoulder_lift_joint": -0.6931192918684786,
        "elbow_joint": 0.9688935001938827,
        "wrist_1_joint": -1.8465705353261026,
        "wrist_2_joint": -1.5707963265904117,
        "wrist_3_joint": 1.5060872198977679,
    }

# Shadow Hand only (policy); UR10e arm joints stay at task pose below (see :attr:`arm_reset_dof_pos_noise_scale`).
# Must match :attr:`hand_joint_expr` on the USD articulation (same count as in-hand task).
UR10E_DUAL_SHADOWHAND_OVER_NUM_HAND_DOFS: int = 24


def _hand_over_obs_dim(num_hand_dofs: int) -> int:
    """Per-agent policy observation size (hand joints + object/goal layout in :class:`UR10eDualShadowHandOverEnv`)."""
    return 3 * num_hand_dofs + 89


def _hand_over_state_dim(num_hand_dofs: int) -> int:
    """Central critic state dimension (both agents + object + goal)."""
    return 6 * num_hand_dofs + 154


@configclass
class UR10eDualShadowHandOverEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.5
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {
        "right_hand": UR10E_DUAL_SHADOWHAND_OVER_NUM_HAND_DOFS,
        "left_hand": UR10E_DUAL_SHADOWHAND_OVER_NUM_HAND_DOFS,
    }
    observation_spaces = {
        "right_hand": _hand_over_obs_dim(UR10E_DUAL_SHADOWHAND_OVER_NUM_HAND_DOFS),
        "left_hand": _hand_over_obs_dim(UR10E_DUAL_SHADOWHAND_OVER_NUM_HAND_DOFS),
    }
    state_space = _hand_over_state_dim(UR10E_DUAL_SHADOWHAND_OVER_NUM_HAND_DOFS)

    # simulation
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

    # Dual UR10e + ShadowHand: world pose from asset; arm joint defaults overridden for this task.
    right_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_RIGHT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_RIGHT_CFG.init_state.replace(
            joint_pos=_ur10e_hand_over_right_arm_joint_pos(),
        ),
    )
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_hand_over_left_arm_joint_pos(),
        ),
    )

    # Arm pose is fixed (above); policy acts on hand joints only.
    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    fingertip_body_names: tuple[str, ...] = (
        "ffdistal",
        "mfdistal",
        "rfdistal",
        "lfdistal",
        "thdistal",
    )

    # Third-person / TacSL (spawned post-clone in :class:`UR10eDualShadowHandDirectMARLBaseEnv`).
    enable_cameras: bool = False

    # in-hand object (between the two arm bases, slightly above the table)
    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.0335,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(density=500.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.48), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/cosmos_assets/1_object_A/banana/banana.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 放在机械臂前方稍远处，避免一开始就贴太近
            pos=(0.6, 0.0, 0.48), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.0335,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
            ),
        },
    )
    # scene
    scene: UR10eDualShadowHandDirectSceneCfg = UR10eDualShadowHandDirectSceneCfg(
        num_envs=2048, env_spacing=1.5, replicate_physics=True
    )

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset (hand only; arm uses scale below)
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    arm_reset_dof_pos_noise_scale: float = 0.0  # 0 = fixed arm joints; hand joints use ``reset_dof_pos_noise``
    # scales and constants
    fall_dist = 0.24
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    # reward-related scales
    dist_reward_scale = 20.0
