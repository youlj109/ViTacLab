# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + Shadow Hand in-hand cube reorientation (DexCube-style), built on the direct base env."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10E_SHADOWHAND_LEFT_CFG,
    UR10eShadowHandTacSLSceneCfg,
)
from ViTacLab.tasks.direct.simple_dexhand.shadow_hand.shadow_hand_env_cfg import EventCfg


def _ur10e_left_inhand_arm_joint_pos() -> dict[str, float]:
    """Arm default pose for in-hand task (UR10e + ShadowHand).

    Computed from ``ur10_palm_pose_to_arm_joints.py``: palm under object, palm-up, with
    ``--offset-preset video-teleop-tag1`` (see task ``object_cfg.init_state.pos``).
    """
    return {
        "shoulder_pan_joint": -0.5495074154030938,
        "shoulder_lift_joint": -1.6317947052323547,
        "elbow_joint": 2.6047321783754982,
        "wrist_1_joint": -0.9729374731541182,
        "wrist_2_joint": 1.0212889113918058,
        "wrist_3_joint": 3.1415889271506519,
    }


def _inhand_policy_obs_dim(*, obs_type: str, num_hand_dofs: int, num_fingertips: int = 5) -> int:
    """Match `InHandManipulationEnv` observation layout (no tactile)."""
    act = num_hand_dofs
    if obs_type == "openai":
        return num_fingertips * 3 + 3 + 4 + act
    # full
    return (
        2 * num_hand_dofs
        + 3
        + 4
        + 3
        + 3  # object pos, rot, linvel, angvel
        + 3
        + 4
        + 4  # in_hand_pos, goal_rot, quat_err
        + num_fingertips * (3 + 4 + 6)
        + act
    )


def _inhand_critic_state_dim(*, num_hand_dofs: int, num_fingertips: int = 5) -> int:
    """Asymmetric critic: full state + fingertip F/T + actions (no tactile)."""
    act = num_hand_dofs
    return (
        2 * num_hand_dofs
        + 3
        + 4
        + 3
        + 3
        + 3
        + 4
        + 4
        + num_fingertips * (3 + 4 + 6)
        + num_fingertips * 6
        + act
    )


# UR10e + left Shadow Hand (revolute hand joints only): 24 DOFs (see ur10e_shadow_left_hand_glb.urdf).
UR10E_SHADOW_LEFT_HAND_DOF_COUNT: int = 24

# 5 * (20*25) + 5 * (20*25) * 2 (must match `inhand_manipulation_env.TACTILE_*`)
INHAND_TACTILE_NORMAL_DIM: int = 5 * 20 * 25
INHAND_TACTILE_SHEAR_DIM: int = 5 * 20 * 25 * 2


def sync_inhand_rl_space_dims(cfg: "UR10eShadowHandInHandEnvCfg") -> None:
    """Set action / policy / critic dims from `obs_type`, `num_hand_dofs`, `reduced_obs`, `asymmetric_obs`."""
    nh = int(cfg.num_hand_dofs)
    cfg.action_space = nh
    pol = _inhand_policy_obs_dim(obs_type=cfg.obs_type, num_hand_dofs=nh, num_fingertips=5)
    if not getattr(cfg, "reduced_obs", True):
        pol += INHAND_TACTILE_NORMAL_DIM + INHAND_TACTILE_SHEAR_DIM
    cfg.observation_space = pol
    if cfg.asymmetric_obs:
        st = _inhand_critic_state_dim(num_hand_dofs=nh, num_fingertips=5)
        if not getattr(cfg, "reduced_obs", True):
            st += INHAND_TACTILE_NORMAL_DIM + INHAND_TACTILE_SHEAR_DIM
        cfg.state_space = st
    else:
        cfg.state_space = 0


@configclass
class UR10eShadowHandInHandSceneCfg(UR10eShadowHandTacSLSceneCfg):
    """TacSL + DexCube object path (SDF / force-field compatible)."""

    @classmethod
    def _tactile_params(cls) -> dict:
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/object",
            "enable_force_field": True,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
        }


@configclass
class UR10eShadowHandInHandEnvCfg(DirectRLEnvCfg):
    """UR10e arm (fixed pose / no policy actions) + Shadow Hand in-hand manipulation."""

    decimation = 2
    episode_length_s = 10.0
    # Filled by `sync_inhand_rl_space_dims` in `InHandManipulationEnv.__init__`.
    action_space = UR10E_SHADOW_LEFT_HAND_DOF_COUNT
    observation_space = 1
    state_space = 0
    asymmetric_obs = False
    # Third-person + TacSL GelSight sensors (see UR10eShadowHandDirectBaseEnv). False = fast headless RL.
    enable_cameras: bool = False
    obs_type: str = "full"
    # Expected hand DOF count (must match `hand_joint_expr` on the USD articulation).
    num_hand_dofs: int = UR10E_SHADOW_LEFT_HAND_DOF_COUNT

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

    scene: UR10eShadowHandInHandSceneCfg = UR10eShadowHandInHandSceneCfg(
        num_envs=128,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_SHADOWHAND_LEFT_CFG.init_state.replace(joint_pos=_ur10e_left_inhand_arm_joint_pos()),
    )

    # Base env uses these to find actuated DOFs; the task env restricts actions to the hand only.
    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    # In-hand cube (same asset family as standalone Shadow Hand task).
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/DexCube/dex_cube_sdf.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            semantic_tags=[("class", "cube")],
        ),
        # Start near the palm (env frame); tune if the cube spawns outside the hand.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.78, 0.0, 0.3), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/DexCube/dex_cube_sdf.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )

    # Fingertip links on the UR10e+Shadow USD (no `robot0_` prefix).
    fingertip_body_names: tuple[str, ...] = (
        "ffdistal",
        "mfdistal",
        "rfdistal",
        "lfdistal",
        "thdistal",
    )

    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    # No arm jitter: only hand joints use `reset_dof_pos_noise` in env.
    arm_reset_dof_pos_noise_scale: float = 0.0

    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 2
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

    # Goal marker pose (env-frame offset added to env_origins in env).
    goal_marker_pos: tuple[float, float, float] = (0.8, -0.20, 0.1)


@configclass
class UR10eShadowHandInHandOpenAIEnvCfg(UR10eShadowHandInHandEnvCfg):
    decimation = 3
    episode_length_s = 8.0
    asymmetric_obs = True
    obs_type = "openai"
    success_tolerance = 0.4
    max_consecutive_success = 50
    fall_penalty = -50.0
    act_moving_average = 0.3
    events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        use_fabric=True,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )


@configclass
class UR10eShadowHandInHandTactileEnvCfg(UR10eShadowHandInHandEnvCfg):
    """Tactile sensors are spawned in the base env after cloning (not as scene cfg fields)."""

    reduced_obs: bool = True
