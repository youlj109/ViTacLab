# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual UR10e + ShadowHand bottle cap unscrewing (articulated bottle + cap)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_cfg import (
    UR10E_DUAL_SHADOWHAND_LEFT_CFG,
    UR10E_DUAL_SHADOWHAND_RIGHT_CFG,
    UR10eDualShadowHandDirectSceneCfg,
)

# -----------------------------------------------------------------------------
# Defaults — must match DOF joint names on ``mobility.usd`` (see ``scripts/debug/inspect_usd_structure.py``).
# PartNet bottle (3517): ``link_0_helper/joint_0`` [PhysicsRevoluteJoint],
# ``link_1/joint_2`` [PhysicsPrismaticJoint]. ``base/joint_1`` is a fixed joint (no DOF).
# -----------------------------------------------------------------------------
CAP_ROTATION_JOINT_NAME: str = "joint_0"
CAP_TRANSLATION_JOINT_NAME: str = "joint_2"

BOTTLE_USD_PATH: str = "source/ViTacLab/ViTacLab/assets/data/Objects/Bottle/3517/mobility.usd"


def _ur10e_unscrew_right_arm_joint_pos() -> dict[str, float]:
    """Default UR10e arm pose for bottle-cap unscrew (tuned via ``run_ur10e_dual_shadowhand_arm_pose_from_marker``)."""

    return {
        "shoulder_pan_joint": -0.2017389824472911,
        "shoulder_lift_joint": -0.7865157379181013,
        "elbow_joint": 0.7678906702100681,
        "wrist_1_joint": -1.5067936692796640,
        "wrist_2_joint": -1.3525333339304586,
        "wrist_3_joint": 1.3640808964941316,
    }


def _ur10e_unscrew_left_arm_joint_pos() -> dict[str, float]:
    """Default UR10e arm pose for bottle-cap unscrew (tuned via ``run_ur10e_dual_shadowhand_arm_pose_from_marker``)."""

    return {
        "shoulder_pan_joint": 0.2778900402896463,
        "shoulder_lift_joint": -0.6620817640379858,
        "elbow_joint": 2.0146139484748380,
        "wrist_1_joint": 1.7890604689396576,
        "wrist_2_joint": -1.8486863670848717,
        "wrist_3_joint": -3.1415963804429605,
    }


def default_bottle_articulation_cfg(
    *,
    cap_rotation_damping: float = 0.22,
    cap_translation_damping: float = 0.06,
    rotation_joint_name: str = CAP_ROTATION_JOINT_NAME,
    translation_joint_name: str = CAP_TRANSLATION_JOINT_NAME,
    spawn_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    init_pos: tuple[float, float, float] = (0.62, 0.0, 0.08),
) -> ArticulationCfg:
    """Passive cap joints: revolute + prismatic with viscous damping (no stiffness).

    Translation is additionally **locked in simulation** by the env until rotation exceeds a threshold
    (see :class:`UR10eDualShadowHandUnscrewBottleCapEnvCfg`).
    """

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOTTLE_USD_PATH,
            scale=spawn_scale,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=True,
                angular_damping=0.02,
                linear_damping=0.02,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.001),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Env frame; z should match ``UR10eDualShadowHandUnscrewBottleCapEnvCfg.bottle_height_m`` / table.
            pos=init_pos,
            rot=(0.70711, 0.70711, 0.0, 0.0),
            joint_pos={
                rotation_joint_name: 0.0,
                translation_joint_name: 0.0,
            },
        ),
        actuators={
            "cap_revolute": ImplicitActuatorCfg(
                joint_names_expr=[rotation_joint_name],
                effort_limit_sim=80.0,
                stiffness=0.0,
                damping=cap_rotation_damping,
                friction=0.04,
            ),
            "cap_prismatic": ImplicitActuatorCfg(
                joint_names_expr=[translation_joint_name],
                effort_limit_sim=500.0,
                stiffness=0.0,
                damping=cap_translation_damping,
                friction=0.04,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def default_visual_goal_bottle_cfg(
    *,
    bottle_spawn_scale: tuple[float, float, float] = (0.1, 0.1, 0.2),
    init_pos: tuple[float, float, float] = (0.62, 0.0, 0.15),
    init_rot_wxyz: tuple[float, float, float, float] = (0.70711, 0.70711, 0.0, 0.0),
) -> RigidObjectCfg:
    """**Visual-only** goal cylinder under each ``/World/envs/env_*`` (kinematic, no collision).

    Uses a real rigid body prim so the Stage **Translate** matches the goal (unlike
    :class:`~isaaclab.markers.VisualizationMarkers`, where instance poses live on PointInstancer ``positions`` and
    the instancer prim often shows ``(0,0,0)``). Size tracks ``bottle_spawn_scale``: diameter = ``max(sx,sy)``,
    height = ``sz``. Material is **opaque** for visibility.
    """

    sx, sy, sz = bottle_spawn_scale
    diameter = max(float(sx), float(sy), 0.008)
    radius = 0.5 * diameter
    height = max(float(sz), 0.02)

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/visual_goal_bottle",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.95, 0.25),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=-0.001,
                collision_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot_wxyz),
    )


UR10E_DUAL_UNSCREW_NUM_ARM_DOFS: int = 6
UR10E_DUAL_UNSCREW_NUM_HAND_DOFS: int = 24
UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS: int = UR10E_DUAL_UNSCREW_NUM_ARM_DOFS + UR10E_DUAL_UNSCREW_NUM_HAND_DOFS

# Observation layout (must match ``_get_observations`` / ``_get_states`` tensor cat order).
_FINGER_OBS_DIM: int = 5 * (3 + 4 + 6)
_BOTTLE_OBS_DIM: int = 3 + 4 + 3 + 3
_CAP_JOINT_OBS_DIM: int = 4


def _unscrew_obs_dim(num_actuated_dofs: int) -> int:
    """Per-agent: joint pos/vel/actions (3n), fingertips, bottle pose/vel, cap joints."""

    n = num_actuated_dofs
    return 3 * n + _FINGER_OBS_DIM + _BOTTLE_OBS_DIM + _CAP_JOINT_OBS_DIM


def _unscrew_state_dim(num_actuated_dofs: int) -> int:
    """Central critic: two agents × (joint pos + vel + actions + fingertips) + shared bottle + cap joints."""

    n = num_actuated_dofs
    per_agent = 3 * n + _FINGER_OBS_DIM
    return 2 * per_agent + _BOTTLE_OBS_DIM + _CAP_JOINT_OBS_DIM


@configclass
class UR10eDualShadowHandUnscrewBottleCapEnvCfg(DirectMARLEnvCfg):
    """Dual UR10e + Shadow Hand (arm + hand) + articulated bottle (cap revolute + prismatic)."""

    decimation = 2
    episode_length_s = 12.0
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {
        "right_hand": UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS,
        "left_hand": UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS,
    }
    observation_spaces = {
        "right_hand": _unscrew_obs_dim(UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS),
        "left_hand": _unscrew_obs_dim(UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS),
    }
    state_space = _unscrew_state_dim(UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS)

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            enable_ccd=True,
        ),
    )

    # Task-specific UR10e arm defaults (hand defaults from USD via partial ``joint_pos``).
    right_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_RIGHT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_RIGHT_CFG.init_state.replace(
            joint_pos=_ur10e_unscrew_right_arm_joint_pos(),
        ),
    )
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_unscrew_left_arm_joint_pos(),
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

    # Joint names (must match USD and ``default_bottle_articulation_cfg``).
    cap_rotation_joint_name: str = CAP_ROTATION_JOINT_NAME
    cap_translation_joint_name: str = CAP_TRANSLATION_JOINT_NAME

    # Passive joint damping (used when building ``bottle_cfg`` in the env if ``bottle_cfg`` is None).
    cap_rotation_damping: float = 0.22
    cap_translation_damping: float = 0.06

    # Uniform scale is typical; adjust if PartNet / asset units do not match the table & hands.
    bottle_spawn_scale: tuple[float, float, float] = (0.1, 0.1, 0.2)

    # World height after scale (m); tune with ``bottle_spawn_scale``; used when setting ``fall_height`` vs init z.
    bottle_height_m: float = 0.16
    # Env-frame bottle root position (m); default ~ tabletop + half height if root near centroid (tune with scene).
    bottle_root_init_pos: tuple[float, float, float] = (0.62, 0.0, 0.15)

    bottle_cfg: ArticulationCfg | None = None

    # --- Task goal: ``env.goal_pos`` is **world** xyz (m); ``goal_quat_w`` is world wxyz -------------------------
    # Nominal goal in **each env's local frame** (same convention as ``bottle_root_init_pos``); reset adds
    # ``scene.env_origins`` so ``env.goal_pos`` is global. Independent of bottle root after noise unless
    # :attr:`goal_uses_reset_position_noise` is True (then ``dpos`` matches the bottle).
    goal_pos_local: tuple[float, float, float] = (0.62, 0.0, 0.5)
    goal_quat_wxyz: tuple[float, float, float, float] = (0.70711, 0.70711, 0.0, 0.0)

    goal_uses_reset_position_noise: bool = True
    """If True, world ``goal_pos`` uses the same ``dpos`` as the physical bottle root (see ``reset_position_noise``)."""

    # Kinematic cylinder under each env (``visual_goal_bottle``); world pose from ``goal_pos`` / ``goal_quat_w``.
    enable_visual_goal_bottle: bool = True
    visual_goal_bottle_cfg: RigidObjectCfg | None = None

    # Runtime: lock prismatic DOF until |revolute| >= this (rad).
    cap_translation_unlock_angle_rad: float = 0.45

    # PhysX solver joint limits (override USD if set). Applied once in env init via
    # :meth:`Articulation.write_joint_position_limit_to_sim` / ``write_joint_velocity_limit_to_sim``.
    apply_bottle_joint_limits: bool = True
    # Revolute ``joint_0`` (rad). PhysX revolute limits must lie in **[-2π, 2π]** per joint (see PhysX error on wider spans).
    cap_rotation_pos_limit: tuple[float, float] = (-6.283185307179586, 6.283185307179586)
    # Prismatic ``joint_2`` (m). Typical cap stroke; tune with asset / ``bottle_spawn_scale``.
    cap_translation_pos_limit: tuple[float, float] = (0.0, 0.1)
    # Max |velocity| in solver (rad/s and m/s). None = do not override PhysX defaults.
    cap_rotation_velocity_limit_sim: float | None = 8.0
    cap_translation_velocity_limit_sim: float | None = 0.4

    # --- Staged reward (soft switching) -------------------------------------------------------------------------
    # progress = w1*grasp_success + w2*rotation_progress + w3*lift_height  (all in [0,1], logged).
    # reward = r_grasp*(1 - rotation_progress) + r_rotate*grasp_success + r_lift*rotation_success  (+ penalties).
    # grasp_success: blend of soft position/orientation match to goal; rotation_progress: |cap_rot| / max;
    # lift_height: cap_trans / max; rotation_success: smooth ramp from unlock angle to ``rotation_success_ramp_max_rad``.
    # ``grasp_rot_align_pos_std``: tighter than ``staged_grasp_pos_std`` so rotation alignment only turns on near goal pos.
    staged_progress_w1: float = 0.4
    staged_progress_w2: float = 0.35
    staged_progress_w3: float = 0.25
    staged_grasp_pos_std: float = 0.10
    staged_grasp_rot_std: float = 0.45
    grasp_rot_align_pos_std: float = 0.038
    grasp_success_pos_blend: float = 0.65
    grasp_success_rot_blend: float = 0.35
    rotation_progress_max_rad: float = 3.14159
    lift_progress_max_m: float = 0.08
    rotation_success_ramp_min_rad: float = 0.45
    rotation_success_ramp_max_rad: float = 0.85
    r_grasp_pos_scale: float = 2.0
    r_grasp_rot_scale: float = 1.0
    unscrew_rotation_sign: float = 1.0
    r_rotate_scale: float = 2.0
    r_lift_scale: float = 5.0
    # Hard gate: no cap revolute/prismatic **positive** rewards until both are satisfied (m, rad).
    cap_joint_reward_pos_thresh_m: float = 0.06
    cap_joint_reward_rot_thresh_rad: float = 0.1

    # Fingertip–bottle grasp shaping (thumb + four fingers contact, per-finger opposition, stability).
    # ``grasp_other_mean_w`` scales the **geometric mean** of ff,mf,rf,lf soft contacts (not plain mean).
    # Opposition = mean over each non-thumb finger of ``clamp(-dot(v_thumb,v_i),0,1)`` vs object center.
    # Fingertip order in env matches ``fingertip_body_names``: ff,mf,rf,lf,th → thumb = index 4.
    # Right hand: object center = this bottle link (cap); left hand: bottle articulation root (body index 0).
    right_grasp_contact_body_name: str = "link_1"
    enable_grasp_contact_reward: bool = True
    grasp_contact_dist_m: float = 0.06
    grasp_thumb_w: float = 0.3
    grasp_other_mean_w: float = 0.3
    grasp_opposition_w: float = 0.4
    grasp_stability_weight: float = 0.2
    grasp_contact_reward_scale: float = 0.35

    # Penalties: per-agent L2 on actions and action-rate (positive coeffs; subtracted in :meth:`_get_rewards`).
    action_l2_penalty: float = 0.005
    action_rate_l2_penalty: float = 0.005
    wrong_rotation_penalty_weight: float = -0.15
    out_of_bound_x: tuple[float, float] = (0.40, 1.30)
    out_of_bound_y: tuple[float, float] = (-0.60, 0.60)
    out_of_bound_z: tuple[float, float] = (0.00, 1.50)
    bottle_out_of_bounds_penalty_weight: float = -0.5

    # Success when axial joint exceeds this (meters), or total rotation exceeds cap (rad).
    success_min_translation_m: float = 0.012
    success_total_rotation_rad: float = 6.0

    scene: UR10eDualShadowHandDirectSceneCfg = UR10eDualShadowHandDirectSceneCfg(
        num_envs=512,
        env_spacing=1.5,
        replicate_physics=True,
    )

    reset_position_noise = 0.008
    reset_dof_pos_noise = 0.15
    reset_dof_vel_noise = 0.0
    arm_reset_dof_pos_noise_scale: float = 0.0

    # Env-frame z: terminate if bottle root drops below this (~ ``bottle_root_init_pos[2] - bottle_height_m``).
    fall_height = 0.05
    vel_obs_scale = 0.2
    act_moving_average = 1.0
