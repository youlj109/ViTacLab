"""UR10e + ShadowHand Factory-style peg insertion / gear mesh / nut threading."""

from __future__ import annotations

import os
import json
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    TacSLSensorPolicyCfg,
    UR10E_SHADOWHAND_LEFT_CFG,
    UR10eShadowHandTacSLSceneCfg,
)
from ViTacLab.tasks.direct.simple_gripper.forge_tasks_cfg import (
    ForgeGearMesh,
    ForgeNutThread,
    ForgePegInsert,
)

_INIT_POSE_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
_INIT_POSE_CONFIG_BY_TASK = {
    "peg_insert": "UR10eShadowHandForgeEnv__init_pose_hand_cfg__peg_insert.json",
    "gear_mesh": "UR10eShadowHandForgeEnv__init_pose_hand_cfg__gear_mesh.json",
    "nut_thread": "UR10eShadowHandForgeEnv__init_pose_hand_cfg__nut_thread.json",
}
_INIT_POSE_CONFIG_LEGACY = "UR10eShadowHandForgeEnv__init_pose_hand_cfg.json"


def _load_forge_init_pose_data(task_name: str = "peg_insert") -> dict:
    cfg_files = []
    task_file = _INIT_POSE_CONFIG_BY_TASK.get(str(task_name))
    if task_file:
        cfg_files.append(os.path.join(_INIT_POSE_CONFIG_DIR, task_file))
    cfg_files.append(os.path.join(_INIT_POSE_CONFIG_DIR, _INIT_POSE_CONFIG_LEGACY))
    for cfg_file in cfg_files:
        if os.path.exists(cfg_file):
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    return {}


@configclass
class ForgeDexhandObjectPosesCfg:
    """Articulation root poses in **environment frame** (m, wxyz). World pose = ``env_origin + pos``."""

    fixed_pos: tuple[float, float, float] = (0.8, 0.0, 0.00)
    # +90 deg around z-axis (wxyz).
    fixed_quat: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678)
    held_pos: tuple[float, float, float] = (0.8, 0.0, 0.14)
    held_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # ``gear_mesh`` + ``add_flanking_gears``: if ``None``, reuse ``fixed_*``.
    small_gear_pos: tuple[float, float, float] | None = None
    small_gear_quat: tuple[float, float, float, float] | None = None
    large_gear_pos: tuple[float, float, float] | None = None
    large_gear_quat: tuple[float, float, float, float] | None = None


def _ur10e_forge_arm_joint_pos(task_name: str = "peg_insert") -> dict[str, float]:
    """Initial UR10e pose for forge tasks."""
    data = _load_forge_init_pose_data(task_name)
    arm_joints = data.get("arm_joint_pos", {})
    if isinstance(arm_joints, dict) and arm_joints:
        return {str(k): float(v) for k, v in arm_joints.items()}

    return {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.57079632679,
        "elbow_joint": 1.57079632679,
        "wrist_1_joint": -1.57079632679,
        "wrist_2_joint": -1.57079632679,
        "wrist_3_joint": 0.0,
    }


def _ur10e_forge_hand_pose_shadow_order(
    name: str, task_name: str = "peg_insert"
) -> tuple[float, ...]:
    """Hand preset in ShadowHand canonical order (WRJ2, WRJ1, FFJ4...THJ1)."""
    data = _load_forge_init_pose_data(task_name)
    pose = data.get("hand_joint_pos_shadow_order", {}).get(str(name), [])
    if isinstance(pose, list) and pose:
        return tuple(float(v) for v in pose)
    return ()


@configclass
class UR10eShadowHandForgeTactileSceneCfg(UR10eShadowHandTacSLSceneCfg):
    """TacSL contacts target the Factory held peg/nut/gear (same prim as Isaac Factory)."""

    @classmethod
    def _tactile_policy(cls) -> TacSLSensorPolicyCfg:
        return TacSLSensorPolicyCfg(
            contact_object_prim_path_expr="/World/envs/env_.*/HeldAsset",
            enable_force_field=True,
            tactile_array_size=(20, 25),
            tactile_margin=0.005,
            use_physx_sparse_anchors=True,
            strict_target_contact_attribution=True,
            require_physx_sparse_anchors=True,
        )

    @classmethod
    def _tactile_params(cls) -> dict:
        # Keep compatibility path; canonical source is `_tactile_policy()`.
        return super()._tactile_params()


@configclass
class UR10eShadowHandForgeEnvCfg(DirectRLEnvCfg):
    """Factory forge assets / rewards / randomization; simulation stack aligned with :class:`UR10eShadowHandPickupEnvCfg`."""

    decimation: int = 2
    # Policy action: EE task-space command [dx, dy, dz, yaw_world_z].
    action_space: int = 4
    observation_space: int = 256
    state_space: int = 0
    asymmetric_obs: bool = False

    episode_length_s: float = 10.0
    task_name: str = "peg_insert"

    task: ForgePegInsert | ForgeGearMesh | ForgeNutThread = ForgePegInsert()

    object_poses: ForgeDexhandObjectPosesCfg = ForgeDexhandObjectPosesCfg()
    # Randomize fixed-asset xy at reset (env frame). Keep z fixed by default.
    fixed_reset_pos_x_range: tuple[float, float] = (-0.003, 0.003)
    fixed_reset_pos_y_range: tuple[float, float] = (-0.003, 0.003)
    fixed_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
    # Held-asset reset randomization (env frame), aligned with fixed randomization.
    held_reset_pos_x_range: tuple[float, float] = (-0.003, 0.003)
    held_reset_pos_y_range: tuple[float, float] = (-0.003, 0.003)
    held_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
    # Observation reference point (env frame) for ``held_rel`` / ``wrist_rel``; tune with ``object_poses``.
    fixed_obs_ref_pos: tuple[float, float, float] = (0.8, 0.0, 0.12)
    # Std dev of Gaussian noise added to ``fixed_obs_ref_pos`` in observations (0 = off).
    fixed_obs_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Match ``hand_pickup`` (light PhysX; fewer substeps per policy step than old Factory-scale settings).
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

    scene: UR10eShadowHandForgeTactileSceneCfg = UR10eShadowHandForgeTactileSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_forge_arm_joint_pos(),
        ),
    )

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    # Fallback EE body name; actual control body can prefer palm via ``prefer_palm_as_ee``.
    ee_body_name: str = "palm"
    # If True, pick palm-like body as EE control point so world-z yaw rotates around hand center.
    # If no palm body is found, fallback to ``ee_body_name`` and then wrist_3-like bodies.
    prefer_palm_as_ee: bool = True

    vel_obs_scale: float = 0.2
    use_full_tactile_obs: bool = False

    enable_cameras: bool = False

    # Optional decorative background (spawned in UR10eShadowHandForgeEnv._setup_scene).
    enable_high_fidelity_scene: bool = False
    high_fidelity_scene_usd_path: str = ""
    high_fidelity_scene_prim_path: str = "/World/envs/env_.*/HighFidelityScene"
    high_fidelity_scene_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    high_fidelity_scene_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    high_fidelity_scene_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Third-person camera override (single-arm), aligned with hand_pickup/v2 defaults.
    third_person_camera_pos: tuple[float, float, float] = (1.5, 0.0, 0.8)
    third_person_camera_rot: tuple[float, float, float, float] = (
        0.64086,
        0.29884,
        0.29884,
        0.64086,
    )
    # hand_pickup/v2 third-person resolution
    third_person_camera_width: int = 640
    third_person_camera_height: int = 480
    # Optional wrist/twist camera (expects camera prim in robot USD).
    enable_twist_camera: bool = True
    twist_camera_prim_path: str = "/World/envs/env_.*/Robot/wrist/twist_camera"
    # Wrist camera resolution.
    twist_camera_width: int =  640
    twist_camera_height: int = 480
    twist_camera_data_types: tuple[str, ...] = ("rgb", "distance_to_image_plane")

    # Hand two-state posture from preset file ``scripts/rsl_rl/full_tra/hand_pose_presets/...``.
    # Order: WRJ2, WRJ1, FFJ4...THJ1.
    hand_open_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order("open")
    hand_close_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order("close")
    # Task-space mode hand target source:
    # True  -> keep hand joints at robot default_joint_pos (loaded from pose keyframe, e.g. pose_003).
    # False -> force closed hand target each step from preset.
    hand_target_from_default_pose: bool = False

    # EE task-space control settings.
    # action[0:3] are xyz offsets (m) around ``fixed_obs_ref_pos``.
    ee_pos_action_bounds: tuple[float, float, float] = (0.08, 0.08, 0.08)
    # Per-step clipping in xyz (m), limits IK target jump.
    ee_pos_step_clip: tuple[float, float, float] = (0.02, 0.02, 0.02)
    # action[3] maps to absolute yaw (deg) around world z-axis.
    ee_yaw_world_range_deg: tuple[float, float] = (-30.0, 30.0)
    # Per-step world-z yaw change clamp (deg).
    ee_yaw_world_step_clip_deg: float = 10.0

    # Differential IK settings for UR10e arm joints.
    ik_method: str = "dls"
    ik_lambda: float = 0.01

    # Contact friction overrides for forge_dexhand.
    # Effective contact friction depends on both sides (robot links and held object).
    held_object_friction: float = 1.5
    robot_friction: float = 1.5

    # Pre-grasp IK target in held-asset local frame (EE control-body pose).
    pregrasp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.12)
    # wxyz, +90deg about local y: EE local-x aligns with held-asset local -z.
    pregrasp_offset_quat: tuple[float, float, float, float] = (
        0.70710678,
        0.0,
        0.70710678,
        0.0,
    )
    # Per-episode pre-grasp randomization (world frame xyz noise and local yaw noise).
    pregrasp_pos_noise: tuple[float, float, float] = (0, 0, 0)
    pregrasp_yaw_noise_deg: float = 0.0
    # IK servo budget / convergence tolerances during reset pre-grasp stage.
    pregrasp_ik_max_steps: int = 24
    pregrasp_ik_pos_tol: float = 0.003
    pregrasp_ik_rot_tol_deg: float = 6.0
    # False: keep reset exactly at pose keyframe joint state (e.g., pose_003).
    # True: run reset-time pregrasp IK and hand-open posture override.
    use_pregrasp_reset: bool = False
    # Debug print EE/object pose every N env steps (0 disables).
    debug_pose_print_interval: int = 0
    debug_pose_print_env_index: int = 0
    # Debug print success-condition terms every N env steps (0 disables).
    debug_success_print_interval: int = 0
    debug_success_print_env_index: int = 0

    act_moving_average: float = 0.3


@configclass
class UR10eShadowHandForgePegInsertEnvCfg(UR10eShadowHandForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    # Use smaller reset randomization range for insert data collection.
    # fixed_reset_pos_x_range: tuple[float, float] = (-0.0005, 0.0005)
    # fixed_reset_pos_y_range: tuple[float, float] = (0, 0)
    # held_reset_pos_x_range: tuple[float, float] = (0, 0)
    # held_reset_pos_y_range: tuple[float, float] = (0, 0)
    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_forge_arm_joint_pos("peg_insert"),
        ),
    )
    hand_open_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order(
        "open", "peg_insert"
    )
    hand_close_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order(
        "close", "peg_insert"
    )
    task.success_threshold = 1
    episode_length_s = 10.0


@configclass
class UR10eShadowHandForgeGearMeshEnvCfg(UR10eShadowHandForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_forge_arm_joint_pos("gear_mesh"),
        ),
    )
    hand_open_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order(
        "open", "gear_mesh"
    )
    hand_close_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order(
        "close", "gear_mesh"
    )
    # Loosen gear-mesh success z-threshold multiplier for replay/data collection.
    task.success_threshold = 0.5
    episode_length_s = 20.0


@configclass
class UR10eShadowHandForgeNutThreadEnvCfg(UR10eShadowHandForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    # Loosen nut-thread yaw success gate (rad). Success checks `curr_yaw < ee_success_yaw`.

    task.ee_success_yaw = 2.0
    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_forge_arm_joint_pos("nut_thread"),
        ),
    )
    hand_open_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order(
        "open", "nut_thread"
    )
    hand_close_joint_pos_shadow_order: tuple[float, ...] = _ur10e_forge_hand_pose_shadow_order(
        "close", "nut_thread"
    )
    task.success_threshold = 0.5
    episode_length_s = 30.0
