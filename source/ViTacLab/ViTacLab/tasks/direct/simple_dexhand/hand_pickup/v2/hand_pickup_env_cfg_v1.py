import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg_v1 import (
    UR10E_SHADOWHAND_LEFT_CFG,
    UR10eShadowHandTacSLSceneCfg,
)


@configclass
class UR10eShadowHandPickupSceneCfgV1(UR10eShadowHandTacSLSceneCfg):
    """Pickup scene (v1).

    TacSL contact / grid matches ``example.inhand_manipulation_env_cfg.UR10eShadowHandInHandSceneCfg``
    so GelSight + force-field behavior is consistent for data collection.
    """

    @classmethod
    def _tactile_params(cls) -> dict:
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/object",
            "enable_force_field": True,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
        }


@configclass
class UR10eShadowHandPickupEnvCfgV1(DirectRLEnvCfg):
    """Pickup task cfg (v1).

    Pickup-specific: object spawn, goals, success, bounds, rewards (unchanged semantics).

    Data-collection knobs mirror ``example.inhand_manipulation_env_cfg.UR10eShadowHandInHandEnvCfg``:
    ``enable_cameras``, ``visual_disturbance*``, third-person overrides. Set ``enable_cameras=True``
    (or play ``--record_data`` / ``ENABLE_CAMERAS``) so TacSL + third-person match inhand recording.
    """

    decimation = 2
    episode_length_s = 6.0
    action_space = 30
    observation_space = 256
    state_space = 0
    asymmetric_obs = False

    # Inhand-style recording-related knobs
    enable_cameras: bool = False
    visual_disturbance: bool = False
    visual_disturbance_type: str = "gaussian_noise"
    visual_disturbance_noise_std: float = 0.08
    visual_disturbance_blur_kernel_size: int = 5
    visual_disturbance_blur_sigma: float = 1.0

    # Keep these fields for compatibility; values follow inhand-style third-person defaults.
    third_person_camera_pos: tuple[float, float, float] = (1.5, 0.0, 0.8)
    third_person_camera_rot: tuple[float, float, float, float] = (0.64086, 0.29884, 0.29884, 0.64086)
    third_person_camera_width: int = 640
    third_person_camera_height: int = 480

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        use_fabric=True,
        physics_material=RigidBodyMaterialCfg(
            static_friction=10.0,
            dynamic_friction=10.0,
        ),
        physx=PhysxCfg(
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Match inhand defaults (``UR10eShadowHandInHandEnvCfg.scene``): spacing / replicate / fabric.
    scene: UR10eShadowHandPickupSceneCfgV1 = UR10eShadowHandPickupSceneCfgV1(
        num_envs=128,
        env_spacing=5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.copy()
    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"
    object_scale: tuple[float, float, float] = (1.0, 2.0, 1.0)

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/SdfCube/SdfCube.usd",
            scale=object_scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.15, 0.05)),
    )

    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_object_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            )
        },
    )

    goal_pos_x_range: tuple[float, float] = (0.75, 1.00)
    goal_pos_y_range: tuple[float, float] = (0.05, 0.25)
    goal_pos_z_range: tuple[float, float] = (0.20, 0.35)
    goal_resample_time_range_s: tuple[float, float] = (10.0, 10.0)

    object_reset_pos_x_range: tuple[float, float] = (-0.04, 0.04)
    object_reset_pos_y_range: tuple[float, float] = (-0.04, 0.04)
    object_reset_pos_z_range: tuple[float, float] = (0.00, 0.00)
    object_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    robot_reset_dof_pos_offset_range: tuple[float, float] = (0.0, 0.0)
    robot_reset_dof_vel_range: tuple[float, float] = (0.0, 0.0)

    vel_obs_scale: float = 0.2
    use_full_tactile_obs: bool = True

    pos_tracking_std: float = 0.10
    pos_tracking_weight: float = 2.0
    # Success rule:
    # - "lift_and_goal_z": object is lifted above init z by ``grasp_lift_min_dz`` and reaches goal z within ``goal_z_tol``.
    # - "xyz_tolerance": legacy mode using ``success_height_tol`` + ``success_pos_tol``.
    success_mode: str = "lift_and_goal_z"
    grasp_lift_min_dz: float = 0.02
    goal_z_tol: float = 0.012
    success_pos_tol: float = 0.03
    success_height_tol: float = 0.02
    success_weight: float = 10.0
    success_ema_alpha: float = 0.1
    max_consecutive_success: int = 0
    episode_success_ema_alpha: float = 0.15
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005

    fall_height: float = 0.02
    out_of_bound_x: tuple[float, float] = (0.40, 1.30)
    out_of_bound_y: tuple[float, float] = (-0.60, 0.60)
    out_of_bound_z: tuple[float, float] = (0.00, 1.50)
    act_moving_average: float = 0.3
