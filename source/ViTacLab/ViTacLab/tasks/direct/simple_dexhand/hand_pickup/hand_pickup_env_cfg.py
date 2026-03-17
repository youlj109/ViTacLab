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


@configclass
class UR10eShadowHandPickupSceneCfg(UR10eShadowHandTacSLSceneCfg):
    """Scene configuration for UR10e + ShadowHand pickup task with TacSL tactile sensors."""
    @classmethod
    def _tactile_params(cls) -> dict:
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/object",
            "enable_force_field": True,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
        }


@configclass
class UR10eShadowHandPickupEnvCfg(DirectRLEnvCfg):
    """Config for UR10e + ShadowHand pickup task."""

    # env meta (keep same structure as pour cfg for consistency)
    decimation = 2
    episode_length_s = 6.0
    # NOTE: must match the number of actuated joints selected in the environment (UR10e arm + ShadowHand hand)
    action_space = 30
    observation_space = 256
    state_space = 0
    asymmetric_obs = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        # Fabric cloning can fail for complex USD hierarchies (and breaks multi-env resets).
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

    # scene with tactile sensors
    scene: UR10eShadowHandPickupSceneCfg = UR10eShadowHandPickupSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # robot cfg
    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.copy()

    # joint name regex for action mapping
    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    # rigid object to lift: DexCube with SDF mesh (for TacSL force-field)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/SdfCube/SdfCube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 放在机械臂前方稍远处，避免一开始就贴太近
            pos=(0.9, 0.15, 0.05),
        ),
    )

    # goal marker (target pose above table)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_object_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            )
        },
    )

    # --- goal sampling (dexsuite-style: command-like target pose)
    goal_pos_x_range: tuple[float, float] = (0.75, 1.00)
    goal_pos_y_range: tuple[float, float] = (0.05, 0.25)
    goal_pos_z_range: tuple[float, float] = (0.20, 0.35)
    # resample goal periodically during episode (seconds)
    goal_resample_time_range_s: tuple[float, float] = (10.0, 10.0)

    # --- object reset randomization (offset from initial spawn pose)
    object_reset_pos_x_range: tuple[float, float] = (-0.10, 0.10)
    object_reset_pos_y_range: tuple[float, float] = (-0.10, 0.10)
    object_reset_pos_z_range: tuple[float, float] = (0.00, 0.10)
    object_reset_yaw_range: tuple[float, float] = (-3.14159, 3.14159)

    # --- robot joint reset randomization (offset from default joint pos)
    robot_reset_dof_pos_offset_range: tuple[float, float] = (-0.50, 0.50)
    robot_reset_dof_vel_range: tuple[float, float] = (0.0, 0.0)

    # --- observations
    vel_obs_scale: float = 0.2
    use_full_tactile_obs: bool = False

    # --- reward composition (dexsuite-like)
    pos_tracking_std: float = 0.10
    pos_tracking_weight: float = 2.0
    # small shaping term: distance from robot base/EE region to object
    ee_object_std: float = 0.10
    ee_object_weight: float = 0.5
    # within this radius, EE→object reward is considered equally good (plateau)
    ee_object_saturation_radius: float = 0.2
    success_pos_tol: float = 0.03
    success_height_tol: float = 0.02
    success_weight: float = 10.0
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005

    # --- terminations
    fall_height: float = 0.02
    out_of_bound_x: tuple[float, float] = (0.40, 1.30)
    out_of_bound_y: tuple[float, float] = (-0.60, 0.60)
    out_of_bound_z: tuple[float, float] = (0.00, 1.50)
    # action smoothing (same meaning as ShadowHandEnvCfg / InHandManipulationEnv)
    act_moving_average: float = 0.3

