import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10E_SHADOWHAND_LEFT_CFG,
    UR10eShadowHandTacSLSceneCfg,
)
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class UR10eShadowHandTactileSceneCfg(UR10eShadowHandTacSLSceneCfg):
    """Scene configuration for UR10e + ShadowHand with TacSL tactile sensors.

    Deformable cup has no rigid SDF mesh; use :class:`~ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_v2.VisuoTactileSensorV2`
    with ``contact_object_is_deformable=True`` (depth-based force field + soft-body nodal velocity).
    """

    @classmethod
    def _tactile_params(cls) -> dict:
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/cup",
            "enable_force_field": True,
            "contact_object_is_deformable": True,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
            "depth_penetration_deadband": 0.0,
        }


@configclass
class UR10eShadowHandPourEnvCfg(DirectRLEnvCfg):
    """Config for UR10e + ShadowHand deformable cup pouring (aligned struct with :class:`UR10eShadowHandPickupEnvCfg`)."""

    # env meta (match pickup)
    decimation = 2
    episode_length_s = 10.0
    action_space = 30
    observation_space = 256
    state_space = 0
    asymmetric_obs = False
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

    scene: UR10eShadowHandTactileSceneCfg = UR10eShadowHandTactileSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.copy()

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    cup_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/RealCups/cup3/cup.usd",
            scale=(1.5, 1.5, 1.5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.05),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        debug_vis=False,
    )

    water_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/water",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/Fluid/MyWater.usd",
            scale=(1, 1, 1),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.1),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        debug_vis=False,
    )

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/Bowl016/Bowl016.usd",
            scale=(1.5, 1.5, 1.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.9, -0.3, 0.10),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pour_goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.1)),
            )
        },
    )

    # --- goal sampling (env-local), reset each episode (like pickup goal ranges)
    goal_cup_pos_x_range: tuple[float, float] = (0.82, 0.98)
    goal_cup_pos_y_range: tuple[float, float] = (-0.35, -0.25)
    goal_cup_pos_z_range: tuple[float, float] = (0.22, 0.38)
    # optional: resample goal during episode (seconds); set min==max large to disable
    goal_resample_time_range_s: tuple[float, float] = (1.0e9, 1.0e9)

    # Goal cup orientation (w, x, y, z) for reward / marker; init buffer before first reset.
    goal_cup_rot_wxyz: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # Goal orientation written on each env reset (episode start).
    goal_cup_rot_reset_wxyz: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Rigid ``target_cup`` (bowl): only **x,y** follow ``goal_cup_pos``; **z** is fixed (table height).
    # ``target_xy = goal_xy + target_xy_offset_from_goal``, ``target_z = target_z_env`` (env frame).
    target_xy_offset_from_goal: tuple[float, float] = (0.0, 0.0)
    target_z_env: float = 0.10

    # --- deformable reset: planar jitter **x,y only** (no z); same offset applied to cup **and** water nodal state.
    reset_cup_pos_noise: float = 0.02
    reset_robot_dof_pos_noise: float = 0.0
    reset_robot_dof_vel_noise: float = 0.0
    robot_reset_dof_pos_offset_range: tuple[float, float] = (0.0, 0.0)
    robot_reset_dof_vel_range: tuple[float, float] = (0.0, 0.0)

    # --- observations
    vel_obs_scale: float = 0.2
    use_full_tactile_obs: bool = True

    # --- rewards (dexsuite-style, same spirit as pickup)
    pos_tracking_std: float = 0.12
    pos_tracking_weight: float = 2.0
    # Rotation vs goal: used only in **phase 1** (pour phase 2 does not penalize / gate on cup–goal rotation).
    rot_tracking_std: float = 0.35
    rot_tracking_weight: float = 1.0
    success_pos_tol: float = 0.05
    success_rot_tol: float = 0.25
    success_weight: float = 10.0
    success_ema_alpha: float = 0.1
    # If >0: episode counts as success only after this many **consecutive** steps with pose+water-in-bowl (pour).
    max_consecutive_success: int = 0
    # EMA of batch fraction of pour-success episodes (see env ``get_episode_success_rate``).
    episode_success_ema_alpha: float = 0.15
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    fall_penalty: float = -5.0

    # --- two-phase pour (uses same counter as ``episode_length_buf``; must be < max episode steps)
    # **Goal pos/rot tracking + action penalties apply every step.** Phase 1: if water–cup dist exceeds
    # ``phase1_water_far_penalty_start_dist``, add fixed ``phase1_water_far_penalty`` (same style as ``fall_penalty``)
    # and optionally multiply reward by ``phase1_water_far_reward_scale``. Phase 2: water→target bowl shaping.
    pour_phase_split_step: int = 400
    # Phase 1 — water should not leave cup (mean positions); hard fail beyond ``phase1_water_cup_max_dist``
    phase1_water_cup_max_dist: float = 0.1
    # Phase 1 — fixed penalty when ``‖water−cup‖ > start_dist`` (keep ``start_dist < max_dist`` for a warning band)
    phase1_water_far_penalty_start_dist: float = 0.1
    phase1_water_far_penalty: float = -5.0
    # When too far: ``reward *= scale`` after adding ``phase1_water_far_penalty`` (1.0 = no multiplicative term)
    phase1_water_far_reward_scale: float = 1.0
    # Phase 2 — extra shaping: water → target bowl (cup–goal is already in ``pos_tracking_*`` / ``success_pos_tol``)
    phase2_cup_goal_max_dist: float = 0.2
    phase2_water_target_xy_std: float = 0.10
    phase2_water_target_xy_weight: float = 2.5
    phase2_water_target_z_std: float = 0.08
    phase2_water_target_z_weight: float = 1.5
    # Phase-2 success: same pos/rot tolerances as phase 1, plus water near bowl
    phase2_success_water_xy_tol: float = 0.07
    phase2_success_water_z_tol: float = 0.10

    # --- terminations (env frame): cup root / water centroid (deformable mean pos)
    fall_height: float = 0.02
    out_of_bound_x: tuple[float, float] = (0.40, 1.30)
    out_of_bound_y: tuple[float, float] = (-0.70, 0.60)
    out_of_bound_z: tuple[float, float] = (-0.1, 1.50)
    # Water OOB (defaults match cup; widen if fluid spills before leaving table)
    water_out_of_bound_x: tuple[float, float] = (0.40, 1.30)
    water_out_of_bound_y: tuple[float, float] = (-0.70, 0.60)
    water_out_of_bound_z: tuple[float, float] = (-0.1, 1.50)

    act_moving_average: float = 0.3
