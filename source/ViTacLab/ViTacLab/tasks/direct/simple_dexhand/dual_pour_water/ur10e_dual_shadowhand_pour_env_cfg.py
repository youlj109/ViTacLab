"""Dual UR10e + ShadowHand deformable cup pouring (MARL; shared cup / water / bowl)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
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

# -----------------------------------------------------------------------------
# Observation layout (per agent): same structure as single-arm pour, own robot + shared task.
# -----------------------------------------------------------------------------
UR10E_DUAL_POUR_NUM_ARM_DOFS: int = 6
UR10E_DUAL_POUR_NUM_HAND_DOFS: int = 24
UR10E_DUAL_POUR_NUM_ACTUATED_DOFS: int = UR10E_DUAL_POUR_NUM_ARM_DOFS + UR10E_DUAL_POUR_NUM_HAND_DOFS

TACTILE_POINTS_PER_SENSOR: int = 20 * 25
TACTILE_NORMAL_DIM: int = 5 * TACTILE_POINTS_PER_SENSOR
TACTILE_SHEAR_DIM: int = 5 * TACTILE_POINTS_PER_SENSOR * 2


def _pour_base_obs_dim() -> int:
    """Joint pos/vel + cup(3+4) + goal(3) + pos_err(3) + actions + action_rate."""

    n = UR10E_DUAL_POUR_NUM_ACTUATED_DOFS
    return n + n + 3 + 4 + 3 + 3 + n + n


def _pour_obs_dim_per_agent(*, use_full_tactile_obs: bool) -> int:
    tactile_dim = (TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM) if use_full_tactile_obs else (5 * (1 + 2))
    return _pour_base_obs_dim() + tactile_dim


@configclass
class UR10eDualShadowHandPourTactileSceneCfg(UR10eDualShadowHandTacSLSceneCfg):
    """TacSL contacts the deformable cup (depth-based + soft body); same idea as single-arm pour."""

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
class UR10eDualShadowHandPourEnvCfg(DirectMARLEnvCfg):
    """Dual UR10e + ShadowHand pour (one shared deformable cup + water + rigid bowl)."""

    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {
        "right_hand": UR10E_DUAL_POUR_NUM_ACTUATED_DOFS,
        "left_hand": UR10E_DUAL_POUR_NUM_ACTUATED_DOFS,
    }
    # Defaults assume ``use_full_tactile_obs=True``; env replaces if toggled.
    observation_spaces = {
        "right_hand": _pour_obs_dim_per_agent(use_full_tactile_obs=True),
        "left_hand": _pour_obs_dim_per_agent(use_full_tactile_obs=True),
    }
    # Central critic: concat both agents' observations (same as ``torch.cat(obs.values(), dim=-1)``).
    state_space = -1
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

    scene: UR10eDualShadowHandPourTactileSceneCfg = UR10eDualShadowHandPourTactileSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    right_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_RIGHT_CFG
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    cup_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/RealCups/cup3/cup.usd",
            scale=(1.5, 1.5, 1.5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, -0.28, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
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
            pos=(0.0, -0.28, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
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
            pos=(0.88, -0.28, 0.10),
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

    goal_cup_pos_x_range: tuple[float, float] = (0.82, 0.98)
    goal_cup_pos_y_range: tuple[float, float] = (-0.35, -0.25)
    goal_cup_pos_z_range: tuple[float, float] = (0.22, 0.38)
    goal_resample_time_range_s: tuple[float, float] = (1.0e9, 1.0e9)

    goal_cup_rot_wxyz: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    goal_cup_rot_reset_wxyz: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    target_xy_offset_from_goal: tuple[float, float] = (0.0, 0.0)
    target_z_env: float = 0.10

    reset_cup_pos_noise: float = 0.02
    robot_reset_dof_pos_offset_range: tuple[float, float] = (0.0, 0.0)
    robot_reset_dof_vel_range: tuple[float, float] = (0.0, 0.0)

    vel_obs_scale: float = 0.2
    use_full_tactile_obs: bool = True

    pos_tracking_std: float = 0.12
    pos_tracking_weight: float = 2.0
    rot_tracking_std: float = 0.35
    rot_tracking_weight: float = 1.0
    success_pos_tol: float = 0.05
    success_rot_tol: float = 0.25
    success_weight: float = 10.0
    success_ema_alpha: float = 0.1
    max_consecutive_success: int = 0
    episode_success_ema_alpha: float = 0.15
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005
    fall_penalty: float = -5.0

    pour_phase_split_step: int = 400
    phase1_water_cup_max_dist: float = 0.1
    phase1_water_far_penalty_start_dist: float = 0.1
    phase1_water_far_penalty: float = -5.0
    phase1_water_far_reward_scale: float = 1.0
    phase2_cup_goal_max_dist: float = 0.2
    phase2_water_target_xy_std: float = 0.10
    phase2_water_target_xy_weight: float = 2.5
    phase2_water_target_z_std: float = 0.08
    phase2_water_target_z_weight: float = 1.5
    phase2_success_water_xy_tol: float = 0.07
    phase2_success_water_z_tol: float = 0.10

    fall_height: float = 0.02
    out_of_bound_x: tuple[float, float] = (0.40, 1.30)
    out_of_bound_y: tuple[float, float] = (-0.70, 0.60)
    out_of_bound_z: tuple[float, float] = (-0.1, 1.50)
    water_out_of_bound_x: tuple[float, float] = (0.40, 1.30)
    water_out_of_bound_y: tuple[float, float] = (-0.70, 0.60)
    water_out_of_bound_z: tuple[float, float] = (-0.1, 1.50)

    act_moving_average: float = 0.3
