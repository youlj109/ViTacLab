import math

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

    触觉传感器设置参考 `shadow_hand_env_cfg.ShadowHandSceneCfg`。
    可变形纸杯 (DeformableObject) 无 SDF/碰撞网格，故关闭 force_field，仅使用 camera_tactile。
    """
    @classmethod
    def _tactile_params(cls) -> dict:
        # Deformable cup has no SDF/collision mesh hierarchy expected by force-field.
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/cup",
            "enable_force_field": False,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
        }


@configclass
class UR10eShadowHandPourEnvCfg(DirectRLEnvCfg):
    """Config for UR10e + ShadowHand deformable cup pouring task with tactile sensing."""

    # env meta
    decimation = 2
    episode_length_s = 10.0
    action_space = 30
    observation_space = 256
    state_space = 0
    asymmetric_obs = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        # Deformables currently do not replicate reliably with Fabric cloning.
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
    scene: UR10eShadowHandTactileSceneCfg = UR10eShadowHandTactileSceneCfg(
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
    # deformable paper cup
    cup_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cup",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.06, 0.06, 0.10),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.002,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=0.4,
                youngs_modulus=5e4,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.8, -0.15, 0.15),
        ),
        debug_vis=False,
    )

    # rigid target cup / bowl
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.06,
            height=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.5, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.9, 0.15, 0.10),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # desired pouring pose (cup over target, tilted)
    goal_cup_pos = (0.9, 0.15, 0.25)
    goal_cup_rot = (0.7071, 0.0, 0.7071, 0.0)

    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_cup_marker",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=0.065,
                height=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            ),
        },
    )

    # reset noise
    reset_cup_pos_noise = 0.02
    reset_robot_dof_pos_noise = 0.2
    reset_robot_dof_vel_noise = 0.0

    # reward & termination params
    cup_pos_reward_scale = -10.0
    cup_rot_reward_scale = 2.0
    action_penalty_scale = -0.001
    success_tolerance_pos = 0.05
    success_tolerance_rot = 0.25
    success_bonus = 250.0
    fall_height = 0.02
    fall_penalty = -100.0
    vel_obs_scale = 0.2
    # action smoothing (same meaning as ShadowHandEnvCfg / InHandManipulationEnv)
    act_moving_average: float = 0.3

