from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg


@configclass
class UR10eShadowHandTactileSceneCfg(InteractiveSceneCfg):
    """Scene configuration for UR10e + ShadowHand with TacSL tactile sensors.

    触觉传感器设置参考 `shadow_hand_env_cfg.ShadowHandSceneCfg`。
    可变形纸杯 (DeformableObject) 无 SDF/碰撞网格，故关闭 force_field，仅使用 camera_tactile。
    """

    tactile_sensor_ff = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_ffdistal/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=False,  # deformable cup has no SDF/collision mesh
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/cup",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_ffdistal/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )
    tactile_sensor_lf = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_lfdistal/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=False,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/cup",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_lfdistal/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )
    tactile_sensor_mf = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mfdistal/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=False,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/cup",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mfdistal/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )
    tactile_sensor_rf = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_rfdistal/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=False,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/cup",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_rfdistal/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )
    tactile_sensor_th = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_thdistal/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=False,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/cup",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_thdistal/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )


def _create_ur10e_shadowhand_left_cfg() -> ArticulationCfg:
    """Base configuration for UR10e + ShadowHand left-hand with tactile.

    配置逻辑参考 TacEx 中的 UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG，
    USD 路径改为当前 ViTacLab 工程要求路径：
    `source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_left_hand_glb_withtac.usd`
    """

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_left_hand_glb_withtac.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"],
                effort_limit_sim=150.0,
                stiffness=400.0,
                damping=80.0,
                friction=0.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class UR10eShadowHandPourEnvCfg(DirectRLEnvCfg):
    """Config for UR10e + ShadowHand deformable cup pouring task with tactile sensing."""

    # env meta
    decimation = 2
    episode_length_s = 10.0
    action_space = 32
    observation_space = 256
    state_space = 0
    asymmetric_obs = False

    # simulation
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

    # scene with tactile sensors
    scene: UR10eShadowHandTactileSceneCfg = UR10eShadowHandTactileSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # robot cfg
    robot_cfg: ArticulationCfg = _create_ur10e_shadowhand_left_cfg()

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

