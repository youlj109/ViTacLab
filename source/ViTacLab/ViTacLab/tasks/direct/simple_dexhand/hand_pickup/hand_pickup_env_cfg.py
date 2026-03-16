from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg


@configclass
class UR10eShadowHandPickupSceneCfg(InteractiveSceneCfg):
    """Scene configuration for UR10e + ShadowHand pickup task with TacSL tactile sensors."""

    # TacSL sensors on five fingertips; contact object is a rigid cube ('object')
    tactile_sensor_ff = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_ffdistal/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/object",
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
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/object",
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
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/object",
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
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/object",
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
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.005,
        contact_object_prim_path_expr="/World/envs/env_.*/object",
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
    """Base configuration for UR10e + ShadowHand left-hand with tactile."""

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
class UR10eShadowHandPickupEnvCfg(DirectRLEnvCfg):
    """Config for UR10e + ShadowHand pickup task."""

    # env meta
    decimation = 2
    episode_length_s = 6.0
    action_space = 32
    observation_space = 256
    state_space = 0
    asymmetric_obs = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        use_fabric=False,
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
    robot_cfg: ArticulationCfg = _create_ur10e_shadowhand_left_cfg()

    # joint name regex for action mapping
    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    # rigid object to lift: DexCube with SDF mesh (for TacSL force-field)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/DexCube/dex_cube_sdf.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 放在机械臂前方稍远处，避免一开始就贴太近
            pos=(2, 0.5, 0.05),
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

    # reward / termination parameters
    goal_height: float = 0.25
    goal_radius_xy: float = 0.05
    success_tolerance_height: float = 0.02
    success_tolerance_xy: float = 0.03
    success_bonus: float = 5.0
    lift_reward_scale: float = 5.0
    action_penalty_scale: float = -0.001
    fall_height: float = 0.02

