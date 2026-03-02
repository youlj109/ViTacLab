# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab_assets.sensors import GELSIGHT_MINI_CFG, GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask

# Tactile sensor: normal force (20*25), shear force (20*25*2), RGB image (240*320*3)
# Two sensors: left and right
# RGB image: 240 * 320 * 3 = 230400 per sensor, total = 460800 for two sensors
OBS_DIM_CFG.update({
    "force_threshold": 1,
    "tactile_normal_force": 1000,  # 2 sensors * 500 points
    "tactile_shear_force": 2000,  # 2 sensors * 500 points * 2 dims
    "tactile_rgb_image": 460800,  # 2 sensors * 240 * 320 * 3
})

STATE_DIM_CFG.update({
    "force_threshold": 1,
    "tactile_normal_force": 1000,
    "tactile_shear_force": 2000,
    "tactile_rgb_image": 460800,
})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone,
        mode="interval",
        interval_range_s=(2.0, 2.0),  # (0.25, 0.25)
    )

class ForgeSceneCfg(InteractiveSceneCfg):
    """Scene configuration with tactile sensors."""
    
    # TacSL Tactile Sensor - Left
    tactile_sensor_left = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/R15_leftfinger/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,  # Disabled to reduce memory usage
        # Sensor configuration
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        # Elastomer configuration
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        # Contact object configuration
        contact_object_prim_path_expr="/World/envs/env_.*/HeldAsset",
        # Force field physics parameters
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        # Camera configuration
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/R15_leftfinger/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        # Debug Visualization
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )

    # TacSL Tactile Sensor - Right
    tactile_sensor_right = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/R15_rightfinger/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,  # Disabled to reduce memory usage
        # Sensor configuration
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        # Elastomer configuration
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        # Contact object configuration
        contact_object_prim_path_expr="/World/envs/env_.*/HeldAsset",
        # Force field physics parameters
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        # Camera configuration
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/R15_rightfinger/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        # Debug Visualization
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )

    # Third-person view camera: one per env (num_envs), fixed in world frame
    third_person_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/ThirdPersonCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 0.8),
            rot=(0.9239, 0.0, 0.3827, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=256,
        height=256,
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    action_space: int = 7
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()
    scene: ForgeSceneCfg = ForgeSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
    
    # Observation mode: "reduce" (without tactile) or "full" (with tactile)
    obs_mode: str = "reduce"  # Options: "reduce", "full"

    # Visual disturbance: when True, apply Gaussian noise or Gaussian blur to third_person_camera images
    visual_disturbance: bool = False
    visual_disturbance_type: str = "gaussian_noise"  # "gaussian_noise" or "gaussian_blur"
    # For gaussian_noise: std of additive noise (on [0,1] normalized image)
    visual_disturbance_noise_std: float = 0.08
    # For gaussian_blur: kernel size (odd) and sigma
    visual_disturbance_blur_kernel_size: int = 5
    visual_disturbance_blur_sigma: float = 1.0

    # Object randomization: when True, held (and optionally fixed) object USDs are randomly chosen from folders
    randomize_objects: bool = False
    # Folder path(s) for USD files. random_objects_usd_dir → held object; random_fixed_objects_usd_dir → fixed object.
    random_objects_usd_dir: str = ""  # e.g. "source/ViTacLab/ViTacLab/assets/data/Objects/held"
    random_fixed_objects_usd_dir: str = ""  # e.g. "source/ViTacLab/ViTacLab/assets/data/Objects/fixed"
    # Default USD paths when randomize_objects is False (optional; used if scene has held_asset/fixed_asset).
    default_held_asset_usd_path: str = ""
    default_fixed_asset_usd_path: str = ""

    # Override robot USD file path
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Robots/Franka/Franka_R15/franka_mimic_edit.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "tactile_normal_force",
        "tactile_shear_force",
        "tactile_rgb_image",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "tactile_normal_force",
        "tactile_shear_force",
        "tactile_rgb_image",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0
