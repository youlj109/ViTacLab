# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene / robot / TacSL config for GelSight R15 short-finger USD pretraining tasks."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorV2Cfg as VisuoTactileSensorCfg

GELSIGHT_FINGER_SHORT_USD: str = (
    "source/ViTacLab/ViTacLab/assets/data/Sensors/Gelsight_finger/gelsight_r15_finger_short_v2.usd"
)


@configclass
class GelsightFingerPretrainSceneCfg(InteractiveSceneCfg):
    """Minimal scene metadata; assets are spawned in :class:`GelsightFingerPretrainBaseEnv`."""

    replicate_physics: bool = True
    clone_in_fabric: bool = False

    @classmethod
    def tactile_params(cls) -> dict:
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/contact_object",
            "enable_force_field": True,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
        }


def build_gelsight_finger_robot_cfg() -> ArticulationCfg:
    """GelSight short finger USD as a single articulation (same pattern as ``tacsl_sensor_gelsight_finger_short``)."""

    spawn = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=GELSIGHT_FINGER_SHORT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        compliant_contact_stiffness=None,
        compliant_contact_damping=None,
        physics_material_prim_path="elastomer",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.003),
    )
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            rot=(0.70711, -0.70711, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )


def build_gelsight_finger_tacsl_sensor_cfg(scene_cfg: GelsightFingerPretrainSceneCfg) -> VisuoTactileSensorCfg:
    """Single VisuoTactile sensor on the short-finger elastomer."""

    tp = type(scene_cfg).tactile_params()
    return VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=tp["enable_force_field"],
        tactile_array_size=tp["tactile_array_size"],
        tactile_margin=tp["tactile_margin"],
        contact_object_prim_path_expr=tp["contact_object_prim_path_expr"],
        contact_object_is_deformable=False,
        depth_penetration_deadband=0.0,
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )


def format_tacsl_cfg_paths(sensor_cfg: VisuoTactileSensorCfg, env_regex_ns: str) -> None:
    """Fill ``ENV_REGEX_NS`` on TacSL + camera paths when placeholders are present."""

    if "{ENV_REGEX_NS}" in sensor_cfg.prim_path:
        sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
    if sensor_cfg.camera_cfg is not None and "{ENV_REGEX_NS}" in sensor_cfg.camera_cfg.prim_path:
        sensor_cfg.camera_cfg.prim_path = sensor_cfg.camera_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
    if sensor_cfg.contact_object_prim_path_expr is not None and "{ENV_REGEX_NS}" in sensor_cfg.contact_object_prim_path_expr:
        sensor_cfg.contact_object_prim_path_expr = sensor_cfg.contact_object_prim_path_expr.format(
            ENV_REGEX_NS=env_regex_ns
        )
