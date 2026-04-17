import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorV2Cfg as VisuoTactileSensorCfg


UR10E_SHADOWHAND_LEFT_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_left_hand_glb_withtac_v2_no_gelsight_articulation.usd",
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
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=-0.002),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # fixed offset w.r.t. Forge/Factory table placement
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        # UR10e arm default pose (rad); matches video teleop IK “elbow up” reference
        joint_pos={
            "shoulder_pan_joint": 0.21119636416180324,
            "shoulder_lift_joint": -1.2847641999317674,
            "elbow_joint": 1.9852784513664496,
            "wrist_1_joint": -0.28662118970668776,
            "wrist_2_joint": 2.019309045922398,
            "wrist_3_joint": 0.1536931628046836,
        },
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
class UR10eShadowHandBaseSceneCfg(InteractiveSceneCfg):
    """Common scene pieces for UR10e + ShadowHand tasks.

    Includes a third-person camera (one per env).
    """


def build_ur10e_shadowhand_third_person_camera_cfg() -> TiledCameraCfg:
    """Build third-person tiled camera cfg.

    IMPORTANT: Do not keep this as a scene cfg field. Same reason as TacSL sensors:
    `InteractiveScene(cfg.scene)` happens before `clone_environments()`.
    """
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/ThirdPersonCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 1.0),
            rot=(0.64086, 0.29884, 0.29884, 0.64086),
            convention="None",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=480,
        height=640,
    )


@configclass
class UR10eShadowHandTacSLSceneCfg(UR10eShadowHandBaseSceneCfg):
    """UR10e + ShadowHand scene with 5 TacSL GelSight sensors (ff/lf/mf/rf/th)."""

    @classmethod
    def _tactile_params(cls) -> dict:
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/object",
            "enable_force_field": True,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
        }

def build_ur10e_shadowhand_tactile_sensor_cfgs(scene_cfg: UR10eShadowHandTacSLSceneCfg) -> dict[str, VisuoTactileSensorCfg]:
    """Build TacSL sensor cfgs.

    IMPORTANT: We intentionally DO NOT put these as fields on the scene cfg.
    `DirectRLEnv` constructs `InteractiveScene(cfg.scene)` *before* calling the env's `_setup_scene()`,
    so sensors defined in the scene cfg would initialize *before* `clone_environments()` and only see `env_0`.
    That causes `_num_envs=1` inside sensors and CUDA indexing asserts when resetting multiple envs.

    Instead, we build these cfgs and let the base env create/register sensors after cloning.
    """
    tp = type(scene_cfg)._tactile_params()

    def _mk(finger: str) -> VisuoTactileSensorCfg:
        return VisuoTactileSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/gelsight_{finger}distal/elastomer/tactile_sensor",
            history_length=0,
            debug_vis=False,
            render_cfg=GELSIGHT_R15_CFG,
            enable_camera_tactile=True,
            enable_force_field=tp["enable_force_field"],
            tactile_array_size=tp["tactile_array_size"],
            tactile_margin=tp["tactile_margin"],
            contact_object_prim_path_expr=tp["contact_object_prim_path_expr"],
            contact_object_is_deformable=tp.get("contact_object_is_deformable", False),
            depth_penetration_deadband=tp.get("depth_penetration_deadband", 0.0),
            normal_contact_stiffness=1.0,
            friction_coefficient=2.0,
            tangential_stiffness=0.1,
            camera_cfg=TiledCameraCfg(
                prim_path=f"/World/envs/env_.*/Robot/gelsight_{finger}distal/elastomer_tip/cam",
                height=GELSIGHT_R15_CFG.image_height,
                width=GELSIGHT_R15_CFG.image_width,
                data_types=["distance_to_image_plane"],
                spawn=None,
            ),
            trimesh_vis_tactile_points=False,
            visualize_sdf_closest_pts=False,
        )

    return {
        "tactile_sensor_ff": _mk("ff"),
        "tactile_sensor_lf": _mk("lf"),
        "tactile_sensor_mf": _mk("mf"),
        "tactile_sensor_rf": _mk("rf"),
        "tactile_sensor_th": _mk("th"),
    }

