import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from ViTacLab.assets.sensor.tacsl_sensor import VisuoTactileSensorV2Cfg as VisuoTactileSensorCfg
#from ViTacLab.assets.sensor.tacsl_sensor import VisuoTactileSensorCfg

# Canonical ordered scene keys and matching robot finger suffixes. Task
# environments, collectors, and policy record adapters must preserve this
# order so tactile tensors have stable sensor semantics across workflows.
UR10E_SHADOWHAND_TACTILE_SENSOR_SPECS: tuple[tuple[str, str], ...] = (
    ("tactile_sensor_ff", "ff"),
    ("tactile_sensor_lf", "lf"),
    ("tactile_sensor_mf", "mf"),
    ("tactile_sensor_rf", "rf"),
    ("tactile_sensor_th", "th"),
)
UR10E_SHADOWHAND_TACTILE_SENSOR_NAMES: tuple[str, ...] = tuple(
    name for name, _finger in UR10E_SHADOWHAND_TACTILE_SENSOR_SPECS
)

UR10E_SHADOWHAND_LEFT_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_left_hand_glb_withtac_v2_no_gelsight_articulation.usd",
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
            # UR10e is a bolted-down tabletop manipulator. Keep this explicit
            # even if a particular USD already authors a fixed root.
            fix_root_link=True,
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
            effort_limit_sim=10.0,
            # Keep hand posture stable while the UR10e arm moves.
            # Previous values (stiffness=3.0, damping=0.1) were too soft and fingers drifted.
            stiffness=80.0,
            damping=10.0,
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
            pos=(0.3899, -1.6833, 1.0833),
            rot=(0.8403, 0.5420, 0.00063, 0.00098),
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
class TacSLSensorPolicyCfg:
    """Unified TacSL policy for UR10e-ShadowHand scene families.

    Task scenes should override this policy (or selected fields) instead of rebuilding
    sensor setup logic per-task.
    """

    contact_object_prim_path_expr: str = "/World/envs/env_.*/object"
    enable_force_field: bool = True
    tactile_array_size: tuple[int, int] = (20, 25)
    tactile_margin: float = 0.005
    contact_object_is_deformable: bool = False
    depth_penetration_deadband: float = 0.0
    use_physx_sparse_anchors: bool = True
    strict_target_contact_attribution: bool = True
    require_physx_sparse_anchors: bool = False


@configclass
class UR10eShadowHandTacSLSceneCfg(UR10eShadowHandBaseSceneCfg):
    """UR10e + ShadowHand scene with 5 TacSL GelSight sensors (ff/lf/mf/rf/th)."""

    @classmethod
    def _tactile_policy(cls) -> TacSLSensorPolicyCfg:
        """Unified TacSL policy provider.

        Keep as classmethod (not dataclass field) so InteractiveScene does not interpret it
        as a scene asset config entry.
        """
        return TacSLSensorPolicyCfg()

    @classmethod
    def _tactile_params(cls) -> dict:
        # Keep this compatibility method so older task configs can still override it.
        # New task configs should override `_tactile_policy()` on the scene cfg.
        p = cls._tactile_policy()
        return {
            "contact_object_prim_path_expr": p.contact_object_prim_path_expr,
            "enable_force_field": p.enable_force_field,
            "tactile_array_size": p.tactile_array_size,
            "tactile_margin": p.tactile_margin,
            "contact_object_is_deformable": p.contact_object_is_deformable,
            "depth_penetration_deadband": p.depth_penetration_deadband,
            "use_physx_sparse_anchors": p.use_physx_sparse_anchors,
            "strict_target_contact_attribution": p.strict_target_contact_attribution,
            "require_physx_sparse_anchors": p.require_physx_sparse_anchors,
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
            use_physx_sparse_anchors=tp.get("use_physx_sparse_anchors", True),
            require_physx_sparse_anchors=tp.get("require_physx_sparse_anchors", False),
            strict_target_contact_attribution=tp.get("strict_target_contact_attribution", True),
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

    return {name: _mk(finger) for name, finger in UR10E_SHADOWHAND_TACTILE_SENSOR_SPECS}

