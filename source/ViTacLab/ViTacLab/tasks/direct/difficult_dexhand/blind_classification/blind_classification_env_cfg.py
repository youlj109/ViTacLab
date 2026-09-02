import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env_cfg import (
    GARBAGE_CAN_USD_PATH,
    UR10eShadowHandBlindGraspEnvCfg,
    UR10eShadowHandBlindGraspSceneCfg,
)


def _common_object_spawn_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        enable_gyroscopic_forces=False,
    )


@configclass
class UR10eShadowHandBlindClassificationSceneCfg(UR10eShadowHandBlindGraspSceneCfg):
    """TacSL contacts any of ``object_cube`` / ``object_sphere`` / ``object_cone``."""

    @classmethod
    def _tactile_params(cls) -> dict:
        p = super()._tactile_params()
        p["contact_object_prim_path_expr"] = "/World/envs/env_.*/object_(cube|sphere|cone)"
        return p


@configclass
class UR10eShadowHandBlindClassificationEnvCfg(UR10eShadowHandBlindGraspEnvCfg):
    """Blind bin + random shape (cube / sphere / cone); policy must classify via actions (last logits)."""

    scene: UR10eShadowHandBlindClassificationSceneCfg = UR10eShadowHandBlindClassificationSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # 30 arm+hand + num_shape_classes logits (same ordering as actuated, then class head).
    num_shape_classes: int = 3
    action_space: int = 33

    # Observation: no goal position / error (blind classification); same tactile as pickup summary by default.
    observation_space: int = 142

    # Park inactive shapes (env frame); keep well below table / workspace.
    park_object_pos: tuple[float, float, float] = (0.88, -0.15, -8.0)

    # Classification reward (dense): +weight when argmax(logits) matches sampled shape.
    classification_reward_weight: float = 2.0
    # Penalties on robot+hand actions only (first 30 dims).
    action_l2_weight: float = -0.005
    action_rate_l2_weight: float = -0.005

    # Disable pickup-style position goal (unused; kept 0 for any legacy reads).
    pos_tracking_weight: float = 0.0
    success_weight: float = 0.0

    trash_can_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trash_can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GARBAGE_CAN_USD_PATH,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=-0.001,
            ),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.88, -0.15, 0.14),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object_cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/SdfCube/SdfCube.usd",
            rigid_props=_common_object_spawn_rigid_props(),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.88, -0.15, 0.11)),
    )

    object_sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.028,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            rigid_props=_common_object_spawn_rigid_props(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.88, -0.15, 0.11)),
    )

    object_cone_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_cone",
        spawn=sim_utils.ConeCfg(
            radius=0.03,
            height=0.055,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.75, 0.35)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            rigid_props=_common_object_spawn_rigid_props(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.88, -0.15, 0.11)),
    )

    object_reset_pos_x_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_y_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
