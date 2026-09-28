import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    TacSLSensorPolicyCfg,
)
from ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env_cfg import (
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
class UR10eShadowHandBlindRetrievalSceneCfg(UR10eShadowHandBlindGraspSceneCfg):
    """TacSL contacts any of ``object_cube`` / ``object_sphere`` / ``object_cone``."""

    @classmethod
    def _tactile_policy(cls) -> TacSLSensorPolicyCfg:
        p = UR10eShadowHandBlindGraspSceneCfg._tactile_policy()
        return TacSLSensorPolicyCfg(
            contact_object_prim_path_expr="/World/envs/env_.*/object_(cube|sphere|cone)",
            enable_force_field=p.enable_force_field,
            tactile_array_size=p.tactile_array_size,
            tactile_margin=p.tactile_margin,
            contact_object_is_deformable=p.contact_object_is_deformable,
            depth_penetration_deadband=p.depth_penetration_deadband,
            use_physx_sparse_anchors=p.use_physx_sparse_anchors,
            strict_target_contact_attribution=False,
            require_physx_sparse_anchors=False,
        )


@configclass
class UR10eShadowHandBlindRetrievalEnvCfg(UR10eShadowHandBlindGraspEnvCfg):
    """Bin with cube + sphere + cone; each episode samples a target class to bring to the goal."""

    scene: UR10eShadowHandBlindRetrievalSceneCfg = UR10eShadowHandBlindRetrievalSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    num_shape_classes: int = 3
    # Same as pickup (arm + hand only).
    action_space: int = 30
    # pickup_obs(133) + target one-hot(3) = 136 + tactile; summary tactile => 151.
    observation_space: int = 151

    # Three slot positions in the bin (env frame, meters). Each reset, cube/sphere/cone are
    # randomly assigned to these slots (per env), so identities permute across locations.
    object_layout_offsets: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (-0.042, -0.035, 0.0),  # cube: back-left
        (0.042, -0.035, 0.0),  # sphere: back-right
        (0.0, 0.048, 0.0),  # cone: forward (opens away from the two backs)
    )
    layout_pos_jitter: tuple[float, float] = (-0.004, 0.004)

    # Inherit trash_can_cfg / trash_can_scale from BlindGrasp (can at ~0.54, 0, 0.20) so
    # teleop keyframes reach inside the bin instead of groping outside at (0.88, -0.15).

    object_cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/SdfCube/SdfCube.usd",
            rigid_props=_common_object_spawn_rigid_props(),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.54, 0.0, 0.11)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.54, 0.0, 0.11)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.54, 0.0, 0.11)),
    )

    object_reset_pos_x_range: tuple[float, float] = (-0.012, 0.012)
    object_reset_pos_y_range: tuple[float, float] = (-0.012, 0.012)
    object_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
