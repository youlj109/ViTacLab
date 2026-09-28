import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10E_SHADOWHAND_LEFT_CFG,
    TacSLSensorPolicyCfg,
)
from ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg import (
    UR10eShadowHandPickupEnvCfg,
    UR10eShadowHandPickupSceneCfg,
)

GARBAGE_CAN_USD_PATH: str = (
    "source/ViTacLab/ViTacLab/assets/data/Objects/cosmos_assets/1_dump_trash/GarbageCan003/GarbageCan003.usd"
)

# BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES: tuple[tuple[float, float, float], ...] = (
#     (0.62, 0.0, 0.1),
#     (0.52, 0.0, 0.1),
#     (0.48, -0.08, 0.1),
# )
# 20260422：for second step, (0.52, 0.0, 0.1) failed too many times. so changing to (0.50, 0.0, 0.1).
BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES: tuple[tuple[float, float, float], ...] = (
    (0.62, 0.0, 0.1),
    (0.50, 0.0, 0.1),
    (0.48, -0.08, 0.1),
)

def _blind_grasp_robot_init_joint_pos() -> dict[str, float]:
    return {
        "shoulder_pan_joint": 0,
        "shoulder_lift_joint": -2,
        "elbow_joint": 1,
        "wrist_1_joint": 1,
        "wrist_2_joint": 1.5,
        "wrist_3_joint": 0.1536931628046836,
    }


@configclass
class UR10eShadowHandBlindGraspSceneCfg(UR10eShadowHandPickupSceneCfg):
    """Same TacSL setup as pickup; manipuland remains ``/World/envs/env_.*/object``."""

    @classmethod
    def _tactile_policy(cls) -> TacSLSensorPolicyCfg:
        """Level-3 blind bin: allow depth-based tactile during bin-wall exploration.

        Strict target-only attribution zeros NF when PhysX sparse anchors miss the cube
        (common during open-loop replay before finger-object contact). For ViTacBench
        demos we still attribute PhysX anchors to ``object``, but depth penetration from
        any contact (bin interior, cube) may produce force-field output.
        """
        return TacSLSensorPolicyCfg(
            contact_object_prim_path_expr="/World/envs/env_.*/object",
            enable_force_field=True,
            tactile_array_size=(20, 25),
            tactile_margin=0.005,
            use_physx_sparse_anchors=True,
            strict_target_contact_attribution=False,
            require_physx_sparse_anchors=False,
        )


@configclass
class UR10eShadowHandBlindGraspEnvCfg(UR10eShadowHandPickupEnvCfg):
    """Pickup task with a fixed garbage can; the cube spawns inside the bin."""

    episode_length_s = 12.0
    include_object_state_in_policy_obs = False
    use_full_tactile_obs = True
    policy_record_profile: str = "tactile_only"

    # Tactile-only learning signal. Object height is retained only as an
    # external benchmark metric and never enters policy observations/rewards.
    tactile_contact_threshold: float = 1.0e-4
    tactile_load_threshold: float = 5.0e-5
    tactile_contact_reward_weight: float = 0.5
    tactile_multi_finger_reward_weight: float = 1.0
    tactile_load_reward_weight: float = 2.0
    tactile_slip_penalty_weight: float = -0.25
    tactile_success_min_fingers: int = 2
    tactile_success_hold_steps: int = 20
    benchmark_lift_height: float = 0.35

    scene: UR10eShadowHandBlindGraspSceneCfg = UR10eShadowHandBlindGraspSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # BlindGrasp-only robot init: lift arm slightly higher for safe exploration above the opaque bin.
    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.copy().replace(
        init_state=UR10E_SHADOWHAND_LEFT_CFG.init_state.replace(joint_pos=_blind_grasp_robot_init_joint_pos())
    )

    object_init_pos_candidates: tuple[tuple[float, float, float], ...] = BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES
    # -1: sample slot A/B each reset (bench). 0/1: pin that branch for collection.
    object_init_choice: int = -1
    object_slot_count: int = 2

    # XY keeps slots A/B inside the opening. Z=1.4 put the rim at ~0.39 m, which
    # collides with recorded Pick wrists (~0.425 m). Z=1.0 + origin z=0.155 keeps
    # the floor near z=0.02 (cube ~0.05) and the rim near z=0.29. XY=2.6 keeps
    # joint-space Pick01_01→02 from hitting the +X inner wall (scale 1.7 wall at
    # ~0.72 m; interpolated fingertips swing to ~0.77 m).
    trash_can_scale: tuple[float, float, float] = (2.6, 2.6, 1.0)

    # Kinematic bin: collision on, no dynamics (does not tip or slide).
    trash_can_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trash_can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GARBAGE_CAN_USD_PATH,
            scale=(2.6, 2.6, 1.0),
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
            pos=(0.54, 0.0, 0.155),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # In-bin cube: light enough for the low-effort Shadow pads, grippier than the
    # inherited pickup default (mass unset / friction 1.0). Advisor enclosure is
    # thumb vs four fingers; gel pads have little friction without this.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        use_fabric=True,
        physics_material=RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=2.5),
        physx=PhysxCfg(enable_ccd=True, bounce_threshold_velocity=0.2),
    )

    # Same SdfCube as hand_pickup; initial pose sits above the can floor (env frame).
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/SdfCube/SdfCube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.88, -0.15, 0.11),
        ),
    )

    @staticmethod
    def resolve_object_init_pos(choice: int) -> tuple[float, float, float]:
        n = min(2, len(BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES))
        idx = 0 if int(choice) < 0 else max(0, min(int(choice), n - 1))
        return tuple(float(v) for v in BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES[idx])

    # Tighter horizontal jitter so random poses stay inside the opening.
    object_reset_pos_x_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_y_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
