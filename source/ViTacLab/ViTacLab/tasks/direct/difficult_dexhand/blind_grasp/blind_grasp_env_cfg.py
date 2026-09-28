import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10E_SHADOWHAND_LEFT_CFG,
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


@configclass
class UR10eShadowHandBlindGraspEnvCfg(UR10eShadowHandPickupEnvCfg):
    """Pickup task with a fixed garbage can; the cube spawns inside the bin."""

    episode_length_s = 12.0

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
    object_init_choice: int = 0

    # Scale applied in env when spawning (tune if the Cosmos asset is not in meters).
    trash_can_scale: tuple[float, float, float] = (1.7, 1.7, 1.4)

    # Kinematic bin: collision on, no dynamics (does not tip or slide).
    trash_can_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trash_can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GARBAGE_CAN_USD_PATH,
            scale=(1.7, 1.7, 1.4),
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
            pos=(0.54, 0.0, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
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
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.88, -0.15, 0.11),
        ),
    )

    @staticmethod
    def resolve_object_init_pos(choice: int) -> tuple[float, float, float]:
        idx = max(0, min(int(choice), len(BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES) - 1))
        return tuple(float(v) for v in BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES[idx])

    # Tighter horizontal jitter so random poses stay inside the opening.
    object_reset_pos_x_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_y_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
