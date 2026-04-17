import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg import (
    UR10eShadowHandPickupEnvCfg,
    UR10eShadowHandPickupSceneCfg,
)

GARBAGE_CAN_USD_PATH: str = (
    "source/ViTacLab/ViTacLab/assets/data/Objects/cosmos_assets/1_dump_trash/GarbageCan003/GarbageCan003.usd"
)


@configclass
class UR10eShadowHandBlindGraspSceneCfg(UR10eShadowHandPickupSceneCfg):
    """Same TacSL setup as pickup; manipuland remains ``/World/envs/env_.*/object``."""


@configclass
class UR10eShadowHandBlindGraspEnvCfg(UR10eShadowHandPickupEnvCfg):
    """Pickup task with a fixed garbage can; the cube spawns inside the bin."""

    scene: UR10eShadowHandBlindGraspSceneCfg = UR10eShadowHandBlindGraspSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # Scale applied in env when spawning (tune if the Cosmos asset is not in meters).
    trash_can_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Kinematic bin: collision on, no dynamics (does not tip or slide).
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

    # Tighter horizontal jitter so random poses stay inside the opening.
    object_reset_pos_x_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_y_range: tuple[float, float] = (-0.025, 0.025)
    object_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
