"""UR10e + ShadowHand Factory-style peg insertion / gear mesh / nut threading."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10E_SHADOWHAND_LEFT_CFG,
    UR10eShadowHandTacSLSceneCfg,
)
from ViTacLab.tasks.direct.simple_gripper.forge_tasks_cfg import ForgeGearMesh, ForgeNutThread, ForgePegInsert


@configclass
class ForgeDexhandObjectPosesCfg:
    """Articulation root poses in **environment frame** (m, wxyz). World pose = ``env_origin + pos``."""

    fixed_pos: tuple[float, float, float] = (0.8, 0.0, 0.12)
    fixed_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    held_pos: tuple[float, float, float] = (0.8, 0.0, 0.20)
    held_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # ``gear_mesh`` + ``add_flanking_gears``: if ``None``, reuse ``fixed_*``.
    small_gear_pos: tuple[float, float, float] | None = None
    small_gear_quat: tuple[float, float, float, float] | None = None
    large_gear_pos: tuple[float, float, float] | None = None
    large_gear_quat: tuple[float, float, float, float] | None = None


@configclass
class UR10eShadowHandForgeTactileSceneCfg(UR10eShadowHandTacSLSceneCfg):
    """TacSL contacts target the Factory held peg/nut/gear (same prim as Isaac Factory)."""

    @classmethod
    def _tactile_params(cls) -> dict:
        return {
            "contact_object_prim_path_expr": "/World/envs/env_.*/HeldAsset",
            "enable_force_field": True,
            "tactile_array_size": (20, 25),
            "tactile_margin": 0.005,
        }


@configclass
class UR10eShadowHandForgeEnvCfg(DirectRLEnvCfg):
    """Factory forge assets / rewards / randomization; simulation stack aligned with :class:`UR10eShadowHandPickupEnvCfg`."""

    decimation: int = 2
    action_space: int = 30
    observation_space: int = 256
    state_space: int = 0
    asymmetric_obs: bool = False

    episode_length_s: float = 10.0
    task_name: str = "peg_insert"

    task: ForgePegInsert | ForgeGearMesh | ForgeNutThread = ForgePegInsert()

    object_poses: ForgeDexhandObjectPosesCfg = ForgeDexhandObjectPosesCfg()
    # Observation reference point (env frame) for ``held_rel`` / ``wrist_rel``; tune with ``object_poses``.
    fixed_obs_ref_pos: tuple[float, float, float] = (0.55, 0.0, 0.12)
    # Std dev of Gaussian noise added to ``fixed_obs_ref_pos`` in observations (0 = off).
    fixed_obs_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Match ``hand_pickup`` (light PhysX; fewer substeps per policy step than old Factory-scale settings).
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        use_fabric=True,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
        ),
    )

    scene: UR10eShadowHandForgeTactileSceneCfg = UR10eShadowHandForgeTactileSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = UR10E_SHADOWHAND_LEFT_CFG.copy()

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    ee_body_name: str = "wrist_3_link"

    vel_obs_scale: float = 0.2
    use_full_tactile_obs: bool = False

    enable_cameras: bool = False

    # Grasp after reset: lerp toward this fraction of hand joint range (0=open, 1=closed).
    reset_grasp_hand_joint_lerp: float = 0.55

    act_moving_average: float = 0.3


@configclass
class UR10eShadowHandForgePegInsertEnvCfg(UR10eShadowHandForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0


@configclass
class UR10eShadowHandForgeGearMeshEnvCfg(UR10eShadowHandForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class UR10eShadowHandForgeNutThreadEnvCfg(UR10eShadowHandForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0
