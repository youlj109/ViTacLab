from isaaclab.utils import configclass

from ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env_cfg import (
    UR10eShadowHandBlindGraspEnvCfg,
    UR10eShadowHandBlindGraspSceneCfg,
)


@configclass
class UR10eShadowHandBlindGraspReplaySceneCfg(UR10eShadowHandBlindGraspSceneCfg):
    """Same TacSL scene as BlindGrasp; replay keyframes use a separate task id."""


@configclass
class UR10eShadowHandBlindGraspReplayEnvCfg(UR10eShadowHandBlindGraspEnvCfg):
    """BlindGrasp phase-3 replay: identical scene/physics, separate registry for data pipelines."""

    scene: UR10eShadowHandBlindGraspReplaySceneCfg = UR10eShadowHandBlindGraspReplaySceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )
