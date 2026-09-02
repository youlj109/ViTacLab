import gymnasium as gym

from ViTacLab.tasks.direct.simple_dexhand.shadow_hand import agents as shadow_hand_agents

from .blind_grasp_env import UR10eShadowHandBlindGraspEnv
from .blind_grasp_env_cfg import (
    UR10eShadowHandBlindGraspEnvCfg,
    UR10eShadowHandBlindGraspSceneCfg,
)

##
# Register Gym environment.
##

gym.register(
    id="Isaac-UR10eShadowHand-BlindGrasp-Direct-v0",
    entry_point=f"{__name__}.blind_grasp_env:UR10eShadowHandBlindGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blind_grasp_env_cfg:UR10eShadowHandBlindGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactilePPORunnerCfg",
    },
)

__all__ = [
    "UR10eShadowHandBlindGraspEnvCfg",
    "UR10eShadowHandBlindGraspSceneCfg",
    "UR10eShadowHandBlindGraspEnv",
]
