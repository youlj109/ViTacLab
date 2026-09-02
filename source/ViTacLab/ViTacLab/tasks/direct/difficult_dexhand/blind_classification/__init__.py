import gymnasium as gym

from ViTacLab.tasks.direct.simple_dexhand.shadow_hand import agents as shadow_hand_agents

from .blind_classification_env import UR10eShadowHandBlindClassificationEnv
from .blind_classification_env_cfg import (
    UR10eShadowHandBlindClassificationEnvCfg,
    UR10eShadowHandBlindClassificationSceneCfg,
)

##
# Register Gym environment.
##

gym.register(
    id="Isaac-UR10eShadowHand-BlindClassification-Direct-v0",
    entry_point=f"{__name__}.blind_classification_env:UR10eShadowHandBlindClassificationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blind_classification_env_cfg:UR10eShadowHandBlindClassificationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactilePPORunnerCfg",
    },
)

__all__ = [
    "UR10eShadowHandBlindClassificationEnvCfg",
    "UR10eShadowHandBlindClassificationSceneCfg",
    "UR10eShadowHandBlindClassificationEnv",
]
