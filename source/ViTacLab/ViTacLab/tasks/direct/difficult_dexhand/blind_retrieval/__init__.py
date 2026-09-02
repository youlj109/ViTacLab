import gymnasium as gym

from ViTacLab.tasks.direct.simple_dexhand.shadow_hand import agents as shadow_hand_agents

from .blind_retrieval_env import UR10eShadowHandBlindRetrievalEnv
from .blind_retrieval_env_cfg import (
    UR10eShadowHandBlindRetrievalEnvCfg,
    UR10eShadowHandBlindRetrievalSceneCfg,
)

##
# Register Gym environment.
##

gym.register(
    id="Isaac-UR10eShadowHand-BlindRetrieval-Direct-v0",
    entry_point=f"{__name__}.blind_retrieval_env:UR10eShadowHandBlindRetrievalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blind_retrieval_env_cfg:UR10eShadowHandBlindRetrievalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactilePPORunnerCfg",
    },
)

__all__ = [
    "UR10eShadowHandBlindRetrievalEnvCfg",
    "UR10eShadowHandBlindRetrievalSceneCfg",
    "UR10eShadowHandBlindRetrievalEnv",
]
