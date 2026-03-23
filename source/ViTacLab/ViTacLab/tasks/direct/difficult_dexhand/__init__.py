import gymnasium as gym

from ..simple_dexhand.shadow_hand import agents as shadow_hand_agents

from .ur10e_shadowhand_pour_env_cfg import (
    UR10eShadowHandPourEnvCfg,
    UR10eShadowHandTactileSceneCfg,
)
from .ur10e_shadowhand_pour_env import UR10eShadowHandPourEnv

##
# Register Gym environments for difficult dexterous tasks.
##


gym.register(
    id="Isaac-UR10eShadowHand-PourDeformable-Direct-v0",
    entry_point=f"{__name__}.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactilePPORunnerCfg",
    },
)

__all__ = [
    "UR10eShadowHandPourEnvCfg",
    "UR10eShadowHandTactileSceneCfg",
    "UR10eShadowHandPourEnv",
]
