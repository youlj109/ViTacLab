import gymnasium as gym

from ..shadow_hand import agents as shadow_hand_agents

from .hand_pickup_env_cfg import (
    UR10eShadowHandPickupEnvCfg,
    UR10eShadowHandPickupSceneCfg,
)
from .hand_pickup_env import UR10eShadowHandPickupEnv

##
# Register Gym environment.
##

gym.register(
    id="Isaac-UR10eShadowHand-Pickup-Direct-v0",
    entry_point=f"{__name__}.hand_pickup_env:UR10eShadowHandPickupEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactilePPORunnerCfg",
    },
)

__all__ = [
    "UR10eShadowHandPickupEnvCfg",
    "UR10eShadowHandPickupSceneCfg",
    "UR10eShadowHandPickupEnv",
]

