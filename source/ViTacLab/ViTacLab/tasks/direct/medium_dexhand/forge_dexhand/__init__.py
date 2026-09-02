import gymnasium as gym

from ...simple_dexhand.shadow_hand import agents as shadow_hand_agents

from .ur10e_shadowhand_forge_env_cfg import (
    ForgeDexhandObjectPosesCfg,
    UR10eShadowHandForgeGearMeshEnvCfg,
    UR10eShadowHandForgeNutThreadEnvCfg,
    UR10eShadowHandForgePegInsertEnvCfg,
)
from .ur10e_shadowhand_forge_env import UR10eShadowHandForgeEnv

##
# Register Gym environments — Factory peg / gear / nut with UR10e + Shadow Hand (joint-space control).
##

gym.register(
    id="Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0",
    entry_point=f"{__name__}.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgePegInsertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-UR10eShadowHand-ForgeGearMesh-Direct-v0",
    entry_point=f"{__name__}.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeGearMeshEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-UR10eShadowHand-ForgeNutThread-Direct-v0",
    entry_point=f"{__name__}.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeNutThreadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
    },
)

__all__ = [
    "ForgeDexhandObjectPosesCfg",
    "UR10eShadowHandForgeEnv",
    "UR10eShadowHandForgePegInsertEnvCfg",
    "UR10eShadowHandForgeGearMeshEnvCfg",
    "UR10eShadowHandForgeNutThreadEnvCfg",
]
