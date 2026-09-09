import gymnasium as gym

from ViTacLab.tasks.direct.simple_dexhand.shadow_hand import agents as shadow_hand_agents

from .blind_grasp_replay_env import UR10eShadowHandBlindGraspReplayEnv
from .blind_grasp_replay_env_cfg import (
    UR10eShadowHandBlindGraspReplayEnvCfg,
    UR10eShadowHandBlindGraspReplaySceneCfg,
)

##
# Register Gym environment.
##

gym.register(
    id="Isaac-UR10eShadowHand-BlindGraspReplay-Direct-v0",
    entry_point=f"{__name__}.blind_grasp_replay_env:UR10eShadowHandBlindGraspReplayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blind_grasp_replay_env_cfg:UR10eShadowHandBlindGraspReplayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactilePPORunnerCfg",
    },
)

__all__ = [
    "UR10eShadowHandBlindGraspReplayEnvCfg",
    "UR10eShadowHandBlindGraspReplaySceneCfg",
    "UR10eShadowHandBlindGraspReplayEnv",
]
