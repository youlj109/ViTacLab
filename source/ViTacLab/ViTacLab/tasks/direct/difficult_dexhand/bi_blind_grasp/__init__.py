import gymnasium as gym

from ViTacLab.tasks.direct.medium_dexhand.bi_peg import agents as bi_peg_agents

from .bi_blind_grasp_env import UR10eDualShadowHandBiBlindGraspEnv
from .bi_blind_grasp_env_cfg import UR10eDualShadowHandBiBlindGraspEnvCfg

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-BiBlindGrasp-Direct-v0",
    entry_point=f"{__name__}.bi_blind_grasp_env:UR10eDualShadowHandBiBlindGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bi_blind_grasp_env_cfg:UR10eDualShadowHandBiBlindGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{bi_peg_agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandBiPegPPORunnerCfg",
    },
)

__all__ = [
    "UR10eDualShadowHandBiBlindGraspEnvCfg",
    "UR10eDualShadowHandBiBlindGraspEnv",
]
