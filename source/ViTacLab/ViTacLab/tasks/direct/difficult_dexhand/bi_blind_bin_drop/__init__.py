import gymnasium as gym

from ViTacLab.tasks.direct.medium_dexhand.bi_peg import agents as bi_peg_agents

from .bi_blind_bin_drop_env import UR10eDualShadowHandBiBlindBinDropEnv
from .bi_blind_bin_drop_env_cfg import UR10eDualShadowHandBiBlindBinDropEnvCfg

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-BiBlindBinDrop-Direct-v0",
    entry_point=f"{__name__}.bi_blind_bin_drop_env:UR10eDualShadowHandBiBlindBinDropEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bi_blind_bin_drop_env_cfg:UR10eDualShadowHandBiBlindBinDropEnvCfg",
        "rsl_rl_cfg_entry_point": f"{bi_peg_agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandBiPegPPORunnerCfg",
    },
)

__all__ = [
    "UR10eDualShadowHandBiBlindBinDropEnvCfg",
    "UR10eDualShadowHandBiBlindBinDropEnv",
]
