import gymnasium as gym

from ViTacLab.tasks.direct.medium_dexhand.bi_peg import agents as bi_peg_agents

from .bi_blind_inhand_env import UR10eDualShadowHandBiBlindInhandEnv
from .bi_blind_inhand_env_cfg import UR10eDualShadowHandBiBlindInhandEnvCfg

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-BiBlindInhand-Direct-v0",
    entry_point=f"{__name__}.bi_blind_inhand_env:UR10eDualShadowHandBiBlindInhandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bi_blind_inhand_env_cfg:UR10eDualShadowHandBiBlindInhandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{bi_peg_agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandBiPegPPORunnerCfg",
    },
)

__all__ = [
    "UR10eDualShadowHandBiBlindInhandEnvCfg",
    "UR10eDualShadowHandBiBlindInhandEnv",
]
