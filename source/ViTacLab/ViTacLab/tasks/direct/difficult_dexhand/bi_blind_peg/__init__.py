# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi blind peg: dual-arm peg insert like ``medium_dexhand.bi_peg``, TacSL without third-person camera."""

import gymnasium as gym

from ViTacLab.tasks.direct.medium_dexhand.bi_peg import agents as bi_peg_agents

from .bi_blind_peg_env_cfg import UR10eDualShadowHandBiBlindPegEnvCfg

_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.bi_blind_peg_env_cfg:UR10eDualShadowHandBiBlindPegEnvCfg",
    "rl_games_cfg_entry_point": f"{bi_peg_agents.__name__}:rl_games_ppo_cfg.yaml",
    "rsl_rl_cfg_entry_point": f"{bi_peg_agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandBiPegPPORunnerCfg",
    "skrl_cfg_entry_point": f"{bi_peg_agents.__name__}:skrl_ppo_cfg.yaml",
    "skrl_ippo_cfg_entry_point": f"{bi_peg_agents.__name__}:skrl_ippo_cfg.yaml",
    "skrl_mappo_cfg_entry_point": f"{bi_peg_agents.__name__}:skrl_mappo_cfg.yaml",
}
_ENTRY = "ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env:UR10eDualShadowHandBiPegEnv"

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-BiBlindPeg-Direct-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs=_KWARGS,
)
# Deprecated alias (former package name ``bi_peg_no_third_person``).
gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-BiPeg-NoThirdPerson-Direct-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs=_KWARGS,
)

__all__ = [
    "UR10eDualShadowHandBiBlindPegEnvCfg",
]
