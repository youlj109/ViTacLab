# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi-Peg: dual UR10e + Shadow Hand, Factory peg (same assets as forge dexhand peg)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-BiPeg-Direct-v0",
    entry_point=f"{__name__}.bi_peg_env:UR10eDualShadowHandBiPegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bi_peg_env_cfg:UR10eDualShadowHandBiPegEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandBiPegPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
