# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e dual-arm hand-over (between two manipulators)."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0",
    entry_point=f"{__name__}.hand_over_env:UR10eDualShadowHandOverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_over_env_cfg:UR10eDualShadowHandOverEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandOverPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
