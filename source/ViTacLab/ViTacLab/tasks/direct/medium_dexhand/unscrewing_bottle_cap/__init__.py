# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual UR10e + ShadowHand bottle cap unscrewing (articulated bottle)."""

import gymnasium as gym

from . import agents
from .unscrewing_bottle_cap_env import UR10eDualShadowHandUnscrewBottleCapEnv
from .unscrewing_bottle_cap_env_cfg import UR10eDualShadowHandUnscrewBottleCapEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0",
    entry_point=f"{__name__}.unscrewing_bottle_cap_env:UR10eDualShadowHandUnscrewBottleCapEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unscrewing_bottle_cap_env_cfg:UR10eDualShadowHandUnscrewBottleCapEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandUnscrewBottleCapPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

__all__ = [
    "UR10eDualShadowHandUnscrewBottleCapEnv",
    "UR10eDualShadowHandUnscrewBottleCapEnvCfg",
]
