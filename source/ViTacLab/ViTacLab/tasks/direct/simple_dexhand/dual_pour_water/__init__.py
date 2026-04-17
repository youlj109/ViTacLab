# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual UR10e + ShadowHand shared deformable cup pouring (MARL)."""

import gymnasium as gym

from . import agents
from .ur10e_dual_shadowhand_pour_env import UR10eDualShadowHandPourEnv
from .ur10e_dual_shadowhand_pour_env_cfg import (
    UR10eDualShadowHandPourEnvCfg,
    UR10eDualShadowHandPourTactileSceneCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR10e-Dual-Shadow-Hand-PourDeformable-Direct-v0",
    entry_point=f"{__name__}.ur10e_dual_shadowhand_pour_env:UR10eDualShadowHandPourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_dual_shadowhand_pour_env_cfg:UR10eDualShadowHandPourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDualShadowHandPourPPORunnerCfg",
    },
)

__all__ = [
    "UR10eDualShadowHandPourEnv",
    "UR10eDualShadowHandPourEnvCfg",
    "UR10eDualShadowHandPourTactileSceneCfg",
]
