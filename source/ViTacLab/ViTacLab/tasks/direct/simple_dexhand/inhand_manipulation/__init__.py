# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + Shadow Hand in-hand manipulation (registered from `shadow_hand` gym namespace)."""

from .inhand_manipulation_env import InHandManipulationEnv
from .inhand_manipulation_env_cfg import (
    UR10eShadowHandInHandEnvCfg,
    UR10eShadowHandInHandOpenAIEnvCfg,
    UR10eShadowHandInHandTactileEnvCfg,
)

__all__ = [
    "InHandManipulationEnv",
    "UR10eShadowHandInHandEnvCfg",
    "UR10eShadowHandInHandOpenAIEnvCfg",
    "UR10eShadowHandInHandTactileEnvCfg",
]
