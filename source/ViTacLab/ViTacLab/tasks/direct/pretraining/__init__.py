# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""GelSight short-finger tactile pretraining (mass / friction / pose) environments."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-GelsightFinger-MassPretrain-Direct-v0",
    entry_point=f"{__name__}.mass_pretrain.gelsight_mass_pretrain_env:GelsightFingerMassPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mass_pretrain.gelsight_mass_pretrain_env_cfg:GelsightFingerMassPretrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GelsightFinger-FrictionPretrain-Direct-v0",
    entry_point=f"{__name__}.friction_pretrain.gelsight_friction_pretrain_env:GelsightFingerFrictionPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.friction_pretrain.gelsight_friction_pretrain_env_cfg:GelsightFingerFrictionPretrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GelsightFinger-PosePretrain-Direct-v0",
    entry_point=f"{__name__}.pose_pretrain.gelsight_pose_pretrain_env:GelsightFingerPosePretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_pretrain.gelsight_pose_pretrain_env_cfg:GelsightFingerPosePretrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
