# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi blind peg: same dynamics as ``bi_peg``; TacSL-only when ``enable_cameras`` is True (no third-person RGB)."""

from isaaclab.utils import configclass

from ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env_cfg import (
    UR10eDualShadowHandBiPegEnvCfg,
    UR10eDualShadowHandBiPegSceneCfg,
)


@configclass
class UR10eDualShadowHandBiBlindPegEnvCfg(UR10eDualShadowHandBiPegEnvCfg):
    """Same as :class:`UR10eDualShadowHandBiPegEnvCfg` but:

    - Reuses :class:`UR10eDualShadowHandBiPegSceneCfg` so ``enable_cameras=True`` spawns
      GelSight / TacSL sensors against the shared ``hole|peg`` object expression.
    - Sets :attr:`enable_third_person_camera` to False so no tiled third-person RGB camera is added.
    """

    scene: UR10eDualShadowHandBiPegSceneCfg = UR10eDualShadowHandBiPegSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    enable_third_person_camera: bool = False
