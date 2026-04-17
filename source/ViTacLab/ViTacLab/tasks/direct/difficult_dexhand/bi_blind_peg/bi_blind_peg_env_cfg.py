# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi blind peg: same dynamics as ``bi_peg``; TacSL-only when ``enable_cameras`` is True (no third-person RGB)."""

from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_cfg import (
    UR10eDualShadowHandTacSLSceneCfg,
)
from ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env_cfg import UR10eDualShadowHandBiPegEnvCfg


@configclass
class UR10eDualShadowHandBiBlindPegSceneCfg(UR10eDualShadowHandTacSLSceneCfg):
    """TacSL contacts ``hole`` or ``peg`` (same prims as Bi-Peg rigid bodies)."""

    @classmethod
    def _tactile_params(cls) -> dict:
        p = super()._tactile_params()
        p["contact_object_prim_path_expr"] = "/World/envs/env_.*/(hole|peg)"
        return p


@configclass
class UR10eDualShadowHandBiBlindPegEnvCfg(UR10eDualShadowHandBiPegEnvCfg):
    """Same as :class:`UR10eDualShadowHandBiPegEnvCfg` but:

    - Uses :class:`UR10eDualShadowHandBiBlindPegSceneCfg` so ``enable_cameras=True`` spawns
      GelSight / TacSL sensors (not the default ``object`` prim).
    - Sets :attr:`enable_third_person_camera` to False so no tiled third-person RGB camera is added.
    """

    scene: UR10eDualShadowHandBiBlindPegSceneCfg = UR10eDualShadowHandBiBlindPegSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    enable_third_person_camera: bool = False
