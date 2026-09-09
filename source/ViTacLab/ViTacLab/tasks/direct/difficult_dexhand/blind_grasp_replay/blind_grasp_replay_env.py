from __future__ import annotations

from ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env import UR10eShadowHandBlindGraspEnv

from .blind_grasp_replay_env_cfg import UR10eShadowHandBlindGraspReplayEnvCfg


class UR10eShadowHandBlindGraspReplayEnv(UR10eShadowHandBlindGraspEnv):
    """Replay-phase alias of :class:`UR10eShadowHandBlindGraspEnv` (phase 3 keyframes)."""

    cfg: UR10eShadowHandBlindGraspReplayEnvCfg
