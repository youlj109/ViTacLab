"""Shared env/cfg entry resolution for ``record_single`` and ``play_single``.

Uses the canonical single-arm Pour/Pickup tasks, adds Forge presets, and supports
Gymnasium registry resolution shared by recording and replay entries.
"""

from __future__ import annotations

import importlib
from typing import Any


_TASK_PRESETS: dict[str, dict[str, str]] = {
    "pour": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
    },
    "pickup": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg",
    },
    "inhand": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandEnvCfg",
    },
    "inhand_openai": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandOpenAIEnvCfg",
    },
    "inhand_tactile": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandTactileEnvCfg",
    },
    "forge_peg": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgePegInsertEnvCfg",
    },
    "forge_gear": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeGearMeshEnvCfg",
    },
    "forge_nut": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeNutThreadEnvCfg",
    },
    "blind_grasp": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env:UR10eShadowHandBlindGraspEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env_cfg:UR10eShadowHandBlindGraspEnvCfg",
    },
    "blind_grasp_replay": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.blind_grasp_replay.blind_grasp_replay_env:UR10eShadowHandBlindGraspReplayEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.blind_grasp_replay.blind_grasp_replay_env_cfg:UR10eShadowHandBlindGraspReplayEnvCfg",
    },
    "blind_classification": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.blind_classification.blind_classification_env:UR10eShadowHandBlindClassificationEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.blind_classification.blind_classification_env_cfg:UR10eShadowHandBlindClassificationEnvCfg",
    },
    "blind_retrieval": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.blind_retrieval.blind_retrieval_env:UR10eShadowHandBlindRetrievalEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.blind_retrieval.blind_retrieval_env_cfg:UR10eShadowHandBlindRetrievalEnvCfg",
    },
}

_TASK_GYM_ID_ALIASES: dict[str, str] = {
    "Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0": "forge_peg",
    "Isaac-UR10eShadowHand-ForgeGearMesh-Direct-v0": "forge_gear",
    "Isaac-UR10eShadowHand-ForgeNutThread-Direct-v0": "forge_nut",
    "Isaac-UR10eShadowHand-Pickup-Direct-v0": "pickup",
    "Isaac-UR10eShadowHand-PourDeformable-Direct-v0": "pour",
    "Isaac-UR10eShadowHand-Repose-Cube-Direct-v0": "inhand",
    "Isaac-UR10eShadowHand-Repose-Cube-OpenAI-FF-Direct-v0": "inhand_openai",
    "Isaac-UR10eShadowHand-Repose-Cube-Tactile-Direct-v0": "inhand_tactile",
    "Isaac-UR10eShadowHand-BlindGrasp-Direct-v0": "blind_grasp",
    "Isaac-UR10eShadowHand-BlindGraspReplay-Direct-v0": "blind_grasp_replay",
    "Isaac-UR10eShadowHand-BlindClassification-Direct-v0": "blind_classification",
    "Isaac-UR10eShadowHand-BlindRetrieval-Direct-v0": "blind_retrieval",
}


def entries_from_gym_registry(task_id: str) -> tuple[str, str]:
    """Resolve ``module:EnvClass`` and ``env_cfg_entry_point`` from a registered Gymnasium id."""

    import gymnasium as gym

    tid = task_id.split(":")[-1].strip()
    spec = gym.spec(tid)
    ep = spec.entry_point
    if callable(ep):
        env_entry = f"{ep.__module__}:{ep.__name__}"
    else:
        env_entry = str(ep)
    kwargs = spec.kwargs or {}
    cfg_ep = kwargs.get("env_cfg_entry_point")
    if not cfg_ep:
        raise ValueError(f"Registry task {tid!r} has no env_cfg_entry_point in spec.kwargs.")
    return env_entry, str(cfg_ep)


def resolve_env_cfg_entries(
    *,
    task: str,
    env: str = "",
    cfg: str = "",
) -> tuple[str, str, str | None]:
    """Resolve env/cfg entry strings.

    Returns ``(env_entry, cfg_entry, preset_key)`` where ``preset_key`` is set when using a
    built-in preset name (after forge aliases), else ``None`` (e.g. arbitrary registered Gym id).
    """

    env_s = str(env or "").strip()
    cfg_s = str(cfg or "").strip()
    if env_s and cfg_s:
        return env_s, cfg_s, None
    if env_s or cfg_s:
        raise SystemExit("Provide both --env and --cfg, or neither and use --task.")

    t = str(task or "").strip()
    if t in _TASK_GYM_ID_ALIASES:
        t = _TASK_GYM_ID_ALIASES[t]
    if t in _TASK_PRESETS:
        p = _TASK_PRESETS[t]
        return p["env"], p["cfg"], t
    try:
        e, c = entries_from_gym_registry(t)
        return e, c, None
    except Exception as exc:
        keys = ", ".join(sorted(_TASK_PRESETS.keys()))
        aliases = ", ".join(sorted(_TASK_GYM_ID_ALIASES.keys()))
        raise SystemExit(
            f"Unknown --task {task!r}. Use one of: {keys}; a forge Gym id alias ({aliases}); "
            f"or any registered Gymnasium id whose spec provides env_cfg_entry_point "
            f"(e.g. Isaac-UR10eShadowHand-BlindGrasp-Direct-v0). ({exc})"
        ) from exc


def task_presets() -> dict[str, dict[str, str]]:
    """Read-only view of built-in presets (for tests or tooling)."""

    return dict(_TASK_PRESETS)
