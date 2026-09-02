"""Shared env/cfg entry resolution for dual-arm full trajectory scripts."""

from __future__ import annotations

from typing import Any


_TASK_PRESETS: dict[str, dict[str, str]] = {
    "bi_blind_grasp": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.bi_blind_grasp.bi_blind_grasp_env:UR10eDualShadowHandBiBlindGraspEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.bi_blind_grasp.bi_blind_grasp_env_cfg:UR10eDualShadowHandBiBlindGraspEnvCfg",
    },
    "hand_over": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env:UR10eDualShadowHandOverEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env_cfg:UR10eDualShadowHandOverEnvCfg",
    },
    "bi_peg": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env:UR10eDualShadowHandBiPegEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env_cfg:UR10eDualShadowHandBiPegEnvCfg",
    },
    "unscrew": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.unscrewing_bottle_cap.unscrewing_bottle_cap_env:UR10eDualShadowHandUnscrewBottleCapEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.unscrewing_bottle_cap.unscrewing_bottle_cap_env_cfg:UR10eDualShadowHandUnscrewBottleCapEnvCfg",
    },
}

_TASK_GYM_ID_ALIASES: dict[str, str] = {
    "Isaac-UR10e-Dual-Shadow-Hand-BiBlindGrasp-Direct-v0": "bi_blind_grasp",
    "Isaac-UR10e-Dual-Shadow-Hand-HandOver-Direct-v0": "hand_over",
    "Isaac-UR10e-Dual-Shadow-Hand-BiPeg-Direct-v0": "bi_peg",
    "Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0": "unscrew",
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
    built-in preset name (after aliases), else ``None``.
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
            f"Unknown --task {task!r}. Use one of: {keys}; a dual Gym id alias ({aliases}); "
            f"or any registered Gymnasium id whose spec provides env_cfg_entry_point. ({exc})"
        ) from exc


def task_presets() -> dict[str, dict[str, str]]:
    """Read-only view of built-in presets (for tests or tooling)."""

    return dict(_TASK_PRESETS)
