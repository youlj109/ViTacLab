"""Shared helpers for full_tra data-collection scripts."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import torch

_EPISODE_SUCCESS_RE = re.compile(r"^episode_(\d+)_success\.npz$")


def max_saved_episode_index(record_dir: str) -> int:
    """Largest ``episode_XXXX_success.npz`` index in *record_dir*, or -1 if none."""

    if not record_dir or not os.path.isdir(record_dir):
        return -1
    max_id = -1
    for name in os.listdir(record_dir):
        m = _EPISODE_SUCCESS_RE.match(name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id


def resolve_start_episode(start_episode: int, *, resume: bool, record_dir: str) -> int:
    """First episode index for ``range(start, num_episodes)``.

    With *resume*, use ``max(saved) + 1`` but never go below *start_episode*.
    """

    start = max(0, int(start_episode))
    if resume:
        saved = max_saved_episode_index(record_dir)
        if saved >= 0:
            start = max(start, saved + 1)
    return start


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def extract_record_row(obs: object, env_index: int) -> dict[str, np.ndarray] | None:
    record = obs.get("record") if isinstance(obs, dict) else None
    if not isinstance(record, dict):
        return None
    return _record_dict_to_row(record, env_index)


def _record_dict_to_row(record: dict, env_index: int) -> dict[str, np.ndarray] | None:
    row: dict[str, np.ndarray] = {}
    for key, value in record.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                row[key] = np.asarray(value.detach().cpu().numpy())
            else:
                ei = max(0, min(int(env_index), int(value.shape[0]) - 1))
                row[key] = np.asarray(value[ei].detach().cpu().numpy())
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                row[key] = value
            else:
                ei = max(0, min(int(env_index), int(value.shape[0]) - 1))
                row[key] = np.asarray(value[ei])
    return row or None


def extract_canonical_record_row(
    obs: object,
    env: Any,
    env_index: int,
    *,
    num_envs: int | None = None,
) -> dict[str, np.ndarray] | None:
    """Resolve one canonical record row from obs or env fallbacks."""

    row = extract_record_row(obs, env_index)
    if row:
        return row

    cur = env
    seen: set[int] = set()
    for _ in range(16):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))

        build_record = getattr(cur, "_build_record_dict", None)
        if callable(build_record):
            try:
                record = build_record()
                if isinstance(record, dict):
                    row = _record_dict_to_row(record, env_index)
                    if row:
                        return row
            except Exception:
                pass

        nxt = getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt

    return None


def merge_record_row(
    obs: object,
    env_index: int,
    *,
    actions: torch.Tensor | None = None,
    robot: Any | None = None,
    action_right: torch.Tensor | None = None,
    action_left: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Build a canonical pickup-compatible record row, including ``action`` when available."""

    row = extract_record_row(obs, env_index) or {}
    if robot is not None:
        row.setdefault("joint_pos", to_numpy(robot.data.joint_pos[env_index]))
        row.setdefault("joint_vel", to_numpy(robot.data.joint_vel[env_index]))
    if actions is not None and "action" not in row:
        row["action"] = to_numpy(actions[env_index])
    if action_right is not None and "action_right" not in row:
        row["action_right"] = to_numpy(action_right[env_index])
    if action_left is not None and "action_left" not in row:
        row["action_left"] = to_numpy(action_left[env_index])
    # Dual-arm scripts pass actions per hand. Keep per-side keys for compatibility,
    # and expose one merged action vector (left -> right) aligned with merged joint_state semantics.
    if "action" not in row and ("action_right" in row and "action_left" in row):
        ar = np.asarray(row["action_right"]).ravel()
        al = np.asarray(row["action_left"]).ravel()
        row["action"] = np.concatenate([al, ar], axis=0)
    return row


def _success_candidate(value: Any, env_index: int) -> bool | None:
    """Convert a common scalar/per-environment success payload to ``bool``.

    ``None`` means that *value* did not contain a usable success signal.  This
    distinction is important for full-trajectory replay: a real ``False``
    must not fall through to a task-specific geometric heuristic.
    """

    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value)
    except Exception:
        return None
    if tensor.numel() == 0:
        return None
    flat = tensor.reshape(-1)
    index = max(0, min(int(env_index), int(flat.numel()) - 1))
    try:
        return bool(flat[index].item())
    except Exception:
        return None


def extract_success_signal(
    infos: Any,
    obs: Any,
    env: Any,
    env_index: int,
) -> tuple[bool, bool, str]:
    """Return ``(available, success, source)`` for one environment.

    The canonical order is runtime ``infos``, the observation ``record``
    group, then attributes on the environment/wrapper chain.  Collectors use
    this shared helper before applying any task-specific fallback criterion.
    """

    named_candidates: list[tuple[str, Any]] = []
    if isinstance(infos, dict):
        for key in ("curr_success_per_env", "successes", "success"):
            named_candidates.append((f"infos.{key}", infos.get(key)))
        for parent in ("extras", "log", "record"):
            nested = infos.get(parent)
            if isinstance(nested, dict):
                for key in ("curr_success_per_env", "successes", "success"):
                    named_candidates.append((f"infos.{parent}.{key}", nested.get(key)))

    try:
        record = obs["record"]
        keys = set(record.keys())
        for key in ("curr_success_per_env", "successes", "success"):
            if key in keys:
                named_candidates.append((f"obs.record.{key}", record[key]))
    except Exception:
        pass

    cur = env
    seen: set[int] = set()
    for depth in range(16):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        for key in ("curr_success_per_env", "successes", "success"):
            named_candidates.append((f"env[{depth}].{key}", getattr(cur, key, None)))
        next_env = getattr(cur, "env", None)
        if next_env is None or next_env is cur:
            next_env = getattr(cur, "unwrapped", None)
        if next_env is None or next_env is cur:
            break
        cur = next_env

    for source, candidate in named_candidates:
        resolved = _success_candidate(candidate, env_index)
        if resolved is not None:
            return True, resolved, source
    return False, False, "unavailable"
