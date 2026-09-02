# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load palm/IK trajectory defaults from YAML for ``train_ik_rl_single`` / ``play_ik_rl_single``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# Keys must match argparse ``dest`` names (underscores) for train/play IK arguments.
# ``task`` = registered Gym id this preset is intended for (merged as default ``--task``; CLI overrides).
IK_YAML_KEYS = (
    "task",
    "trajectory",
    "object_to_palm_offset",
    "palm_in_wrist_pos",
    "palm_in_wrist_euler",
    "palm_orient",
    "palm_normal_local",
    "palm_yaw_offset",
    "world_down",
    "palm_euler",
    "palm_euler_in_anchor",
    "ee_body",
    "ik_method",
    "ik_lambda",
    # Optional: freeze hand joints to a fixed grasp pose during pickup.
    # (Used by ik_rl pickup only; full_ik has its own hand staging.)
    "hand_freeze_phase_target",
    "hand_freeze_yaml",
)


def default_pickup_ik_yaml_path() -> Path:
    """``ik_rl/configs/ik_rl_pickup.yaml`` (``utils`` → parent ``ik_rl``)."""
    # This file lives in ``.../ik_rl/utils/``; configs are in ``.../ik_rl/configs/``.
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "scripts" / "data_collection" / "ik" / "configs" / "ik_rl" / "ik_rl_pickup.yaml"


def _coerce(name: str, val: Any) -> Any:
    if name in (
        "object_to_palm_offset",
        "palm_in_wrist_pos",
        "palm_in_wrist_euler",
        "palm_normal_local",
        "world_down",
        "palm_euler",
        "palm_euler_in_anchor",
    ):
        if isinstance(val, (list, tuple)) and len(val) == 3:
            return tuple(float(x) for x in val)
    if name == "palm_yaw_offset":
        return float(val)
    if name == "ik_lambda" and val is None:
        return None
    if name == "ik_lambda" and val is not None:
        return float(val)
    if name == "task":
        return None if val is None else str(val).strip()
    if name in ("hand_freeze_phase_target", "hand_freeze_yaml"):
        return None if val is None else str(val).strip()
    return val


def resolve_ik_config_path(argv: list[str], default_file: Path | None) -> Path | None:
    """Pick YAML path: explicit ``--ik-config PATH`` or ``default_file`` if it exists."""
    if "--ik-config" in argv:
        i = argv.index("--ik-config")
        if i + 1 < len(argv):
            raw = argv[i + 1].strip()
            low = raw.lower()
            if low in ("none", "false", ""):
                return None
            return Path(raw).expanduser()
        return None
    if default_file is not None and default_file.is_file():
        return default_file
    return None


def load_ik_yaml_into_parser(parser: Any, yaml_path: Path | None) -> None:
    """``parser.set_defaults`` from YAML (only known IK keys)."""
    if yaml_path is None or not yaml_path.is_file():
        return
    data = yaml.safe_load(yaml_path.read_text()) or {}
    kwargs: dict[str, Any] = {}
    for k in IK_YAML_KEYS:
        if k in data:
            kwargs[k] = _coerce(k, data[k])
    if kwargs:
        parser.set_defaults(**kwargs)


def apply_sys_argv_ik_yaml_defaults(parser: Any, default_file: Path | None = None) -> Path | None:
    """Resolve path from ``sys.argv`` + default file, apply to ``parser``, return resolved path (or None)."""
    if default_file is None:
        default_file = default_pickup_ik_yaml_path()
    resolved = resolve_ik_config_path(sys.argv, default_file)
    load_ik_yaml_into_parser(parser, resolved)
    return resolved


def warn_if_task_mismatch_with_ik_yaml(resolved_yaml_path: Path | None, cli_task: str | None) -> None:
    """If YAML declares ``task`` and CLI ``--task`` differs, print a warning."""
    if resolved_yaml_path is None or not resolved_yaml_path.is_file() or not cli_task:
        return
    data = yaml.safe_load(resolved_yaml_path.read_text()) or {}
    yt = data.get("task")
    if yt is None:
        return
    ys = str(yt).strip()
    cs = str(cli_task).strip()
    if ys and cs and ys != cs:
        print(
            f"[WARN] IK config YAML task={ys!r} does not match CLI --task={cs!r}. "
            "Using CLI task; check that palm/IK settings suit this environment."
        )
