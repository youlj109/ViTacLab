# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load IK+RL trajectory defaults from YAML for train/play scripts (single- and dual-arm)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# Keys must match argparse ``dest`` names (underscores) for train/play IK arguments.
# ``task`` = registered Gym id this preset is intended for (merged as default ``--task``; CLI overrides).
# ``trajectory`` = list of {pos: [x,y,z], quat: [w,x,y,z], steps: int}; quat is Isaac Lab **wxyz**.
IK_YAML_KEYS = (
    "task",
    "trajectory",
    "trajectory_right",
    "trajectory_left",
    "ee_body",
    "ik_method",
    "ik_lambda",
    "ik_k_val",
    "ik_delta_scale",
    "ik_waypoints_world_frame",
)


def default_pickup_ik_yaml_path() -> Path:
    """``ik_rl/configs/ik_rl_pickup.yaml`` (``utils`` → parent ``ik_rl``)."""
    return Path(__file__).resolve().parent.parent / "configs" / "ik_rl_pickup.yaml"


def default_unscrew_dual_ik_yaml_path() -> Path:
    """``ik_rl/configs/ik_rl_unscrew_dual.yaml``."""
    return Path(__file__).resolve().parent.parent / "configs" / "ik_rl_unscrew_dual.yaml"


def _coerce(name: str, val: Any) -> Any:
    if name == "task":
        return None if val is None else str(val).strip()
    if name == "ik_lambda" and val is None:
        return None
    if name == "ik_lambda" and val is not None:
        return float(val)
    if name == "ik_k_val" and val is None:
        return None
    if name == "ik_k_val" and val is not None:
        return float(val)
    if name == "ik_delta_scale":
        return 1.0 if val is None else float(val)
    if name == "ik_waypoints_world_frame":
        return bool(val) if val is not None else False
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
    """``parser.set_defaults`` from YAML (IK keys)."""
    if yaml_path is None or not yaml_path.is_file():
        return
    data = yaml.safe_load(yaml_path.read_text()) or {}
    kwargs: dict[str, Any] = {}
    for k in IK_YAML_KEYS:
        if k in data:
            kwargs[k] = _coerce(k, data[k])
    # Dual: if only ``trajectory`` is set, duplicate to both arms
    if "trajectory" in data:
        if "trajectory_right" not in data:
            kwargs["trajectory_right"] = kwargs["trajectory"]
        if "trajectory_left" not in data:
            kwargs["trajectory_left"] = kwargs["trajectory"]
    if kwargs:
        parser.set_defaults(**kwargs)


def apply_sys_argv_ik_yaml_defaults(
    parser: Any,
    default_file: Path | None = None,
) -> Path | None:
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
            "Using CLI task; check that IK trajectories suit this environment."
        )
