# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for RSL-RL: checkpoint log layout (``logs/rsl_rl/``) and ``rsl-rl-lib`` version check."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys

from packaging import version


def check_rsl_rl_lib_version(min_version: str = "3.0.1") -> None:
    """Exit with message if ``rsl-rl-lib`` is older than ``min_version`` (ik_rl train/play scripts)."""

    installed = importlib.metadata.version("rsl-rl-lib")
    if version.parse(installed) >= version.parse(min_version):
        return
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={min_version}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={min_version}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed}'"
        f" and required version is: '{min_version}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    sys.exit(1)


def get_rsl_rl_log_root(task_id: str, experiment_name_override: str | None = None) -> str:
    """Root directory for a training run's checkpoints (``.pt``) and logs.

    By default the folder name matches the registered Gymnasium task id (e.g.
    ``Isaac-UR10eShadowHand-Pickup-Direct-v0``) so different tasks do not share one
    ``experiment_name`` from the agent YAML.

    Args:
        task_id: Full task id passed to ``--task`` (may contain ``:`` for namespaced ids).
        experiment_name_override: If set (e.g. CLI ``--experiment_name``), use this folder name instead.

    Returns:
        Absolute path ``.../logs/rsl_rl/<folder_name>``.
    """
    if experiment_name_override:
        folder = experiment_name_override
    else:
        folder = task_id.replace(":", "_").replace(os.sep, "_").strip()
    return os.path.abspath(os.path.join("logs", "rsl_rl", folder))
