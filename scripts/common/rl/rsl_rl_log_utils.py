# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for RSL-RL checkpoint / log directory layout under ``logs/rsl_rl/``."""

from __future__ import annotations

import os


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
