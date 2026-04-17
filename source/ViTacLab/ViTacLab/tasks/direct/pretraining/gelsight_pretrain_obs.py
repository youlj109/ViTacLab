# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared observation size for GelSight finger pretraining tasks."""


def pretrain_obs_dim(use_full_tactile_obs: bool) -> int:
    """Object position (3) + tactile compact (3) or full TacSL normal+shear."""

    base = 3
    tactile = (20 * 25 + 20 * 25 * 2) if use_full_tactile_obs else 3
    return base + tactile
