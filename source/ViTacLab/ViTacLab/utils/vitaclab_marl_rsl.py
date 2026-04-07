# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL compatibility for Isaac Lab :func:`multi_agent_to_single_agent` (ViTacLab-only).

Upstream's MARL→single-agent wrapper does not implement :meth:`_get_observations` and does not expose
buffers like ``episode_length_buf`` on the wrapper; :class:`~isaaclab_rl.rsl_rl.vecenv_wrapper.RslRlVecEnvWrapper`
expects both. We patch the nested wrapper class each call without modifying the Isaac Lab install.
"""

from __future__ import annotations

import typing

import torch

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, multi_agent_to_single_agent as _isaac_multi_agent_to_single_agent

if typing.TYPE_CHECKING:
    from isaaclab.envs.common import VecEnvObs


def multi_agent_to_single_agent(
    env: DirectMARLEnv, state_as_observation: bool = False
) -> DirectRLEnv:
    """Drop-in replacement for ``isaaclab.envs.multi_agent_to_single_agent`` when using RSL-RL."""
    out = _isaac_multi_agent_to_single_agent(env, state_as_observation)
    cls = type(out)

    def _get_observations(self) -> VecEnvObs:
        obs = self.env._get_observations()
        if self._state_as_observation:
            return {"policy": self.env.state()}
        return {
            "policy": torch.cat(
                [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents], dim=-1
            )
        }

    def __getattr__(self, key: str) -> typing.Any:
        return getattr(self.env, key)

    cls._get_observations = _get_observations
    cls.__getattr__ = __getattr__
    return out
