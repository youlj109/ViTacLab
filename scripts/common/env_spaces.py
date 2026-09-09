"""Compatibility helpers for single-agent and Isaac Lab DirectMARLEnv spaces.

DirectRLEnv exposes batched ``observation_space`` and ``action_space``
attributes.  DirectMARLEnv instead exposes per-agent methods with those names
and stores unbatched spaces in ``observation_spaces``/``action_spaces``.  Smoke
agents use this module so both APIs produce correctly batched torch actions.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from isaaclab.envs.utils.spaces import sample_space


def resolve_env_space(env: Any, name: str) -> tuple[gym.Space, bool]:
    """Resolve an environment space and report whether it needs vector batching.

    Args:
        env: Gym environment or wrapper.
        name: ``"observation_space"`` or ``"action_space"``.

    Returns:
        ``(space, needs_batching)``. DirectMARLEnv methods are converted to a
        Dict keyed by ``possible_agents`` and require batching by ``num_envs``.
        DirectRLEnv attributes are already vectorized and do not.
    """

    exposed = getattr(env, name)
    if not callable(exposed):
        return exposed, False

    unwrapped = env.unwrapped
    agents = tuple(getattr(unwrapped, "possible_agents", ()))
    if not agents:
        raise RuntimeError(
            f"Environment exposes callable {name} but has no possible_agents: {type(unwrapped).__name__}"
        )
    method = getattr(unwrapped, name)
    return gym.spaces.Dict({agent: method(agent) for agent in agents}), True


def sample_resolved_action(
    space: gym.Space,
    *,
    needs_batching: bool,
    num_envs: int,
    device: str,
    fill_value: float | None,
) -> Any:
    """Create correctly batched torch actions from a resolved action space.

    ``fill_value=0.0`` creates zero actions; ``None`` preserves random samples.
    This helper is used only for DirectMARLEnv spaces. DirectRLEnv action
    spaces are already batched and continue through the agents' existing
    tensor conversion functions.
    """

    if not needs_batching:
        raise ValueError("sample_resolved_action is only for unbatched DirectMARLEnv spaces")
    return sample_space(space, device=device, batch_size=num_envs, fill_value=fill_value)
