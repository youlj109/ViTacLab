#!/usr/bin/env python3
"""Replay a generic normalized-action full trajectory for any ViTacLab task.

Trajectories are JSON files produced by ``record_action.py``.  This entry is
the schema-independent fallback for Franka, standalone ShadowHand, GelSight
pretraining, and any task that cannot use the UR10e marker/IK replay engines.
It records canonical ``record`` observations plus the normalized action sent
to ``env.step`` and prefers the task's own success signal.

Example::

    python scripts/data_collection/full_trajectory/play_action.py \
      --trajectory-file scripts/data_collection/full_trajectory/records_action/example.json \
      --record-data --record-path play_records/action_replay \
      --num-episodes 1 --save-outcome all --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-file", type=str, default="", help="JSON from record_action.py (required for execution).")
    parser.add_argument("--task", type=str, default="", help="Optional Gym task override; default comes from trajectory JSON.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of replay attempts.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum steps per attempt; 0 uses all frame hold durations.")
    parser.add_argument("--record-data", action="store_true", help="Write replay observations as compressed NPZ.")
    parser.add_argument("--record-path", type=str, default=None, help="NPZ output directory.")
    parser.add_argument("--record-env-index", type=int, default=0, help="Parallel environment index to record.")
    parser.add_argument("--record-step-interval", type=int, default=1, help="Record one row every N environment steps.")
    parser.add_argument(
        "--save-outcome",
        choices=("success", "completed", "all"),
        default="success",
        help="Save only canonical successes, completed trajectories, or all attempts.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed override.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _tensor_action(value: Any, *, device: str, num_envs: int, needs_batching: bool):
    import torch

    if isinstance(value, dict):
        return {key: _tensor_action(item, device=device, num_envs=num_envs, needs_batching=True) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_action(item, device=device, num_envs=num_envs, needs_batching=True) for item in value)
    array = np.asarray(value, dtype=np.float32)
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if needs_batching:
        tensor = tensor.unsqueeze(0).expand(num_envs, *tensor.shape).clone()
    elif tensor.ndim > 0 and tensor.shape[0] == 1 and num_envs > 1:
        tensor = tensor.expand(num_envs, *tensor.shape[1:]).clone()
    return tensor


def _task_env(env: Any) -> Any:
    cur = env
    seen: set[int] = set()
    for _ in range(16):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        inner = getattr(cur, "env", None)
        if inner is None or inner is cur:
            inner = getattr(cur, "unwrapped", None)
        if inner is None or inner is cur:
            break
        cur = inner
    return cur


def _record_dict(obs: Any, env: Any) -> dict[str, Any] | None:
    if isinstance(obs, dict) and isinstance(obs.get("record"), dict):
        return obs["record"]
    task = _task_env(env)
    builder = getattr(task, "_build_record_dict", None)
    if callable(builder):
        value = builder()
        if isinstance(value, dict):
            return value
    getter = getattr(task, "_get_observations", None)
    if callable(getter):
        value = getter()
        if isinstance(value, dict) and isinstance(value.get("record"), dict):
            return value["record"]
    return None


def _action_row(action: Any, env_index: int) -> np.ndarray:
    import torch

    chunks: list[np.ndarray] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in value:
                visit(value[key])
            return
        if isinstance(value, tuple):
            for item in value:
                visit(item)
            return
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if array.ndim > 1:
            index = max(0, min(int(env_index), int(array.shape[0]) - 1))
            array = array[index]
        chunks.append(np.asarray(array, dtype=np.float32).reshape(-1))

    visit(action)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)


def main() -> int:
    args = _parser().parse_args()
    if not str(args.trajectory_file).strip():
        raise SystemExit("--trajectory-file is required when executing action-trajectory replay.")
    if int(args.num_envs) != 1:
        raise SystemExit("play_action.py currently requires --num-envs 1; replay each recorded action stream deterministically.")

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import ViTacLab.tasks  # noqa: F401

    if str(HERE) not in os.sys.path:
        os.sys.path.insert(0, str(HERE))
    from common_record_utils import extract_success_signal

    trajectory_path = Path(args.trajectory_file).expanduser()
    if not trajectory_path.is_absolute():
        trajectory_path = (Path.cwd() / trajectory_path).resolve()
    doc = json.loads(trajectory_path.read_text(encoding="utf-8"))
    frames = list(doc.get("frames") or [])
    if not frames:
        raise RuntimeError(f"Trajectory has no frames: {trajectory_path}")
    task = str(args.task).strip() or str(doc.get("task", "")).strip()
    if not task:
        raise RuntimeError("No task in trajectory JSON; pass --task explicitly.")

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=max(1, int(args.num_envs)))
    if args.seed is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = int(args.seed)
    if hasattr(env_cfg, "enable_cameras"):
        env_cfg.enable_cameras = True
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")

    exposed_action_space = getattr(env, "action_space")
    if callable(exposed_action_space):
        agents = tuple(getattr(env.unwrapped, "possible_agents", ()))
        action_space = gym.spaces.Dict({agent: env.unwrapped.action_space(agent) for agent in agents})
        needs_batching = True
    else:
        action_space = exposed_action_space
        needs_batching = False
    action_dim = int(gym.spaces.flatdim(action_space))
    expected_dim = int(doc.get("action_dim", action_dim))
    if action_dim != expected_dim:
        raise RuntimeError(
            f"Trajectory action_dim={expected_dim} does not match task action_dim={action_dim}."
        )

    record_dir = None
    if args.record_data:
        record_dir = (
            Path(args.record_path).expanduser().resolve()
            if args.record_path
            else Path.cwd() / "play_records" / f"action_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / "meta.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "trajectory_file": str(trajectory_path),
                    "num_episodes": int(args.num_episodes),
                    "save_outcome": str(args.save_outcome),
                    "action_dim": action_dim,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    env_index = max(0, min(int(args.record_env_index), int(env.unwrapped.num_envs) - 1))
    interval = max(1, int(args.record_step_interval))
    successes = 0
    saved = 0

    try:
        for episode in range(max(1, int(args.num_episodes))):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            buffers: dict[str, list[np.ndarray]] = {}
            step = 0
            completed = True
            success_latched = False
            success_source = "unavailable"

            for frame in frames:
                flat = np.asarray(frame.get("action_flat", []), dtype=np.float32).reshape(-1)
                if flat.size != action_dim:
                    raise RuntimeError(
                        f"Frame action size {flat.size} does not match task action dim {action_dim}."
                    )
                flat = np.clip(flat, -1.0, 1.0)
                unflattened = gym.spaces.unflatten(action_space, flat)
                action = _tensor_action(
                    unflattened,
                    device=str(env.unwrapped.device),
                    num_envs=int(env.unwrapped.num_envs),
                    needs_batching=bool(needs_batching),
                )
                for _ in range(max(1, int(frame.get("hold_steps", 1)))):
                    with torch.inference_mode():
                        obs, _, terminated, truncated, infos = env.step(action)
                    step += 1
                    available, succeeded, source = extract_success_signal(infos, obs, env, env_index)
                    if available:
                        success_source = source
                        success_latched = success_latched or succeeded

                    if args.record_data and step % interval == 0:
                        record = _record_dict(obs, env)
                        if record is None:
                            raise RuntimeError(f"Task {task} exposes no canonical record dictionary.")
                        row: dict[str, np.ndarray] = {}
                        for key, value in record.items():
                            if torch.is_tensor(value):
                                if value.ndim == 0:
                                    row[key] = np.asarray(value.detach().cpu().numpy())
                                else:
                                    row[key] = np.asarray(value[env_index].detach().cpu().numpy())
                        row["action"] = _action_row(action, env_index)
                        for key, value in row.items():
                            buffers.setdefault(key, []).append(value)

                    done = False
                    if torch.is_tensor(terminated):
                        done = bool(torch.any(terminated).item())
                    elif isinstance(terminated, dict):
                        done = any(bool(torch.any(value).item()) for value in terminated.values())
                    if torch.is_tensor(truncated):
                        done = done or bool(torch.any(truncated).item())
                    elif isinstance(truncated, dict):
                        done = done or any(bool(torch.any(value).item()) for value in truncated.values())
                    if done:
                        completed = False
                        break
                    if int(args.max_steps) > 0 and step >= int(args.max_steps):
                        completed = False
                        break
                if not completed:
                    break

            outcome = "success" if success_latched else ("completed" if completed else "done")
            should_save = (
                outcome == "success"
                or (outcome == "completed" and args.save_outcome in ("completed", "all"))
                or (outcome == "done" and args.save_outcome == "all")
            )
            if outcome == "success":
                successes += 1
            if should_save and args.record_data and record_dir is not None:
                if not buffers:
                    raise RuntimeError("Replay selected for saving but no record rows were buffered.")
                lengths = {key: len(value) for key, value in buffers.items()}
                if len(set(lengths.values())) != 1:
                    raise RuntimeError(f"Inconsistent record lengths before save: {lengths}")
                path = record_dir / f"episode_{episode:04d}_{outcome}.npz"
                np.savez_compressed(path, **{key: np.stack(value, axis=0) for key, value in buffers.items()})
                saved += 1
                print(f"[INFO] saved {outcome}: {path} (T={next(iter(lengths.values()))}, source={success_source})")
            else:
                print(f"[INFO] skipped {outcome} episode {episode} (source={success_source})")
    finally:
        env.close()
        app.close()

    print(f"[INFO] action replay complete: success={successes}, saved={saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
