#!/usr/bin/env python3
"""Record normalized Gym-action keyframes for any registered ViTacLab task.

This is the robot-schema-independent full-trajectory recorder.  Use the
marker/IK recorders for UR10e tasks when Cartesian editing is desired; use this
entry for Franka, standalone ShadowHand, pretraining, or any task whose action
space is Box/Dict/Tuple.  The GUI exposes one normalized ``[-1, 1]`` slider per
flattened action dimension and writes a timestamped JSON trajectory.

Examples::

    python scripts/data_collection/full_trajectory/record_action.py \
      --task Isaac-ViTac-Forge-GearMesh-Direct-v0 --enable_cameras

    python scripts/data_collection/full_trajectory/record_action.py \
      --task Isaac-ViTac-Shadow-Hand-Over-Direct-v0 --enable_cameras

Headless parser/runtime smoke test (writes one zero-action keyframe)::

    python scripts/data_collection/full_trajectory/record_action.py \
      --task Isaac-GelsightFinger-MassPretrain-Direct-v0 --no-gui \
      --max-steps 2 --record-dir play_records/action_smoke --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default=None, help="Registered ViTacLab Gym task ID (required for execution).")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments; interactive recording requires 1.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target environment stepping frequency.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps; 0 runs until the app closes.")
    parser.add_argument("--record-dir", type=str, default="scripts/data_collection/full_trajectory/records_action", help="JSON output directory.")
    parser.add_argument("--record-name", type=str, default="action_trajectory", help="Output filename prefix.")
    parser.add_argument("--hold-steps", type=int, default=30, help="Default replay hold duration stored with each snapshot.")
    parser.add_argument("--initial-action", type=float, default=0.0, help="Initial value for every normalized action slider.")
    gui = parser.add_mutually_exclusive_group()
    gui.add_argument("--gui", dest="gui", action="store_true", help="Show normalized action sliders and recorder buttons (default).")
    gui.add_argument("--no-gui", dest="gui", action="store_false", help="Disable action GUI; write one constant-action keyframe for smoke testing.")
    parser.set_defaults(gui=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _tensor_action(value: Any, *, device: str, num_envs: int, needs_batching: bool):
    import numpy as np
    import torch

    if isinstance(value, dict):
        return {key: _tensor_action(item, device=device, num_envs=num_envs, needs_batching=True) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_action(item, device=device, num_envs=num_envs, needs_batching=True) for item in value)
    array = np.asarray(value, dtype=np.float32)
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if needs_batching:
        tensor = tensor.unsqueeze(0).expand(num_envs, *tensor.shape).clone()
    return tensor


def main() -> int:
    args = _parser().parse_args()
    if not args.task:
        raise SystemExit("--task is required when executing action-trajectory recording.")
    if int(args.num_envs) != 1:
        raise SystemExit("record_action.py currently requires --num-envs 1 so one GUI command maps to one environment.")

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import ViTacLab.tasks  # noqa: F401

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if hasattr(env_cfg, "enable_cameras"):
        env_cfg.enable_cameras = True
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    exposed_action_space = getattr(env, "action_space")
    if callable(exposed_action_space):
        agents = tuple(getattr(env.unwrapped, "possible_agents", ()))
        if not agents:
            raise RuntimeError("Callable action_space exposed without possible_agents.")
        action_space = gym.spaces.Dict(
            {agent: env.unwrapped.action_space(agent) for agent in agents}
        )
        needs_batching = True
    else:
        action_space = exposed_action_space
        needs_batching = False
    action_dim = int(gym.spaces.flatdim(action_space))
    initial = float(np.clip(args.initial_action, -1.0, 1.0))
    flat_action = np.full((action_dim,), initial, dtype=np.float32)
    frames: list[dict[str, Any]] = []
    recording = {"active": False}
    status_model = None
    slider_models: list[Any] = []

    def _set_status(message: str) -> None:
        if status_model is not None:
            status_model.set_value(str(message))
        print(f"[ACTION-RECORDER] {message}")

    def _current_flat() -> np.ndarray:
        if slider_models:
            return np.asarray([model.get_value_as_float() for model in slider_models], dtype=np.float32)
        return flat_action.copy()

    def _snapshot() -> None:
        if not recording["active"]:
            _set_status("Not recording")
            return
        frames.append(
            {
                "action_flat": _current_flat().tolist(),
                "hold_steps": max(1, int(args.hold_steps)),
                "t_wall": float(time.time()),
            }
        )
        _set_status(f"Snapshots: {len(frames)}")

    def _output_path() -> Path:
        root = Path(args.record_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"{args.record_name}_{stamp}.json"

    def _save() -> Path:
        if not frames:
            frames.append(
                {
                    "action_flat": _current_flat().tolist(),
                    "hold_steps": max(1, int(args.hold_steps)),
                    "t_wall": float(time.time()),
                }
            )
        path = _output_path()
        payload = {
            "format": "vitaclab_action_trajectory_v1",
            "task": str(args.task),
            "created_at": datetime.now().isoformat(),
            "action_dim": action_dim,
            "action_space": repr(action_space),
            "needs_batching": bool(needs_batching),
            "num_frames": len(frames),
            "frames": frames,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        recording["active"] = False
        _set_status(f"Saved: {path}")
        return path

    window = None
    if args.gui:
        import omni.ui  # type: ignore

        window = omni.ui.Window("Generic Action Trajectory Recorder", width=560, height=900, visible=True)
        with window.frame:
            with omni.ui.ScrollingFrame():
                with omni.ui.VStack(spacing=4, height=0):
                    omni.ui.Label(f"Task: {args.task}", word_wrap=True)
                    omni.ui.Label(f"Flattened action dim: {action_dim}")
                    with omni.ui.HStack(spacing=4):
                        omni.ui.Button("Start Recording", clicked_fn=lambda: (frames.clear(), recording.update(active=True), _set_status("Recording")))
                        omni.ui.Button("Snapshot", clicked_fn=_snapshot)
                        omni.ui.Button("Stop + Save", clicked_fn=_save)
                    status_model = omni.ui.SimpleStringModel("Idle")
                    omni.ui.StringField(model=status_model, read_only=True)
                    for index in range(action_dim):
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label(f"action[{index}]", width=90)
                            model = omni.ui.SimpleFloatModel(initial)
                            slider_models.append(model)
                            omni.ui.FloatSlider(model=model, min=-1.0, max=1.0, step=0.01)
    else:
        recording["active"] = True
        _snapshot()

    target_dt = 1.0 / max(1e-3, float(args.fps))
    steps = 0
    try:
        while app.is_running():
            started = time.time()
            current_flat = np.clip(_current_flat(), -1.0, 1.0)
            unflattened = gym.spaces.unflatten(action_space, current_flat)
            action = _tensor_action(
                unflattened,
                device=str(env.unwrapped.device),
                num_envs=int(env.unwrapped.num_envs),
                needs_batching=bool(needs_batching),
            )
            with torch.inference_mode():
                env.step(action)
            steps += 1
            if int(args.max_steps) > 0 and steps >= int(args.max_steps):
                break
            remaining = target_dt - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if frames and (not args.gui or recording["active"]):
            _save()
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
