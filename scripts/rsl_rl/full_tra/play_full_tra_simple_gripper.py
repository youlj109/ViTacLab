#!/usr/bin/env python3
"""Replay open-loop action trajectories for ``simple_gripper`` (Franka + Factory FORGE).

Unlike ``play_full_tra_single_v5.py`` (UR10e + Shadow Hand marker/IK), this script drives
``ForgeEnv`` with the native 7-D factory action vector per frame:

    [pos_x, pos_y, pos_z, roll, pitch, yaw, success_pred]  (each in [-1, 1])

Trajectory JSON (``vitatlab_forge_action_trajectory_v1``) example::

    {
      "format": "vitatlab_forge_action_trajectory_v1",
      "task": "simple_forge_peg",
      "frames": [
        {"action": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1.0], "hold_steps": 8},
        ...
      ]
    }

Also accepts ``.npz`` with an ``action`` array of shape ``(T, 7)``.

Usage (Isaac Sim python)::

    python scripts/rsl_rl/full_tra/play_full_tra_simple_gripper.py \\
        --task simple_forge_peg \\
        --trajectory-file path/to/expert_actions.json \\
        --num_envs 1 --enable_cameras \\
        --record-data --record-path ~/output/simple_forge/peg \\
        --record-max-episodes 100
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from full_tra_task_entries import resolve_env_cfg_entries

_TASK_PRESETS: dict[str, dict[str, str]] = {
    "simple_forge_peg": {
        "env": "ViTacLab.tasks.direct.simple_gripper.forge_env:ForgeEnv",
        "cfg": "ViTacLab.tasks.direct.simple_gripper.forge_env_cfg:ForgeTaskPegInsertCfg",
    },
    "simple_forge_gear": {
        "env": "ViTacLab.tasks.direct.simple_gripper.forge_env:ForgeEnv",
        "cfg": "ViTacLab.tasks.direct.simple_gripper.forge_env_cfg:ForgeTaskGearMeshCfg",
    },
    "simple_forge_nut": {
        "env": "ViTacLab.tasks.direct.simple_gripper.forge_env:ForgeEnv",
        "cfg": "ViTacLab.tasks.direct.simple_gripper.forge_env_cfg:ForgeTaskNutThreadCfg",
    },
}

TACTILE_SENSOR_NAMES = ("tactile_sensor_left", "tactile_sensor_right")
ACTION_DIM = 7


def _repo_root() -> Path:
    p = _HERE
    for _ in range(10):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _load_symbol(entry: str) -> Any:
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _img_to_uint8_torch(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.dtype == torch.uint8:
        return rgb
    mx = float(rgb.max().item())
    if mx <= 1.0:
        return torch.clamp(rgb * 255.0, 0.0, 255.0).to(torch.uint8)
    return torch.clamp(rgb, 0.0, 255.0).to(torch.uint8)


def _save_episode_npz(out_dir: str, fname: str, bufs: dict[str, list[np.ndarray]]) -> tuple[str | None, int]:
    if not bufs:
        return None, 0
    first_key = next(iter(bufs.keys()))
    if len(bufs[first_key]) == 0:
        return None, 0
    t_len = len(bufs[first_key])
    path = os.path.join(out_dir, fname)
    payload = {k: np.stack(v, axis=0) for k, v in bufs.items() if len(v) > 0}
    np.savez_compressed(path, **payload)
    for k in bufs:
        bufs[k].clear()
    return path, t_len


def _clear_bufs(bufs: dict[str, list[np.ndarray]] | None) -> None:
    if bufs is None:
        return
    for k in bufs:
        bufs[k].clear()


def _warmup_tactile_nominal(env: Any) -> list[str]:
    warmed: list[str] = []
    scene = env.scene
    for name in TACTILE_SENSOR_NAMES:
        if name not in scene.sensors:
            continue
        sensor = scene[name]
        for _ in range(3):
            try:
                sensor.get_initial_render()
                warmed.append(name)
                break
            except RuntimeError:
                continue
            except Exception:
                break
    return warmed


def _read_tactile_from_scene(env: Any, env_index: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    scene = env.scene
    ei = max(0, min(int(env_index), int(env.num_envs) - 1))
    norm_list: list[torch.Tensor] = []
    shear_list: list[torch.Tensor] = []
    rgb_list: list[torch.Tensor] = []
    for name in TACTILE_SENSOR_NAMES:
        if name not in scene.sensors:
            continue
        try:
            data = scene[name].data
        except RuntimeError:
            try:
                scene[name].get_initial_render()
                data = scene[name].data
            except Exception:
                continue
        nf = getattr(data, "tactile_normal_force", None)
        sf = getattr(data, "tactile_shear_force", None)
        rgb = getattr(data, "tactile_rgb_image", None)
        if nf is not None:
            nf_e = torch.nan_to_num(nf[ei].detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            norm_list.append(nf_e)
        if sf is not None:
            sf_e = torch.nan_to_num(sf[ei].detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            shear_list.append(sf_e)
        if rgb is not None:
            rgb_list.append(_img_to_uint8_torch(rgb[ei]).detach().cpu())

    tactile_hw: tuple[int, int] | None = None
    if hasattr(env.cfg, "scene") and hasattr(env.cfg.scene, "tactile_sensor_left"):
        tactile_hw = tuple(env.cfg.scene.tactile_sensor_left.tactile_array_size)

    num_tactile = len(TACTILE_SENSOR_NAMES)
    if len(norm_list) == num_tactile and tactile_hw is not None:
        stacked = torch.stack(norm_list, dim=0)
        out["tactile_normal_force"] = stacked.reshape(num_tactile, tactile_hw[0], tactile_hw[1], 1).numpy()
    if len(shear_list) == num_tactile and tactile_hw is not None:
        stacked = torch.stack(shear_list, dim=0)
        out["tactile_shear_force"] = stacked.reshape(num_tactile, tactile_hw[0], tactile_hw[1], 2).numpy()
    if len(rgb_list) == num_tactile:
        out["tactile_rgb_image"] = torch.stack(rgb_list, dim=0).numpy()

    if "third_person_camera" in scene.sensors:
        cam_rgb = scene["third_person_camera"].data.output.get("rgb", None)
        if cam_rgb is not None:
            rgb_np = cam_rgb[ei].detach().cpu().numpy()
            out["third_person_camera"] = rgb_np
            out["camera_rgb"] = rgb_np
    return out


def _build_record_row(env: Any, env_index: int, action_record: np.ndarray) -> dict[str, np.ndarray]:
    ei = max(0, min(int(env_index), int(env.num_envs) - 1))
    row: dict[str, np.ndarray] = {
        "joint_pos": _to_numpy(env.robot.data.joint_pos[ei]).astype(np.float32),
        "joint_vel": _to_numpy(env.robot.data.joint_vel[ei]).astype(np.float32),
        "action": np.asarray(action_record, dtype=np.float32).ravel()[:ACTION_DIM].copy(),
    }
    row.update(_read_tactile_from_scene(env, ei))
    return row


def _load_trajectory_frames(traj_path: Path) -> list[dict[str, Any]]:
    if traj_path.suffix.lower() == ".npz":
        data = np.load(traj_path, allow_pickle=True)
        if "action" not in data:
            raise RuntimeError(f"NPZ trajectory missing 'action' key: {traj_path}")
        actions = np.asarray(data["action"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise RuntimeError(
                f"NPZ 'action' must have shape (T, {ACTION_DIM}), got {actions.shape} in {traj_path}"
            )
        hold = int(data["hold_steps"]) if "hold_steps" in data else 1
        return [{"action": actions[t].tolist(), "hold_steps": hold} for t in range(actions.shape[0])]

    doc = json.loads(traj_path.read_text(encoding="utf-8"))
    frames = list(doc.get("frames") or [])
    if frames:
        return frames
    frames = list(doc.get("keyframes") or [])
    if not frames:
        raise RuntimeError(f"Trajectory has neither 'frames' nor 'keyframes': {traj_path}")
    return frames


def _frame_action(fr: dict[str, Any], default_hold: int) -> tuple[np.ndarray, int]:
    if "action" not in fr:
        raise RuntimeError("Each trajectory frame must contain an 'action' field with 7 floats.")
    action = np.asarray(fr["action"], dtype=np.float32).ravel()
    if action.shape[0] != ACTION_DIM:
        raise RuntimeError(f"Expected action dim {ACTION_DIM}, got shape {action.shape}")
    hold = int(fr.get("hold_steps", default_hold))
    return np.clip(action, -1.0, 1.0), max(1, hold)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Play open-loop Factory FORGE action trajectories for simple_gripper tasks."
    )
    p.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="simple_forge_peg")
    p.add_argument("--env", type=str, default="")
    p.add_argument("--cfg", type=str, default="")
    p.add_argument("--trajectory-file", type=str, required=True, help="JSON or NPZ with per-step 7-D actions.")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = until app closed.")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = unlimited episode resets.")
    p.add_argument(
        "--steps-per-frame",
        type=int,
        default=1,
        help="Default sim steps to hold each trajectory frame (overridden by frame hold_steps).",
    )
    p.add_argument(
        "--max-steps-per-trajectory",
        type=int,
        default=0,
        help="If >0, fail episode when sim steps exceed this while replaying one trajectory.",
    )
    p.add_argument(
        "--action-smoothing",
        type=float,
        default=0.0,
        help="EMA on actions in [0,1); 0=off, 0.5=moderate smoothing.",
    )
    p.add_argument("--obs-mode", choices=("reduce", "full"), default="full", help="Forge observation mode.")
    p.add_argument("--record-data", action="store_true", help="Save successful episodes to npz.")
    p.add_argument("--record-step-interval", type=int, default=1, help="Record every N sim steps (>=1).")
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--record-env-index", type=int, default=0)
    p.add_argument("--record-max-episodes", type=int, default=0, help="Stop after N successful saves.")
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401

    repo_root = _repo_root()
    record_interval = max(1, int(args.record_step_interval))
    default_hold = max(1, int(args.steps_per_frame))

    traj_path = Path(str(args.trajectory_file)).expanduser()
    if not traj_path.is_absolute():
        traj_path = (repo_root / traj_path).resolve()
    frames = _load_trajectory_frames(traj_path)
    print(f"[INFO] Loaded trajectory {traj_path} with {len(frames)} frames.")

    env_entry, cfg_entry, preset_key = resolve_env_cfg_entries(
        task=str(args.task),
        env=str(args.env),
        cfg=str(args.cfg),
    )
    if preset_key is None and str(args.task) in _TASK_PRESETS:
        preset_key = str(args.task)
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    cfg.obs_mode = str(args.obs_mode)
    if bool(getattr(args, "enable_cameras", False)):
        cfg.enable_cameras = True
    elif str(args.obs_mode) == "full":
        print("[WARN] obs_mode=full without --enable_cameras; tactile obs will be zero-filled.")
    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(cfg, "seed"):
            setattr(cfg, "seed", seed)
        print(f"[INFO] Using seed={seed}")

    env = EnvCls(cfg)
    env.reset()
    warmed = _warmup_tactile_nominal(env)
    if warmed:
        print(f"[INFO] TacSL nominal render warmed: {warmed}")

    action_dim = int(getattr(cfg, "action_space", ACTION_DIM))
    if action_dim != ACTION_DIM:
        print(f"[WARN] cfg.action_space={action_dim}, expected {ACTION_DIM} for simple_gripper replay.")

    record_dir = None
    rec_bufs: dict[str, list[np.ndarray]] | None = None
    if bool(args.record_data):
        if args.record_path:
            record_dir = os.path.abspath(os.path.expanduser(args.record_path))
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            record_dir = os.path.join(os.getcwd(), "play_records", f"simple_forge_{ts}")
        os.makedirs(record_dir, exist_ok=True)
        rec_bufs = {}
        meta = {
            "task": str(args.task),
            "preset_key": preset_key,
            "env_entry": env_entry,
            "cfg_entry": cfg_entry,
            "trajectory_file": str(traj_path),
            "num_frames": len(frames),
            "num_envs": int(env.num_envs),
            "record_env_index": int(args.record_env_index),
            "record_step_interval": record_interval,
            "steps_per_frame_default": default_hold,
            "action_dim": ACTION_DIM,
            "action_smoothing": float(args.action_smoothing),
        }
        with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] --record-data: saving successful episodes under {record_dir}")

    record_ei = max(0, min(int(args.record_env_index), env.num_envs - 1))
    smoothing = float(np.clip(args.action_smoothing, 0.0, 0.999))
    target_dt = 1.0 / max(1e-3, float(args.fps))

    total_steps = 0
    episode_idx = 0
    success_count = 0
    frame_idx = 0
    hold_left = 0
    episode_sim_steps = 0
    last_recorded_episode_step = -1
    prev_action = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)

    def _is_success_now() -> bool:
        try:
            check_rot = bool(getattr(env.cfg_task, "name", "") == "nut_thread")
            success_threshold = float(getattr(env.cfg_task, "success_threshold", 1.0))
            curr = env._get_curr_successes(  # noqa: SLF001
                success_threshold=success_threshold, check_rot=check_rot
            )
            if torch.is_tensor(curr):
                if curr.ndim == 0:
                    return bool(curr.item())
                ei = max(0, min(record_ei, int(curr.shape[0]) - 1))
                return bool(curr[ei].item())
        except Exception:
            pass
        return False

    def _append_record_row(action_record: np.ndarray) -> None:
        nonlocal last_recorded_episode_step
        if not bool(args.record_data) or rec_bufs is None:
            return
        row = _build_record_row(env, record_ei, action_record)
        if not rec_bufs:
            for k in row:
                rec_bufs[k] = []
        for k, v in row.items():
            rec_bufs.setdefault(k, []).append(np.asarray(v))
        last_recorded_episode_step = episode_sim_steps

    def _reset_episode() -> None:
        nonlocal frame_idx, hold_left, episode_sim_steps, last_recorded_episode_step, prev_action
        env.reset()
        _warmup_tactile_nominal(env)
        frame_idx = 0
        hold_left = 0
        episode_sim_steps = 0
        last_recorded_episode_step = -1
        prev_action = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
        _clear_bufs(rec_bufs)

    _reset_episode()
    hold_left = 0

    while simulation_app.is_running():
        t0 = time.time()
        total_steps += 1

        if frame_idx >= len(frames):
            episode_idx += 1
            if args.max_episodes > 0 and episode_idx >= int(args.max_episodes):
                break
            _reset_episode()
            continue

        if hold_left <= 0:
            action_np, hold_left = _frame_action(frames[frame_idx], default_hold)
            frame_idx += 1

        cmd = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)
        if smoothing > 0.0:
            cmd = smoothing * prev_action + (1.0 - smoothing) * cmd
        prev_action = cmd.clone()
        if env.num_envs > 1:
            actions = cmd.expand(env.num_envs, -1).clone()
        else:
            actions = cmd

        last_action_record = np.asarray(actions[record_ei].detach().cpu().numpy(), dtype=np.float32)
        _obs, _rew, terminated, truncated, _extras = env.step(actions)
        episode_sim_steps += 1
        hold_left -= 1

        if bool(args.record_data) and rec_bufs is not None and episode_sim_steps % record_interval == 0:
            _append_record_row(last_action_record)

        if _is_success_now():
            success_count += 1
            if bool(args.record_data) and rec_bufs is not None and last_recorded_episode_step != episode_sim_steps:
                _append_record_row(last_action_record)
            print(
                f"[SUCCESS] step={total_steps} episode_step={episode_sim_steps} "
                f"frames_consumed={frame_idx}/{len(frames)} success_count={success_count}"
            )
            if bool(args.record_data) and rec_bufs is not None and record_dir is not None:
                path, t_len = _save_episode_npz(
                    record_dir, f"episode_{success_count - 1:04d}.npz", rec_bufs
                )
                if path is not None:
                    print(f"[INFO] recorded successful episode {path} (T={t_len})")
            episode_idx += 1
            if args.record_max_episodes > 0 and success_count >= int(args.record_max_episodes):
                break
            if args.max_episodes > 0 and episode_idx >= int(args.max_episodes):
                break
            _reset_episode()
            continue

        mpt = int(args.max_steps_per_trajectory)
        if mpt > 0 and episode_sim_steps > mpt:
            print(
                f"[TRAJ-LIMIT] episode exceeded max_steps_per_trajectory={mpt} "
                f"(episode_sim_steps={episode_sim_steps}), resetting."
            )
            episode_idx += 1
            _clear_bufs(rec_bufs)
            if args.max_episodes > 0 and episode_idx >= int(args.max_episodes):
                break
            _reset_episode()
            continue

        done_any = bool(torch.any(terminated | truncated).item()) if torch.is_tensor(terminated) else False
        if done_any:
            episode_idx += 1
            if args.max_episodes > 0 and episode_idx >= int(args.max_episodes):
                break
            _reset_episode()
            continue

        if args.max_steps > 0 and total_steps >= int(args.max_steps):
            break

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    env.close()
    simulation_app.close()
    print(
        f"[INFO] play finished: total_steps={total_steps}, episodes={episode_idx}, "
        f"success={success_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
