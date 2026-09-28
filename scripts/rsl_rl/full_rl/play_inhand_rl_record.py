#!/usr/bin/env python3
"""Roll out a trained RSL-RL checkpoint on UR10e Shadow Hand In-Hand and save NPZ episodes.

In-Hand (``Isaac-Repose-Cube-Shadow-Tactile-Direct-v0``): fixed UR10e arm; Shadow Hand
reorients a DexCube in the palm toward a goal orientation. Policy uses privileged
proprio + object/goal state (not RGB); TacSL is recorded for demo panel videos only.

Usage (Isaac Sim python)::

    python scripts/rsl_rl/full_rl/play_inhand_rl_record.py \\
        --task Isaac-Repose-Cube-Shadow-Tactile-Direct-v0 \\
        --checkpoint logs/rsl_rl/shadow_hand/2026-03-21_01-54-19/model_55250.pt \\
        --headless --enable_cameras --num_envs 4 \\
        --save_data --data_path logs/demo_inhand/collection \\
        --num_episodes 1 --save-mode best_episode
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

_RSL_RL_ROOT = Path(__file__).resolve().parent
_IK_UTILS = _RSL_RL_ROOT.parent / "ik_rl" / "utils"
_FULL_TRA = _RSL_RL_ROOT.parent / "full_tra"
for _p in (_IK_UTILS, _FULL_TRA):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cli_args  # isort: skip
from play_full_tra_simple_gripper import _save_episode_npz  # noqa: E402

HAND_ACTION_DIM = 24
INHAND_ENV_TYPE = "InHandManipulationEnv"

parser = argparse.ArgumentParser(description="RSL-RL In-Hand rollout + NPZ collection.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Repose-Cube-Shadow-Tactile-Direct-v0",
    help="Gym id for UR10e Shadow Hand in-hand reorientation.",
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RL agent configuration entry point.",
)
parser.add_argument("--save_data", action="store_true", help="Save rollout NPZ episodes.")
parser.add_argument("--data_path", type=str, default=None, help="Output directory for NPZ files.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1,
    help="Stop after this many episodes saved (total across envs).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=600,
    help="Max sim steps per rollout attempt before reset; 0 disables.",
)
parser.add_argument(
    "--max_total_steps",
    type=int,
    default=0,
    help="Stop entire search after this many env steps (0 = unlimited).",
)
parser.add_argument("--record_env_index", type=int, default=0, help="Which parallel env to record.")
parser.add_argument("--record_step_interval", type=int, default=1, help="Record every N sim steps.")
parser.add_argument(
    "--success-tolerance",
    dest="success_tolerance",
    type=float,
    default=0.1,
    help="Orientation success when |rotation_distance(object, goal)| <= this (radians).",
)
parser.add_argument(
    "--success-streak",
    dest="success_streak",
    type=int,
    default=8,
    help="Consecutive orientation-success steps required before saving NPZ.",
)
parser.add_argument(
    "--save-mode",
    dest="save_mode",
    type=str,
    default="best_episode",
    choices=("success", "best_episode", "first_episode"),
    help=(
        "success: save when orientation within tolerance for success_streak steps; "
        "best_episode: on exit save the attempt with smallest min |rot_err|; "
        "first_episode: save the first completed rollout regardless of success."
    ),
)
parser.add_argument(
    "--record-any-env",
    dest="record_any_env",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="When saving, trigger success on any parallel env (not only record_env_index).",
)
parser.add_argument(
    "--max-record-frames",
    dest="max_record_frames",
    type=int,
    default=320,
    help="Cap in-memory NPZ buffer length per attempt (prevents RAM blow-up).",
)
parser.add_argument(
    "--tactile-settle-steps",
    dest="tactile_settle_steps",
    type=int,
    default=16,
    help="Zero-action settle steps after reset before TacSL nominal render warmup.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.save_data:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import ViTacLab.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env import (
    rotation_distance,
)


def _unwrap_inhand_env(vec_env: object) -> object:
    """Return the underlying :class:`InHandManipulationEnv`, not wrapper layers."""
    cur = vec_env
    seen: set[int] = set()
    for _ in range(24):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if type(cur).__name__ == INHAND_ENV_TYPE:
            return cur
        for attr in ("env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and type(nxt).__name__ == INHAND_ENV_TYPE:
                return nxt
        nxt = getattr(cur, "env", None)
        if nxt is None:
            nxt = getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise RuntimeError(f"Could not unwrap to {INHAND_ENV_TYPE} for scene-based recording.")


def _warmup_tacsl(base_env: object) -> list[str]:
    warmed: list[str] = []
    scene = base_env.scene
    for name in scene.sensors:
        if "tactile_sensor" not in name:
            continue
        sensor = scene[name]
        if not getattr(sensor.cfg, "enable_camera_tactile", False):
            continue
        for _ in range(3):
            try:
                sensor.get_initial_render()
                warmed.append(name)
                break
            except Exception:
                break
    return warmed


def _settle_after_reset(vec_env: object, settle_steps: int) -> None:
    if settle_steps <= 0:
        return
    num_envs = int(vec_env.num_envs)
    device = vec_env.unwrapped.device
    zero = torch.zeros(num_envs, HAND_ACTION_DIM, device=device)
    # Do not wrap env.step in inference_mode (corrupts env internal buffers on DirectRLEnv).
    for _ in range(settle_steps):
        vec_env.step(zero)


def _build_record_row(base_env: object, env_index: int, action_vec: np.ndarray) -> dict[str, np.ndarray]:
    record = base_env._build_record_dict()  # noqa: SLF001
    row: dict[str, np.ndarray] = {}
    ei = max(0, min(int(env_index), int(base_env.num_envs) - 1))
    for key, value in record.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                row[key] = np.asarray(value.detach().cpu().numpy())
            else:
                idx = max(0, min(ei, int(value.shape[0]) - 1))
                row[key] = np.asarray(value[idx].detach().cpu().numpy())
    act = np.asarray(action_vec, dtype=np.float32).ravel()
    if act.size >= HAND_ACTION_DIM:
        row["action"] = act[:HAND_ACTION_DIM]
    return row


def _inhand_rot_err(base_env: object, env_index: int) -> float:
    base_env._compute_intermediate_values()  # noqa: SLF001
    ei = max(0, min(int(env_index), int(base_env.object_rot.shape[0]) - 1))
    rd = rotation_distance(base_env.object_rot, base_env.goal_rot)
    return float(torch.abs(rd[ei]).item())


def _inhand_fell(base_env: object, env_index: int) -> bool:
    base_env._compute_intermediate_values()  # noqa: SLF001
    ei = max(0, min(int(env_index), int(base_env.object_pos.shape[0]) - 1))
    palm_dist = torch.norm(base_env.object_pos - base_env.in_hand_pos, dim=-1)
    return float(palm_dist[ei].item()) >= float(base_env.cfg.fall_dist)


def _step_orientation_ok(base_env: object, env_index: int) -> bool:
    if _inhand_fell(base_env, env_index):
        return False
    return _inhand_rot_err(base_env, env_index) <= float(args_cli.success_tolerance)


def _is_inhand_success(base_env: object, env_index: int, *, streak: int) -> bool:
    return _step_orientation_ok(base_env, env_index) and streak >= int(args_cli.success_streak)


def _clear_bufs(bufs: dict[str, list[np.ndarray]] | None) -> None:
    if bufs is None:
        return
    for k in bufs:
        bufs[k].clear()


def _copy_bufs(bufs: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    return {k: list(v) for k, v in bufs.items()}


def _bufs_nonempty(bufs: dict[str, list[np.ndarray]] | None) -> bool:
    if not bufs:
        return False
    return any(len(v) > 0 for v in bufs.values())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.save_data or args_cli.enable_cameras:
        env_cfg.enable_cameras = True
        print("[INFO] In-Hand TacSL + third-person enabled (enable_cameras=True for NPZ recording)")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    resume_path = os.path.abspath(resume_path)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    base_env = _unwrap_inhand_env(env)
    base_env.reset_episode_success_statistics()
    obs, _ = env.reset()
    _settle_after_reset(env, int(args_cli.tactile_settle_steps))
    warmed = _warmup_tacsl(base_env)
    if warmed:
        print(f"[INFO] TacSL nominal render warmed: {len(warmed)} sensors")

    record_ei = max(0, min(int(args_cli.record_env_index), int(env.num_envs) - 1))
    record_interval = max(1, int(args_cli.record_step_interval))
    max_steps = int(args_cli.max_steps) if int(args_cli.max_steps) > 0 else None
    max_total_steps = int(args_cli.max_total_steps) if int(args_cli.max_total_steps) > 0 else None
    num_envs = int(env.num_envs)
    save_mode = str(args_cli.save_mode)

    data_dir: str | None = None
    rec_bufs: dict[str, list[np.ndarray]] | None = None
    if args_cli.save_data:
        if args_cli.data_path:
            data_dir = os.path.abspath(os.path.expanduser(args_cli.data_path))
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("logs", "demo_inhand", stamp))
        os.makedirs(data_dir, exist_ok=True)
        rec_bufs = {}
        meta = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "num_envs": num_envs,
            "record_env_index": record_ei,
            "max_steps": max_steps,
            "max_total_steps": max_total_steps,
            "success_tolerance": float(args_cli.success_tolerance),
            "success_streak": int(args_cli.success_streak),
            "save_mode": save_mode,
            "record_any_env": bool(args_cli.record_any_env),
            "action_source": "rsl_rl_policy",
        }
        import json

        with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(
            f"[INFO] --save_data: dir={data_dir} record_env={record_ei} "
            f"save_mode={save_mode} max_total_steps={max_total_steps or 'unlimited'}"
        )

    episodes_saved = 0
    rollout_steps = 0
    total_steps = 0
    failed_attempts = 0
    best_rot_err = float("inf")
    best_bufs: dict[str, list[np.ndarray]] | None = None
    best_attempt = 0
    best_success_env = record_ei
    saved_via: str | None = None
    record_any_env = bool(args_cli.record_any_env)
    env_streaks = [0] * num_envs
    rollout_bufs: dict[str, list[np.ndarray]] | None = None
    active_record_ei = record_ei
    max_record_frames = max(64, int(getattr(args_cli, "max_record_frames", 320)))

    def _pick_active_record_env() -> int:
        """Single NPZ buffer: follow the in-hand env with best (streak, orientation)."""
        best_ei = record_ei
        best_streak = -1
        best_rot = float("inf")
        for ei in (range(num_envs) if record_any_env else (record_ei,)):
            if _inhand_fell(base_env, ei):
                continue
            streak = env_streaks[ei]
            rot_e = _inhand_rot_err(base_env, ei)
            if streak > best_streak or (streak == best_streak and rot_e < best_rot):
                best_streak = streak
                best_rot = rot_e
                best_ei = ei
            elif best_streak <= 0 and rot_e < best_rot:
                best_rot = rot_e
                best_ei = ei
        return best_ei

    def _trim_rollout_bufs(bufs: dict[str, list[np.ndarray]]) -> None:
        n = max(len(v) for v in bufs.values()) if bufs else 0
        if n <= max_record_frames:
            return
        drop = n - max_record_frames
        for k in bufs:
            bufs[k] = bufs[k][drop:]

    def _append_frame_to(bufs: dict[str, list[np.ndarray]], env_index: int, action_vec: np.ndarray) -> None:
        row = _build_record_row(base_env, env_index, action_vec)
        for k, v in row.items():
            bufs.setdefault(k, []).append(np.asarray(v))

    def _save_bufs(
        bufs: dict[str, list[np.ndarray]],
        *,
        via: str,
        success_env: int,
        rot_err: float,
    ) -> None:
        nonlocal episodes_saved, saved_via
        if data_dir is None or not _bufs_nonempty(bufs):
            return
        path, t_len = _save_episode_npz(data_dir, f"episode_{episodes_saved:04d}.npz", bufs)
        if path is None:
            raise RuntimeError(f"{via} save requested but NPZ buffer was empty.")
        saved_via = via
        print(
            f"[{via.upper()}] saved {path} (T={t_len}, env={success_env}, "
            f"rollout_steps={rollout_steps}, attempt={failed_attempts + 1}, "
            f"total_steps={total_steps}, rot_err={rot_err:.4f})"
        )
        episodes_saved += 1

    def _reset_rollout_state() -> None:
        nonlocal rollout_steps, env_streaks, rollout_bufs, active_record_ei
        rollout_steps = 0
        env_streaks = [0] * num_envs
        rollout_bufs = None
        active_record_ei = record_ei

    while simulation_app.is_running():
        if args_cli.save_data and episodes_saved >= int(args_cli.num_episodes):
            break
        if max_total_steps is not None and total_steps >= max_total_steps:
            print(
                f"[STOP] max_total_steps={max_total_steps} reached with "
                f"episodes_saved={episodes_saved}, failed_attempts={failed_attempts}"
            )
            break

        with torch.inference_mode():
            actions = policy(obs)
        obs, _rew, dones, _infos = env.step(actions)
        policy_nn.reset(dones)

        rollout_steps += 1
        total_steps += 1

        monitor_envs = range(num_envs) if record_any_env else (record_ei,)
        min_inhand_rot = float("inf")
        max_streak = 0
        success_env: int | None = None
        for ei in monitor_envs:
            rot_e = _inhand_rot_err(base_env, ei)
            fell = _inhand_fell(base_env, ei)
            if not fell:
                min_inhand_rot = min(min_inhand_rot, rot_e)
            else:
                env_streaks[ei] = 0
            if _step_orientation_ok(base_env, ei):
                env_streaks[ei] += 1
            else:
                env_streaks[ei] = 0
            max_streak = max(max_streak, env_streaks[ei])
            if _is_inhand_success(base_env, ei, streak=env_streaks[ei]):
                success_env = ei
                break

        active_record_ei = _pick_active_record_env()

        if rec_bufs is not None and rollout_steps % record_interval == 0:
            if not _inhand_fell(base_env, active_record_ei):
                if rollout_bufs is None:
                    rollout_bufs = {}
                act_e = np.asarray(
                    actions[active_record_ei].detach().cpu().numpy(), dtype=np.float32
                ).ravel()
                _append_frame_to(rollout_bufs, active_record_ei, act_e)
                _trim_rollout_bufs(rollout_bufs)
                rot_e = _inhand_rot_err(base_env, active_record_ei)
                if rot_e < best_rot_err:
                    best_rot_err = rot_e
                    best_bufs = _copy_bufs(rollout_bufs)
                    best_attempt = failed_attempts + 1
                    best_success_env = active_record_ei
            else:
                rollout_bufs = None

        if total_steps == 1 or total_steps % 50 == 0:
            rot_log = _inhand_rot_err(base_env, record_ei)
            extra = ""
            if best_rot_err < float("inf"):
                extra = f" best_min={best_rot_err:.4f}@env{best_success_env}"
            n_ok, n_ep, rate = base_env.get_episode_success_stats()
            print(
                f"[PROGRESS] total={total_steps} rollout={rollout_steps} "
                f"attempts={failed_attempts} saved={episodes_saved} "
                f"rot_err={rot_log:.4f} min_inhand={min_inhand_rot:.4f} "
                f"max_streak={max_streak} env_succ={n_ok}/{n_ep}({rate:.2f}){extra}"
            )

        if success_env is not None:
            if rec_bufs is not None and data_dir is not None:
                act_ok = np.asarray(actions[success_env].detach().cpu().numpy(), dtype=np.float32).ravel()
                save_bufs = rollout_bufs if (rollout_bufs and active_record_ei == success_env) else {}
                _append_frame_to(save_bufs, success_env, act_ok)
                rot_ok = _inhand_rot_err(base_env, success_env)
                _save_bufs(save_bufs, via="success", success_env=success_env, rot_err=rot_ok)
            else:
                print(f"[SUCCESS] rollout_steps={rollout_steps} env={success_env} (no --save_data)")
            if episodes_saved >= int(args_cli.num_episodes):
                break
            failed_attempts += 1
            _reset_rollout_state()
            obs, _ = env.reset()
            _settle_after_reset(env, int(args_cli.tactile_settle_steps))
            _warmup_tacsl(base_env)
            base_env = _unwrap_inhand_env(env)
            continue

        done_any = bool(torch.any(dones).item()) if torch.is_tensor(dones) else False
        if (max_steps is not None and rollout_steps >= max_steps) or done_any:
            failed_attempts += 1
            if rec_bufs is not None:
                ep_min = min_inhand_rot if min_inhand_rot < float("inf") else float("inf")
                if failed_attempts <= 3 or failed_attempts % 10 == 0:
                    print(
                        f"[TIMEOUT] attempt={failed_attempts} rollout_step={rollout_steps} "
                        f"ep_min_inhand_rot_err={ep_min:.4f} (done={done_any}); resetting."
                    )
                if save_mode == "first_episode" and rollout_bufs and _bufs_nonempty(rollout_bufs):
                    _save_bufs(
                        rollout_bufs,
                        via="first_episode",
                        success_env=active_record_ei,
                        rot_err=_inhand_rot_err(base_env, active_record_ei),
                    )
                    break
            _reset_rollout_state()
            if episodes_saved >= int(args_cli.num_episodes):
                break
            obs, _ = env.reset()
            _settle_after_reset(env, int(args_cli.tactile_settle_steps))
            _warmup_tacsl(base_env)
            base_env = _unwrap_inhand_env(env)

    if (
        episodes_saved < int(args_cli.num_episodes)
        and best_bufs is not None
        and data_dir is not None
    ):
        _save_bufs(
            best_bufs,
            via="best_episode",
            success_env=best_success_env,
            rot_err=best_rot_err,
        )

    if data_dir is not None and saved_via is not None:
        meta_path = os.path.join(data_dir, "meta.json")
        if os.path.isfile(meta_path):
            import json

            with open(meta_path, "r", encoding="utf-8") as f:
                meta_doc = json.load(f)
            meta_doc["saved_via"] = saved_via
            meta_doc["best_ep_min_rot_err"] = best_rot_err if best_rot_err < float("inf") else None
            meta_doc["best_success_env"] = best_success_env
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_doc, f, indent=2)

    env.close()
    n_ok, n_ep, rate = base_env.get_episode_success_stats()
    print(
        f"[INFO] Finished: total_steps={total_steps}, episodes_saved={episodes_saved}, "
        f"env_episode_success={n_ok}/{n_ep} ({rate:.3f}), data_dir={data_dir or '(none)'}"
    )
    if args_cli.save_data and episodes_saved < int(args_cli.num_episodes):
        hint = (
            "Try SAVE_MODE=best_episode or first_episode, NUM_ENVS=8, another SEED, "
            "higher MAX_TOTAL_STEPS, SUCCESS_STREAK=2 (matches training), or relax SUCCESS_TOLERANCE."
        )
        raise SystemExit(
            f"Only saved {episodes_saved}/{int(args_cli.num_episodes)} episodes "
            f"(save_mode={save_mode}) after total_steps={total_steps}, "
            f"failed_attempts={failed_attempts}. {hint}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
    finally:
        simulation_app.close()
