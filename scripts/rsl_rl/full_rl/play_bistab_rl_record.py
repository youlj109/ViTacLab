#!/usr/bin/env python3
"""Roll out a trained RSL-RL checkpoint on dual-arm BiStab and save NPZ episodes.

BiStab (``Isaac-UR10e-Dual-Shadow-Hand-BiStab-Direct-v0``): fixed UR10e arms; both Shadow
Hands steer a plate to stabilize an orange toward a random goal marker.

Usage (Isaac Sim python)::

    python scripts/rsl_rl/full_rl/play_bistab_rl_record.py \\
        --task Isaac-UR10e-Dual-Shadow-Hand-BiStab-Direct-v0 \\
        --checkpoint logs/rsl_rl/Isaac-UR10e-Dual-Shadow-Hand-BiStab-Direct-v0/.../model_9999.pt \\
        --headless --enable_cameras --num_envs 4 \\
        --save_data --data_path logs/demo_bistab/collection \\
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
BISTAB_ENV_TYPE = "UR10eDualShadowHandBiStabEnv"

parser = argparse.ArgumentParser(description="RSL-RL BiStab rollout + NPZ collection.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-UR10e-Dual-Shadow-Hand-BiStab-Direct-v0",
    help="Gym id for dual-arm bimanual stabilization.",
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
    default=650,
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
    "--success-goal-dist",
    dest="success_goal_dist",
    type=float,
    default=0.04,
    help="Stabilization success when ||object_pos - goal_pos|| < this (meters).",
)
parser.add_argument(
    "--success-vel-thresh",
    dest="success_vel_thresh",
    type=float,
    default=0.08,
    help="Stabilization success when ||object_linvel|| < this (m/s) near the goal.",
)
parser.add_argument(
    "--success-streak",
    dest="success_streak",
    type=int,
    default=8,
    help="Consecutive successful steps required before saving NPZ.",
)
parser.add_argument(
    "--save-mode",
    dest="save_mode",
    type=str,
    default="best_episode",
    choices=("success", "best_episode", "first_episode"),
    help=(
        "success: save only when near goal + low speed for success_streak steps; "
        "best_episode: on exit save the attempt with smallest min goal_dist; "
        "first_episode: save the first completed rollout regardless of success."
    ),
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

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import ViTacLab.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from ViTacLab.tasks.direct.simple_dexhand.bi_stab.bi_stab_env_cfg import (
    UR10eDualShadowHandBiStabTactileSceneCfg,
)
from ViTacLab.utils.vitaclab_marl_rsl import multi_agent_to_single_agent


def _unwrap_bistab_env(vec_env: object) -> object:
    """Return the underlying BiStab MARL env, not the single-agent tensor wrapper."""
    cur = vec_env
    seen: set[int] = set()
    for _ in range(24):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if type(cur).__name__ == BISTAB_ENV_TYPE:
            return cur
        for attr in ("env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and type(nxt).__name__ == BISTAB_ENV_TYPE:
                return nxt
        nxt = getattr(cur, "env", None)
        if nxt is None:
            nxt = getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise RuntimeError(f"Could not unwrap to {BISTAB_ENV_TYPE} for scene-based recording.")


def _warmup_tacsl(marl_env: object) -> list[str]:
    warmed: list[str] = []
    scene = marl_env.scene
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
    zero = torch.zeros(num_envs, 2 * HAND_ACTION_DIM, device=device)
    with torch.inference_mode():
        for _ in range(settle_steps):
            vec_env.step(zero)


def _build_dual_record_row(marl_env: object, env_index: int, action_vec: np.ndarray) -> dict[str, np.ndarray]:
    record = marl_env.build_npz_record_dict()
    row: dict[str, np.ndarray] = {}
    ei = max(0, min(int(env_index), int(marl_env.num_envs) - 1))
    for key, value in record.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                row[key] = np.asarray(value.detach().cpu().numpy())
            else:
                idx = max(0, min(ei, int(value.shape[0]) - 1))
                row[key] = np.asarray(value[idx].detach().cpu().numpy())
    jp_l = row.get("joint_pos_left")
    jp_r = row.get("joint_pos_right")
    if jp_l is not None and jp_r is not None:
        row["joint_pos"] = np.concatenate([np.asarray(jp_l).ravel(), np.asarray(jp_r).ravel()])
    act = np.asarray(action_vec, dtype=np.float32).ravel()
    if act.size >= 2 * HAND_ACTION_DIM:
        row["action_right"] = act[:HAND_ACTION_DIM]
        row["action_left"] = act[HAND_ACTION_DIM : 2 * HAND_ACTION_DIM]
        row["action"] = np.concatenate([row["action_left"], row["action_right"]], axis=0)
    return row


def _bistab_metrics(marl_env: object, env_index: int) -> tuple[float, float]:
    marl_env._compute_intermediate_values()  # noqa: SLF001
    ei = max(0, min(int(env_index), int(marl_env.object_pos.shape[0]) - 1))
    goal_dist = float(torch.norm(marl_env.object_pos - marl_env.goal_pos, dim=-1)[ei].item())
    ball_speed = float(torch.norm(marl_env.object_linvel, dim=-1)[ei].item())
    return goal_dist, ball_speed


def _bistab_fell(marl_env: object, env_index: int) -> bool:
    marl_env._compute_intermediate_values()  # noqa: SLF001
    ei = max(0, min(int(env_index), int(marl_env.object_pos.shape[0]) - 1))
    return float(marl_env.object_pos[ei, 2].item()) <= float(marl_env.cfg.fall_dist)


def _step_meets_stab_criteria(goal_dist: float, ball_speed: float) -> bool:
    return goal_dist < float(args_cli.success_goal_dist) and ball_speed < float(args_cli.success_vel_thresh)


def _is_bistab_success(marl_env: object, env_index: int, *, streak: int) -> bool:
    if _bistab_fell(marl_env, env_index):
        return False
    goal_dist, ball_speed = _bistab_metrics(marl_env, env_index)
    return _step_meets_stab_criteria(goal_dist, ball_speed) and streak >= int(args_cli.success_streak)


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


def _apply_tactile_scene(env_cfg: DirectMARLEnvCfg) -> None:
    old = env_cfg.scene
    env_cfg.scene = UR10eDualShadowHandBiStabTactileSceneCfg(
        num_envs=old.num_envs,
        env_spacing=old.env_spacing,
        replicate_physics=getattr(old, "replicate_physics", True),
        clone_in_fabric=getattr(old, "clone_in_fabric", False),
    )
    env_cfg.enable_cameras = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.save_data or args_cli.enable_cameras:
        _apply_tactile_scene(env_cfg)
        print("[INFO] BiStab TacSL scene enabled (plate contact, enable_cameras=True)")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    resume_path = os.path.abspath(resume_path)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    marl_env = _unwrap_bistab_env(env)
    with torch.inference_mode():
        obs, _ = env.reset()
    _settle_after_reset(env, int(args_cli.tactile_settle_steps))
    warmed = _warmup_tacsl(marl_env)
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
    success_streak = 0
    if args_cli.save_data:
        if args_cli.data_path:
            data_dir = os.path.abspath(os.path.expanduser(args_cli.data_path))
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("logs", "demo_bistab", stamp))
        os.makedirs(data_dir, exist_ok=True)
        rec_bufs = {}
        meta = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "num_envs": num_envs,
            "record_env_index": record_ei,
            "max_steps": max_steps,
            "max_total_steps": max_total_steps,
            "success_goal_dist": float(args_cli.success_goal_dist),
            "success_vel_thresh": float(args_cli.success_vel_thresh),
            "success_streak": int(args_cli.success_streak),
            "save_mode": save_mode,
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
    episode_min_goal_dist = float("inf")
    best_goal_dist = float("inf")
    best_bufs: dict[str, list[np.ndarray]] | None = None
    best_attempt = 0

    def _append_frame(action_vec: np.ndarray) -> None:
        if rec_bufs is None:
            return
        row = _build_dual_record_row(marl_env, record_ei, action_vec)
        for k, v in row.items():
            rec_bufs.setdefault(k, []).append(np.asarray(v))

    def _save_success(action_vec: np.ndarray) -> None:
        nonlocal episodes_saved, success_streak
        if rec_bufs is None or data_dir is None:
            return
        _append_frame(action_vec)
        gd, spd = _bistab_metrics(marl_env, record_ei)
        path, t_len = _save_episode_npz(data_dir, f"episode_{episodes_saved:04d}.npz", rec_bufs)
        if path is None:
            raise RuntimeError("Success detected but NPZ buffer was empty.")
        print(
            f"[SUCCESS] saved {path} (T={t_len}, rollout_steps={rollout_steps}, "
            f"attempt={failed_attempts + 1}, total_steps={total_steps}, "
            f"goal_dist={gd:.4f}, ball_speed={spd:.4f})"
        )
        episodes_saved += 1
        success_streak = 0
        _clear_bufs(rec_bufs)

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

        gd, spd = _bistab_metrics(marl_env, record_ei)
        episode_min_goal_dist = min(episode_min_goal_dist, gd)

        if total_steps == 1 or total_steps % 50 == 0:
            extra = ""
            if save_mode == "best_episode" and best_goal_dist < float("inf"):
                extra = f" best_min={best_goal_dist:.4f}"
            print(
                f"[PROGRESS] total={total_steps} rollout={rollout_steps} "
                f"attempts={failed_attempts} saved={episodes_saved} "
                f"goal_dist={gd:.4f} ball_speed={spd:.4f} ep_min={episode_min_goal_dist:.4f} "
                f"streak={success_streak}{extra}"
            )

        act_rec = np.asarray(actions[record_ei].detach().cpu().numpy(), dtype=np.float32).ravel()
        if rec_bufs is not None and rollout_steps % record_interval == 0:
            _append_frame(act_rec)

        if not _bistab_fell(marl_env, record_ei) and _step_meets_stab_criteria(gd, spd):
            success_streak += 1
        else:
            success_streak = 0

        if _is_bistab_success(marl_env, record_ei, streak=success_streak):
            if rec_bufs is not None and data_dir is not None:
                _save_success(act_rec)
            else:
                print(f"[SUCCESS] rollout_steps={rollout_steps} (no --save_data)")
            if episodes_saved >= int(args_cli.num_episodes):
                break
            rollout_steps = 0
            success_streak = 0
            episode_min_goal_dist = float("inf")
            with torch.inference_mode():
                obs, _ = env.reset()
            _settle_after_reset(env, int(args_cli.tactile_settle_steps))
            _warmup_tacsl(marl_env)
            marl_env = _unwrap_bistab_env(env)
            continue

        done_any = bool(torch.any(dones).item()) if torch.is_tensor(dones) else False
        if (max_steps is not None and rollout_steps >= max_steps) or done_any:
            failed_attempts += 1
            if rec_bufs is not None:
                if failed_attempts <= 3 or failed_attempts % 10 == 0:
                    print(
                        f"[TIMEOUT] attempt={failed_attempts} rollout_step={rollout_steps} "
                        f"ep_min_goal_dist={episode_min_goal_dist:.4f} (done={done_any}); resetting."
                    )
                if save_mode == "first_episode" and _bufs_nonempty(rec_bufs):
                    path, t_len = _save_episode_npz(
                        data_dir, f"episode_{episodes_saved:04d}.npz", rec_bufs
                    )
                    print(
                        f"[FIRST_EPISODE] saved {path} (T={t_len}, attempt={failed_attempts}, "
                        f"ep_min_goal_dist={episode_min_goal_dist:.4f})"
                    )
                    episodes_saved += 1
                    _clear_bufs(rec_bufs)
                    break
                if (
                    save_mode == "best_episode"
                    and _bufs_nonempty(rec_bufs)
                    and episode_min_goal_dist < best_goal_dist
                ):
                    best_goal_dist = episode_min_goal_dist
                    best_bufs = _copy_bufs(rec_bufs)
                    best_attempt = failed_attempts
                    print(
                        f"[BEST_SO_FAR] attempt={best_attempt} ep_min_goal_dist={best_goal_dist:.4f}"
                    )
                _clear_bufs(rec_bufs)
            success_streak = 0
            episode_min_goal_dist = float("inf")
            rollout_steps = 0
            if episodes_saved >= int(args_cli.num_episodes):
                break
            with torch.inference_mode():
                obs, _ = env.reset()
            _settle_after_reset(env, int(args_cli.tactile_settle_steps))
            _warmup_tacsl(marl_env)
            marl_env = _unwrap_bistab_env(env)

    if (
        save_mode == "best_episode"
        and episodes_saved < int(args_cli.num_episodes)
        and best_bufs is not None
        and data_dir is not None
    ):
        path, t_len = _save_episode_npz(data_dir, f"episode_{episodes_saved:04d}.npz", best_bufs)
        print(
            f"[BEST_EPISODE] saved {path} (T={t_len}, attempt={best_attempt}, "
            f"ep_min_goal_dist={best_goal_dist:.4f}, failed_attempts={failed_attempts})"
        )
        episodes_saved += 1

    env.close()
    print(
        f"[INFO] Finished: total_steps={total_steps}, episodes_saved={episodes_saved}, "
        f"data_dir={data_dir or '(none)'}"
    )
    if args_cli.save_data and episodes_saved < int(args_cli.num_episodes):
        hint = (
            "Try SAVE_MODE=best_episode or first_episode, NUM_ENVS=8, another SEED, "
            "higher MAX_TOTAL_STEPS, or relax SUCCESS_GOAL_DIST / SUCCESS_VEL_THRESH."
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
