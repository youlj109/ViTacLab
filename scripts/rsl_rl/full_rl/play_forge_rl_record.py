#!/usr/bin/env python3
"""Roll out a trained RSL-RL checkpoint on Franka two-finger FORGE tasks and save NPZ episodes.

Per mentor guidance, Peg / Gear / Nut (``Isaac-Forge-*-Direct-v0``) demos should be driven by
RL checkpoints, not open-loop pose trajectories.  This script loads ``model_*.pt``, runs the
policy until task success, and writes canonical tactile + camera NPZ under ``--data_path``.

Usage (Isaac Sim python)::

    python scripts/rsl_rl/full_rl/play_forge_rl_record.py \\
        --task Isaac-Forge-PegInsert-Direct-v0 \\
        --checkpoint logs/rsl_rl/forge_peg_insert_direct/2026-02-10_11-42-29/model_1200.pt \\
        --headless --enable_cameras --num_envs 1 \\
        --save_data --data_path logs/demo_forge_franka/forge_peg_insert/collection \\
        --num_episodes 1 --max_steps 800
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
from play_full_tra_simple_gripper import (  # noqa: E402
    ACTION_DIM,
    _build_record_row,
    _save_episode_npz,
    _warmup_tactile_nominal,
)

parser = argparse.ArgumentParser(description="RSL-RL Forge (Franka) rollout + NPZ collection.")
parser.add_argument("--task", type=str, required=True, help="Gym id, e.g. Isaac-Forge-PegInsert-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--save_data", action="store_true", help="Save successful rollout NPZ episodes.")
parser.add_argument("--data_path", type=str, default=None, help="Output directory for NPZ files.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1,
    help="Stop after this many successful episodes saved (total across envs).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=800,
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
    "--obs-mode",
    dest="obs_mode",
    choices=("reduce", "full"),
    default="",
    help="Forge obs mode. Default: read from checkpoint params/env.yaml, else 'reduce'. "
    "Legacy peg/gear checkpoints on this machine were trained with reduce (21-dim policy obs).",
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
from ViTacLab.utils.vitaclab_marl_rsl import multi_agent_to_single_agent


def _obs_mode_from_checkpoint_params(checkpoint_path: str) -> str | None:
    """Read ``obs_mode`` saved next to the training run (``params/env.yaml``)."""
    import re

    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(env_yaml):
        return None
    try:
        text = Path(env_yaml).read_text(encoding="utf-8")
        match = re.search(r"^obs_mode:\s*(\w+)\s*$", text, flags=re.MULTILINE)
        if match and match.group(1) in ("reduce", "full"):
            return match.group(1)
    except Exception as exc:
        print(f"[WARN] Could not parse obs_mode from {env_yaml}: {exc}")
    return None


def _resolve_obs_mode(checkpoint_path: str) -> str:
    cli_mode = str(getattr(args_cli, "obs_mode", "") or "").strip()
    if cli_mode in ("reduce", "full"):
        return cli_mode
    from_params = _obs_mode_from_checkpoint_params(checkpoint_path)
    if from_params is not None:
        return from_params
    return "reduce"


def _unwrap_forge_env(env: object) -> object:
    """Unwrap RSL/gym wrappers to Isaac Lab Forge/Factory env (uses ``_robot``, not ``robot``)."""
    cur = env
    seen: set[int] = set()
    for _ in range(24):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        scene = getattr(cur, "scene", None)
        if scene is not None and hasattr(scene, "sensors"):
            if callable(getattr(cur, "_get_curr_successes", None)) or hasattr(cur, "cfg_task"):
                return cur
        nxt = getattr(cur, "unwrapped", None)
        if nxt is None:
            nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise RuntimeError("Could not unwrap to ForgeEnv for scene-based recording.")


def _is_forge_success(forge_env: object, env_index: int) -> bool:
    try:
        check_rot = bool(getattr(forge_env.cfg_task, "name", "") == "nut_thread")
        success_threshold = float(getattr(forge_env.cfg_task, "success_threshold", 1.0))
        curr = forge_env._get_curr_successes(  # noqa: SLF001
            success_threshold=success_threshold, check_rot=check_rot
        )
        if torch.is_tensor(curr):
            if curr.ndim == 0:
                return bool(curr.item())
            ei = max(0, min(env_index, int(curr.shape[0]) - 1))
            return bool(curr[ei].item())
    except Exception:
        pass
    return False


def _clear_bufs(bufs: dict[str, list[np.ndarray]] | None) -> None:
    if bufs is None:
        return
    for k in bufs:
        bufs[k].clear()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    resume_path = os.path.abspath(resume_path)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    # Tactile + third-person recording needs sensors; policy obs mode must match checkpoint.
    setattr(env_cfg, "enable_cameras", True)
    obs_mode = _resolve_obs_mode(resume_path)
    if hasattr(env_cfg, "obs_mode"):
        env_cfg.obs_mode = obs_mode
    print(f"[INFO] Forge obs_mode={obs_mode} (enable_cameras=True for NPZ tactile/camera recording)")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.unwrapped._use_rl_control = True

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

    forge_env = _unwrap_forge_env(env)
    with torch.inference_mode():
        obs, _ = env.reset()
    warmed = _warmup_tactile_nominal(forge_env)
    if warmed:
        print(f"[INFO] TacSL nominal render warmed: {warmed}")

    record_ei = max(0, min(int(args_cli.record_env_index), int(env.num_envs) - 1))
    record_interval = max(1, int(args_cli.record_step_interval))
    max_steps = int(args_cli.max_steps) if int(args_cli.max_steps) > 0 else None
    max_total_steps = int(args_cli.max_total_steps) if int(args_cli.max_total_steps) > 0 else None
    num_envs = int(env.num_envs)

    data_dir: str | None = None
    rec_bufs_list: list[dict[str, list[np.ndarray]]] | None = None
    if args_cli.save_data:
        if args_cli.data_path:
            data_dir = os.path.abspath(os.path.expanduser(args_cli.data_path))
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("logs", "demo_forge_franka", stamp))
        os.makedirs(data_dir, exist_ok=True)
        rec_bufs_list = [dict() for _ in range(num_envs)]
        meta = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "obs_mode": obs_mode,
            "num_envs": num_envs,
            "record_env_index": record_ei,
            "max_steps": max_steps,
            "max_total_steps": max_total_steps,
            "record_step_interval": record_interval,
            "action_source": "rsl_rl_policy",
        }
        with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
            import json

            json.dump(meta, f, indent=2)
        print(
            f"[INFO] --save_data: dir={data_dir} num_envs={num_envs} "
            f"max_total_steps={max_total_steps or 'unlimited'}"
        )

    episodes_saved = 0
    rollout_steps = 0
    total_steps = 0
    failed_attempts = 0

    def _append_frame_for_env(env_i: int, action_vec: np.ndarray) -> None:
        if rec_bufs_list is None:
            return
        row = _build_record_row(forge_env, env_i, action_vec)
        buf = rec_bufs_list[env_i]
        for k, v in row.items():
            buf.setdefault(k, []).append(np.asarray(v))

    def _save_success_from_env(env_i: int, action_vec: np.ndarray) -> None:
        nonlocal episodes_saved
        if rec_bufs_list is None or data_dir is None:
            return
        _append_frame_for_env(env_i, action_vec)
        path, t_len = _save_episode_npz(
            data_dir, f"episode_{episodes_saved:04d}.npz", rec_bufs_list[env_i]
        )
        if path is None:
            raise RuntimeError("Success detected but NPZ buffer was empty.")
        print(
            f"[SUCCESS] env={env_i} saved {path} (T={t_len}, rollout_steps={rollout_steps}, "
            f"attempt={failed_attempts + 1}, total_steps={total_steps})"
        )
        episodes_saved += 1
        for buf in rec_bufs_list:
            _clear_bufs(buf)

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

        if total_steps == 1 or total_steps % 50 == 0:
            print(
                f"[PROGRESS] total={total_steps} rollout={rollout_steps} "
                f"attempts={failed_attempts} saved={episodes_saved}"
            )

        if rec_bufs_list is not None and rollout_steps % record_interval == 0:
            for env_i in range(num_envs):
                act_i = np.asarray(actions[env_i].detach().cpu().numpy(), dtype=np.float32).ravel()[:ACTION_DIM]
                _append_frame_for_env(env_i, act_i)

        success_env: int | None = None
        for env_i in range(num_envs):
            if _is_forge_success(forge_env, env_i):
                success_env = env_i
                break

        if success_env is not None:
            act_ok = np.asarray(
                actions[success_env].detach().cpu().numpy(), dtype=np.float32
            ).ravel()[:ACTION_DIM]
            if rec_bufs_list is not None and data_dir is not None:
                _save_success_from_env(success_env, act_ok)
            else:
                print(f"[SUCCESS] env={success_env} rollout_steps={rollout_steps} (no --save_data)")

            if episodes_saved >= int(args_cli.num_episodes):
                break

            rollout_steps = 0
            with torch.inference_mode():
                obs, _ = env.reset()
                policy_nn.reset(torch.ones(num_envs, dtype=torch.bool, device=env.unwrapped.device))
            forge_env = _unwrap_forge_env(env)
            _warmup_tactile_nominal(forge_env)
            continue

        done_any = bool(torch.any(dones).item()) if torch.is_tensor(dones) else False
        if (max_steps is not None and rollout_steps >= max_steps) or done_any:
            failed_attempts += 1
            if rec_bufs_list is not None:
                if failed_attempts <= 3 or failed_attempts % 10 == 0:
                    print(
                        f"[TIMEOUT] attempt={failed_attempts} rollout_step={rollout_steps} "
                        f"(done={done_any}); resetting all envs."
                    )
                for buf in rec_bufs_list:
                    _clear_bufs(buf)
            rollout_steps = 0
            with torch.inference_mode():
                obs, _ = env.reset()
                policy_nn.reset(torch.ones(num_envs, dtype=torch.bool, device=env.unwrapped.device))
            forge_env = _unwrap_forge_env(env)
            _warmup_tactile_nominal(forge_env)

    env.close()
    print(
        f"[INFO] Finished: total_steps={total_steps}, episodes_saved={episodes_saved}, "
        f"data_dir={data_dir or '(none)'}"
    )
    if args_cli.save_data and episodes_saved < int(args_cli.num_episodes):
        raise SystemExit(
            f"Only saved {episodes_saved}/{int(args_cli.num_episodes)} successful episodes "
            f"after total_steps={total_steps}, failed_attempts={failed_attempts}. "
            "Try NUM_ENVS=8, another SEED, higher MAX_TOTAL_STEPS, or verify the checkpoint."
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
