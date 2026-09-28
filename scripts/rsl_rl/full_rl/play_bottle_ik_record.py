#!/usr/bin/env python3
"""Bottle cap task demo: GPU IK arm waypoints + optional hand RL checkpoint + NPZ/video collection.

Uses the in-package stack for ``Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0``:
  - env cfg arm poses + bottle layout from ``unscrewing_bottle_cap_env_cfg.py``
  - EE waypoints from ``scripts/rsl_rl/ik_rl/configs/ik_rl_unscrew_dual.yaml`` (auto-loaded)
  - optional ``--checkpoint`` for learned hand policy (``play_ik_rl_dual.py`` stack)

Without a checkpoint, hand actions are zero (arms still follow IK toward the bottle).

Usage::

    python scripts/rsl_rl/full_rl/play_bottle_ik_record.py \\
        --headless --enable_cameras --num_envs 1 \\
        --save_data --data_path logs/demo_bottle/collection \\
        --max_steps 750 --save-mode first_episode
"""

from __future__ import annotations

import argparse
import json
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

HAND_ACTION_DIM = 48  # 24 left + 24 right (single-agent tensor after MARL merge)

parser = argparse.ArgumentParser(description="Bottle cap IK (+ optional hand RL) rollout + NPZ collection.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--save_data", action="store_true")
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=750, help="Max sim steps per rollout; 0 disables.")
parser.add_argument("--max_total_steps", type=int, default=0)
parser.add_argument("--record_env_index", type=int, default=0)
parser.add_argument("--record_step_interval", type=int, default=1)
parser.add_argument(
    "--save-mode",
    dest="save_mode",
    type=str,
    default="first_episode",
    choices=("success", "best_episode", "first_episode"),
)
parser.add_argument(
    "--success-cap-trans",
    dest="success_cap_trans",
    type=float,
    default=0.012,
    help="Cap prismatic travel (m) treated as success when save-mode=success.",
)
parser.add_argument(
    "--tactile-settle-steps",
    dest="tactile_settle_steps",
    type=int,
    default=16,
)
parser.add_argument(
    "--ik-config",
    type=str,
    default=None,
    help="IK YAML (default: ik_rl/configs/ik_rl_unscrew_dual.yaml). Use 'none' to disable.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

from ik_rl_load_config import (  # noqa: E402
    apply_sys_argv_ik_yaml_defaults,
    default_unscrew_dual_ik_yaml_path,
)

apply_sys_argv_ik_yaml_defaults(parser, default_unscrew_dual_ik_yaml_path())
args_cli, hydra_args = parser.parse_known_args()
if args_cli.save_data:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from ik_rl_hand_vec_env import (  # noqa: E402
    DualArmIkHandActionExpander,
    IkHandRslRlVecEnvWrapper,
    build_ik_cfg_from_trajectory_args,
)
from ik_rl_load_config import resolve_ik_config_path  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import ViTacLab.tasks  # noqa: F401
from ViTacLab.utils.vitaclab_marl_rsl import multi_agent_to_single_agent


def _unwrap_bottle_env(vec_env: object) -> object:
    cur = vec_env
    seen: set[int] = set()
    for _ in range(24):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if type(cur).__name__ == "UR10eDualShadowHandUnscrewBottleCapEnv":
            return cur
        for attr in ("env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and type(nxt).__name__ == "UR10eDualShadowHandUnscrewBottleCapEnv":
                return nxt
        nxt = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise RuntimeError("Could not unwrap to UR10eDualShadowHandUnscrewBottleCapEnv.")


def _warmup_tacsl(marl_env: object) -> list[str]:
    warmed: list[str] = []
    for name in marl_env.scene.sensors:
        if "tactile_sensor" not in name:
            continue
        sensor = marl_env.scene[name]
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
    zero = torch.zeros(int(vec_env.num_envs), HAND_ACTION_DIM, device=vec_env.unwrapped.device)
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
    if act.size >= HAND_ACTION_DIM:
        row["action_left"] = act[:24]
        row["action_right"] = act[24:48]
        row["action"] = act[:48]
    return row


def _bottle_metrics(marl_env: object, env_index: int) -> tuple[float, float, float, bool]:
    marl_env._compute_intermediate_values()  # noqa: SLF001
    ei = max(0, min(int(env_index), int(marl_env.num_envs) - 1))
    cap_trans = float(marl_env.cap_trans[ei].item())
    cap_rot = float(torch.abs(marl_env.cap_rot[ei]).item())
    bottle_z = float(marl_env.bottle_pos[ei, 2].item())
    fell = bottle_z <= float(marl_env.cfg.fall_height)
    return cap_trans, cap_rot, bottle_z, fell


def _resolve_checkpoint_optional() -> str | None:
    cp = getattr(args_cli, "checkpoint", None)
    if cp and os.path.isfile(os.path.expanduser(cp)):
        return retrieve_file_path(cp)
    if getattr(args_cli, "resume", False) and getattr(args_cli, "load_run", None) and cp:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.task))
        return get_checkpoint_path(log_root, args_cli.load_run, cp)
    return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.save_data or args_cli.enable_cameras:
        setattr(env_cfg, "enable_cameras", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    ik_cfg_right = build_ik_cfg_from_trajectory_args(args_cli, arm="right")
    ik_cfg_left = build_ik_cfg_from_trajectory_args(args_cli, arm="left")
    expander = DualArmIkHandActionExpander(env.unwrapped, ik_cfg_right, ik_cfg_left)
    env = IkHandRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions, expander=expander)
    print(
        f"[INFO] Bottle IK: waypoints right={len(ik_cfg_right.waypoints)} "
        f"left={len(ik_cfg_left.waypoints)}"
    )

    resume_path = _resolve_checkpoint_optional()
    policy = None
    policy_nn = None
    if resume_path:
        print(f"[INFO] Loading hand policy checkpoint: {resume_path}")
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
            policy_nn = getattr(runner.alg, "actor_critic", None)
    else:
        print("[INFO] No checkpoint: zero hand actions (arms follow IK waypoints only).")

    marl_env = _unwrap_bottle_env(env)
    obs, _ = env.reset()
    _settle_after_reset(env, int(args_cli.tactile_settle_steps))
    warmed = _warmup_tacsl(marl_env)
    if warmed:
        print(f"[INFO] TacSL nominal render warmed: {len(warmed)} sensors")

    record_ei = max(0, min(int(args_cli.record_env_index), int(env.num_envs) - 1))
    record_interval = max(1, int(args_cli.record_step_interval))
    max_steps = int(args_cli.max_steps) if int(args_cli.max_steps) > 0 else None
    max_total_steps = int(args_cli.max_total_steps) if int(args_cli.max_total_steps) > 0 else None
    save_mode = str(args_cli.save_mode)

    data_dir: str | None = None
    rec_bufs: dict[str, list[np.ndarray]] | None = None
    if args_cli.save_data:
        data_dir = os.path.abspath(os.path.expanduser(args_cli.data_path or "logs/demo_bottle/collection"))
        os.makedirs(data_dir, exist_ok=True)
        rec_bufs = {}
        meta = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "action_source": "rsl_rl_policy" if resume_path else "ik_only_zero_hand",
            "ik_config": str(resolve_ik_config_path(sys.argv, default_unscrew_dual_ik_yaml_path()) or ""),
            "num_envs": int(env.num_envs),
            "record_env_index": record_ei,
            "max_steps": max_steps,
            "save_mode": save_mode,
        }
        with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] --save_data: dir={data_dir} record_env={record_ei} save_mode={save_mode}")

    episodes_saved = 0
    rollout_steps = 0
    total_steps = 0
    failed_attempts = 0
    best_cap_trans = -1.0
    best_bufs: dict[str, list[np.ndarray]] | None = None

    def _append_frame(action_vec: np.ndarray) -> None:
        if rec_bufs is None:
            return
        row = _build_dual_record_row(marl_env, record_ei, action_vec)
        for k, v in row.items():
            rec_bufs.setdefault(k, []).append(np.asarray(v))

    while simulation_app.is_running():
        if args_cli.save_data and episodes_saved >= int(args_cli.num_episodes):
            break
        if max_total_steps is not None and total_steps >= max_total_steps:
            break

        if policy is not None:
            with torch.inference_mode():
                actions = policy(obs)
        else:
            actions = torch.zeros(env.num_envs, HAND_ACTION_DIM, device=env.unwrapped.device)
        obs, _rew, dones, _infos = env.step(actions)
        if policy_nn is not None:
            with torch.inference_mode():
                policy_nn.reset(dones)

        rollout_steps += 1
        total_steps += 1
        cap_trans, cap_rot, bottle_z, fell = _bottle_metrics(marl_env, record_ei)

        if total_steps == 1 or total_steps % 50 == 0:
            print(
                f"[PROGRESS] step={total_steps} cap_trans={cap_trans:.4f} cap_rot={cap_rot:.4f} "
                f"bottle_z={bottle_z:.4f} fell={fell} saved={episodes_saved}"
            )

        act_rec = np.asarray(actions[record_ei].detach().cpu().numpy(), dtype=np.float32).ravel()
        if rec_bufs is not None and rollout_steps % record_interval == 0:
            _append_frame(act_rec)

        success = (not fell) and cap_trans >= float(args_cli.success_cap_trans)
        if success and save_mode == "success" and rec_bufs is not None and data_dir is not None:
            _append_frame(act_rec)
            path, t_len = _save_episode_npz(data_dir, f"episode_{episodes_saved:04d}_env{record_ei}_success.npz", rec_bufs)
            print(f"[SUCCESS] saved {path} (T={t_len})")
            episodes_saved += 1
            for k in rec_bufs:
                rec_bufs[k].clear()
            rollout_steps = 0
            obs, _ = env.reset()
            _settle_after_reset(env, int(args_cli.tactile_settle_steps))
            _warmup_tacsl(marl_env)
            marl_env = _unwrap_bottle_env(env)
            continue

        done_any = bool(torch.any(dones).item()) if torch.is_tensor(dones) else False
        if (max_steps is not None and rollout_steps >= max_steps) or done_any or fell:
            failed_attempts += 1
            if rec_bufs is not None and data_dir is not None and any(len(v) > 0 for v in rec_bufs.values()):
                if save_mode == "first_episode":
                    suffix = "success" if success else "fail"
                    path, t_len = _save_episode_npz(
                        data_dir, f"episode_{episodes_saved:04d}_env{record_ei}_{suffix}.npz", rec_bufs
                    )
                    print(f"[SAVE] {save_mode} -> {path} (T={t_len}, cap_trans={cap_trans:.4f}, bottle_z={bottle_z:.4f})")
                    episodes_saved += 1
                    break
                if save_mode == "best_episode" and cap_trans > best_cap_trans:
                    best_cap_trans = cap_trans
                    best_bufs = {k: list(v) for k, v in rec_bufs.items()}
                for k in rec_bufs:
                    rec_bufs[k].clear()
            rollout_steps = 0
            if save_mode != "first_episode" and episodes_saved < int(args_cli.num_episodes):
                obs, _ = env.reset()
                _settle_after_reset(env, int(args_cli.tactile_settle_steps))
                _warmup_tacsl(marl_env)
                marl_env = _unwrap_bottle_env(env)

    if save_mode == "best_episode" and best_bufs and data_dir:
        path, t_len = _save_episode_npz(
            data_dir, f"episode_{episodes_saved:04d}_env{record_ei}_best.npz", best_bufs
        )
        print(f"[BEST] saved {path} (T={t_len}, cap_trans={best_cap_trans:.4f})")
        episodes_saved += 1

    env.close()
    print(f"[DONE] episodes_saved={episodes_saved} total_steps={total_steps}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
    finally:
        if simulation_app.is_running():
            simulation_app.close()
