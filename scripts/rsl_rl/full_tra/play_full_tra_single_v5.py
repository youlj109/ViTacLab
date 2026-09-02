#!/usr/bin/env python3
"""Replay pure trajectory control (v5).

Based on v3, with one recording change:
- Keep real joint_pos from env record unchanged.
- Additionally save commanded action (the normalized action sent to env.step) as key ``action``.
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher

_TASK_PRESETS: dict[str, dict[str, str]] = {
    "pour": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.pour_water.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.pour_water.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
    },
    "pickup": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_v1:UR10eShadowHandPickupEnvV1",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg_v1:UR10eShadowHandPickupEnvCfgV1",
    },
    "inhand": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandEnvCfg",
    },
    "forge_gear": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeGearMeshEnvCfg",
    },
    "forge_insert": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgePegInsertEnvCfg",
    },
    "forge_nut": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeNutThreadEnvCfg",
    },
}


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
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


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


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


def _extract_record_row(obs: object, env_index: int) -> dict[str, np.ndarray] | None:
    record = None
    try:
        record = obs["record"]  # type: ignore[index]
    except Exception:
        if isinstance(obs, dict):
            record = obs.get("record", None)
    if not isinstance(record, dict):
        return None
    row: dict[str, np.ndarray] = {}
    for k, v in record.items():
        if torch.is_tensor(v):
            if v.ndim == 0:
                row[k] = np.asarray(v.detach().cpu().numpy())
            else:
                ei = max(0, min(int(env_index), int(v.shape[0]) - 1))
                row[k] = np.asarray(v[ei].detach().cpu().numpy())
    return row if row else None


def _sample_forge_fixed_offset_env(env: Any, env_index: int) -> np.ndarray:
    """Read forge fixed reset offset from env (fallback zeros)."""
    out = np.zeros(3, dtype=np.float64)
    try:
        t = getattr(env, "fixed_pos_env_random", None)
        if torch.is_tensor(t) and t.ndim >= 2 and t.shape[-1] >= 3:
            ei = max(0, min(int(env_index), int(t.shape[0]) - 1))
            out = _to_numpy(t[ei]).ravel()[:3]
    except Exception:
        pass
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play pure full_tra trajectory from teleop recorder JSON (v5).")
    p.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="pickup")
    p.add_argument("--env", type=str, default="")
    p.add_argument("--cfg", type=str, default="")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=None, help="Random seed for deterministic env resets/replay.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = until app closed.")
    p.add_argument(
        "--max-steps-per-trajectory",
        type=int,
        default=0,
        help="If >0, max env.step count per episode while replaying one JSON; exceed => fail episode and reset.",
    )
    p.add_argument("--max-episodes", type=int, default=0, help="0 = unlimited resets.")
    p.add_argument("--trajectory-file", type=str, required=True, help="JSON generated by GUI recorder.")
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-pos-tol", type=float, default=0.05, help="Arm joint convergence tolerance (rad).")
    p.add_argument("--hand-pos-tol", type=float, default=0.08, help="Hand joint convergence tolerance (rad).")
    p.add_argument("--stable-steps", type=int, default=30, help="Need N consecutive converged steps to switch frame.")
    p.add_argument("--max-steps-per-frame", type=int, default=240, help="Force switch if a frame cannot converge.")
    p.add_argument(
        "--post-arm-reached-steps",
        type=int,
        default=100,
        help="After arm reaches target, wait at most N more steps before switching frame.",
    )
    p.add_argument("--goal-z-tol", type=float, default=0.012, help="Success z window: abs(object_z-goal_z) <= tol.")
    p.add_argument(
        "--grasp-lift-min-dz",
        type=float,
        default=0.02,
        help="Minimum lift above init z to consider object as grasped/lifted.",
    )
    p.add_argument(
        "--arm-max-step-rad",
        type=float,
        default=0.0,
        help="Per sim step, cap each arm joint change toward IK (rad); 0=off. Try 0.012-0.04 to slow post-grasp lift.",
    )
    p.add_argument(
        "--arm-slew-only-when-lifted",
        action="store_true",
        help="If set, --arm-max-step-rad applies only when object z is already lifted (grasp-lift-min-dz).",
    )
    p.add_argument(
        "--hand-max-step-rad",
        type=float,
        default=0.0,
        help="Per sim step, cap each Shadow Hand joint toward IK (rad); 0=off.",
    )
    p.add_argument("--record-data", action="store_true", help="Record successful episodes to npz.")
    p.add_argument(
        "--record-step-interval",
        type=int,
        default=2,
        help="Within each episode, append one data row every N env.step (>=1). Default 2 = every other step.",
    )
    p.add_argument("--record-path", type=str, default=None, help="Output dir for npz records.")
    p.add_argument("--record-env-index", type=int, default=0, help="Env index to record.")
    p.add_argument("--record-max-episodes", type=int, default=0, help="Stop after saving N successful episodes.")
    p.add_argument(
        "--max-success-xy-dist",
        type=float,
        default=-1.0,
        help="If >0 and source=env_success, require xy_dist <= this to accept/save success episode.",
    )
    p.add_argument(
        "--max-success-z-disp-ratio",
        type=float,
        default=-1.0,
        help="If >0 and source=env_success, require z_disp <= ratio * height_threshold to accept/save success.",
    )
    p.add_argument(
        "--require-tactile-success",
        action="store_true",
        help="Require tactile-feedback gate in addition to env geometric success before saving success episode.",
    )
    p.add_argument(
        "--min-tactile-normal-total",
        type=float,
        default=0.0,
        help="Tactile gate: require max(total_normal_force) over recent window >= this value.",
    )
    p.add_argument(
        "--tactile-point-force-threshold",
        type=float,
        default=1e-4,
        help="Tactile gate: point-level threshold used for active-ratio computation.",
    )
    p.add_argument(
        "--min-tactile-active-ratio",
        type=float,
        default=0.0,
        help="Tactile gate: require max(active_ratio) over recent window >= this value.",
    )
    p.add_argument(
        "--tactile-gate-window",
        type=int,
        default=10,
        help="Recent-step window length for tactile success gate.",
    )
    p.add_argument(
        "--success-hold-steps",
        type=int,
        default=1,
        help="Require N consecutive accepted-success steps before saving success episode.",
    )
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401

    repo_root = _repo_root()
    source_dir = repo_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import VideoTeleopControl
    from scipy.spatial.transform import Rotation as R

    record_interval = max(1, int(args.record_step_interval))

    traj_path = Path(str(args.trajectory_file)).expanduser()
    if not traj_path.is_absolute():
        traj_path = (repo_root / traj_path).resolve()
    doc = json.loads(traj_path.read_text(encoding="utf-8"))
    frames = list(doc.get("frames") or [])
    if frames:
        print(f"[INFO] Loaded trajectory {traj_path} with {len(frames)} frames (source=frames).")
    else:
        frames = list(doc.get("keyframes") or [])
        if not frames:
            raise RuntimeError(
                f"trajectory has neither 'frames' nor 'keyframes': {traj_path}"
            )
        print(
            f"[INFO] Loaded trajectory {traj_path} with {len(frames)} frames "
            "(source=keyframes fallback)."
        )

    preset = _TASK_PRESETS[str(args.task)]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
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
    if hasattr(cfg, "goal_resample_time_range_s"):
        setattr(cfg, "goal_resample_time_range_s", (1.0e9, 1.0e9))
    if hasattr(cfg, "enable_cameras") and not bool(getattr(cfg, "enable_cameras", False)):
        setattr(cfg, "enable_cameras", True)
        print("[INFO] Forced cfg.enable_cameras=True for pickup v1 tactile pipeline.")
    env = EnvCls(cfg)
    if not hasattr(env, "_use_rl_control"):
        env._use_rl_control = True
    robot = env.robot
    joint_names = list(robot.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    sh_names = shadowhand_joint_names()
    env.reset()
    action_dim = env.num_actions

    def _sample_object_pos_env() -> np.ndarray:
        if hasattr(env, "object") and hasattr(env.object, "data"):
            return _to_numpy(env.object.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3]
        if hasattr(env, "cup") and hasattr(env.cup, "data"):
            return _to_numpy(env.cup.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3]
        return np.zeros(3, dtype=np.float64)

    def _sample_goal_pos_env() -> np.ndarray:
        if hasattr(env, "goal_object_pos"):
            return _to_numpy(env.goal_object_pos[0]).ravel()[:3]
        if hasattr(env, "goal_cup_pos"):
            return _to_numpy(env.goal_cup_pos[0]).ravel()[:3]
        return np.zeros(3, dtype=np.float64)

    def _object_init_pos_env() -> np.ndarray:
        if hasattr(cfg, "object_cfg"):
            pos = getattr(getattr(cfg, "object_cfg"), "init_state", None)
            if pos is not None and hasattr(pos, "pos"):
                return np.asarray(pos.pos, dtype=np.float64).ravel()[:3]
        if hasattr(cfg, "cup_cfg"):
            pos = getattr(getattr(cfg, "cup_cfg"), "init_state", None)
            if pos is not None and hasattr(pos, "pos"):
                return np.asarray(pos.pos, dtype=np.float64).ravel()[:3]
        return np.zeros(3, dtype=np.float64)

    def _hand_joint_for_robot_name(name: str, hand_joints: np.ndarray) -> float:
        for sh_idx, sh_name in enumerate(sh_names):
            if sh_name in name or name.endswith(sh_name):
                return float(hand_joints[sh_idx])
        return 0.0

    def _current_arm_joints() -> np.ndarray:
        jpos = robot.data.joint_pos[0].detach().cpu().numpy()
        return np.array([float(jpos[idx]) for idx in arm_indices], dtype=np.float64)

    def _current_hand_shadow24() -> np.ndarray:
        out = np.zeros(24, dtype=np.float64)
        jpos = robot.data.joint_pos[0].detach().cpu().numpy()
        for sh_i, sh_name in enumerate(sh_names):
            for idx in hand_indices:
                n = joint_names[idx]
                if sh_name in n or n.endswith(sh_name):
                    out[sh_i] = float(jpos[idx])
                    break
        return out

    def _build_action(arm_joints: np.ndarray, hand_joints: np.ndarray) -> torch.Tensor:
        full_dof = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_indices):
            if i < len(arm_joints):
                full_dof[idx] = float(arm_joints[i])
        for idx in hand_indices:
            full_dof[idx] = _hand_joint_for_robot_name(joint_names[idx], hand_joints)
        actuated = full_dof[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lower = env.robot_dof_lower_limits[0, env.actuated_dof_indices].cpu().numpy()
        upper = env.robot_dof_upper_limits[0, env.actuated_dof_indices].cpu().numpy()
        eps = 1e-6
        scale = np.where(upper - lower > eps, 2.0 * (actuated - lower) / (upper - lower) - 1.0, 0.0)
        scale = np.clip(scale, -1.0, 1.0)
        return torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)

    def _set_goal_above_object() -> None:
        if not hasattr(env, "goal_object_pos"):
            return
        obj = _sample_object_pos_env()
        goal = _sample_goal_pos_env()
        env.goal_object_pos[0, 0] = float(obj[0])
        env.goal_object_pos[0, 1] = float(obj[1])
        env.goal_object_pos[0, 2] = float(goal[2])

    def _is_success_now() -> tuple[bool, str]:
        # Prefer task-native success logic when available (e.g., forge gear_mesh).
        try:
            if hasattr(env, "_get_curr_successes") and hasattr(env, "cfg_task"):
                check_rot = bool(getattr(env.cfg_task, "name", "") == "nut_thread")
                success_threshold = float(getattr(env.cfg_task, "success_threshold", 1.0))
                curr_successes = env._get_curr_successes(  # noqa: SLF001
                    success_threshold=success_threshold, check_rot=check_rot
                )
                if torch.is_tensor(curr_successes):
                    if curr_successes.ndim == 0:
                        ok = bool(curr_successes.item())
                    else:
                        ei = max(0, min(int(record_ei), int(curr_successes.shape[0]) - 1))
                        ok = bool(curr_successes[ei].item())
                    return ok, "env_success"
        except Exception:
            pass

        # Fallback for environments without native success API.
        z_obj = float(_sample_object_pos_env()[2])
        z_goal = float(_sample_goal_pos_env()[2])
        z_init = float(_object_init_pos_env()[2])
        is_lifted_ok = (z_obj - z_init) >= float(args.grasp_lift_min_dz)
        reached_goal_z_ok = abs(z_obj - z_goal) <= float(args.goal_z_tol)
        return bool(is_lifted_ok and reached_goal_z_ok), "lifted_and_goal_z"

    def _success_debug_values() -> tuple[float | None, float | None, float | None]:
        """Return (xy_dist, z_disp, height_times_success_threshold) for current step."""
        try:
            if not (hasattr(env, "_held_base_pos") and hasattr(env, "_target_held_base_pos") and hasattr(env, "cfg_task")):
                return None, None, None
            ei = max(0, min(int(record_ei), int(env.num_envs) - 1))
            held_base_pos = env._held_base_pos  # noqa: SLF001
            target_held_base_pos = env._target_held_base_pos  # noqa: SLF001
            xy_dist_t = torch.linalg.vector_norm(
                target_held_base_pos[ei, 0:2] - held_base_pos[ei, 0:2], dim=0
            )
            z_disp_t = held_base_pos[ei, 2] - target_held_base_pos[ei, 2]
            success_threshold = float(getattr(env.cfg_task, "success_threshold", 1.0))
            fixed_cfg = getattr(env.cfg_task, "fixed_asset_cfg", None)
            task_name = str(getattr(env.cfg_task, "name", ""))
            if task_name in {"peg_insert", "gear_mesh"} and fixed_cfg is not None:
                hst = float(getattr(fixed_cfg, "height")) * success_threshold
            elif task_name == "nut_thread" and fixed_cfg is not None:
                hst = float(getattr(fixed_cfg, "thread_pitch")) * success_threshold
            else:
                hst = None
            return float(xy_dist_t.item()), float(z_disp_t.item()), hst
        except Exception:
            return None, None, None

    def _make_T(pos_xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
        t[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
        return t

    record_dir = None
    rec_bufs: dict[str, list[np.ndarray]] | None = None
    if bool(args.record_data):
        if args.record_path:
            record_dir = os.path.abspath(os.path.expanduser(args.record_path))
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            record_dir = os.path.join(os.getcwd(), "play_records", f"full_tra_v5_{ts}")
        os.makedirs(record_dir, exist_ok=True)
        rec_bufs = {}
        meta = {
            "task": str(args.task),
            "trajectory_file": str(traj_path),
            "num_envs": int(env.num_envs),
            "record_env_index": int(args.record_env_index),
            "goal_z_tol": float(args.goal_z_tol),
            "grasp_lift_min_dz": float(args.grasp_lift_min_dz),
            "arm_max_step_rad": float(args.arm_max_step_rad),
            "arm_slew_only_when_lifted": bool(args.arm_slew_only_when_lifted),
            "hand_max_step_rad": float(args.hand_max_step_rad),
            "max_steps_per_trajectory": int(args.max_steps_per_trajectory),
            "record_step_interval": int(record_interval),
            "action_source": "normalized_env_action_sent_to_env_step",
            "success_criterion": "prefer_env_native_success_else_lifted_and_goal_z",
        }
        with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(
            f"[INFO] --record-data: saving successful episodes under {record_dir} "
            f"(every {record_interval} sim step(s), success frame always included)"
        )

    T_world_arm_base = _make_T(np.array(args.arm_base_pos, dtype=np.float64), np.array(args.arm_base_euler, dtype=np.float64))
    control = VideoTeleopControl(T_world_arm_base=T_world_arm_base)

    target_dt = 1.0 / max(1e-3, float(args.fps))
    total_steps = 0
    episode_idx = 0
    success_count = 0
    record_ei = max(0, min(int(args.record_env_index), env.num_envs - 1))

    frame_idx = 0
    stable_count = 0
    arm_reached_countdown = -1
    step_in_frame = 0
    episode_sim_steps = 0
    last_recorded_episode_step = -1
    last_action_record: np.ndarray | None = None
    success_hold_count = 0
    tactile_window = max(1, int(args.tactile_gate_window))
    tactile_fn_hist: deque[float] = deque(maxlen=tactile_window)
    tactile_active_hist: deque[float] = deque(maxlen=tactile_window)

    def _append_record_row(obs: object, action_record: np.ndarray | None) -> None:
        nonlocal last_recorded_episode_step
        if not bool(args.record_data) or rec_bufs is None:
            return
        row = _extract_record_row(obs, record_ei)
        if row is None:
            print(
                f"[WARN] step={total_steps} episode_step={episode_sim_steps} "
                "obs['record'] missing; skip saving this step."
            )
            return
        if action_record is not None:
            row["action"] = np.asarray(action_record, dtype=np.float32).copy()
        if not rec_bufs:
            for k in row:
                rec_bufs[k] = []
        for k, v in row.items():
            rec_bufs.setdefault(k, []).append(np.asarray(v))
        last_recorded_episode_step = episode_sim_steps

    def _reset_episode() -> None:
        nonlocal frame_idx, stable_count, arm_reached_countdown, step_in_frame, episode_sim_steps, last_recorded_episode_step, last_action_record, success_hold_count
        env.reset()
        _set_goal_above_object()
        frame_idx = 0
        stable_count = 0
        arm_reached_countdown = -1
        step_in_frame = 0
        episode_sim_steps = 0
        last_recorded_episode_step = -1
        last_action_record = None
        success_hold_count = 0
        _clear_bufs(rec_bufs)
        tactile_fn_hist.clear()
        tactile_active_hist.clear()

    def _tactile_metrics_from_obs(obs_now: object) -> tuple[float, float]:
        row = _extract_record_row(obs_now, record_ei)
        if row is None:
            return 0.0, 0.0
        nf = row.get("tactile_normal_force", None)
        if nf is None:
            return 0.0, 0.0
        nfa = np.asarray(nf, dtype=np.float64)
        total_nf = float(np.abs(nfa).sum())
        thr = float(args.tactile_point_force_threshold)
        active_ratio = float((np.abs(nfa) > thr).mean()) if nfa.size > 0 else 0.0
        return total_nf, active_ratio

    _set_goal_above_object()

    while simulation_app.is_running():
        t0 = time.time()
        total_steps += 1

        if frame_idx >= len(frames):
            episode_idx += 1
            if args.max_episodes > 0 and episode_idx >= int(args.max_episodes):
                break
            _reset_episode()
            continue

        z_init_pre = float(_object_init_pos_env()[2])
        z_obj_pre = float(_sample_object_pos_env()[2])
        is_lifted_pre = (z_obj_pre - z_init_pre) >= float(args.grasp_lift_min_dz)
        fr = frames[frame_idx]
        marker_pos = np.asarray(fr.get("marker_pos_w", [0.65, 0.12, 0.42]), dtype=np.float64).ravel()[:3]
        marker_euler = np.asarray(fr.get("marker_euler_xyz", [0.0, 2.2, 0.0]), dtype=np.float64).ravel()[:3]
        fixed_rand = np.asarray(
            fr.get("fixed_pos_env_random", fr.get("object_pos_env_random", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        ).ravel()[:3]
        hand_j = np.asarray(fr.get("hand_joint_pos_shadow_order", [0.0] * 24), dtype=np.float64).ravel()
        fixed_cur_rand = _sample_forge_fixed_offset_env(env, record_ei)
        fixed_rand[2] = 0.0
        fixed_cur_rand[2] = 0.0
        # Keep the first snapshot (pose_001) unchanged; apply random-offset compensation from frame 2 onward.
        if frame_idx == 0:
            wrist_pos = marker_pos
        else:
            wrist_pos = marker_pos - fixed_rand + fixed_cur_rand
        targets = control.compute(wrist_pos, marker_euler, hand_j)
        if targets is None:
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            converged_now = False
            arm_reached_now = False
        else:
            des_arm = np.asarray(targets.arm_joints, dtype=np.float64).ravel()
            des_hand = np.asarray(targets.hand_joints, dtype=np.float64).ravel()
            cur_arm = _current_arm_joints()
            cur_hand = _current_hand_shadow24()
            mar = float(args.arm_max_step_rad)
            use_arm_slew = mar > 0.0 and (not bool(args.arm_slew_only_when_lifted) or is_lifted_pre)
            if use_arm_slew and des_arm.shape == cur_arm.shape:
                arm_cmd = cur_arm + np.clip(des_arm - cur_arm, -mar, mar)
            else:
                arm_cmd = des_arm
            mhr = float(args.hand_max_step_rad)
            use_hand_slew = mhr > 0.0 and des_hand.shape == cur_hand.shape
            if use_hand_slew:
                hand_cmd = cur_hand + np.clip(des_hand - cur_hand, -mhr, mhr)
            else:
                hand_cmd = des_hand
            actions = _build_action(arm_cmd, hand_cmd)
            arm_err = float(np.max(np.abs(cur_arm - des_arm)))
            hand_err = float(np.max(np.abs(cur_hand - des_hand)))
            arm_reached_now = arm_err <= float(args.arm_pos_tol)
            converged_now = (arm_err <= float(args.arm_pos_tol)) and (hand_err <= float(args.hand_pos_tol))
        if env.num_envs > 1:
            actions = actions.expand(env.num_envs, -1).clone()

        last_action_record = np.asarray(actions[record_ei].detach().cpu().numpy(), dtype=np.float32)
        obs, _rew, terminated, truncated, _extras = env.step(actions)
        t_nf, t_active = _tactile_metrics_from_obs(obs)
        tactile_fn_hist.append(t_nf)
        tactile_active_hist.append(t_active)
        episode_sim_steps += 1
        step_in_frame += 1
        if converged_now:
            stable_count += 1
        else:
            stable_count = 0
        if arm_reached_countdown < 0 and arm_reached_now:
            arm_reached_countdown = max(0, int(args.post_arm_reached_steps))
        elif arm_reached_countdown >= 0:
            arm_reached_countdown -= 1
        if (
            stable_count >= max(1, int(args.stable_steps))
            or step_in_frame >= max(1, int(args.max_steps_per_frame))
            or (arm_reached_countdown == 0)
        ):
            frame_idx += 1
            stable_count = 0
            arm_reached_countdown = -1
            step_in_frame = 0
            if frame_idx >= len(frames):
                # Log immediately when one full trajectory sequence is consumed.
                xy_dist, z_disp, height_times_success_threshold = _success_debug_values()
                xy_str = "None" if xy_dist is None else f"{xy_dist:.6f}"
                z_str = "None" if z_disp is None else f"{z_disp:.6f}"
                hst_str = "None" if height_times_success_threshold is None else f"{height_times_success_threshold:.6f}"
                print(
                    f"[Episode{episode_idx + 1}] step={total_steps} "
                    f"consumed_frames={len(frames)} episode_steps={episode_sim_steps} "
                    f"success_count={success_count} "
                    f"xy_dist={xy_str} z_disp={z_str} height_times_success_threshold={hst_str}"
                )

        if bool(args.record_data) and rec_bufs is not None:
            if episode_sim_steps % record_interval == 0:
                _append_record_row(obs, last_action_record)

        success_now, success_source = _is_success_now()
        if success_now:
            # Optional stricter gate for "visually obvious insertion" demos.
            if success_source == "env_success":
                xy_dist, z_disp, height_times_success_threshold = _success_debug_values()
                strict_xy = float(args.max_success_xy_dist)
                strict_ratio = float(args.max_success_z_disp_ratio)
                reject_reasons: list[str] = []
                if strict_xy > 0.0 and xy_dist is not None and xy_dist > strict_xy:
                    reject_reasons.append(f"xy_dist>{strict_xy:.6f}")
                if (
                    strict_ratio > 0.0
                    and z_disp is not None
                    and height_times_success_threshold is not None
                    and z_disp > strict_ratio * height_times_success_threshold
                ):
                    reject_reasons.append(
                        f"z_disp>{strict_ratio:.3f}*threshold"
                    )
                if len(reject_reasons) > 0:
                    xy_str = "None" if xy_dist is None else f"{xy_dist:.6f}"
                    z_str = "None" if z_disp is None else f"{z_disp:.6f}"
                    hst_str = "None" if height_times_success_threshold is None else f"{height_times_success_threshold:.6f}"
                    print(
                        f"[REJECT-SUCCESS] step={total_steps} episode_step={episode_sim_steps} "
                        f"xy_dist={xy_str} z_disp={z_str} threshold={hst_str} "
                        f"reasons={reject_reasons}"
                    )
                    success_now = False

        if success_now and bool(args.require_tactile_success):
            fn_gate = float(args.min_tactile_normal_total)
            act_gate = float(args.min_tactile_active_ratio)
            fn_peak = max(tactile_fn_hist) if len(tactile_fn_hist) > 0 else 0.0
            act_peak = max(tactile_active_hist) if len(tactile_active_hist) > 0 else 0.0
            reject_reasons: list[str] = []
            if fn_gate > 0.0 and fn_peak < fn_gate:
                reject_reasons.append(f"tactile_nf_peak<{fn_gate:.6f}")
            if act_gate > 0.0 and act_peak < act_gate:
                reject_reasons.append(f"tactile_active_peak<{act_gate:.6f}")
            if len(reject_reasons) > 0:
                print(
                    f"[REJECT-SUCCESS-TACTILE] step={total_steps} episode_step={episode_sim_steps} "
                    f"nf_peak={fn_peak:.6f} active_peak={act_peak:.6f} "
                    f"window={len(tactile_fn_hist)}/{tactile_window} reasons={reject_reasons}"
                )
                success_now = False

        if success_now:
            success_hold_count += 1
            req_hold = max(1, int(args.success_hold_steps))
            if success_hold_count < req_hold:
                print(
                    f"[WAIT-SUCCESS-HOLD] step={total_steps} episode_step={episode_sim_steps} "
                    f"hold={success_hold_count}/{req_hold}"
                )
                success_now = False
        else:
            success_hold_count = 0

        if success_now:
            success_count += 1
            if bool(args.record_data) and rec_bufs is not None and last_recorded_episode_step != episode_sim_steps:
                _append_record_row(obs, last_action_record)
            if success_source == "env_success":
                xy_dist, z_disp, height_times_success_threshold = _success_debug_values()
                xy_str = "None" if xy_dist is None else f"{xy_dist:.6f}"
                z_str = "None" if z_disp is None else f"{z_disp:.6f}"
                hst_str = "None" if height_times_success_threshold is None else f"{height_times_success_threshold:.6f}"
                print(
                    f"[SUCCESS] step={total_steps} episode_step={episode_sim_steps} "
                    f"xy_dist={xy_str} z_disp={z_str} height_times_success_threshold={hst_str} "
                    f"(source={success_source}, saving episode)"
                )
            else:
                z_obj = float(_sample_object_pos_env()[2])
                z_goal = float(_sample_goal_pos_env()[2])
                print(
                    f"[SUCCESS] step={total_steps} episode_step={episode_sim_steps} z_obj={z_obj:.4f} z_goal={z_goal:.4f} "
                    f"(source={success_source}, saving episode)"
                )
            if bool(args.record_data) and rec_bufs is not None and record_dir is not None:
                path, t_len = _save_episode_npz(record_dir, f"episode_{success_count - 1:04d}.npz", rec_bufs)
                if path is not None:
                    print(f"[INFO] recorded successful episode {path} (T={t_len}, record_step_interval={record_interval})")
                else:
                    print(
                        "[WARN] success reached but nothing was saved "
                        "(record buffer empty; check observation record stream)."
                    )
            else:
                _clear_bufs(rec_bufs)
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
                f"(episode_sim_steps={episode_sim_steps}), failing episode."
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
    print(f"[INFO] play finished: total_steps={total_steps}, episodes={episode_idx}, success={success_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

