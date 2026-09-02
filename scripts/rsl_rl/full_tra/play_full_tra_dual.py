#!/usr/bin/env python3
"""Play dual-arm full trajectory json and optionally dump record npz."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R

_FULL_TRA_DIR = Path(__file__).resolve().parent
if str(_FULL_TRA_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_TRA_DIR))
from full_tra_task_entries_dual import resolve_env_cfg_entries


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


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)


def _quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).ravel()
    if q.size != 4:
        return np.zeros(3, dtype=np.float64)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_euler("xyz")


def _root_T_world_np(robot, env_idx: int) -> np.ndarray:
    pos = robot.data.root_pos_w[int(env_idx)].detach().cpu().numpy().ravel()[:3]
    q = robot.data.root_quat_w[int(env_idx)].detach().cpu().numpy().ravel()[:4]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_matrix()
    t[:3, 3] = pos
    return t


def _urdf_path(repo_root: Path, side: str) -> str:
    side = side.lower()
    return str(
        repo_root
        / "source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e"
        / f"ur10e_shadow_{side}_hand_glb.urdf"
    )


def _extract_record_row(obs: object, env_idx: int) -> dict[str, np.ndarray] | None:
    rec = obs.get("record") if isinstance(obs, dict) else None
    if not isinstance(rec, dict):
        return None
    out: dict[str, np.ndarray] = {}
    for k, v in rec.items():
        if torch.is_tensor(v):
            out[k] = np.asarray(v[max(0, min(env_idx, int(v.shape[0]) - 1))].detach().cpu().numpy())
    return out or None


def _save_npz(out_dir: str, name: str, bufs: dict[str, list[np.ndarray]]):
    if not bufs:
        return None, 0
    first = next(iter(bufs.keys()))
    if not bufs[first]:
        return None, 0
    path = os.path.join(out_dir, name)
    t_len = len(bufs[first])
    np.savez_compressed(path, **{k: np.stack(v, 0) for k, v in bufs.items() if v})
    return path, t_len


def _T(pos, euler):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = R.from_euler("xyz", np.asarray(euler, dtype=np.float64).ravel()[:3], degrees=False).as_matrix()
    t[:3, 3] = np.asarray(pos, dtype=np.float64).ravel()[:3]
    return t


def _parser():
    p = argparse.ArgumentParser(description="Play dual full trajectory and record env outputs.")
    p.add_argument("--trajectory-json", "--keyframe-json", dest="trajectory_json", type=str, required=True)
    p.add_argument("--task", type=str, default="")
    p.add_argument("--env", type=str, default="")
    p.add_argument("--cfg", type=str, default="")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--object-init-index", type=int, default=0)
    p.add_argument("--object-xy-noise", type=float, default=0.0, help="Gaussian XY noise (m) added to marker targets.")
    p.add_argument("--hand-noise", type=float, default=0.0, help="Gaussian noise (rad) added to hand targets.")
    p.add_argument("--arm-base-right-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-base-right-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-base-left-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-base-left-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-pos-tol", type=float, default=0.05)
    p.add_argument("--hand-pos-tol", type=float, default=0.08)
    p.add_argument("--stable-steps", type=int, default=12)
    p.add_argument("--max-steps-per-frame", type=int, default=180)
    p.add_argument("--post-arm-reached-steps", type=int, default=0, help="Extra settle steps after each frame converges.")
    p.add_argument("--action-smoothing", type=float, default=0.75)
    p.add_argument("--record-step-interval", type=int, default=1)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--record-env-index", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args = _parser().parse_args()
    app = AppLauncher(args).app
    import ViTacLab.tasks  # noqa: F401

    repo = _repo_root()
    src = repo / "source"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import VideoTeleopControl

    traj_path = Path(args.trajectory_json).expanduser()
    if not traj_path.is_absolute():
        traj_path = (repo / traj_path).resolve()
    doc = json.loads(traj_path.read_text(encoding="utf-8"))
    frames = list(doc.get("frames") or [])
    if not frames:
        raise RuntimeError("No frames in trajectory json.")

    task = str(args.task).strip() or str(doc.get("task", "")).strip()
    env_override = str(args.env).strip()
    cfg_override = str(args.cfg).strip()
    env_entry_doc = str(doc.get("env_entry", "")).strip()
    cfg_entry_doc = str(doc.get("cfg_entry", "")).strip()
    if env_override and cfg_override:
        env_entry, cfg_entry, _ = resolve_env_cfg_entries(task=task, env=env_override, cfg=cfg_override)
    elif env_entry_doc and cfg_entry_doc:
        env_entry, cfg_entry = env_entry_doc, cfg_entry_doc
    else:
        env_entry, cfg_entry, _ = resolve_env_cfg_entries(task=task, env="", cfg="")

    env_cls = _load_symbol(env_entry)
    cfg_cls = _load_symbol(cfg_entry)

    cfg = cfg_cls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    if hasattr(cfg, "object_init_choice"):
        cfg.object_init_choice = int(args.object_init_index if args.object_init_index >= 0 else doc.get("object_init_index", 0))
    setattr(cfg, "enable_cameras", True)

    env = env_cls(cfg)
    env.reset()
    right = env.right_hand
    left = env.left_hand
    env_idx = max(0, min(int(args.record_env_index), env.num_envs - 1))

    joint_names = list(right.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_ids = sorted([i for i, n in enumerate(joint_names) if re.match(arm_expr, n)])
    hand_ids = sorted([i for i, n in enumerate(joint_names) if re.match(hand_expr, n)])
    sh = shadowhand_joint_names()
    act_dim = len(env.actuated_dof_indices)
    right_base_offset = _T(args.arm_base_right_pos, args.arm_base_right_euler)
    left_base_offset = _T(args.arm_base_left_pos, args.arm_base_left_euler)

    control_right = VideoTeleopControl(
        urdf_path=_urdf_path(repo, "right"), T_world_arm_base=_root_T_world_np(right, env_idx) @ right_base_offset
    )
    control_left = VideoTeleopControl(
        urdf_path=_urdf_path(repo, "left"), T_world_arm_base=_root_T_world_np(left, env_idx) @ left_base_offset
    )

    def hand_for_name(name: str, hv: np.ndarray) -> float:
        for si, sn in enumerate(sh):
            if sn in name or name.endswith(sn):
                return float(hv[si])
        return 0.0

    def cur_arm(robot: Any):
        q = _np(robot.data.joint_pos[env_idx])
        return np.array([float(q[i]) for i in arm_ids], dtype=np.float64)

    def cur_hand(robot: Any):
        out = np.zeros(24, dtype=np.float64)
        q = _np(robot.data.joint_pos[env_idx])
        for si, sn in enumerate(sh):
            for idx in hand_ids:
                if sn in joint_names[idx] or joint_names[idx].endswith(sn):
                    out[si] = float(q[idx])
                    break
        return out

    def build_action(robot: Any, arm_j: np.ndarray, hand_j: np.ndarray) -> torch.Tensor:
        full = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_ids):
            if i < len(arm_j):
                full[idx] = float(arm_j[i])
        for idx in hand_ids:
            full[idx] = hand_for_name(joint_names[idx], hand_j)
        act = full[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lo = _np(env.robot_dof_lower_limits[0, env.actuated_dof_indices])
        hi = _np(env.robot_dof_upper_limits[0, env.actuated_dof_indices])
        scale = np.where(hi - lo > 1e-6, 2.0 * (act - lo) / (hi - lo) - 1.0, 0.0)
        return torch.tensor(np.clip(scale, -1.0, 1.0), dtype=torch.float32, device=env.device).unsqueeze(0)

    record_dir = (
        os.path.abspath(os.path.expanduser(args.record_path))
        if args.record_path
        else os.path.join(os.getcwd(), "play_records", f"dual_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    )
    os.makedirs(record_dir, exist_ok=True)
    with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "trajectory_json": str(traj_path),
                "task": task,
                "env_entry": env_entry,
                "cfg_entry": cfg_entry,
                "object_init_index": int(getattr(cfg, "object_init_choice", 0)),
                "object_xy_noise": float(args.object_xy_noise),
                "hand_noise": float(args.hand_noise),
                "num_frames": len(frames),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    target_dt = 1.0 / max(1e-3, float(args.fps))
    success_eps = 0
    for ep in range(int(args.num_episodes)):
        env.reset()
        buffers: dict[str, list[np.ndarray]] = {}
        prev_actions = {
            "right_hand": torch.zeros(env.num_envs, act_dim, device=env.device),
            "left_hand": torch.zeros(env.num_envs, act_dim, device=env.device),
        }
        fi = 0
        stable = 0
        sif = 0
        post_reached = 0
        ep_steps = 0
        done_ok = True
        while app.is_running() and fi < len(frames):
            t0 = time.time()
            fr = frames[fi]

            control_right.T_world_arm_base = _root_T_world_np(right, env_idx) @ right_base_offset
            control_left.T_world_arm_base = _root_T_world_np(left, env_idx) @ left_base_offset

            pos_r = np.asarray(fr.get("marker_right_pos_w", []), dtype=np.float64).ravel()[:3]
            pos_l = np.asarray(fr.get("marker_left_pos_w", []), dtype=np.float64).ravel()[:3]
            euler_r = np.asarray(fr.get("marker_right_euler_xyz", []), dtype=np.float64).ravel()[:3]
            euler_l = np.asarray(fr.get("marker_left_euler_xyz", []), dtype=np.float64).ravel()[:3]
            if euler_r.size != 3:
                q = np.asarray(fr.get("marker_right_quat_wxyz", []), dtype=np.float64).ravel()[:4]
                euler_r = _quat_wxyz_to_euler_xyz(q)
            if euler_l.size != 3:
                q = np.asarray(fr.get("marker_left_quat_wxyz", []), dtype=np.float64).ravel()[:4]
                euler_l = _quat_wxyz_to_euler_xyz(q)
            if float(args.object_xy_noise) > 0.0:
                pos_r[:2] = pos_r[:2] + np.random.normal(0.0, float(args.object_xy_noise), size=2)
                pos_l[:2] = pos_l[:2] + np.random.normal(0.0, float(args.object_xy_noise), size=2)

            h_r = np.asarray(fr.get("hand_right_joint_pos_shadow_order", []), dtype=np.float64).ravel()[:24]
            h_l = np.asarray(fr.get("hand_left_joint_pos_shadow_order", []), dtype=np.float64).ravel()[:24]
            if h_r.size != 24:
                h_r = cur_hand(right)
            if h_l.size != 24:
                h_l = cur_hand(left)
            if float(args.hand_noise) > 0.0:
                h_r = h_r + np.random.normal(0.0, float(args.hand_noise), size=h_r.shape)
                h_l = h_l + np.random.normal(0.0, float(args.hand_noise), size=h_l.shape)

            tgt_r = control_right.compute(pos_r, euler_r, h_r)
            tgt_l = control_left.compute(pos_l, euler_l, h_l)

            arm_r = cur_arm(right) if tgt_r is None else np.asarray(tgt_r.arm_joints, dtype=np.float64)
            arm_l = cur_arm(left) if tgt_l is None else np.asarray(tgt_l.arm_joints, dtype=np.float64)
            act_r = build_action(right, arm_r, h_r)
            act_l = build_action(left, arm_l, h_l)

            smooth = float(np.clip(args.action_smoothing, 0.0, 0.999))
            actions = {
                "right_hand": smooth * prev_actions["right_hand"] + (1.0 - smooth) * act_r.expand(env.num_envs, -1).clone(),
                "left_hand": smooth * prev_actions["left_hand"] + (1.0 - smooth) * act_l.expand(env.num_envs, -1).clone(),
            }
            prev_actions = {"right_hand": actions["right_hand"].clone(), "left_hand": actions["left_hand"].clone()}

            out = env.step(actions)
            obs, term, trunc = (out[0], out[2], out[3]) if isinstance(out, tuple) and len(out) >= 5 else (out, None, None)
            ep_steps += 1
            sif += 1

            if ep_steps % max(1, int(args.record_step_interval)) == 0:
                row = _extract_record_row(obs, env_idx)
                if row is None:
                    row = {
                        "joint_pos_right": _np(right.data.joint_pos[env_idx]),
                        "joint_pos_left": _np(left.data.joint_pos[env_idx]),
                        "action_right": _np(actions["right_hand"][env_idx]),
                        "action_left": _np(actions["left_hand"][env_idx]),
                    }
                if not buffers:
                    for k in row:
                        buffers[k] = []
                    buffers["task_episode_sim_step"] = []
                for k, v in row.items():
                    buffers.setdefault(k, []).append(np.asarray(v))
                buffers["task_episode_sim_step"].append(np.asarray(float(ep_steps), dtype=np.float32))

            ae_r = float(np.max(np.abs(cur_arm(right) - arm_r)))
            ae_l = float(np.max(np.abs(cur_arm(left) - arm_l)))
            he_r = float(np.max(np.abs(cur_hand(right) - h_r)))
            he_l = float(np.max(np.abs(cur_hand(left) - h_l)))
            conv = (ae_r <= float(args.arm_pos_tol)) and (ae_l <= float(args.arm_pos_tol)) and (he_r <= float(args.hand_pos_tol)) and (he_l <= float(args.hand_pos_tol))
            stable = stable + 1 if conv else 0
            if stable >= max(1, int(args.stable_steps)):
                if post_reached < max(0, int(args.post_arm_reached_steps)):
                    post_reached += 1
                else:
                    fi += 1
                    stable = 0
                    sif = 0
                    post_reached = 0
            elif sif >= max(1, int(args.max_steps_per_frame)):
                fi += 1
                stable = 0
                sif = 0
                post_reached = 0

            if torch.is_tensor(term) and torch.is_tensor(trunc) and bool(torch.any(term | trunc).item()):
                done_ok = False
                env.reset()
                stable = 0
                sif = 0
                post_reached = 0

            dt = target_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

        if fi >= len(frames) and done_ok:
            success_eps += 1
            p, t_len = _save_npz(record_dir, f"episode_{ep:04d}_success.npz", buffers)
            print(f"[INFO] saved success -> {p} (T={t_len})")
        else:
            p, t_len = _save_npz(record_dir, f"episode_{ep:04d}_fail.npz", buffers)
            print(f"[INFO] saved fail -> {p} (T={t_len})")
        if not app.is_running():
            break

    print(f"[INFO] done: {success_eps}/{int(args.num_episodes)} episodes reached final frame")
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
