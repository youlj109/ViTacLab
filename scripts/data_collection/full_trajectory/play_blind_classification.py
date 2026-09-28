#!/usr/bin/env python3
"""Replay Blind Classification keyframes and record tactile + camera NPZ data.

Blind Classification uses 33-dim actions (30 robot + 3 class logits). Open-loop
teleop replay pads class logits; on Finish* keyframes the correct label logit is
asserted when ``--emit-class-logits`` is enabled (default).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common_record_utils import extract_canonical_record_row, extract_success_signal, merge_record_row
from play_single_phase import (
    _T,
    _load_symbol,
    _np,
    _read_tactile_from_scene,
    _repo_root,
    _reset_and_warmup,
    _save_npz,
)
from task_entries_single import resolve_env_cfg_entries

DEFAULT_SEQUENCE = [
    "Start01",
    "Start02",
    "Touch01",
    "Touch01_01",
    "Touch01_02",
    "Touch02_01",
    "Touch02_02",
    "Pick02_01",
    "Pick02_02",
    "Pick02_03",
    "Finish02",
]
DEFAULT_KEYFRAME = (
    "UR10eShadowHandBlindClassificationEnv__UR10eShadowHandBlindClassificationEnvCfg.json"
)
SHAPE_NAMES = ("cube", "sphere", "cone")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play Blind Classification keyframe sequence and record NPZ.")
    p.add_argument("--keyframe-json", type=str, default="", help="Keyframe JSON path override.")
    p.add_argument(
        "--sequence",
        type=str,
        default="",
        help="Comma-separated keyframe names; default is the canonical classification demo sequence.",
    )
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument(
        "--shape-label",
        type=int,
        default=0,
        help="Fixed manipuland shape for the episode: 0=cube, 1=sphere, 2=cone. Use -1 for random.",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for hand noise / env reset.")
    p.add_argument("--hand-noise", type=float, default=0.02)
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-pos-tol", type=float, default=0.05)
    p.add_argument("--hand-pos-tol", type=float, default=0.08)
    p.add_argument("--stable-steps", type=int, default=15)
    p.add_argument("--max-steps-per-frame", type=int, default=240)
    p.add_argument("--post-arm-reached-steps", type=int, default=30)
    p.add_argument("--action-smoothing", type=float, default=0.75)
    p.add_argument("--record-step-interval", type=int, default=1)
    p.add_argument("--tactile-reset-settle-steps", type=int, default=8)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--record-env-index", type=int, default=0)
    p.add_argument(
        "--emit-class-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On Finish* keyframes, append one-hot class logits matching the spawned shape label.",
    )
    p.add_argument(
        "--save-outcome",
        choices=("success", "completed", "all"),
        default="completed",
    )
    AppLauncher.add_app_launcher_args(p)
    return p


def _active_object_z(env, ei: int) -> float:
    if hasattr(env, "shape_labels") and hasattr(env, "_shape_objects"):
        label = int(env.shape_labels[ei].item())
        obj = env._shape_objects[label]
        return float((obj.data.root_pos_w[ei] - env.scene.env_origins[ei])[2].item())
    if hasattr(env, "object") and env.object is not None:
        return float((env.object.data.root_pos_w[ei] - env.scene.env_origins[ei])[2].item())
    return 0.0


def main() -> int:
    args = _parser().parse_args()
    app = AppLauncher(args).app
    import ViTacLab.tasks  # noqa: F401

    repo = _repo_root()
    src = repo / "source"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if args.seed is not None:
        seed = int(args.seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import VideoTeleopControl

    sequence = (
        [s.strip() for s in str(args.sequence).split(",") if s.strip()]
        if str(args.sequence).strip()
        else list(DEFAULT_SEQUENCE)
    )
    kf_path = Path(args.keyframe_json).expanduser() if args.keyframe_json else HERE / "pose_keyframes" / DEFAULT_KEYFRAME
    if not kf_path.is_absolute():
        kf_path = (repo / kf_path).resolve()
    doc = json.loads(kf_path.read_text(encoding="utf-8"))
    kfs = {str(k["name"]): k for k in doc.get("keyframes", [])}
    miss = [n for n in sequence if n not in kfs]
    if miss:
        raise RuntimeError(f"Missing keyframes: {miss}")
    frames = [kfs[n] for n in sequence]
    frame_names = list(sequence)

    env_entry, cfg_entry, _ = resolve_env_cfg_entries(task="blind_classification")
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)
    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    cfg.enable_cameras = True
    if hasattr(cfg, "goal_resample_time_range_s"):
        cfg.goal_resample_time_range_s = (1.0e9, 1.0e9)
    if hasattr(cfg, "fixed_shape_label"):
        cfg.fixed_shape_label = int(args.shape_label)

    env = EnvCls(cfg)
    act_dim = env.num_actions
    _reset_and_warmup(env, act_dim=act_dim, settle_steps=args.tactile_reset_settle_steps)

    robot = env.robot
    jn = list(robot.joint_names)
    sh = shadowhand_joint_names()
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_ids = [i for i, n in enumerate(jn) if re.match(arm_expr, n)]
    hand_ids = [i for i, n in enumerate(jn) if re.match(hand_expr, n)]
    lo_all = _np(env.robot_dof_lower_limits[0])
    hi_all = _np(env.robot_dof_upper_limits[0])
    sh_lo, sh_hi = np.full(24, -1.5), np.full(24, 1.5)
    for si, sn in enumerate(sh):
        for idx in hand_ids:
            if sn in jn[idx] or jn[idx].endswith(sn):
                sh_lo[si], sh_hi[si] = lo_all[idx], hi_all[idx]
                break

    robot_action_dim = 30
    class_dim = max(0, act_dim - robot_action_dim)

    def hand_for_name(name, hv):
        for si, sn in enumerate(sh):
            if sn in name or name.endswith(sn):
                return float(hv[si])
        return 0.0

    def cur_arm():
        q = _np(robot.data.joint_pos[0])
        return np.array([float(q[i]) for i in arm_ids], dtype=np.float64)

    def cur_hand():
        out = np.zeros(24, dtype=np.float64)
        q = _np(robot.data.joint_pos[0])
        for si, sn in enumerate(sh):
            for idx in hand_ids:
                if sn in jn[idx] or jn[idx].endswith(sn):
                    out[si] = float(q[idx])
                    break
        return out

    def class_logits_for_frame(frame_name: str, ei: int) -> np.ndarray:
        logits = np.zeros(class_dim, dtype=np.float64)
        if class_dim <= 0 or not args.emit_class_logits:
            return logits
        if not (frame_name.startswith("Finish") or frame_name.startswith("Pick02_03")):
            return logits
        label = int(env.shape_labels[ei].item())
        if 0 <= label < class_dim:
            logits[label] = 1.0
        return logits

    def build_action(arm_j, hand_j, frame_name: str, ei: int):
        full = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_ids):
            if i < len(arm_j):
                full[idx] = float(arm_j[i])
        for idx in hand_ids:
            full[idx] = hand_for_name(jn[idx], hand_j)
        act = full[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lo = _np(env.robot_dof_lower_limits[0, env.actuated_dof_indices])
        hi = _np(env.robot_dof_upper_limits[0, env.actuated_dof_indices])
        robot_scaled = np.where(hi - lo > 1e-6, 2.0 * (act - lo) / (hi - lo) - 1.0, 0.0)
        robot_scaled = np.clip(robot_scaled[:robot_action_dim], -1.0, 1.0)
        if class_dim > 0:
            logits = class_logits_for_frame(frame_name, ei)
            scaled = np.concatenate([robot_scaled, logits], axis=0)
        else:
            scaled = robot_scaled
        return torch.tensor(scaled, dtype=torch.float32, device=env.device).unsqueeze(0)

    control = VideoTeleopControl(T_world_arm_base=_T(np.array(args.arm_base_pos), np.array(args.arm_base_euler)))
    action_smoothing = float(np.clip(args.action_smoothing, 0.0, 0.999))
    rec_dir = (
        os.path.abspath(os.path.expanduser(args.record_path))
        if args.record_path
        else os.path.join(
            os.getcwd(),
            f"play_records/blind_classification_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    )
    os.makedirs(rec_dir, exist_ok=True)
    shape_name = SHAPE_NAMES[int(args.shape_label)] if 0 <= int(args.shape_label) < len(SHAPE_NAMES) else "random"
    with open(os.path.join(rec_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "blind_classification",
                "sequence": sequence,
                "shape_label": int(args.shape_label),
                "shape_name": shape_name,
                "hand_noise": float(args.hand_noise),
                "emit_class_logits": bool(args.emit_class_logits),
                "keyframe_json": str(kf_path),
                "save_outcome": str(args.save_outcome),
            },
            f,
            indent=2,
        )

    ei = max(0, min(int(args.record_env_index), env.num_envs - 1))
    target_dt = 1.0 / max(1e-3, float(args.fps))
    succ = 0

    for ep in range(int(args.num_episodes)):
        _reset_and_warmup(env, act_dim=act_dim, settle_steps=args.tactile_reset_settle_steps)
        bufs: dict[str, list[np.ndarray]] = {}
        ep_steps = 0
        success_latched = False
        success_signal_seen = False
        success_source = "unavailable"
        prev_actions = torch.zeros(env.num_envs, act_dim, device=env.device)
        noisy_hands = [
            np.clip(
                np.asarray(fr["hand_joint_pos_shadow_order"], dtype=np.float64)[:24]
                + np.random.uniform(-float(args.hand_noise), float(args.hand_noise), size=24),
                sh_lo,
                sh_hi,
            )
            for fr in frames
        ]
        fi = 0
        stable = 0
        countdown = -1
        sif = 0
        label_now = int(env.shape_labels[ei].item())
        print(f"[INFO] episode {ep}: shape_label={label_now} ({SHAPE_NAMES[label_now]})")

        while app.is_running() and fi < len(frames):
            t0 = time.time()
            fr = frames[fi]
            frame_name = frame_names[fi]
            marker_pos = np.asarray(fr["marker_pos_w"], dtype=np.float64)[:3]
            marker_euler = np.asarray(fr["marker_euler_xyz"], dtype=np.float64)[:3]
            hand_j = noisy_hands[fi]
            tgt = control.compute(marker_pos, marker_euler, hand_j)
            if tgt is None:
                actions = torch.zeros(env.num_envs, act_dim, device=env.device)
                conv = False
                arm_ok = False
            else:
                actions = build_action(np.asarray(tgt.arm_joints, dtype=np.float64), hand_j, frame_name, ei)
                ae = float(np.max(np.abs(cur_arm() - np.asarray(tgt.arm_joints, dtype=np.float64))))
                he = float(np.max(np.abs(cur_hand() - hand_j)))
                arm_ok = ae <= float(args.arm_pos_tol)
                conv = arm_ok and he <= float(args.hand_pos_tol)
            if env.num_envs > 1:
                actions = actions.expand(env.num_envs, -1).clone()
            actions = action_smoothing * prev_actions + (1.0 - action_smoothing) * actions
            prev_actions = actions.clone()
            out = env.step(actions)
            if isinstance(out, tuple) and len(out) >= 5:
                obs, term, trunc, infos = out[0], out[2], out[3], out[4]
            elif isinstance(out, tuple) and len(out) >= 4:
                obs, term, trunc, infos = out[0], out[2], out[3], {}
            else:
                obs, term, trunc, infos = out, None, None, {}
            ep_steps += 1
            if ep_steps % max(1, int(args.record_step_interval)) == 0:
                row = extract_canonical_record_row(obs, env, ei) or {}
                for key, value in merge_record_row(obs, ei, actions=actions, robot=robot).items():
                    row.setdefault(key, value)
                for key, value in _read_tactile_from_scene(env, ei).items():
                    row.setdefault(key, value)
                if actions is not None:
                    row["action"] = _np(actions[ei])
                if not bufs:
                    for k in row:
                        bufs[k] = []
                    bufs["task_episode_sim_step"] = []
                    bufs["shape_label"] = []
                for k, v in row.items():
                    bufs.setdefault(k, []).append(np.asarray(v))
                bufs["task_episode_sim_step"].append(np.asarray(float(ep_steps), dtype=np.float32))
                bufs["shape_label"].append(np.asarray(float(env.shape_labels[ei].item()), dtype=np.float32))
            signal_available, signal_success, signal_source_now = extract_success_signal(infos, obs, env, ei)
            if signal_available:
                success_signal_seen = True
                success_latched = success_latched or signal_success
                success_source = signal_source_now
            sif += 1
            stable = stable + 1 if conv else 0
            if countdown < 0 and arm_ok and not conv:
                countdown = max(0, int(args.post_arm_reached_steps))
            elif countdown >= 0 and not conv:
                countdown -= 1
            if (
                stable >= max(1, int(args.stable_steps))
                or sif >= max(1, int(args.max_steps_per_frame))
                or (arm_ok and not conv and countdown == 0)
            ):
                fi += 1
                stable = 0
                countdown = -1
                sif = 0
            if torch.is_tensor(term) and torch.is_tensor(trunc) and bool(torch.any(term | trunc).item()):
                _reset_and_warmup(env, act_dim=act_dim, settle_steps=args.tactile_reset_settle_steps)
                prev_actions.zero_()
                stable = 0
                countdown = -1
                sif = 0
                continue
            dt = target_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

        final_success = bool(success_latched if success_signal_seen else False)
        if not success_signal_seen:
            success_source = "classification_logits_unavailable"
        completed = fi >= len(frames)
        outcome = "success" if final_success else ("completed" if completed else "done")
        should_save = (
            outcome == "success"
            or (outcome == "completed" and args.save_outcome in ("completed", "all"))
            or (outcome == "done" and args.save_outcome == "all")
        )
        if outcome == "success":
            succ += 1
        if should_save:
            t_rows = len(bufs.get("task_episode_sim_step", []))
            if t_rows > 0:
                bufs["outcome_success"] = [np.asarray(final_success, dtype=np.bool_) for _ in range(t_rows)]
            fname = f"episode_{ep:04d}_{shape_name}_{outcome}.npz"
            p, t_len = _save_npz(rec_dir, fname, bufs)
            print(f"[INFO] saved {outcome} -> {p} (T={t_len}, success_source={success_source})")
        else:
            print(f"[INFO] skipped {outcome} episode {ep:04d} (success_source={success_source})")
        if not app.is_running():
            break

    print(f"[INFO] done: {succ}/{int(args.num_episodes)} success")
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
