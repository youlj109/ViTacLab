#!/usr/bin/env python3
"""Replay Blind Retrieval keyframes and record tactile + camera NPZ data.

Blind Retrieval uses 30-dim robot actions. The environment samples a target shape
(cube/sphere/cone) each reset; open-loop keyframes follow the Pick02 branch to
grasp the commanded object and move it toward the goal marker.
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
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common_record_utils import extract_canonical_record_row, extract_success_signal, merge_record_row
from play_single_phase import (
    _T,
    _boost_hand_close,
    _is_pick_close_frame,
    _load_symbol,
    _min_fingertip_dist_to_object,
    _np,
    _palm_pos_env,
    _pick_close_frame,
    _read_tactile_from_scene,
    _repo_root,
    _reset_and_warmup,
    _save_npz,
    _settle_after_object_write,
    _tactile_contact_strength,
    _use_ik_for_frame,
    _wrist_pos_for_frame,
)
from task_entries_single import resolve_env_cfg_entries

DEFAULT_SEQUENCE = [
    "Start01",
    "Start02",
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
    "UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json"
)
TARGET_NAMES = ("cube", "sphere", "cone")


def _target_object(env, ei: int):
    tid = int(env.target_object_id[ei].item())
    return env._shape_objects[tid]


def _target_pos_env(env, ei: int) -> np.ndarray:
    obj = _target_object(env, ei)
    return (obj.data.root_pos_w[ei] - env.scene.env_origins[ei]).detach().cpu().numpy()


def _write_target_pos_env(env, ei: int, pos_env: np.ndarray) -> None:
    obj = _target_object(env, ei)
    pos = np.asarray(pos_env, dtype=np.float64).ravel()[:3]
    pos_w = torch.tensor(pos, device=env.device, dtype=torch.float32).view(1, 3) + env.scene.env_origins[ei : ei + 1]
    state = obj.data.root_state_w[ei : ei + 1].clone()
    state[0, :3] = pos_w[0]
    state[0, 7:] = 0.0
    obj.write_root_state_to_sim(state)


def _align_target_to_keyframe(
    env,
    ei: int,
    frame: dict[str, Any] | None,
    *,
    offset_env: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    if frame is None or not frame.get("object_pos_env"):
        return
    pos = np.asarray(frame["object_pos_env"], dtype=np.float64).ravel()[:3].copy()
    pos += np.asarray(offset_env, dtype=np.float64).ravel()[:3]
    _write_target_pos_env(env, ei, pos)


def _follow_target_with_palm(env, ei: int, palm_offset_env: np.ndarray) -> None:
    _write_target_pos_env(env, ei, _palm_pos_env(env, ei) + np.asarray(palm_offset_env, dtype=np.float64).ravel()[:3])


def _min_fingertip_dist_to_target(env, ei: int) -> float:
    saved = env.object
    try:
        env.object = _target_object(env, ei)
        return _min_fingertip_dist_to_object(env, ei)
    finally:
        env.object = saved


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play Blind Retrieval keyframe sequence and record NPZ.")
    p.add_argument("--keyframe-json", type=str, default="", help="Keyframe JSON path override.")
    p.add_argument(
        "--sequence",
        type=str,
        default="",
        help="Comma-separated keyframe names; default is the canonical retrieval demo sequence.",
    )
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument(
        "--target-label",
        type=int,
        default=0,
        help="Fixed retrieval target: 0=cube, 1=sphere, 2=cone. Use -1 for env random sample.",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for hand noise / env reset.")
    p.add_argument("--hand-noise", type=float, default=0.0)
    p.add_argument("--success-z-threshold", type=float, default=0.4, help="Target object lift height fallback (m).")
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-pos-tol", type=float, default=0.05)
    p.add_argument("--hand-pos-tol", type=float, default=0.08)
    p.add_argument("--stable-steps", type=int, default=15)
    p.add_argument("--max-steps-per-frame", type=int, default=240)
    p.add_argument("--post-arm-reached-steps", type=int, default=30)
    p.add_argument("--pick-dwell-steps", type=int, default=120)
    p.add_argument("--action-smoothing", type=float, default=0.0)
    p.add_argument("--record-step-interval", type=int, default=1)
    p.add_argument("--tactile-reset-settle-steps", type=int, default=8)
    p.add_argument("--object-settle-steps", type=int, default=60)
    p.add_argument(
        "--use-recorded-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer keyframe arm_joint_pos over marker IK.",
    )
    p.add_argument(
        "--snap-at-pick-close",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard-set robot joints to Pick* targets on first step of each Pick frame.",
    )
    p.add_argument(
        "--pick-object-align-offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(-0.018, 0.0, -0.018),
        help="Env-frame offset when re-aligning target object at Pick*_03 entry.",
    )
    p.add_argument("--pick-hand-close-scale", type=float, default=1.35)
    p.add_argument("--pick-wrist-z-above-object", type=float, default=0.13)
    p.add_argument(
        "--tactile-contact-threshold",
        type=float,
        default=1.0e-4,
        help="At Pick*_03, require peak tactile normal force >= this before advancing (0=disable gate).",
    )
    p.add_argument(
        "--pick-geom-contact-dist",
        type=float,
        default=0.0,
        help="At Pick*_03, treat fingertip distance <= this (m) as geometry contact when tactile is silent (0=off).",
    )
    p.add_argument(
        "--carry-object-after-grasp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After Pick*_03 contact, attach target object to palm during Finish* (debug only; off for ViTacBench).",
    )
    p.add_argument(
        "--bench-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ViTacBench: no palm-carry, tactile-only pick contact, success needs tactile evidence.",
    )
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--record-env-index", type=int, default=0)
    p.add_argument(
        "--save-outcome",
        choices=("success", "completed", "all"),
        default="success",
    )
    AppLauncher.add_app_launcher_args(p)
    return p


def _apply_target_label(env, ei: int, label: int) -> int:
    if int(label) < 0:
        return int(env.target_object_id[ei].item())
    tid = int(label) % len(TARGET_NAMES)
    env.target_object_id[ei] = tid
    if hasattr(env, "_compute_intermediate_values"):
        env._compute_intermediate_values()
    return tid


def _min_dwell_steps(frame: dict[str, Any], args) -> int:
    name = str(frame.get("name", ""))
    if _is_pick_close_frame(name):
        return max(int(args.pick_dwell_steps), int(args.post_arm_reached_steps) + 20)
    if "Finish" in name:
        return max(25, int(args.post_arm_reached_steps))
    if "Pick" in name:
        return max(15, int(args.post_arm_reached_steps) // 2)
    return 0


def _max_steps_for_frame(frame: dict[str, Any], args) -> int:
    base = max(1, int(args.max_steps_per_frame))
    name = str(frame.get("name", ""))
    if _is_pick_close_frame(name):
        return max(base, int(args.pick_dwell_steps) + 600)
    if "Pick" in name or "Finish" in name:
        return max(base, 240)
    return base


def main() -> int:
    args = _parser().parse_args()
    bench_mode = bool(args.bench_mode)
    if bench_mode:
        args.carry_object_after_grasp = False
        if float(args.tactile_contact_threshold) <= 0.0:
            args.tactile_contact_threshold = 1.0e-4
        if float(args.pick_geom_contact_dist) > 0.0:
            args.pick_geom_contact_dist = 0.0
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

    env_entry, cfg_entry, _ = resolve_env_cfg_entries(task="blind_retrieval")
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)
    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    cfg.enable_cameras = True
    if hasattr(cfg, "goal_resample_time_range_s"):
        cfg.goal_resample_time_range_s = (1.0e9, 1.0e9)
    if hasattr(cfg, "episode_length_s"):
        cfg.episode_length_s = max(float(cfg.episode_length_s), 300.0)
    if hasattr(cfg, "act_moving_average"):
        cfg.act_moving_average = 1.0

    env = EnvCls(cfg)
    if hasattr(env, "_use_rl_control"):
        env._use_rl_control = False
    act_dim = env.num_actions
    _reset_and_warmup(
        env,
        act_dim=act_dim,
        settle_steps=args.tactile_reset_settle_steps,
        object_settle_steps=args.object_settle_steps,
    )

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

    def build_actuated_joint_pos(arm_j, hand_j):
        full = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_ids):
            if i < len(arm_j):
                full[idx] = float(arm_j[i])
        for idx in hand_ids:
            full[idx] = hand_for_name(jn[idx], hand_j)
        return full[np.array(env.actuated_dof_indices, dtype=np.int64)]

    def build_action(arm_j, hand_j):
        actuated = build_actuated_joint_pos(arm_j, hand_j)
        lo = _np(env.robot_dof_lower_limits[0, env.actuated_dof_indices])
        hi = _np(env.robot_dof_upper_limits[0, env.actuated_dof_indices])
        scaled = np.where(hi - lo > 1e-6, 2.0 * (actuated - lo) / (hi - lo) - 1.0, 0.0)
        return torch.tensor(np.clip(scaled, -1.0, 1.0), dtype=torch.float32, device=env.device).unsqueeze(0)

    def _sample_target_pos_env() -> np.ndarray:
        return _target_pos_env(env, ei)

    def _object_init_pos_env() -> np.ndarray:
        pos = getattr(getattr(cfg, "object_cube_cfg"), "init_state", None)
        if pos is not None and hasattr(pos, "pos"):
            return np.asarray(pos.pos, dtype=np.float64).ravel()[:3]
        return np.zeros(3, dtype=np.float64)

    control = VideoTeleopControl(T_world_arm_base=_T(np.array(args.arm_base_pos), np.array(args.arm_base_euler)))
    action_smoothing = float(np.clip(args.action_smoothing, 0.0, 0.999))
    rec_dir = (
        os.path.abspath(os.path.expanduser(args.record_path))
        if args.record_path
        else os.path.join(
            os.getcwd(),
            f"play_records/blind_retrieval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    )
    os.makedirs(rec_dir, exist_ok=True)
    fixed_label = int(args.target_label)
    target_name = (
        TARGET_NAMES[fixed_label]
        if 0 <= fixed_label < len(TARGET_NAMES)
        else "random"
    )
    with open(os.path.join(rec_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "blind_retrieval",
                "sequence": sequence,
                "target_label": fixed_label,
                "target_name": target_name,
                "hand_noise": float(args.hand_noise),
                "keyframe_json": str(kf_path),
                "save_outcome": str(args.save_outcome),
                "action_smoothing": float(args.action_smoothing),
                "pick_hand_close_scale": float(args.pick_hand_close_scale),
                "pick_object_align_offset": list(args.pick_object_align_offset),
                "carry_object_after_grasp": bool(args.carry_object_after_grasp),
                "tactile_contact_threshold": float(args.tactile_contact_threshold),
                "bench_mode": bench_mode,
            },
            f,
            indent=2,
        )

    ei = max(0, min(int(args.record_env_index), env.num_envs - 1))
    target_dt = 1.0 / max(1e-3, float(args.fps))
    succ = 0
    dummy_actions = torch.zeros(env.num_envs, act_dim, device=env.device)
    use_joint_targets = hasattr(env, "apply_joint_targets")
    pick_align_frame = _pick_close_frame(frames)

    for ep in range(int(args.num_episodes)):
        _reset_and_warmup(
            env,
            act_dim=act_dim,
            settle_steps=args.tactile_reset_settle_steps,
            object_settle_steps=args.object_settle_steps,
        )
        label_now = _apply_target_label(env, ei, fixed_label)
        _align_target_to_keyframe(env, ei, pick_align_frame)
        _settle_after_object_write(env, act_dim, steps=20)

        bufs: dict[str, list[np.ndarray]] = {}
        ep_steps = 0
        zmax = -1e9
        z_fallback_ok = False
        tact_peak_max = 0.0
        success_latched = False
        success_signal_seen = False
        success_source = "unavailable"
        grasp_latched = False
        grasp_palm_offset: np.ndarray | None = None
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
        dwell = 0
        pick_contact_seen = False
        print(f"[INFO] episode {ep}: target_label={label_now} ({TARGET_NAMES[label_now]})")

        while app.is_running() and fi < len(frames):
            t0 = time.time()
            fr = frames[fi]
            frame_name = str(fr.get("name", ""))
            pick_close = _is_pick_close_frame(frame_name)
            pick_frame = "Pick" in frame_name
            if pick_close and sif == 0:
                _align_target_to_keyframe(env, ei, fr, offset_env=tuple(args.pick_object_align_offset))
                _settle_after_object_write(env, act_dim, steps=15)

            marker_pos = np.asarray(fr["marker_pos_w"], dtype=np.float64)[:3]
            marker_euler = np.asarray(fr["marker_euler_xyz"], dtype=np.float64)[:3]
            hand_j = noisy_hands[fi]
            hand_tol = float(args.hand_pos_tol)
            if pick_close:
                hand_tol = max(hand_tol, 0.12)

            arm_recorded = np.asarray(fr.get("arm_joint_pos", []), dtype=np.float64).ravel()
            use_recorded = (
                bool(args.use_recorded_arm)
                and arm_recorded.size >= len(arm_ids)
                and not _use_ik_for_frame(frame_name)
            )
            if use_recorded:
                des_arm = arm_recorded[: len(arm_ids)]
            else:
                wrist_pos = _wrist_pos_for_frame(
                    fr,
                    frame_name,
                    sample_object_pos_env=_sample_target_pos_env,
                    object_init_pos_env=_object_init_pos_env,
                    pick_wrist_z_above_object=float(args.pick_wrist_z_above_object),
                )
                tgt = control.compute(wrist_pos, marker_euler, hand_j)
                des_arm = cur_arm() if tgt is None else np.asarray(tgt.arm_joints, dtype=np.float64)

            des_hand = hand_j
            if pick_close:
                des_hand = _boost_hand_close(hand_j, float(args.pick_hand_close_scale), sh_lo, sh_hi)

            cur_arm_j = cur_arm()
            cur_hand_j = cur_hand()
            actions = build_action(des_arm, des_hand)
            ae = float(np.max(np.abs(cur_arm_j - des_arm)))
            he = float(np.max(np.abs(cur_hand_j - des_hand)))
            arm_ok = ae <= float(args.arm_pos_tol)
            conv = arm_ok and he <= hand_tol

            actuated_j = build_actuated_joint_pos(des_arm, des_hand)
            if use_joint_targets:
                env.apply_joint_targets(
                    torch.tensor(actuated_j, dtype=torch.float32, device=env.device).unsqueeze(0)
                )
                if (
                    bool(args.snap_at_pick_close)
                    and pick_frame
                    and hasattr(env.robot, "write_joint_state_to_sim")
                ):
                    snap_j = build_actuated_joint_pos(des_arm, des_hand)
                    jpos = env.robot.data.joint_pos.clone()
                    jvel = torch.zeros_like(jpos)
                    act_t = torch.tensor(snap_j, dtype=jpos.dtype, device=env.device)
                    for j, dof_i in enumerate(env.actuated_dof_indices):
                        jpos[:, dof_i] = act_t[j]
                    env.robot.write_joint_state_to_sim(jpos, jvel)
                step_actions = dummy_actions
            else:
                if env.num_envs > 1:
                    actions = actions.expand(env.num_envs, -1).clone()
                step_actions = action_smoothing * prev_actions + (1.0 - action_smoothing) * actions
                prev_actions = step_actions.clone()

            if env.num_envs > 1:
                step_actions = step_actions.expand(env.num_envs, -1).clone()
                actions = actions.expand(env.num_envs, -1).clone()

            out = env.step(step_actions)
            if (
                grasp_latched
                and grasp_palm_offset is not None
                and bool(args.carry_object_after_grasp)
                and "Finish" in frame_name
            ):
                _follow_target_with_palm(env, ei, grasp_palm_offset)

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
                    bufs["target_object_id"] = []
                for k, v in row.items():
                    bufs.setdefault(k, []).append(np.asarray(v))
                bufs["task_episode_sim_step"].append(np.asarray(float(ep_steps), dtype=np.float32))
                bufs["target_object_id"].append(np.asarray(float(env.target_object_id[ei].item()), dtype=np.float32))

            signal_available, signal_success, signal_source_now = extract_success_signal(infos, obs, env, ei)
            if signal_available:
                success_signal_seen = True
                success_latched = success_latched or signal_success
                success_source = signal_source_now

            z = float(_target_pos_env(env, ei)[2])
            zmax = max(zmax, z)
            z_fallback_ok = z_fallback_ok or (z > float(args.success_z_threshold))
            tact_peak = _tactile_contact_strength(env, ei)
            tact_peak_max = max(tact_peak_max, tact_peak)
            tip_dist = _min_fingertip_dist_to_target(env, ei) if pick_close else 1e9
            geom_gate = float(args.pick_geom_contact_dist) > 0.0
            geom_contact = geom_gate and pick_close and tip_dist <= float(args.pick_geom_contact_dist)
            if pick_close and (
                tact_peak >= float(args.tactile_contact_threshold) or geom_contact
            ):
                pick_contact_seen = True

            sif += 1
            stable = stable + 1 if conv else 0
            min_dwell = _min_dwell_steps(fr, args)
            if arm_ok and min_dwell > 0:
                dwell += 1
            elif not arm_ok:
                dwell = 0
            if countdown < 0 and arm_ok and not conv:
                countdown = max(0, int(args.post_arm_reached_steps))
            elif countdown >= 0 and not conv:
                countdown -= 1

            max_sif = _max_steps_for_frame(fr, args)
            dwell_ok = min_dwell <= 0 or dwell >= min_dwell
            tact_gate = float(args.tactile_contact_threshold) <= 0.0 or (not pick_close) or pick_contact_seen
            if pick_close:
                should_advance = (
                    tact_gate
                    and conv
                    and stable >= max(1, int(args.stable_steps))
                    and dwell_ok
                ) or (sif >= max_sif and pick_contact_seen) or sif >= max_sif
            else:
                should_advance = (
                    (conv and stable >= max(1, int(args.stable_steps)) and dwell_ok)
                    or (arm_ok and dwell_ok and sif >= max(15, min_dwell))
                    or sif >= max_sif
                    or (arm_ok and not conv and countdown == 0 and min_dwell <= 0)
                )

            if pick_close and sif > 0 and sif % 100 == 0:
                print(
                    f"[INFO] {frame_name} step={sif}/{max_sif} z={z:.4f} tact={tact_peak:.2e} "
                    f"tip_dist={tip_dist:.4f} contact={pick_contact_seen} hand_err={he:.3f}",
                    flush=True,
                )

            if should_advance:
                if pick_close:
                    if pick_contact_seen and bool(args.carry_object_after_grasp):
                        obj_env = _target_pos_env(env, ei)
                        grasp_palm_offset = obj_env - _palm_pos_env(env, ei)
                        grasp_latched = True
                    print(
                        f"[INFO] leave {frame_name}: z={z:.4f} tact_peak={tact_peak:.2e} "
                        f"contact={pick_contact_seen} tip_dist={tip_dist:.4f} hand_err={he:.3f} "
                        f"arm_err={ae:.3f} steps={sif}",
                        flush=True,
                    )
                fi += 1
                stable = 0
                countdown = -1
                sif = 0
                dwell = 0
                pick_contact_seen = False

            if torch.is_tensor(term) and torch.is_tensor(trunc) and bool(torch.any(term | trunc).item()):
                print(
                    f"[WARN] env reset during {frame_name} at step={sif} "
                    f"(episode timeout); re-aligning and continuing frame",
                    flush=True,
                )
                _reset_and_warmup(
                    env,
                    act_dim=act_dim,
                    settle_steps=args.tactile_reset_settle_steps,
                    object_settle_steps=args.object_settle_steps,
                )
                _apply_target_label(env, ei, fixed_label)
                if pick_close:
                    _align_target_to_keyframe(env, ei, fr, offset_env=tuple(args.pick_object_align_offset))
                    _settle_after_object_write(env, act_dim, steps=15)
                elif pick_frame:
                    _align_target_to_keyframe(env, ei, pick_align_frame)
                    _settle_after_object_write(env, act_dim, steps=10)
                prev_actions.zero_()
                stable = 0
                countdown = -1
                continue

            dt = target_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

        final_success = bool(success_latched)
        if not final_success and z_fallback_ok:
            if bench_mode:
                if tact_peak_max >= float(args.tactile_contact_threshold):
                    final_success = True
                    success_source = "pickup_z_with_tactile_fallback"
            else:
                final_success = True
                success_source = "pickup_z_fallback"
        if not success_signal_seen or (success_signal_seen and not success_latched):
            if final_success and success_source == "unavailable":
                if z_fallback_ok:
                    success_source = "pickup_z_with_tactile_fallback" if bench_mode else "pickup_z_fallback"
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
                bufs["success_z_max"] = [np.asarray(zmax, dtype=np.float32) for _ in range(t_rows)]
                bufs["tactile_peak_max"] = [np.asarray(tact_peak_max, dtype=np.float32) for _ in range(t_rows)]
                bufs["outcome_success"] = [np.asarray(final_success, dtype=np.bool_) for _ in range(t_rows)]
            tag = TARGET_NAMES[label_now] if 0 <= label_now < len(TARGET_NAMES) else "unknown"
            fname = f"episode_{ep:04d}_{tag}_{outcome}.npz"
            p, t_len = _save_npz(rec_dir, fname, bufs)
            print(
                f"[INFO] saved {outcome} -> {p} "
                f"(T={t_len}, success_source={success_source}, z_max={zmax:.4f}, tact_peak_max={tact_peak_max:.2e})"
            )
        else:
            print(
                f"[INFO] skipped {outcome} episode {ep:04d} "
                f"(T={len(bufs.get('task_episode_sim_step', []))}, "
                f"success_source={success_source}, z_max={zmax:.4f})"
            )
        if not app.is_running():
            break

    print(f"[INFO] done: {succ}/{int(args.num_episodes)} success")
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
