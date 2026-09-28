#!/usr/bin/env python3
"""Privileged Blind Grasp demo collector (replaces GUI teleop).

Object pose is used only to generate the expert motion (approach / pinch / lift).
Recorded NPZ and policy observations stay tactile + proprioception. Do not train
a policy that reads object pose from this script.

Arm seed is the in-bin pose_008 reach. The hand is an advisor enclosure:
thumb on one side, FF/MF/RF/LF closed on the other. Cube mass/friction are
set on the BlindGrasp env (light + mu 3.0/2.5).

Usage (repo root, env_isaaclab_510test; close other Isaac GUIs first):

    ../IsaacLab/isaaclab.sh -p scripts/data_collection/full_trajectory/collect_blind_grasp_privileged.py \\
        --headless --enable_cameras --device cuda:0 --object-init-index 0 --num-episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common_record_utils import merge_record_row
from play_single_phase import (
    TACTILE_SENSOR_NAMES,
    _POLICY_RECORD_KEYS,
    _T,
    _load_symbol,
    _np,
    _read_tactile_from_scene,
    _read_third_person_from_scene,
    _repo_root,
    _reset_and_warmup,
    _save_npz,
    _tactile_contact_strength,
    _tendon_couple_distal,
)
from task_entries_single import resolve_env_cfg_entries

_PINCH_FINGERS = (
    "tactile_sensor_ff",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_lf",
    "tactile_sensor_th",
)
_PINCH_TIPS = ("ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal")
_DEFAULT_KF = HERE / "pose_keyframes" / "UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json"
# Advisor enclosure: four-finger pad vs opposed thumb (pickup-cube thumb + recorded close).
_ENCLOSURE_HAND = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.79,
        0.72,
        0.49,
        0.0,
        0.79,
        0.72,
        0.49,
        0.0,
        0.79,
        0.72,
        0.49,
        0.08,
        0.0,
        0.66,
        0.59,
        0.41,
        0.10,
        0.40,
        0.22,
        0.55,
        0.85,
    ],
    dtype=np.float64,
)


def _parser():
    p = argparse.ArgumentParser(description="Privileged Blind Grasp expert collector.")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--object-init-index", type=int, default=0)
    p.add_argument("--record-path", type=str, default="logs/demo_blind_grasp/privileged/collection")
    p.add_argument("--success-z-threshold", type=float, default=0.35)
    p.add_argument("--tactile-contact-threshold", type=float, default=1.0e-4)
    p.add_argument("--min-pinch-fingers", type=int, default=2)
    p.add_argument("--seed-pose", type=str, default="pose_008")
    p.add_argument("--keyframe-json", type=str, default=str(_DEFAULT_KF))
    p.add_argument("--lift-dz", type=float, default=0.45, help="World-z raise of wrist_3 after pinch.")
    p.add_argument("--arm-step-rad", type=float, default=0.02)
    p.add_argument("--lift-arm-step-rad", type=float, default=0.008)
    p.add_argument("--hand-step-rad", type=float, default=0.02)
    p.add_argument("--phase-steps", type=int, default=280)
    p.add_argument("--probe-dz", type=float, default=0.015)
    p.add_argument("--min-wrist-z", type=float, default=0.22)
    p.add_argument("--record-step-interval", type=int, default=5)
    p.add_argument("--object-settle-steps", type=int, default=60)
    AppLauncher.add_app_launcher_args(p)
    return p


def _load_seed_pose(path: str, name: str) -> dict:
    doc = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    for kf in doc.get("keyframes", []):
        if str(kf.get("name")) == name:
            return kf
    available = [str(kf.get("name")) for kf in doc.get("keyframes", [])]
    raise KeyError(f"seed pose {name!r} not in {path}; have {available}")


def _finger_peaks(env, ei: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in TACTILE_SENSOR_NAMES:
        out[name] = 0.0
        if name not in env.scene.sensors:
            continue
        try:
            nf = env.scene[name].data.tactile_normal_force
            out[name] = float(torch.nan_to_num(nf[ei], nan=0.0).abs().max().item())
        except Exception:
            continue
    return out


def _cube_pos_env(env) -> np.ndarray:
    return _np(env.object.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3]


def _body_pos_env(env, *keys: str) -> np.ndarray | None:
    try:
        idx = env._resolve_body_index_by_keywords(keys)
        return _np(env.robot.data.body_pos_w[0, idx] - env.scene.env_origins[0]).ravel()[:3]
    except Exception:
        found = env.robot.find_bodies(keys[0])
        if not found or not found[0]:
            return None
        return _np(env.robot.data.body_pos_w[0, found[0][0]] - env.scene.env_origins[0]).ravel()[:3]


def _pinch_centroid(env) -> np.ndarray | None:
    pts = []
    for tip in _PINCH_TIPS:
        p = _body_pos_env(env, tip)
        if p is not None and np.all(np.isfinite(p)):
            pts.append(p)
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def _tip_summary(env, cube: np.ndarray) -> str:
    bits = []
    names = list(_PINCH_TIPS)
    for idx, bname in enumerate(list(getattr(env.robot, "body_names", []))):
        key = str(bname).lower()
        if any(tok in key for tok in ("elastomer", "fftip", "thtip", "mftip")):
            names.append(str(bname))
    seen = set()
    for tip in names:
        if tip in seen:
            continue
        seen.add(tip)
        p = _body_pos_env(env, tip)
        if p is None:
            continue
        bits.append(f"{tip}=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) d={float(np.linalg.norm(p - cube)):.3f}")
    return " ".join(bits)


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

    seed = _load_seed_pose(args.keyframe_json, args.seed_pose)
    seed_arm = np.asarray(seed["arm_joint_pos"], dtype=np.float64).ravel()[:6]
    seed_hand = np.asarray(seed["hand_joint_pos_shadow_order"], dtype=np.float64).ravel()[:24]
    seed_cube = np.asarray(seed.get("object_pos_env", [0.62, 0.0, 0.054]), dtype=np.float64).ravel()[:3]
    seed_euler = np.asarray(seed.get("marker_euler_xyz", [np.pi, 1.13, np.pi]), dtype=np.float64).ravel()[:3]

    env_entry, cfg_entry, _ = resolve_env_cfg_entries(task="blind_grasp")
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)
    cfg = CfgCls()
    cfg.scene.num_envs = 1
    cfg.device = getattr(args, "device", None) or "cuda:0"
    cfg.object_init_choice = int(args.object_init_index)
    cfg.enable_cameras = True
    if hasattr(cfg, "episode_length_s"):
        cfg.episode_length_s = 300.0
    if hasattr(cfg, "act_moving_average"):
        cfg.act_moving_average = 1.0
    if hasattr(cfg, "object_reset_pos_x_range"):
        cfg.object_reset_pos_x_range = (0.0, 0.0)
        cfg.object_reset_pos_y_range = (0.0, 0.0)
    fingers_act = cfg.robot_cfg.actuators.get("fingers") if hasattr(cfg.robot_cfg, "actuators") else None
    if fingers_act is not None and hasattr(fingers_act, "effort_limit_sim"):
        fingers_act.effort_limit_sim = 25.0
    print(
        "[INFO] privileged expert uses object pose for motion only; "
        f"policy obs object state={getattr(cfg, 'include_object_state_in_policy_obs', None)} "
        f"cube_mass={getattr(getattr(cfg.object_cfg.spawn, 'mass_props', None), 'mass', None)} "
        f"mu=({getattr(getattr(cfg.sim, 'physics_material', None), 'static_friction', None)},"
        f"{getattr(getattr(cfg.sim, 'physics_material', None), 'dynamic_friction', None)}) "
        f"finger_effort={getattr(fingers_act, 'effort_limit_sim', None)}",
        flush=True,
    )
    env = EnvCls(cfg)
    if hasattr(env, "_use_rl_control"):
        env._use_rl_control = True
    env._replay_q = None
    _base_apply = type(env)._apply_action

    def _apply_replay(_env=env):
        q = getattr(_env, "_replay_q", None)
        if q is not None:
            _env.apply_joint_targets(q)
            return
        _base_apply(_env)

    env._apply_action = _apply_replay
    _reset_and_warmup(env, act_dim=env.num_actions, settle_steps=8, object_settle_steps=args.object_settle_steps)
    robot = env.robot
    jn = list(robot.joint_names)
    sh = shadowhand_joint_names()
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_ids = [i for i, n in enumerate(jn) if re.match(arm_expr, n)]
    hand_ids = [i for i, n in enumerate(jn) if re.match(hand_expr, n)]
    sh_robot_idx = [-1] * 24
    sh_lo = np.full(24, -1.5)
    sh_hi = np.full(24, 1.5)
    lo_all = _np(env.robot_dof_lower_limits[0])
    hi_all = _np(env.robot_dof_upper_limits[0])
    for idx in hand_ids:
        for si, sn in enumerate(sh):
            if re.search(rf"(^|[^A-Za-z0-9]){re.escape(sn)}([^A-Za-z0-9]|$)", jn[idx]):
                sh_robot_idx[si] = idx
                sh_lo[si], sh_hi[si] = lo_all[idx], hi_all[idx]
                break
    seed_hand = np.clip(_tendon_couple_distal(seed_hand), sh_lo, sh_hi)
    closed_hand = _ENCLOSURE_HAND.copy()
    closed_hand[0:2] = seed_hand[0:2]
    closed_hand = np.clip(_tendon_couple_distal(closed_hand), sh_lo, sh_hi)
    approach_hand = closed_hand.copy()
    approach_hand[2:] *= 0.15
    approach_hand = np.clip(_tendon_couple_distal(approach_hand), sh_lo, sh_hi)
    squeeze_hand = closed_hand.copy()
    for i in (3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 17, 18, 22, 23):
        squeeze_hand[i] = float(np.clip(squeeze_hand[i] + 0.10, sh_lo[i], sh_hi[i]))
    squeeze_hand = np.clip(_tendon_couple_distal(squeeze_hand), sh_lo, sh_hi)
    print(
        f"[INFO] arm_joints={[jn[i] for i in arm_ids]} seed_arm={np.round(seed_arm, 3).tolist()}",
        flush=True,
    )
    control = VideoTeleopControl(T_world_arm_base=_T(np.zeros(3), np.zeros(3)))

    def cur_arm():
        q = _np(robot.data.joint_pos[0])
        return np.array([float(q[i]) for i in arm_ids], dtype=np.float64)

    def cur_hand():
        q = _np(robot.data.joint_pos[0])
        out = np.zeros(24, dtype=np.float64)
        for si, idx in enumerate(sh_robot_idx):
            if idx >= 0:
                out[si] = float(q[idx])
        return out

    def build_full(arm_j, hand_j):
        full = _np(robot.data.joint_pos[0]).astype(np.float64, copy=True)
        for i, idx in enumerate(arm_ids):
            if i < len(arm_j):
                full[idx] = float(arm_j[i])
        for si, idx in enumerate(sh_robot_idx):
            if idx >= 0 and si < len(hand_j):
                full[idx] = float(hand_j[si])
        return full

    def build_action(arm_j, hand_j):
        actuated = build_full(arm_j, hand_j)[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lo = _np(env.robot_dof_lower_limits[0, env.actuated_dof_indices])
        hi = _np(env.robot_dof_upper_limits[0, env.actuated_dof_indices])
        scale = np.where(hi - lo > 1e-6, 2.0 * (actuated - lo) / (hi - lo) - 1.0, 0.0)
        return torch.tensor(np.clip(scale, -1.0, 1.0), dtype=torch.float32, device=env.device).unsqueeze(0)

    def wrist_pos_env() -> np.ndarray:
        p = _body_pos_env(env, "wrist_3", "wrist3", "palm")
        return p if p is not None else np.full(3, np.nan)

    def wrist_euler_env() -> np.ndarray:
        try:
            idx = env._resolve_body_index_by_keywords(("wrist_3", "wrist3", "palm"))
            quat = _np(robot.data.body_quat_w[0, idx]).ravel()
            return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
        except Exception:
            return seed_euler.copy()

    def ik_arm(wrist_pos: np.ndarray, euler: np.ndarray, hand_j: np.ndarray) -> np.ndarray | None:
        control._last_arm_joints = cur_arm()
        tgt = control.compute(wrist_pos, euler, hand_j)
        if tgt is None:
            return None
        return np.asarray(tgt.arm_joints, dtype=np.float64)

    rec_dir = os.path.abspath(os.path.expanduser(args.record_path))
    os.makedirs(rec_dir, exist_ok=True)
    succ = 0
    for ep in range(int(args.num_episodes)):
        env._replay_q = None
        _reset_and_warmup(env, act_dim=env.num_actions, settle_steps=8, object_settle_steps=args.object_settle_steps)
        cube0 = _cube_pos_env(env)
        print(
            f"[INFO] ep={ep} seed={args.seed_pose} cube=({cube0[0]:.3f},{cube0[1]:.3f},{cube0[2]:.3f}) "
            f"seed_cube=({seed_cube[0]:.3f},{seed_cube[1]:.3f},{seed_cube[2]:.3f}) "
            f"seed_arm={np.round(seed_arm, 3).tolist()}",
            flush=True,
        )
        arm_cmd = cur_arm()
        hand_cmd = cur_hand()
        bufs: dict[str, list[np.ndarray]] = {}
        zmax = float(cube0[2])
        lift_tact_max = 0.0
        ep_steps = 0
        pinch_ok = False

        def _step_once(phase_name: str, sif: int, des_arm: np.ndarray, hand_des: np.ndarray, arm_step: float) -> tuple[np.ndarray, float, dict[str, float], int]:
            nonlocal arm_cmd, hand_cmd, ep_steps, zmax, lift_tact_max
            arm_cmd = arm_cmd + np.clip(des_arm - arm_cmd, -arm_step, arm_step)
            hand_cmd = hand_cmd + np.clip(hand_des - hand_cmd, -float(args.hand_step_rad), float(args.hand_step_rad))
            actions = build_action(arm_cmd, hand_cmd)
            env._replay_q = torch.tensor(build_full(arm_cmd, hand_cmd), dtype=torch.float32, device=env.device).unsqueeze(0)
            out = env.step(actions)
            obs = out[0] if isinstance(out, tuple) and len(out) >= 5 else out
            ep_steps += 1
            cube = _cube_pos_env(env)
            zmax = max(zmax, float(cube[2]))
            peaks = _finger_peaks(env, 0)
            tact = _tactile_contact_strength(env, 0)
            n_pinch = sum(1 for n in _PINCH_FINGERS if peaks[n] >= float(args.tactile_contact_threshold))
            if phase_name == "lift":
                lift_tact_max = max(lift_tact_max, tact)
            if ep_steps % max(1, int(args.record_step_interval)) == 0:
                row = merge_record_row(obs, 0, actions=actions, robot=robot)
                row.update(_read_tactile_from_scene(env, 0))
                row.update(_read_third_person_from_scene(env, 0))
                row = {k: v for k, v in row.items() if k in _POLICY_RECORD_KEYS}
                if not bufs:
                    for k in row:
                        bufs[k] = []
                    bufs["task_episode_sim_step"] = []
                for k, v in row.items():
                    bufs.setdefault(k, []).append(np.asarray(v))
                bufs["task_episode_sim_step"].append(np.asarray(float(ep_steps), dtype=np.float32))
            wrist = wrist_pos_env()
            if sif % 30 == 0:
                print(
                    f"[INFO] {phase_name} step={sif} cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
                    f"wrist=({wrist[0]:.3f},{wrist[1]:.3f},{wrist[2]:.3f}) "
                    f"tact={tact:.2e} ff={peaks['tactile_sensor_ff']:.2e} "
                    f"mf={peaks['tactile_sensor_mf']:.2e} th={peaks['tactile_sensor_th']:.2e} "
                    f"fingers={n_pinch} {_tip_summary(env, cube)}",
                    flush=True,
                )
            return cube, tact, peaks, n_pinch

        def _hold(phase_name: str, des_arm: np.ndarray, hand_des: np.ndarray, steps: int, arm_step: float, latch_fingers: int = 2):
            last = (_cube_pos_env(env), 0.0, _finger_peaks(env, 0), 0)
            best = last
            for sif in range(max(1, int(steps))):
                last = _step_once(phase_name, sif, des_arm, hand_des, arm_step)
                if last[3] > best[3] or last[1] > best[1]:
                    best = last
                if last[3] >= int(latch_fingers):
                    return last
            return best if best[3] > last[3] else last

        def _grasp_score(cube: np.ndarray, tact: float, n_pinch: int) -> float:
            tips = _pinch_centroid(env)
            dist = float(np.linalg.norm(tips - cube)) if tips is not None else 1.0
            return dist - 2.0 * n_pinch - 0.15 * min(tact, 5.0) - 4.0 * max(0.0, float(cube[2] - cube0[2]))

        def _ik_to(pos: np.ndarray, euler: np.ndarray, hand_des: np.ndarray) -> np.ndarray:
            solved = ik_arm(pos, euler, hand_des)
            if solved is None:
                print(f"[WARN] IK failed at {pos}; keeping current arm command.", flush=True)
                return arm_cmd.copy()
            return solved

        def _snap_q(arm_j: np.ndarray, hand_j: np.ndarray) -> None:
            nonlocal arm_cmd, hand_cmd
            arm_cmd = np.asarray(arm_j, dtype=np.float64).copy()
            hand_cmd = np.asarray(hand_j, dtype=np.float64).copy()
            full = build_full(arm_cmd, hand_cmd)
            q = torch.tensor(full, dtype=torch.float32, device=env.device).unsqueeze(0)
            qpos = env.robot.data.joint_pos.clone()
            qpos[0] = q[0]
            env.robot.write_joint_state_to_sim(qpos, torch.zeros_like(qpos))
            env.apply_joint_targets(q)
            env._replay_q = q

        def _hand_err() -> str:
            actual = cur_hand()
            bits = []
            for i, name in enumerate(sh):
                e = float(actual[i] - squeeze_hand[i])
                if abs(e) >= 0.08:
                    bits.append(f"{name}={actual[i]:.2f}/{squeeze_hand[i]:.2f}")
            return " ".join(bits) or "ok"

        # 1) Joint-space approach to the recorded in-bin grasp, fingers still mostly open.
        for sif in range(max(1, int(args.phase_steps))):
            cube, tact, peaks, n_pinch = _step_once("approach", sif, seed_arm, approach_hand, float(args.arm_step_rad))
            actual_arm = cur_arm()
            if sif % 30 == 0:
                print(
                    f"[INFO] approach track cmd_err={float(np.max(np.abs(arm_cmd - seed_arm))):.3f} "
                    f"act_err={float(np.max(np.abs(actual_arm - seed_arm))):.3f}",
                    flush=True,
                )
            if float(np.max(np.abs(actual_arm - seed_arm))) <= 0.05 and sif > 30:
                break

        # Force the recorded hand (PD was ~0.2 rad short, so the gel never reached pose_008).
        lift_hand = closed_hand.copy()
        cube, tact, peaks, n_pinch = _cube_pos_env(env), 0.0, _finger_peaks(env, 0), 0
        for attempt in range(3):
            _snap_q(seed_arm, lift_hand)
            cube, tact, peaks, n_pinch = _hold(
                "snap", seed_arm, lift_hand, 36, float(args.arm_step_rad), latch_fingers=1
            )
            print(
                f"[INFO] after snap attempt={attempt} tact={tact:.2e} fingers={n_pinch} "
                f"hand={_hand_err()}",
                flush=True,
            )
            if n_pinch >= 1:
                break
        if n_pinch >= 1:
            pinch_ok = True
            print(f"[INFO] latching gel contact fingers={n_pinch} tact={tact:.2e}; press down next.", flush=True)
            # Thumb flexion raised the tip; lower the whole hand so the thumb pad meets the cube top.
            press_arm = seed_arm.copy()
            for dz_try, d_elb, d_lift in ((1, 0.05, -0.03), (2, 0.10, -0.05), (3, 0.14, -0.07)):
                trial = seed_arm.copy()
                trial[1] = float(seed_arm[1] + d_lift)
                trial[2] = float(seed_arm[2] + d_elb)
                _snap_q(trial, lift_hand)
                cube, tact, peaks, n_pinch = _hold(
                    "press", trial, lift_hand, 28, float(args.arm_step_rad), latch_fingers=2
                )
                print(
                    f"[INFO] press k={dz_try} tact={tact:.2e} fingers={n_pinch} "
                    f"cube_z={cube[2]:.3f} wrist_z={float(wrist_pos_env()[2]):.3f}",
                    flush=True,
                )
                press_arm = trial
                if n_pinch >= 1 and cube[2] > float(cube0[2] + 0.008):
                    pinch_ok = True
                    print("[INFO] press raised the cube with gel contact; lifting from this pose.", flush=True)
                    break
            best_arm = press_arm
        else:
            thumb_hand = closed_hand.copy()
            thumb_hand[22] = float(np.clip(thumb_hand[22] - 0.12, sh_lo[22], sh_hi[22]))
            thumb_hand[23] = float(np.clip(thumb_hand[23] + 0.18, sh_lo[23], sh_hi[23]))
            thumb_hand = np.clip(_tendon_couple_distal(thumb_hand), sh_lo, sh_hi)
            cube, tact, peaks, n_pinch = _hold(
                "thumb", seed_arm, thumb_hand, 40, float(args.arm_step_rad), latch_fingers=1
            )
            print(f"[INFO] after thumb dip tact={tact:.2e} fingers={n_pinch} cube_z={cube[2]:.3f}", flush=True)
            if n_pinch >= 1:
                pinch_ok = True
                lift_hand = thumb_hand

        # 2) Only keep searching if the recorded pose never touched gel.
        if not pinch_ok:
            cube, tact, peaks, n_pinch = _hold("close", seed_arm, closed_hand, 90, float(args.arm_step_rad))
            cube, tact, peaks, n_pinch = _hold("squeeze", seed_arm, squeeze_hand, 60, float(args.arm_step_rad))
            if n_pinch >= 1:
                pinch_ok = True
                print(f"[INFO] pinch ready at seed fingers={n_pinch} tact={tact:.2e}", flush=True)

        # 3) Joint-space search around the seed. IK probe was twisting the wrist off the gel.
        if "best_arm" not in locals():
            best_arm = seed_arm.copy()
        best_score = _grasp_score(cube, tact, n_pinch)
        if not pinch_ok:
            print(
                f"[INFO] joint probe start score={best_score:.3f} cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
                f"hand_err={float(np.max(np.abs(cur_hand() - squeeze_hand))):.3f}",
                flush=True,
            )
            deltas = (
                (0, -0.05), (0, 0.05),
                (1, -0.05), (1, 0.05),
                (2, -0.05), (2, 0.05),
                (3, -0.06), (3, 0.06),
            )
            for it in range(5):
                improved = False
                for j, dlt in deltas:
                    trial = best_arm.copy()
                    trial[j] = float(trial[j] + dlt)
                    cube, tact, peaks, n_pinch = _hold("probe", trial, lift_hand, 22, float(args.arm_step_rad))
                    score = _grasp_score(cube, tact, n_pinch)
                    if n_pinch >= 1:
                        print(
                            f"[INFO] probe it={it} j={j} d={dlt:+.2f} fingers={n_pinch} "
                            f"tact={tact:.2e} cube_z={cube[2]:.3f} score={score:.3f}",
                            flush=True,
                        )
                    if n_pinch >= 1 or cube[2] > float(cube0[2] + 0.015):
                        best_arm = trial
                        pinch_ok = True
                        print(f"[INFO] pinch ready after joint probe fingers={n_pinch} tact={tact:.2e}", flush=True)
                        break
                    if score < best_score - 1e-3:
                        best_arm = trial
                        best_score = score
                        improved = True
                if pinch_ok:
                    break
                if not improved:
                    break
            cube, tact, peaks, n_pinch = _hold("probe", best_arm, lift_hand, 20, float(args.arm_step_rad))
        if not pinch_ok:
            print(f"[WARN] joint probe finished score={best_score:.3f}; lifting only if cube moved.", flush=True)

        # 4) Joint-space lift: keep wrist_1/2/3 and the hand, unfold lift/elbow.
        #    Cartesian IK was twisting the graze contact off the gel.
        hold_arm = best_arm.copy()
        wrist0 = wrist_pos_env()
        if not pinch_ok and zmax < float(cube0[2] + 0.012):
            print("[WARN] skip lift: no tactile pinch and cube never rose.", flush=True)
        else:
            print(
                f"[INFO] joint lift from wrist_z={float(wrist0[2]):.3f} cube_z={float(_cube_pos_env(env)[2]):.3f}",
                flush=True,
            )
            for k in range(18):
                des_arm = hold_arm.copy()
                des_arm[1] = float(hold_arm[1] + 0.028 * (k + 1))
                des_arm[2] = float(hold_arm[2] - 0.040 * (k + 1))
                cube, tact, peaks, n_pinch = _hold(
                    "lift", des_arm, lift_hand, 20, float(args.lift_arm_step_rad), latch_fingers=99
                )
                if zmax > float(args.success_z_threshold) and lift_tact_max >= float(args.tactile_contact_threshold):
                    break
                if pinch_ok and cube[2] < float(cube0[2] + 0.008) and float(wrist_pos_env()[2]) > float(wrist0[2] + 0.10):
                    print("[WARN] cube slipped back during joint lift; stopping raise.", flush=True)
                    break

        extracted = zmax > float(args.success_z_threshold)
        pinched_lift = lift_tact_max >= float(args.tactile_contact_threshold)
        ok = bool(extracted and pinched_lift)
        outcome = "success" if ok else "done"
        if ok:
            succ += 1
        t_rows = len(bufs.get("task_episode_sim_step", []))
        if t_rows > 0:
            bufs["success_z_max"] = [np.asarray(zmax, dtype=np.float32) for _ in range(t_rows)]
            bufs["lift_tact_max"] = [np.asarray(lift_tact_max, dtype=np.float32) for _ in range(t_rows)]
            bufs["outcome_success"] = [np.asarray(ok, dtype=np.bool_) for _ in range(t_rows)]
            pth, t_len = _save_npz(rec_dir, f"episode_{ep:04d}_{outcome}.npz", bufs)
            print(
                f"[INFO] saved {outcome} -> {pth} T={t_len} z_max={zmax:.4f} lift_tact={lift_tact_max:.2e}",
                flush=True,
            )
        else:
            print(f"[INFO] skipped empty episode {ep} z_max={zmax:.4f}", flush=True)
    print(f"[INFO] done: {succ}/{int(args.num_episodes)} success", flush=True)
    env.close()
    app.close()
    return 0 if succ > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
