#!/usr/bin/env python3
"""Low-level Blind Grasp layout check: slots A/B, bin, cube, recorded-keyframe reach.

Does not teleport the cube onto the hand or retarget fingers. Prints whether
probe/pick keyframes can physically reach the spawned object.

Modes:
  snap     — teleport each listed keyframe independently after reset
  seq-kin  — interpolate recorded joints with write_joint_state (layout diagnostic)
  seq-pd   — same interpolation using only env PD (matches bench replay)
  both     — seq-kin then reset and seq-pd
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/ViTacLab"))
sys.path.insert(0, str(ROOT / "source/video_teleop"))
sys.path.insert(0, str(ROOT / "scripts/data_collection/full_trajectory"))

TIP_BODIES = ("palm", "ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal")
TACTILE_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)
SEQ_A = ["Start01", "Start02", "Touch01_01", "Touch01_02", "Pick01_01", "Pick01_02", "Pick01_03", "Finish01"]
SEQ_B = [
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


def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _env_xyz(env, body_w: torch.Tensor) -> np.ndarray:
    return _np(body_w - env.scene.env_origins[0])


def _tactile_peaks(env) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in TACTILE_NAMES:
        if name not in env.scene.sensors:
            continue
        nf = getattr(env.scene[name].data, "tactile_normal_force", None)
        if nf is None:
            continue
        out[name.replace("tactile_sensor_", "")] = float(torch.nan_to_num(nf[0]).amax().item())
    return out


def _tactile_peak(env) -> float:
    peaks = _tactile_peaks(env)
    return max(peaks.values()) if peaks else 0.0


def _patch_replay_apply(env) -> None:
    if hasattr(env, "_layout_apply_patched"):
        return
    base = type(env)._apply_action

    def _apply(_env=env, _base=base):
        rq = getattr(_env, "_replay_q", None)
        if rq is not None:
            _env.apply_joint_targets(rq)
            return
        _base(_env)

    env._apply_action = _apply
    env._layout_apply_patched = True
    if hasattr(env, "_use_rl_control"):
        env._use_rl_control = True


def _keyframe_full_q(env, kf: dict) -> torch.Tensor:
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names

    jn = list(env.robot.joint_names)
    sh = shadowhand_joint_names()
    arm_expr = getattr(env.cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(env.cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_ids = [i for i, n in enumerate(jn) if re.match(arm_expr, n)]
    hand_ids = [i for i, n in enumerate(jn) if re.match(hand_expr, n)]
    arm_j = np.asarray(kf["arm_joint_pos"], dtype=np.float64).ravel()
    hand_j = np.asarray(kf["hand_joint_pos_shadow_order"], dtype=np.float64)[:24]

    def hand_for_name(name: str) -> float:
        best_i, best_len = None, -1
        for si, sn in enumerate(sh):
            if re.search(rf"(^|[^A-Za-z0-9]){re.escape(sn)}([^A-Za-z0-9]|$)", name):
                if len(sn) > best_len:
                    best_i, best_len = si, len(sn)
        return float(hand_j[best_i]) if best_i is not None else 0.0

    full = _np(env.robot.data.joint_pos[0]).astype(np.float64, copy=True)
    for i, idx in enumerate(arm_ids):
        if i < arm_j.size:
            full[idx] = float(arm_j[i])
    for idx in hand_ids:
        full[idx] = hand_for_name(jn[idx])
    return torch.tensor(full, dtype=torch.float32, device=env.device).unsqueeze(0)


def _drive_q(env, q: torch.Tensor, steps: int, kinematic: bool) -> None:
    _patch_replay_apply(env)
    env._replay_q = q
    z = torch.zeros(1, env.num_actions, device=env.device)
    for _ in range(max(1, int(steps))):
        if kinematic:
            qpos = env.robot.data.joint_pos.clone()
            qpos[0] = q[0]
            env.robot.write_joint_state_to_sim(qpos, torch.zeros_like(qpos))
        env.apply_joint_targets(q)
        env.step(z)


def _apply_keyframe_joints(env, kf: dict, steps: int, kinematic: bool = True) -> None:
    _drive_q(env, _keyframe_full_q(env, kf), steps, kinematic=kinematic)


def _print_extents(env) -> None:
    origin = env.scene.env_origins[0]
    print("[extents]", flush=True)
    for label, asset in (("cube", env.object), ("bin", env.trash_can)):
        pos = _env_xyz(env, asset.data.root_pos_w[0])
        line = f"  {label} pos={pos}"
        view = getattr(asset, "root_physx_view", None)
        if view is not None:
            for meth in ("get_aabbs", "get_aabb"):
                if hasattr(view, meth):
                    try:
                        aabb = getattr(view, meth)()
                        arr = _np(aabb)
                        line += f" aabb={arr}"
                    except Exception as exc:
                        line += f" aabb_err={exc}"
                    break
        print(line, flush=True)
    names = list(getattr(env.robot, "body_names", []))
    interesting = [n for n in names if any(k in n.lower() for k in ("elastomer", "palm", "distal", "wrist_3", "case"))]
    print(f"  interesting_bodies={interesting}", flush=True)
    for idx, name in enumerate(names):
        if name not in interesting:
            continue
        print(f"  {name:28s} {_env_xyz(env, env.robot.data.body_pos_w[0, idx])}", flush=True)


def _print_pose(env, title: str) -> None:
    cube = _env_xyz(env, env.object.data.root_pos_w[0])
    bin_p = _env_xyz(env, env.trash_can.data.root_pos_w[0])
    slot = int(env.object_slot_index[0].item()) if hasattr(env, "object_slot_index") else -1
    peaks = _tactile_peaks(env)
    peak_s = " ".join(f"{k}={v:.3e}" for k, v in peaks.items()) or "none"
    print(f"\n[{title}]", flush=True)
    print(
        f"  slot={slot} cube={cube} bin={bin_p} tact_peak={_tactile_peak(env):.3e} [{peak_s}]",
        flush=True,
    )
    print(
        f"  policy_obs_dim={tuple(env._get_observations()['policy'].shape)} "
        f"include_object_state={bool(getattr(env.cfg, 'include_object_state_in_policy_obs', True))}",
        flush=True,
    )
    for key in ("wrist_3_link", "wrist_3", "ee_link"):
        found = env.robot.find_bodies(key)
        if found and found[0]:
            wrist = _env_xyz(env, env.robot.data.body_pos_w[0, found[0][0]])
            print(f"  {key:10s} {wrist}", flush=True)
            break
    try:
        qerr = float(torch.max(torch.abs(env.robot.data.joint_pos[0] - env.robot.data.joint_pos_target[0])).item())
        print(f"  joint_track_err={qerr:.4f} rad cube_z={cube[2]:.4f}", flush=True)
        jn = list(env.robot.joint_names)
        arm_expr = getattr(env.cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
        diffs = torch.abs(env.robot.data.joint_pos[0] - env.robot.data.joint_pos_target[0])
        arm_bits = []
        for i, name in enumerate(jn):
            if re.match(arm_expr, name):
                arm_bits.append(f"{name}={float(diffs[i]):.3f}")
        print("  arm_err " + " ".join(arm_bits), flush=True)
    except Exception:
        print(f"  cube_z={cube[2]:.4f}", flush=True)
    for name in TIP_BODIES:
        found = env.robot.find_bodies(name)
        if not found or not found[0]:
            continue
        tip = _env_xyz(env, env.robot.data.body_pos_w[0, found[0][0]])
        dist = float(np.linalg.norm(tip - cube))
        print(f"  {name:10s} {tip}  dist_cube={dist:.4f}", flush=True)
    for idx, bname in enumerate(list(getattr(env.robot, "body_names", []))):
        key = str(bname).lower()
        if not any(token in key for token in ("elastomer", "case", "fftip", "thtip", "mftip", "rftip", "lftip")):
            continue
        gel = _env_xyz(env, env.robot.data.body_pos_w[0, idx])
        print(f"  {bname:28s} {gel}  dist_cube={float(np.linalg.norm(gel - cube)):.4f}", flush=True)


def _sequence_for_slot(slot: int) -> list[str]:
    return list(SEQ_B if int(slot) == 1 else SEQ_A)


def _run_sequence(env, kfs: dict, names: list[str], kinematic: bool, lerp_steps: int, hold_steps: int) -> None:
    tag = "kin" if kinematic else "pd"
    prev_q = env.robot.data.joint_pos[0].detach().clone().unsqueeze(0)
    for name in names:
        if name not in kfs:
            print(f"[WARN] missing keyframe {name}", flush=True)
            continue
        tgt = _keyframe_full_q(env, kfs[name])
        n = max(1, int(lerp_steps))
        for s in range(n):
            a = float(s + 1) / float(n)
            q = (1.0 - a) * prev_q + a * tgt
            _drive_q(env, q, 1, kinematic=kinematic)
        _drive_q(env, tgt, int(hold_steps), kinematic=kinematic)
        prev_q = tgt
        marker = np.asarray(kfs[name].get("marker_pos_w", []), dtype=np.float64).ravel()[:3]
        print(f"[INFO] {tag} {name} recorded marker_pos_w={marker}", flush=True)
        _print_pose(env, f"{tag} {name}")


def _maybe_hide_bin(env, hide: bool) -> None:
    if not hide:
        return
    pose = env.trash_can.data.root_pose_w.clone()
    pose[0, 0] = pose[0, 0] + 10.0
    env.trash_can.write_root_pose_to_sim(pose)
    env.trash_can.write_root_velocity_to_sim(torch.zeros_like(env.trash_can.data.root_vel_w))
    print("[INFO] hide-bin: trash_can moved +10 m in x (PD diagnostic)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--object-init-index", type=int, default=0, help="0=A, 1=B, -1=random")
    p.add_argument("--settle-steps", type=int, default=80)
    p.add_argument("--hold-steps", type=int, default=12)
    p.add_argument("--lerp-steps", type=int, default=40)
    p.add_argument("--mode", choices=("snap", "seq-kin", "seq-pd", "both"), default="both")
    p.add_argument(
        "--hide-bin",
        action="store_true",
        help="Diagnostic: move the kinematic bin far away so PD tracking is tested without rim collision.",
    )
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app

    import ViTacLab.tasks  # noqa: F401
    from task_entries_single import resolve_env_cfg_entries

    kf_path = (
        ROOT
        / "scripts/data_collection/full_trajectory/pose_keyframes"
        / "UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json"
    )
    doc = json.loads(kf_path.read_text(encoding="utf-8"))
    kfs = {str(k["name"]): k for k in doc["keyframes"]}

    env_e, cfg_e, _ = resolve_env_cfg_entries(task="blind_grasp")
    EnvCls = getattr(importlib.import_module(env_e.rsplit(":", 1)[0]), env_e.rsplit(":", 1)[1])
    CfgCls = getattr(importlib.import_module(cfg_e.rsplit(":", 1)[0]), cfg_e.rsplit(":", 1)[1])
    cfg = CfgCls()
    cfg.scene.num_envs = 1
    cfg.object_init_choice = int(args.object_init_index)
    # Third-person is optional for layout; GelSight TacSL still updates force-field
    # without AppLauncher --enable_cameras. Set True only when that CLI flag is passed.
    cfg.enable_cameras = bool(getattr(args, "enable_cameras", False))
    env = EnvCls(cfg)
    env.reset()
    z = torch.zeros(1, env.num_actions, device=env.device)
    for _ in range(int(args.settle_steps)):
        env.step(z)
    _maybe_hide_bin(env, bool(args.hide_bin))

    print("[INFO] Blind Grasp layout", flush=True)
    print(f"  object_init_choice={cfg.object_init_choice}", flush=True)
    print(f"  slots={cfg.object_init_pos_candidates[: cfg.object_slot_count]}", flush=True)
    print(f"  bin_init={cfg.trash_can_cfg.init_state.pos} scale={cfg.trash_can_scale}", flush=True)
    print(f"  robot_init={cfg.robot_cfg.init_state.pos}", flush=True)
    _print_pose(env, "after reset settle")
    _print_extents(env)

    names = _sequence_for_slot(int(args.object_init_index))
    if args.mode == "snap":
        for name in names:
            if name not in kfs:
                print(f"[WARN] missing keyframe {name}", flush=True)
                continue
            marker = np.asarray(kfs[name].get("marker_pos_w", []), dtype=np.float64).ravel()[:3]
            print(f"[INFO] {name} recorded marker_pos_w={marker}", flush=True)
            env.reset()
            for _ in range(int(args.settle_steps)):
                env.step(z)
            _maybe_hide_bin(env, bool(args.hide_bin))
            _apply_keyframe_joints(env, kfs[name], int(args.hold_steps), kinematic=True)
            _print_pose(env, f"snapped {name}")
    else:
        if args.mode in ("seq-kin", "both"):
            _run_sequence(env, kfs, names, kinematic=True, lerp_steps=int(args.lerp_steps), hold_steps=int(args.hold_steps))
        if args.mode in ("seq-pd", "both"):
            env.reset()
            for _ in range(int(args.settle_steps)):
                env.step(z)
            _maybe_hide_bin(env, bool(args.hide_bin))
            _run_sequence(env, kfs, names, kinematic=False, lerp_steps=int(args.lerp_steps), hold_steps=int(args.hold_steps))

    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
