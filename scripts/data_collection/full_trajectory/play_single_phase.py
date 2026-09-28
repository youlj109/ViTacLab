#!/usr/bin/env python3
"""Play one canonical single-arm BlindGrasp business phase and record NPZ data.

Use ``--phase 1`` through ``--phase 3``. Phases 1/2 use BlindGrasp keyframes;
phase 3 uses the BlindGraspReplay keyframe/environment by default. All phases
share this implementation and command-line interface.
"""

from __future__ import annotations
import argparse, importlib, json, os, re, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np, torch
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from common_record_utils import extract_success_signal, merge_record_row
from task_entries_single import resolve_env_cfg_entries

_POLICY_RECORD_KEYS = {
    "joint_pos",
    "joint_vel",
    "action",
    "tactile_normal_force",
    "tactile_shear_force",
    "tactile_rgb_image",
    "third_person_camera",
}

_PHASES = {
    1: {
        "task": "blind_grasp",
        "sequence": ["Start01", "Start02", "Touch01_01", "Touch01_02", "Pick01_01", "Pick01_02", "Pick01_03", "Finish01"],
        "keyframe": "UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json",
    },
    2: {
        "task": "blind_grasp",
        "sequence": ["Start01", "Start02", "Touch01_01", "Touch01_02", "Touch02_01", "Touch02_02", "Pick02_01", "Pick02_02", "Pick02_03", "Finish02"],
        "keyframe": "UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json",
    },
    3: {
        "task": "blind_grasp_replay",
        "sequence": ["Start01", "Start02", "Touch01_01", "Touch01_02", "Touch02_01", "Touch02_02", "Pick02_01", "Pick02_02", "Pick02_03", "Finish02"],
        "keyframe": "UR10eShadowHandBlindGraspReplayEnv__UR10eShadowHandBlindGraspReplayEnvCfg.json",
    },
}


def _repo_root() -> Path:
    p = HERE
    for _ in range(10):
        if (p / "source").is_dir(): return p
        if p.parent == p: break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _load_symbol(entry: str) -> Any:
    m, s = entry.split(":", 1)
    return getattr(importlib.import_module(m), s)


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)


def _img_to_uint8_torch(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.dtype == torch.uint8:
        return rgb
    mx = rgb.max().item()
    if mx <= 1.0:
        return torch.clamp(rgb * 255.0, 0.0, 255.0).to(torch.uint8)
    return torch.clamp(rgb, 0.0, 255.0).to(torch.uint8)


TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)


def _warmup_tactile_nominal(env) -> list[str]:
    """Initialize tactile nominal render for all TacSL sensors, retrying after resets."""
    warmed: list[str] = []
    scene = env.scene
    for name in TACTILE_SENSOR_NAMES:
        if name not in scene.sensors:
            continue
        sensor = scene[name]
        for _ in range(3):
            try:
                sensor.get_initial_render()
                warmed.append(name)
                break
            except RuntimeError:
                continue
            except Exception:
                break
    return warmed


def _reset_and_warmup(env, act_dim: int, settle_steps: int, object_settle_steps: int = 0) -> list[str]:
    """Reset env, settle with zero actions, then capture fresh tactile nominal frames."""
    env.reset()
    zero_actions = torch.zeros(env.num_envs, act_dim, device=env.device)
    for _ in range(max(0, int(settle_steps))):
        env.step(zero_actions)
    for _ in range(max(0, int(object_settle_steps))):
        env.step(zero_actions)
    return _warmup_tactile_nominal(env)


def _read_tactile_from_scene(env, ei: int) -> dict[str, np.ndarray]:
    """Read tactile data directly from scene sensors (same as record_single.py)."""
    out: dict[str, np.ndarray] = {}
    scene = env.scene
    num_tactile = len(TACTILE_SENSOR_NAMES)
    norm_list, shear_list, rgb_list = [], [], []
    for name in TACTILE_SENSOR_NAMES:
        if name not in scene.sensors:
            continue
        try:
            data = scene[name].data
        except RuntimeError:
            try:
                scene[name].get_initial_render()
                data = scene[name].data
            except Exception:
                continue
        nf = getattr(data, "tactile_normal_force", None)
        sf = getattr(data, "tactile_shear_force", None)
        rgb = getattr(data, "tactile_rgb_image", None)
        if nf is not None:
            nf_e = nf[ei].detach().cpu()
            nf_e = torch.nan_to_num(nf_e, nan=0.0, posinf=0.0, neginf=0.0)
            norm_list.append(nf_e)
        if sf is not None:
            sf_e = sf[ei].detach().cpu()
            sf_e = torch.nan_to_num(sf_e, nan=0.0, posinf=0.0, neginf=0.0)
            shear_list.append(sf_e)
        if rgb is not None:
            rgb_e = _img_to_uint8_torch(rgb[ei]).detach().cpu()
            rgb_list.append(rgb_e)
    tactile_hw = None
    try:
        tactile_hw = env.cfg.scene._tactile_params()["tactile_array_size"]
    except Exception:
        pass
    if len(norm_list) == num_tactile and tactile_hw is not None:
        stacked = torch.stack(norm_list, dim=0)
        out["tactile_normal_force"] = stacked.reshape(num_tactile, tactile_hw[0], tactile_hw[1], 1).numpy()
    if len(shear_list) == num_tactile and tactile_hw is not None:
        stacked = torch.stack(shear_list, dim=0)
        out["tactile_shear_force"] = stacked.reshape(num_tactile, tactile_hw[0], tactile_hw[1], 2).numpy()
    if len(rgb_list) == num_tactile:
        out["tactile_rgb_image"] = torch.stack(rgb_list, dim=0).numpy()
    return out


def _read_third_person_from_scene(env, ei: int) -> dict[str, np.ndarray]:
    """RGB only. Camera pose stays out of the control/IL contract."""
    if "third_person_camera" not in env.scene.sensors:
        return {}
    cam = env.scene["third_person_camera"]
    rgb = cam.data.output.get("rgb", None) if hasattr(cam, "data") else None
    if rgb is None:
        return {}
    return {"third_person_camera": _img_to_uint8_torch(rgb[ei]).detach().cpu().numpy()}


def _tactile_contact_strength(env, ei: int) -> float:
    """Peak normal force across TacSL finger sensors (PhysX-grounded when V2 enabled)."""
    peak = 0.0
    for name in TACTILE_SENSOR_NAMES:
        if name not in env.scene.sensors:
            continue
        try:
            data = env.scene[name].data
            nf = getattr(data, "tactile_normal_force", None)
            if nf is None:
                continue
            val = float(torch.nan_to_num(nf[ei], nan=0.0).abs().max().item())
            peak = max(peak, val)
        except Exception:
            continue
    return peak


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
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(euler, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _is_pick_close_frame(name: str) -> bool:
    return "Pick" in name and name.endswith("_03")


def _use_ik_for_frame(name: str) -> bool:
    """Recorded arm_joint_pos is authoritative for replay; IK only if arm data missing."""
    return False


def _wrist_pos_for_frame(frame: dict[str, Any]) -> np.ndarray:
    """Return the recorded open-loop wrist waypoint without object-state adaptation."""
    return np.asarray(frame["marker_pos_w"], dtype=np.float64)[:3]


# Shadow order: WRJ2, WRJ1, FFJ4/3/2/1, MFJ*, RFJ*, LFJ5/4/3/2/1, THJ5/4/3/2/1.
_SHADOW_FLEX_IDX = (3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 17, 18, 21, 22, 23)


def _shadow_token_index(joint_name: str, shadow_names: list[str]) -> int | None:
    """Map a robot joint name onto ShadowHand order with token boundaries."""
    best_i = None
    best_len = -1
    for si, sn in enumerate(shadow_names):
        if re.search(rf"(^|[^A-Za-z0-9]){re.escape(sn)}([^A-Za-z0-9]|$)", joint_name):
            if len(sn) > best_len:
                best_i = si
                best_len = len(sn)
    return best_i


def _tendon_couple_distal(hand_j: np.ndarray) -> np.ndarray:
    """Shadow distal joints follow PIP (FJ1/THJ1 were recorded as 0 by MediaPipe)."""
    out = np.asarray(hand_j, dtype=np.float64).copy()
    for i1, i2 in ((5, 4), (9, 8), (13, 12), (18, 17), (23, 22)):
        if i1 < out.size and i2 < out.size and abs(out[i1]) < 0.05:
            out[i1] = float(out[i2])
    return out


def _overlay_thumb(hand_j: np.ndarray, close_pose: np.ndarray) -> np.ndarray:
    """Keep recorded finger joints; take thumb opposition from the close pose."""
    out = np.asarray(hand_j, dtype=np.float64).copy()
    src = np.asarray(close_pose, dtype=np.float64)
    if out.size >= 24 and src.size >= 24:
        out[19:24] = src[19:24]
    return out


def _boost_hand_close(hand_j: np.ndarray, scale: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Optional extra flexion. Scale 1.0 keeps the recorded close pose."""
    out = np.asarray(hand_j, dtype=np.float64).copy()
    if abs(float(scale) - 1.0) < 1e-6:
        return out
    for i in _SHADOW_FLEX_IDX:
        if i >= min(out.size, lo.size, hi.size) or out[i] <= 0.05:
            continue
        cur = float(out[i])
        hi_i, lo_i = float(hi[i]), float(lo[i])
        toward_hi = cur + 0.35 * (hi_i - cur) * min(max(float(scale) - 1.0, 0.0), 1.0)
        out[i] = float(np.clip(max(cur * float(scale), toward_hi), lo_i, hi_i))
    return out


def _squeeze_hand_from_tactile(hand_j: np.ndarray, steps: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """After tactile contact, keep closing flexors. Trigger is contact, not object pose."""
    out = np.asarray(hand_j, dtype=np.float64).copy()
    amt = min(0.35, max(0.0, float(steps)) * 0.001)
    for i in _SHADOW_FLEX_IDX:
        if i >= min(out.size, lo.size, hi.size):
            continue
        out[i] = float(np.clip(out[i] + amt * (float(hi[i]) - out[i]), float(lo[i]), float(hi[i])))
    return out


def _parser():
    p = argparse.ArgumentParser(description="Play a canonical single-arm full-trajectory business phase.")
    p.add_argument("--phase", type=int, choices=(1, 2, 3), default=1, help="Single-arm business phase (default: 1).")
    p.add_argument("--keyframe-json", type=str, default="", help="Optional keyframe JSON override; default is selected by --phase.")
    p.add_argument("--task", type=str, default="", help="Optional task alias or registered Gym ID; default is selected by --phase.")
    p.add_argument("--num-envs", type=int, default=1, help='Number of parallel simulation environments.')
    p.add_argument("--fps", type=float, default=30.0, help='Target control and recording loop frequency in frames per second.')
    p.add_argument("--num-episodes", type=int, default=1, help='Number of episodes to attempt and save.')
    p.add_argument("--object-init-index", type=int, default=0, help='Object initialization preset index; -1 uses the trajectory/keyframe default.')
    p.add_argument("--object-xy-noise", type=float, default=0.03, help='Uniform object-reset position noise in the world XY plane, in meters.')
    p.add_argument("--hand-noise", type=float, default=0.0, help='Uniform random hand-joint perturbation magnitude in radians.')
    p.add_argument("--success-z-threshold", type=float, default=0.4, help='Object world-height threshold in meters used to label successful pickup.')
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), help='Robot arm-base world position as X Y Z in meters.')
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0), help='Robot arm-base XYZ Euler orientation in radians.')
    p.add_argument("--arm-pos-tol", type=float, default=0.02, help='Maximum arm-joint absolute error in radians for a keyframe to count as reached.')
    p.add_argument("--hand-pos-tol", type=float, default=0.08, help='Maximum hand-joint absolute error in radians for a keyframe to count as reached.')
    p.add_argument("--stable-steps", type=int, default=15, help='Consecutive in-tolerance steps required before advancing to the next keyframe.')
    p.add_argument("--max-steps-per-frame", type=int, default=400, help='Maximum simulation steps allowed for each trajectory keyframe.')
    p.add_argument("--post-arm-reached-steps", type=int, default=30, help='Extra settling steps after the arm reaches a keyframe before advancing.')
    p.add_argument("--action-smoothing", type=float, default=0.0, help='Exponential smoothing factor for commanded actions in [0, 1). High values prevent grasp closure.')
    p.add_argument("--record-step-interval", type=int, default=1, help='Record one observation every N simulation steps.')
    p.add_argument("--tactile-reset-settle-steps", type=int, default=8, help='Zero-action steps after reset so tactile and camera sensors settle.')
    p.add_argument("--object-settle-steps", type=int, default=60, help='Extra zero-action steps after reset so the object rests on the bin floor before replay.')
    p.add_argument(
        "--pick-dwell-steps",
        type=int,
        default=120,
        help="Minimum sim steps at Pick*_03 with arm reached, for PhysX contact/friction before advancing.",
    )
    p.add_argument(
        "--use-recorded-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer keyframe arm_joint_pos over marker IK (matches recorded trajectory).",
    )
    p.add_argument(
        "--pick-hand-close-scale",
        type=float,
        default=1.22,
        help="Multiply Pick*_03 finger flexion joints (>0.05 rad) by this factor for tighter PhysX grasp.",
    )
    p.add_argument(
        "--tactile-contact-threshold",
        type=float,
        default=1.0e-4,
        help="At Pick*_03, require peak tactile normal force >= this before advancing (0=disable gate).",
    )
    p.add_argument(
        "--bench-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strict ViTacBench contract: no object teleport/alignment, no object-pose IK adaptation, "
            "no fingertip/object distance gate, no palm carry, and no robot joint teleport."
        ),
    )
    p.add_argument(
        "--hand-max-step-rad",
        type=float,
        default=0.0,
        help="Per-step cap on all hand joints toward recorded targets (0=unlimited). Do not use with coupled distals.",
    )
    p.add_argument(
        "--thumb-oppose-step-rad",
        type=float,
        default=0.02,
        help="At Pick*_03, per-step cap on THJ5 only so opposition rotates without collapsing THJ2 (0=unlimited).",
    )
    p.add_argument(
        "--arm-max-step-rad",
        type=float,
        default=0.0,
        help="Per-step cap on arm joint motion toward target (0=unlimited).",
    )
    p.add_argument(
        "--lift-arm-step-rad",
        type=float,
        default=0.008,
        help="At Finish, per-step cap on arm joints so the lift is slow enough to keep a tactile pinch.",
    )
    p.add_argument("--record-path", type=str, default=None, help='Output directory for episode NPZ files and metadata.')
    p.add_argument("--record-env-index", type=int, default=0, help='Parallel environment index whose observation stream is saved.')
    p.add_argument(
        "--save-outcome",
        choices=("success", "completed", "all"),
        default="success",
        help=(
            "Which attempts may be written: success uses canonical environment success signals with the "
            "legacy pickup-height fallback; completed also saves trajectories that reach their final frame; "
            "all additionally saves interrupted/done attempts."
        ),
    )
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args = _parser().parse_args()
    bench_mode = bool(args.bench_mode)
    if not bench_mode:
        raise ValueError("Blind-grasp phase replay only supports strict tactile ViTacBench mode.")
    if float(args.tactile_contact_threshold) <= 0.0:
        args.tactile_contact_threshold = 1.0e-4
    app = AppLauncher(args).app
    import ViTacLab.tasks  # noqa: F401
    repo = _repo_root()
    src = repo / "source"
    if str(src) not in sys.path: sys.path.insert(0, str(src))
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import VideoTeleopControl

    phase_cfg = _PHASES[args.phase]
    sequence = list(phase_cfg["sequence"])
    default_keyframe = HERE / "pose_keyframes" / str(phase_cfg["keyframe"])
    kf_path = Path(args.keyframe_json).expanduser() if args.keyframe_json else default_keyframe
    if not kf_path.is_absolute(): kf_path = (repo / kf_path).resolve()
    doc = json.loads(kf_path.read_text(encoding="utf-8"))
    kfs = {str(k["name"]): k for k in doc.get("keyframes", [])}
    miss = [n for n in sequence if n not in kfs]
    if miss: raise RuntimeError(f"Missing keyframes: {miss}")
    frames = [kfs[n] for n in sequence]

    task = str(args.task).strip() or str(phase_cfg["task"])
    if args.task:
        env_entry, cfg_entry, _ = resolve_env_cfg_entries(task=task)
    else:
        env_entry, cfg_entry = str(doc["env_entry"]), str(doc["cfg_entry"])
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)
    cfg = CfgCls(); cfg.scene.num_envs = max(1, int(args.num_envs)); cfg.device = getattr(args, "device", None) or "cuda:0"
    cfg.object_init_choice = int(args.object_init_index); cfg.enable_cameras = True
    noise = max(0.0, float(args.object_xy_noise))
    if hasattr(cfg, "object_reset_pos_x_range"):
        cfg.object_reset_pos_x_range = (-noise, noise)
    if hasattr(cfg, "object_reset_pos_y_range"):
        cfg.object_reset_pos_y_range = (-noise, noise)
    if hasattr(cfg, "episode_length_s"):
        cfg.episode_length_s = max(float(cfg.episode_length_s), 300.0)
    if hasattr(cfg, "act_moving_average"):
        cfg.act_moving_average = 1.0  # replay: track joint targets in one step (0.3 prevents hand closure)
    print(
        "[INFO] replay uses registered env physics; tactile follows the BlindGrasp scene "
        "(bin contact allowed during probe). No object-pose IK or finger retarget. "
        f"MediaPipe FJ1 follows FJ2; Pick*_03 THJ5 slew={float(args.thumb_oppose_step_rad):.3f} rad/step.",
        flush=True,
    )
    env = EnvCls(cfg)
    # Drive the env through its normal joint-position action path. Dummy-zero
    # steps with _use_rl_control=False left the fingers stuck open (hand_err~0.6).
    if hasattr(env, "_use_rl_control"):
        env._use_rl_control = True
    if hasattr(cfg, "act_moving_average"):
        env.cfg.act_moving_average = 1.0
    env._replay_q = None
    _base_apply_action = type(env)._apply_action

    def _apply_replay_action(_env=env):
        # Position targets must be written every physics substep. A one-shot
        # apply_joint_targets() before env.step() is overwritten or dropped.
        q = getattr(_env, "_replay_q", None)
        if q is not None:
            _env.apply_joint_targets(q)
            return
        _base_apply_action(_env)

    env._apply_action = _apply_replay_action
    _reset_and_warmup(env, act_dim=env.num_actions, settle_steps=args.tactile_reset_settle_steps, object_settle_steps=args.object_settle_steps)
    robot = env.robot; jn = list(robot.joint_names); sh = shadowhand_joint_names(); act_dim = env.num_actions
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_ids = [i for i, n in enumerate(jn) if re.match(arm_expr, n)]
    hand_ids = [i for i, n in enumerate(jn) if re.match(hand_expr, n)]
    lo_all = _np(env.robot_dof_lower_limits[0]); hi_all = _np(env.robot_dof_upper_limits[0])
    sh_lo, sh_hi = np.full(24, -1.5), np.full(24, 1.5)
    sh_robot_idx = [-1] * 24
    for idx in hand_ids:
        si = _shadow_token_index(jn[idx], sh)
        if si is None:
            continue
        sh_lo[si], sh_hi[si] = lo_all[idx], hi_all[idx]
        sh_robot_idx[si] = idx
    print("[INFO] replay joint map:", flush=True)
    print(f"  robot.joint_names ({len(jn)}): {jn}", flush=True)
    print(f"  actuated={[(int(i), jn[int(i)]) for i in env.actuated_dof_indices]}", flush=True)
    for si, sn in enumerate(sh):
        idx = sh_robot_idx[si]
        mapped = f"{idx}:{jn[idx]}" if idx >= 0 else "UNMAPPED"
        print(f"  {sn:5s} -> {mapped} lo={sh_lo[si]:.3f} hi={sh_hi[si]:.3f}", flush=True)
    print(
        f"[INFO] object_init_choice={int(args.object_init_index)} "
        f"( -1=sample A/B each reset; 0=slot A 0.62; 1=slot B 0.50 )",
        flush=True,
    )
    print(
        f"[INFO] policy obs includes object state: {bool(getattr(cfg, 'include_object_state_in_policy_obs', True))}",
        flush=True,
    )

    def cur_arm():
        q = _np(robot.data.joint_pos[0]); return np.array([float(q[i]) for i in arm_ids], dtype=np.float64)
    def cur_hand():
        out = np.zeros(24, dtype=np.float64); q = _np(robot.data.joint_pos[0])
        for si, idx in enumerate(sh_robot_idx):
            if idx >= 0:
                out[si] = float(q[idx])
        return out
    def build_full_joint_pos(arm_j, hand_j):
        full = _np(robot.data.joint_pos[0]).astype(np.float64, copy=True)
        for i, idx in enumerate(arm_ids):
            if i < len(arm_j):
                full[idx] = float(arm_j[i])
        for si, idx in enumerate(sh_robot_idx):
            if idx >= 0 and si < len(hand_j):
                full[idx] = float(hand_j[si])
        return full
    def build_actuated_joint_pos(arm_j, hand_j):
        full = build_full_joint_pos(arm_j, hand_j)
        return full[np.array(env.actuated_dof_indices, dtype=np.int64)]

    def build_action(arm_j, hand_j):
        actuated = build_actuated_joint_pos(arm_j, hand_j)
        lo = _np(env.robot_dof_lower_limits[0, env.actuated_dof_indices])
        hi = _np(env.robot_dof_upper_limits[0, env.actuated_dof_indices])
        scale = np.where(hi - lo > 1e-6, 2.0 * (actuated - lo) / (hi - lo) - 1.0, 0.0)
        return torch.tensor(np.clip(scale, -1.0, 1.0), dtype=torch.float32, device=env.device).unsqueeze(0)

    def _min_dwell_steps(frame: dict[str, Any]) -> int:
        name = str(frame.get("name", ""))
        if _is_pick_close_frame(name):
            return max(int(args.pick_dwell_steps), int(args.post_arm_reached_steps) + 20)
        if "Finish" in name:
            return max(25, int(args.post_arm_reached_steps))
        if "Touch" in name:
            return max(80, int(args.post_arm_reached_steps) + 40)
        if "Pick" in name:
            return max(80, int(args.post_arm_reached_steps))
        return 0

    control = VideoTeleopControl(T_world_arm_base=_T(np.array(args.arm_base_pos), np.array(args.arm_base_euler)))
    action_smoothing = float(np.clip(args.action_smoothing, 0.0, 0.999))

    def _max_steps_for_frame(frame: dict[str, Any]) -> int:
        base = max(1, int(args.max_steps_per_frame))
        name = str(frame.get("name", ""))
        if _is_pick_close_frame(name):
            return max(base, int(args.pick_dwell_steps) + 600)
        if "Finish" in name:
            return max(base, 800)
        if "Pick" in name:
            return max(base, 400)
        return base
    rec_dir = os.path.abspath(os.path.expanduser(args.record_path)) if args.record_path else os.path.join(os.getcwd(), "play_records", f"single_phase{args.phase}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(rec_dir, exist_ok=True)
    with open(os.path.join(rec_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": int(args.phase),
                "task": task,
                "sequence": sequence,
                "object_init_index": int(args.object_init_index),
                "object_xy_noise": float(args.object_xy_noise),
                "hand_noise": float(args.hand_noise),
                "success_z_threshold": float(args.success_z_threshold),
                "keyframe_json": str(kf_path),
                "save_outcome": str(args.save_outcome),
                "pick_dwell_steps": int(args.pick_dwell_steps),
                "action_smoothing": float(args.action_smoothing),
                "use_recorded_arm": bool(args.use_recorded_arm),
                "tactile_contact_threshold": float(args.tactile_contact_threshold),
                "bench_mode": bench_mode,
                "policy_input_contract": "tactile_plus_robot_proprioception",
                "object_geometry_in_control": False,
                "demo_third_person_camera": True,
            },
            f,
            indent=2,
        )
    ei = max(0, min(int(args.record_env_index), env.num_envs - 1)); target_dt = 1.0 / max(1e-3, float(args.fps)); succ = 0
    for ep in range(int(args.num_episodes)):
        env._replay_q = None
        _reset_and_warmup(env, act_dim=act_dim, settle_steps=args.tactile_reset_settle_steps, object_settle_steps=args.object_settle_steps)
        pick_contact_seen = False
        pick_contact_streak = 0
        closed_hand = None
        squeeze_steps = 0
        lift_tact_max = 0.0
        bufs: dict[str, list[np.ndarray]] = {}; ep_steps = 0; zmax = -1e9; z_fallback_ok = False; tact_peak_max = 0.0
        success_latched = False; success_signal_seen = False; success_source = "unavailable"
        prev_actions = torch.zeros(env.num_envs, act_dim, device=env.device)
        noisy_hands = [np.clip(np.asarray(fr["hand_joint_pos_shadow_order"], dtype=np.float64)[:24] + np.random.uniform(-float(args.hand_noise), float(args.hand_noise), size=24), sh_lo, sh_hi) for fr in frames]
        fi = 0; stable = 0; countdown = -1; sif = 0; dwell = 0; pick_contact_seen = False; pick_contact_streak = 0
        while app.is_running() and fi < len(frames):
            t0 = time.time(); fr = frames[fi]
            frame_name = str(fr.get("name", ""))
            pick_close = _is_pick_close_frame(frame_name)
            marker_euler = np.asarray(fr["marker_euler_xyz"], dtype=np.float64)[:3]
            hand_j = noisy_hands[fi]
            hand_tol = float(args.hand_pos_tol)
            if _is_pick_close_frame(frame_name):
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
                wrist_pos = _wrist_pos_for_frame(fr)
                tgt = control.compute(wrist_pos, marker_euler, hand_j)
                if tgt is None:
                    des_arm = cur_arm()
                else:
                    des_arm = np.asarray(tgt.arm_joints, dtype=np.float64)
            des_hand = _tendon_couple_distal(np.asarray(hand_j, dtype=np.float64)[:24])
            if closed_hand is not None and ("Pick" in frame_name or "Finish" in frame_name):
                des_hand = _squeeze_hand_from_tactile(closed_hand, squeeze_steps, sh_lo, sh_hi)
            cur_arm_j = cur_arm()
            cur_hand_j = cur_hand()
            mar = float(args.arm_max_step_rad)
            if mar > 0.0:
                arm_cmd = cur_arm_j + np.clip(des_arm - cur_arm_j, -mar, mar)
            else:
                arm_cmd = np.asarray(des_arm, dtype=np.float64).copy()
            lift_step = float(args.lift_arm_step_rad)
            if "Finish" in frame_name and lift_step > 0.0:
                if sif == 0 or not hasattr(env, "_lift_arm_cmd"):
                    env._lift_arm_cmd = cur_arm_j.copy()
                env._lift_arm_cmd = env._lift_arm_cmd + np.clip(
                    des_arm - env._lift_arm_cmd, -lift_step, lift_step
                )
                arm_cmd = env._lift_arm_cmd
            elif hasattr(env, "_lift_arm_cmd"):
                delattr(env, "_lift_arm_cmd")
            mhr = float(args.hand_max_step_rad)
            if mhr > 0.0:
                hand_cmd = cur_hand_j + np.clip(des_hand - cur_hand_j, -mhr, mhr)
            else:
                hand_cmd = np.asarray(des_hand, dtype=np.float64).copy()
            # Recorded Pick*_03 jumps THJ5 by ~1.3 rad in one keyframe. Command that
            # instantly and PhysX drives THJ2 the wrong way. Other joints stay at target.
            th_step = float(args.thumb_oppose_step_rad)
            if pick_close and th_step > 0.0 and hand_cmd.size > 19:
                if sif == 0 or not hasattr(env, "_thj5_cmd"):
                    env._thj5_cmd = float(cur_hand_j[19])
                env._thj5_cmd = float(
                    env._thj5_cmd + np.clip(float(des_hand[19]) - float(env._thj5_cmd), -th_step, th_step)
                )
                hand_cmd[19] = env._thj5_cmd
            elif hasattr(env, "_thj5_cmd"):
                delattr(env, "_thj5_cmd")
            full_j = build_full_joint_pos(arm_cmd, hand_cmd)
            actions = build_action(arm_cmd, hand_cmd)
            ae = float(np.max(np.abs(cur_arm_j - des_arm)))
            he = float(np.max(np.abs(cur_hand_j - des_hand)))
            arm_ok = ae <= float(args.arm_pos_tol)
            conv = arm_ok and he <= hand_tol
            if env.num_envs > 1:
                actions = actions.expand(env.num_envs, -1).clone()
            step_actions = action_smoothing * prev_actions + (1.0 - action_smoothing) * actions
            prev_actions = step_actions.clone()
            replay_q = torch.tensor(full_j, dtype=torch.float32, device=env.device).unsqueeze(0)
            if env.num_envs > 1:
                replay_q = replay_q.expand(env.num_envs, -1).clone()
            env._replay_q = replay_q
            out = env.step(step_actions)
            if isinstance(out, tuple) and len(out) >= 5:
                obs, term, trunc, infos = out[0], out[2], out[3], out[4]
            elif isinstance(out, tuple) and len(out) >= 4:
                obs, term, trunc, infos = out[0], out[2], out[3], {}
            else:
                obs, term, trunc, infos = out, None, None, {}
            ep_steps += 1
            if ep_steps % max(1, int(args.record_step_interval)) == 0:
                row = merge_record_row(obs, ei, actions=actions, robot=robot)
                tactile_row = _read_tactile_from_scene(env, ei)
                row.update(tactile_row)
                row.update(_read_third_person_from_scene(env, ei))
                row = {k: v for k, v in row.items() if k in _POLICY_RECORD_KEYS}
                if not bufs:
                    for k in row: bufs[k] = []
                    bufs["task_episode_sim_step"] = []
                for k, v in row.items(): bufs.setdefault(k, []).append(np.asarray(v))
                bufs["task_episode_sim_step"].append(np.asarray(float(ep_steps), dtype=np.float32))
            signal_available, signal_success, signal_source_now = extract_success_signal(infos, obs, env, ei)
            if signal_available:
                success_signal_seen = True
                success_latched = success_latched or signal_success
                success_source = signal_source_now
            z = float((env.object.data.root_pos_w[0] - env.scene.env_origins[0])[2].item()); zmax = max(zmax, z); z_fallback_ok = z_fallback_ok or (z > float(args.success_z_threshold))
            tact_peak = _tactile_contact_strength(env, ei)
            tact_peak_max = max(tact_peak_max, tact_peak)
            if "Finish" in frame_name:
                lift_tact_max = max(lift_tact_max, tact_peak)
            if pick_close and tact_peak >= float(args.tactile_contact_threshold):
                pick_contact_streak += 1
            else:
                pick_contact_streak = 0
            if pick_close and pick_contact_streak >= 20:
                if closed_hand is None:
                    closed_hand = np.asarray(hand_cmd, dtype=np.float64).copy()
                squeeze_steps += 1
            sif += 1; stable = stable + 1 if conv else 0
            min_dwell = _min_dwell_steps(fr)
            if arm_ok and min_dwell > 0:
                dwell += 1
            elif not arm_ok:
                dwell = 0
            if countdown < 0 and arm_ok and not conv:
                countdown = max(0, int(args.post_arm_reached_steps))
            elif countdown >= 0 and not conv:
                countdown -= 1
            max_sif = _max_steps_for_frame(fr)
            dwell_ok = min_dwell <= 0 or dwell >= min_dwell
            live_contact = pick_close and pick_contact_streak >= 20
            pick_contact_seen = live_contact
            if pick_close:
                # Lift only after an early tactile pinch plus a short squeeze.
                # A last-frame brush at max_sif scoops the cube on the fingertips.
                should_advance = live_contact and arm_ok and squeeze_steps >= 40
                if live_contact and sif % 20 == 0:
                    print(
                        f"[INFO] {frame_name} tactile grasp ready: tact={tact_peak:.2e} "
                        f"streak={pick_contact_streak} squeeze={squeeze_steps} "
                        f"arm_err={ae:.3f} hand_err={he:.3f} "
                        f"THJ5={cur_hand_j[19]:.3f}/{des_hand[19]:.3f} "
                        f"THJ2={cur_hand_j[22]:.3f}/{des_hand[22]:.3f}",
                        flush=True,
                    )
                if sif >= 200 and not live_contact:
                    print(
                        f"[WARN] {frame_name} no early tactile pinch by step {sif} "
                        f"(tact_peak={tact_peak:.2e}, streak={pick_contact_streak}); aborting lift.",
                        flush=True,
                    )
                    break
                if sif >= max_sif and not live_contact:
                    print(
                        f"[WARN] {frame_name} ended without sustained tactile contact "
                        f"(tact_peak={tact_peak:.2e}, streak={pick_contact_streak}); aborting lift.",
                        flush=True,
                    )
                    break
            elif "Pick" in frame_name:
                hand_ready = he <= max(hand_tol, 0.18)
                should_advance = (
                    (arm_ok and hand_ready and dwell_ok and stable >= max(1, int(args.stable_steps)))
                    or sif >= max_sif
                )
            else:
                should_advance = (
                    (conv and stable >= max(1, int(args.stable_steps)) and dwell_ok)
                    or (arm_ok and dwell_ok and sif >= max(15, min_dwell))
                    or sif >= max_sif
                    or (arm_ok and not conv and countdown == 0 and min_dwell <= 0)
                )
            if "Finish" in frame_name and sif > 0 and sif % 25 == 0:
                print(
                    f"[INFO] {frame_name} lift step={sif} cube_z={z:.4f} zmax={zmax:.4f} "
                    f"tact={tact_peak:.2e} arm_err={ae:.3f}",
                    flush=True,
                )
            if ("Pick" in frame_name or "Touch" in frame_name or "Finish" in frame_name) and sif > 0 and sif % 100 == 0:
                herr = np.abs(cur_hand_j - des_hand)
                wi = int(np.argmax(herr))
                tgt_q = _np(robot.data.joint_pos_target[0])
                wi_idx = sh_robot_idx[wi]
                tgt_w = float(tgt_q[wi_idx]) if wi_idx >= 0 else float("nan")
                obj_xy = _np(env.object.data.root_pos_w[0] - env.scene.env_origins[0])
                try:
                    ee_idx = env._resolve_body_index_by_keywords(("wrist_3", "wrist3", "palm"))
                    wrist_xy = _np(robot.data.body_pos_w[0, ee_idx] - env.scene.env_origins[0])
                except Exception:
                    wrist_xy = np.array([float("nan")] * 3)
                def _link_xyz(*keys):
                    try:
                        idx = env._resolve_body_index_by_keywords(keys)
                        return _np(robot.data.body_pos_w[0, idx] - env.scene.env_origins[0])
                    except Exception:
                        return np.array([float("nan")] * 3)
                palm_xyz = _link_xyz("palm")
                ff_xyz = _link_xyz("ffdistal")
                th_xyz = _link_xyz("thdistal")
                print(
                    f"[INFO] {frame_name} step={sif}/{max_sif} tact={tact_peak:.2e} "
                    f"contact={pick_contact_seen} streak={pick_contact_streak} "
                    f"hand_err={he:.3f} worst={sh[wi]}:{herr[wi]:.3f} "
                    f"cur={cur_hand_j[wi]:.3f} des={des_hand[wi]:.3f} tgt={tgt_w:.3f} "
                    f"THJ5={cur_hand_j[19]:.3f}/{des_hand[19]:.3f} "
                    f"THJ2={cur_hand_j[22]:.3f}/{des_hand[22]:.3f} "
                    f"FFJ3={cur_hand_j[3]:.3f}/{des_hand[3]:.3f} "
                    f"cube=({obj_xy[0]:.3f},{obj_xy[1]:.3f},{obj_xy[2]:.3f}) "
                    f"wrist=({wrist_xy[0]:.3f},{wrist_xy[1]:.3f},{wrist_xy[2]:.3f}) "
                    f"palm=({palm_xyz[0]:.3f},{palm_xyz[1]:.3f},{palm_xyz[2]:.3f}) "
                    f"ff=({ff_xyz[0]:.3f},{ff_xyz[1]:.3f},{ff_xyz[2]:.3f}) "
                    f"th=({th_xyz[0]:.3f},{th_xyz[1]:.3f},{th_xyz[2]:.3f})",
                    flush=True,
                )
            if should_advance:
                if _is_pick_close_frame(frame_name):
                    print(
                        f"[INFO] leave {frame_name}: tact_peak={tact_peak:.2e} "
                        f"contact={pick_contact_seen} hand_err={he:.3f} "
                        f"arm_err={ae:.3f} steps={sif}",
                        flush=True,
                    )
                fi += 1; stable = 0; countdown = -1; sif = 0; dwell = 0; pick_contact_seen = False; pick_contact_streak = 0
            if torch.is_tensor(term) and torch.is_tensor(trunc) and bool(torch.any(term | trunc).item()):
                print(
                    f"[WARN] env timed out during {frame_name} at step={sif}; "
                    f"ending episode instead of resetting the object back into the bin.",
                    flush=True,
                )
                break
            dt = target_dt - (time.time() - t0)
            if dt > 0 and not bool(getattr(args, "headless", False)):
                time.sleep(dt)
        # Extraction only counts if the cube was pinched during the lift.
        # Probe tactile + a fingertip scoop (Finish tact=0) is not a grasp.
        extracted = zmax > float(args.success_z_threshold)
        pinched_lift = lift_tact_max >= float(args.tactile_contact_threshold)
        final_success = bool(extracted and pinched_lift)
        if final_success:
            success_source = "extracted_with_lift_tactile"
        elif extracted and not pinched_lift:
            success_source = "scooped_without_lift_tactile"
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
            p, t_len = _save_npz(rec_dir, f"episode_{ep:04d}_{outcome}.npz", bufs)
            print(
                f"[INFO] saved {outcome} -> {p} "
                f"(T={t_len}, success_source={success_source}, z_max={zmax:.4f}, "
                f"tact_peak_max={tact_peak_max:.2e}, lift_tact_max={lift_tact_max:.2e})"
            )
        else:
            print(
                f"[INFO] skipped {outcome} episode {ep:04d} "
                f"(T={len(bufs.get('task_episode_sim_step', []))}, "
                f"success_source={success_source}, z_max={zmax:.4f})"
            )
        if not app.is_running(): break
    print(f"[INFO] done: {succ}/{int(args.num_episodes)} success")
    env.close(); app.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
