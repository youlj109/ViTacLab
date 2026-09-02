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


def _reset_and_warmup(env, act_dim: int, settle_steps: int) -> list[str]:
    """Reset env, settle with zero actions, then capture fresh tactile nominal frames."""
    env.reset()
    zero_actions = torch.zeros(env.num_envs, act_dim, device=env.device)
    for _ in range(max(0, int(settle_steps))):
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
    if "third_person_camera" in scene.sensors:
        cam = scene["third_person_camera"]
        cam_rgb = cam.data.output.get("rgb", None)
        if cam_rgb is not None:
            out["third_person_camera"] = cam_rgb[ei].detach().cpu().numpy()
    return out



def _save_npz(out_dir: str, name: str, bufs: dict[str, list[np.ndarray]]):
    if not bufs: return None, 0
    first = next(iter(bufs.keys()))
    if not bufs[first]: return None, 0
    path = os.path.join(out_dir, name)
    t_len = len(bufs[first])
    np.savez_compressed(path, **{k: np.stack(v, 0) for k, v in bufs.items() if v})
    return path, t_len


def _T(pos, euler):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(euler, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


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
    p.add_argument("--hand-noise", type=float, default=0.02, help='Uniform random hand-joint perturbation magnitude in radians.')
    p.add_argument("--success-z-threshold", type=float, default=0.4, help='Object world-height threshold in meters used to label successful pickup.')
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), help='Robot arm-base world position as X Y Z in meters.')
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0), help='Robot arm-base XYZ Euler orientation in radians.')
    p.add_argument("--arm-pos-tol", type=float, default=0.05, help='Maximum arm-joint absolute error in radians for a keyframe to count as reached.')
    p.add_argument("--hand-pos-tol", type=float, default=0.08, help='Maximum hand-joint absolute error in radians for a keyframe to count as reached.')
    p.add_argument("--stable-steps", type=int, default=15, help='Consecutive in-tolerance steps required before advancing to the next keyframe.')
    p.add_argument("--max-steps-per-frame", type=int, default=240, help='Maximum simulation steps allowed for each trajectory keyframe.')
    p.add_argument("--post-arm-reached-steps", type=int, default=30, help='Extra settling steps after the arm reaches a keyframe before advancing.')
    p.add_argument("--action-smoothing", type=float, default=0.75, help='Exponential smoothing factor for commanded actions in [0, 1).')
    p.add_argument("--record-step-interval", type=int, default=1, help='Record one observation every N simulation steps.')
    p.add_argument("--tactile-reset-settle-steps", type=int, default=8, help='Zero-action steps after reset so tactile and camera sensors settle.')
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
    env = EnvCls(cfg); _reset_and_warmup(env, act_dim=env.num_actions, settle_steps=args.tactile_reset_settle_steps)
    robot = env.robot; jn = list(robot.joint_names); sh = shadowhand_joint_names(); act_dim = env.num_actions
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_ids = [i for i, n in enumerate(jn) if re.match(arm_expr, n)]
    hand_ids = [i for i, n in enumerate(jn) if re.match(hand_expr, n)]
    lo_all = _np(env.robot_dof_lower_limits[0]); hi_all = _np(env.robot_dof_upper_limits[0])
    sh_lo, sh_hi = np.full(24, -1.5), np.full(24, 1.5)
    for si, sn in enumerate(sh):
        for idx in hand_ids:
            if sn in jn[idx] or jn[idx].endswith(sn): sh_lo[si], sh_hi[si] = lo_all[idx], hi_all[idx]; break

    def hand_for_name(name, hv):
        for si, sn in enumerate(sh):
            if sn in name or name.endswith(sn): return float(hv[si])
        return 0.0
    def cur_arm():
        q = _np(robot.data.joint_pos[0]); return np.array([float(q[i]) for i in arm_ids], dtype=np.float64)
    def cur_hand():
        out = np.zeros(24, dtype=np.float64); q = _np(robot.data.joint_pos[0])
        for si, sn in enumerate(sh):
            for idx in hand_ids:
                if sn in jn[idx] or jn[idx].endswith(sn): out[si] = float(q[idx]); break
        return out
    def build_action(arm_j, hand_j):
        full = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_ids):
            if i < len(arm_j): full[idx] = float(arm_j[i])
        for idx in hand_ids: full[idx] = hand_for_name(jn[idx], hand_j)
        act = full[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lo = _np(env.robot_dof_lower_limits[0, env.actuated_dof_indices]); hi = _np(env.robot_dof_upper_limits[0, env.actuated_dof_indices])
        scale = np.where(hi - lo > 1e-6, 2.0 * (act - lo) / (hi - lo) - 1.0, 0.0)
        return torch.tensor(np.clip(scale, -1.0, 1.0), dtype=torch.float32, device=env.device).unsqueeze(0)

    control = VideoTeleopControl(T_world_arm_base=_T(np.array(args.arm_base_pos), np.array(args.arm_base_euler)))
    action_smoothing = float(np.clip(args.action_smoothing, 0.0, 0.999))
    rec_dir = os.path.abspath(os.path.expanduser(args.record_path)) if args.record_path else os.path.join(os.getcwd(), "play_records", f"single_phase{args.phase}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(rec_dir, exist_ok=True)
    with open(os.path.join(rec_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"phase": int(args.phase), "task": task, "sequence": sequence, "object_init_index": int(args.object_init_index), "object_xy_noise": float(args.object_xy_noise), "hand_noise": float(args.hand_noise), "success_z_threshold": float(args.success_z_threshold), "keyframe_json": str(kf_path), "save_outcome": str(args.save_outcome)}, f, indent=2)
    ei = max(0, min(int(args.record_env_index), env.num_envs - 1)); target_dt = 1.0 / max(1e-3, float(args.fps)); succ = 0

    for ep in range(int(args.num_episodes)):
        _reset_and_warmup(env, act_dim=act_dim, settle_steps=args.tactile_reset_settle_steps); bufs: dict[str, list[np.ndarray]] = {}; ep_steps = 0; zmax = -1e9; z_fallback_ok = False
        success_latched = False; success_signal_seen = False; success_source = "unavailable"
        prev_actions = torch.zeros(env.num_envs, act_dim, device=env.device)
        noisy_hands = [np.clip(np.asarray(fr["hand_joint_pos_shadow_order"], dtype=np.float64)[:24] + np.random.uniform(-float(args.hand_noise), float(args.hand_noise), size=24), sh_lo, sh_hi) for fr in frames]
        fi = 0; stable = 0; countdown = -1; sif = 0
        while app.is_running() and fi < len(frames):
            t0 = time.time(); fr = frames[fi]
            marker_pos = np.asarray(fr["marker_pos_w"], dtype=np.float64)[:3]
            marker_euler = np.asarray(fr["marker_euler_xyz"], dtype=np.float64)[:3]
            wrist_pos = marker_pos; hand_j = noisy_hands[fi]
            tgt = control.compute(wrist_pos, marker_euler, hand_j)
            if tgt is None:
                actions = torch.zeros(env.num_envs, act_dim, device=env.device); conv = False; arm_ok = False
            else:
                actions = build_action(np.asarray(tgt.arm_joints, dtype=np.float64), hand_j)
                ae = float(np.max(np.abs(cur_arm() - np.asarray(tgt.arm_joints, dtype=np.float64))))
                he = float(np.max(np.abs(cur_hand() - hand_j)))
                arm_ok = ae <= float(args.arm_pos_tol); conv = arm_ok and he <= float(args.hand_pos_tol)
            if env.num_envs > 1: actions = actions.expand(env.num_envs, -1).clone()
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
                row = merge_record_row(obs, ei, actions=actions, robot=robot)
                tactile_row = _read_tactile_from_scene(env, ei)
                row.update(tactile_row)
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
            sif += 1; stable = stable + 1 if conv else 0
            if countdown < 0 and arm_ok and not conv:
                countdown = max(0, int(args.post_arm_reached_steps))
            elif countdown >= 0 and not conv:
                countdown -= 1
            if stable >= max(1, int(args.stable_steps)) or sif >= max(1, int(args.max_steps_per_frame)) or (arm_ok and not conv and countdown == 0):
                fi += 1; stable = 0; countdown = -1; sif = 0
            if torch.is_tensor(term) and torch.is_tensor(trunc) and bool(torch.any(term | trunc).item()):
                _reset_and_warmup(env, act_dim=act_dim, settle_steps=args.tactile_reset_settle_steps)
                prev_actions.zero_()
                stable = 0; countdown = -1; sif = 0
                continue
            dt = target_dt - (time.time() - t0)
            if dt > 0: time.sleep(dt)
        final_success = bool(success_latched if success_signal_seen else z_fallback_ok)
        if not success_signal_seen:
            success_source = "pickup_z_fallback"
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
                bufs["outcome_success"] = [np.asarray(final_success, dtype=np.bool_) for _ in range(t_rows)]
            p, t_len = _save_npz(rec_dir, f"episode_{ep:04d}_{outcome}.npz", bufs)
            print(
                f"[INFO] saved {outcome} -> {p} "
                f"(T={t_len}, success_source={success_source}, z_max={zmax:.4f})"
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
