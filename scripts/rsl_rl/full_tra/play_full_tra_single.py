#!/usr/bin/env python3
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
SEQ = ["Start01","Start02","Touch01_01","Touch01_02","Pick01_01","Pick01_02","Pick01_03","Finish01"]
KF_JSON = HERE / "pose_keyframes/UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json"


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


def _read_tactile_from_scene(env, ei: int) -> dict[str, np.ndarray]:
    """Read tactile data directly from scene sensors (same as record_full_tra_single.py)."""
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


def _extract_record_row(obs: object, ei: int) -> dict[str, np.ndarray] | None:
    rec = obs.get("record") if isinstance(obs, dict) else None
    if not isinstance(rec, dict): return None
    out = {}
    for k, v in rec.items():
        if torch.is_tensor(v): out[k] = np.asarray(v[max(0, min(ei, int(v.shape[0]) - 1))].detach().cpu().numpy())
    return out or None


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
    p = argparse.ArgumentParser()
    p.add_argument("--keyframe-json", type=str, default=str(KF_JSON))
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--object-init-index", type=int, default=0)
    p.add_argument("--object-xy-noise", type=float, default=0.03)
    p.add_argument("--hand-noise", type=float, default=0.02)
    p.add_argument("--success-z-threshold", type=float, default=0.4)
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--arm-pos-tol", type=float, default=0.05)
    p.add_argument("--hand-pos-tol", type=float, default=0.08)
    p.add_argument("--stable-steps", type=int, default=15)
    p.add_argument("--max-steps-per-frame", type=int, default=240)
    p.add_argument("--post-arm-reached-steps", type=int, default=30)
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
    if str(src) not in sys.path: sys.path.insert(0, str(src))
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import VideoTeleopControl

    kf_path = Path(args.keyframe_json).expanduser()
    if not kf_path.is_absolute(): kf_path = (repo / kf_path).resolve()
    doc = json.loads(kf_path.read_text(encoding="utf-8"))
    kfs = {str(k["name"]): k for k in doc.get("keyframes", [])}
    miss = [n for n in SEQ if n not in kfs]
    if miss: raise RuntimeError(f"Missing keyframes: {miss}")
    frames = [kfs[n] for n in SEQ]

    EnvCls = _load_symbol(str(doc["env_entry"]))
    CfgCls = _load_symbol(str(doc["cfg_entry"]))
    cfg = CfgCls(); cfg.scene.num_envs = max(1, int(args.num_envs)); cfg.device = getattr(args, "device", None) or "cuda:0"
    cfg.object_init_choice = int(args.object_init_index); cfg.enable_cameras = True
    env = EnvCls(cfg); env.reset(); _warmup_tactile_nominal(env)
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
    rec_dir = os.path.abspath(os.path.expanduser(args.record_path)) if args.record_path else os.path.join(os.getcwd(), "play_records", f"step1_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(rec_dir, exist_ok=True)
    with open(os.path.join(rec_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"sequence": SEQ, "object_init_index": int(args.object_init_index), "object_xy_noise": float(args.object_xy_noise), "hand_noise": float(args.hand_noise), "success_z_threshold": float(args.success_z_threshold), "keyframe_json": str(kf_path)}, f, indent=2)
    ei = max(0, min(int(args.record_env_index), env.num_envs - 1)); target_dt = 1.0 / max(1e-3, float(args.fps)); succ = 0

    for ep in range(int(args.num_episodes)):
        env.reset(); _warmup_tactile_nominal(env); bufs: dict[str, list[np.ndarray]] = {}; ep_steps = 0; zmax = -1e9; ok = False
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
            obs, term, trunc = (out[0], out[2], out[3]) if isinstance(out, tuple) and len(out) >= 5 else (out, None, None)
            ep_steps += 1
            if ep_steps % max(1, int(args.record_step_interval)) == 0:
                row = _extract_record_row(obs, ei) or {"joint_pos": _np(robot.data.joint_pos[ei]), "joint_vel": _np(robot.data.joint_vel[ei]), "action": _np(actions[ei])}
                tactile_row = _read_tactile_from_scene(env, ei)
                row.update(tactile_row)
                if not bufs:
                    for k in row: bufs[k] = []
                    bufs["task_episode_sim_step"] = []
                for k, v in row.items(): bufs.setdefault(k, []).append(np.asarray(v))
                bufs["task_episode_sim_step"].append(np.asarray(float(ep_steps), dtype=np.float32))
            z = float((env.object.data.root_pos_w[0] - env.scene.env_origins[0])[2].item()); zmax = max(zmax, z); ok = ok or (z > float(args.success_z_threshold))
            sif += 1; stable = stable + 1 if conv else 0
            if countdown < 0 and arm_ok and not conv:
                countdown = max(0, int(args.post_arm_reached_steps))
            elif countdown >= 0 and not conv:
                countdown -= 1
            if stable >= max(1, int(args.stable_steps)) or sif >= max(1, int(args.max_steps_per_frame)) or (arm_ok and not conv and countdown == 0):
                fi += 1; stable = 0; countdown = -1; sif = 0
            if torch.is_tensor(term) and torch.is_tensor(trunc) and bool(torch.any(term | trunc).item()):
                env.reset(); _warmup_tactile_nominal(env)
                stable = 0; countdown = -1; sif = 0
                continue
            dt = target_dt - (time.time() - t0)
            if dt > 0: time.sleep(dt)
        if ok:
            succ += 1; bufs.setdefault("success_z_max", []).append(np.asarray(zmax, dtype=np.float32)); p, t_len = _save_npz(rec_dir, f"episode_{ep:04d}_success.npz", bufs); print(f"[INFO] saved success -> {p} (T={t_len}, z_max={zmax:.4f})")
        else:
            print(f"[INFO] episode {ep:04d} failed (T={len(bufs.get('task_episode_sim_step', []))}, z_max={zmax:.4f})")
        if not app.is_running(): break
    print(f"[INFO] done: {succ}/{int(args.num_episodes)} success")
    env.close(); app.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())