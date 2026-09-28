#!/usr/bin/env python3
"""Sweep Pick01_03 object-align offsets; report min fingertip distance to cube."""

from __future__ import annotations

import argparse
import importlib
import itertools
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
sys.path.insert(0, str(ROOT / "scripts/data_collection/full_trajectory"))


def _measure(env, pick: dict, offset: np.ndarray, *, pick_hand_close_scale: float = 1.22) -> dict[str, float]:
    z = torch.zeros(1, env.num_actions, device=env.device)
    pos = np.asarray(pick["object_pos_env"], dtype=np.float64).ravel()[:3] + offset
    state = env.object.data.root_state_w[0:1].clone()
    state[0, :3] = torch.tensor(pos, device=env.device, dtype=torch.float32) + env.scene.env_origins[0]
    state[0, 7:] = 0.0
    env.object.write_root_state_to_sim(state)
    for _ in range(20):
        env.step(z)

    jn = list(env.robot.joint_names)
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names

    sh = shadowhand_joint_names()
    hand_expr = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"
    arm_expr = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_ids = [i for i, n in enumerate(jn) if re.match(hand_expr, n)]
    arm_ids = [i for i, n in enumerate(jn) if re.match(arm_expr, n)]
    hand_j = np.asarray(pick["hand_joint_pos_shadow_order"], dtype=np.float64)[:24]
    from play_single_phase import _boost_hand_close, _tactile_contact_strength

    sh_lo = np.full(24, -np.pi)
    sh_hi = np.full(24, np.pi)
    hand_j = _boost_hand_close(hand_j, float(pick_hand_close_scale), sh_lo, sh_hi)
    arm_j = np.asarray(pick["arm_joint_pos"], dtype=np.float64)

    def hand_for_name(name: str, hv: np.ndarray) -> float:
        for si, sn in enumerate(sh):
            if sn in name or name.endswith(sn):
                return float(hv[si])
        return 0.0

    full = np.zeros(env.robot.num_joints, dtype=np.float64)
    for i, idx in enumerate(arm_ids):
        full[idx] = arm_j[i]
    for idx in hand_ids:
        full[idx] = hand_for_name(jn[idx], hand_j)
    act = full[np.array(env.actuated_dof_indices, dtype=np.int64)]
    env.apply_joint_targets(torch.tensor(act, dtype=torch.float32, device=env.device).unsqueeze(0))
    jpos = env.robot.data.joint_pos.clone()
    jvel = torch.zeros_like(jpos)
    act_t = torch.tensor(act, dtype=jpos.dtype, device=env.device)
    for j, dof_i in enumerate(env.actuated_dof_indices):
        jpos[:, dof_i] = act_t[j]
    env.robot.write_joint_state_to_sim(jpos, jvel)
    for _ in range(80):
        env.step(z)

    obj = (env.object.data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu().numpy()
    out: dict[str, float] = {"obj_z": float(obj[2])}
    dists = []
    for tip in ("ffdistal", "mfdistal", "thdistal"):
        bi = env.robot.find_bodies(tip)[0][0]
        tip_p = (env.robot.data.body_pos_w[0, bi] - env.scene.env_origins[0]).detach().cpu().numpy()
        d = float(np.linalg.norm(tip_p - obj))
        out[f"{tip}_dist"] = d
        dists.append(d)
    out["min_tip_dist"] = float(min(dists))
    out["mean_tip_dist"] = float(np.mean(dists))

    out["tact_peak"] = float(_tactile_contact_strength(env, 0))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Scan Pick01_03 object align offset (env frame, meters).")
    p.add_argument("--object-init-index", type=int, default=0)
    p.add_argument("--keyframe-name", type=str, default="Pick01_03")
    p.add_argument("--dx", type=float, nargs="+", default=[-0.030, -0.025, -0.020, -0.015, -0.010])
    p.add_argument("--dy", type=float, nargs="+", default=[-0.005, 0.0, 0.005])
    p.add_argument("--dz", type=float, nargs="+", default=[-0.015, -0.010, -0.005, 0.0])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--pick-hand-close-scale", type=float, default=1.22)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app

    import ViTacLab.tasks  # noqa: F401
    from task_entries_single import resolve_env_cfg_entries

    kf_path = (
        ROOT
        / "scripts/data_collection/full_trajectory/pose_keyframes/UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json"
    )
    doc = json.loads(kf_path.read_text(encoding="utf-8"))
    pick = next(k for k in doc["keyframes"] if k["name"] == args.keyframe_name)

    env_e, cfg_e, _ = resolve_env_cfg_entries(task="blind_grasp")
    EnvCls = getattr(importlib.import_module(env_e.rsplit(":", 1)[0]), env_e.rsplit(":", 1)[1])
    CfgCls = getattr(importlib.import_module(cfg_e.rsplit(":", 1)[0]), cfg_e.rsplit(":", 1)[1])
    cfg = CfgCls()
    cfg.scene.num_envs = 1
    cfg.object_init_choice = int(args.object_init_index)
    cfg.enable_cameras = True
    env = EnvCls(cfg)
    env._use_rl_control = False
    env.reset()
    z = torch.zeros(1, env.num_actions, device=env.device)
    for _ in range(60):
        env.step(z)

    results: list[tuple[tuple[float, float, float], dict[str, float]]] = []
    grid = list(itertools.product(args.dx, args.dy, args.dz))
    print(f"[scan] {len(grid)} offsets for {args.keyframe_name} object_init_index={args.object_init_index}", flush=True)
    for dx, dy, dz in grid:
        off = np.array([dx, dy, dz], dtype=np.float64)
        m = _measure(env, pick, off, pick_hand_close_scale=float(args.pick_hand_close_scale))
        results.append(((dx, dy, dz), m))
        print(
            f"  offset=({dx:+.3f},{dy:+.3f},{dz:+.3f}) "
            f"min_tip={m['min_tip_dist']:.4f} ff={m['ffdistal_dist']:.4f} tact={m['tact_peak']:.2e} obj_z={m['obj_z']:.4f}",
            flush=True,
        )

    results.sort(key=lambda x: x[1]["min_tip_dist"])
    print("\n[scan] top offsets by min fingertip distance:", flush=True)
    for i, ((dx, dy, dz), m) in enumerate(results[: int(args.top_k)]):
        print(
            f"  #{i + 1} ({dx:+.4f}, {dy:+.4f}, {dz:+.4f}) "
            f"min_tip={m['min_tip_dist']:.4f} ff={m['ffdistal_dist']:.4f} tact={m['tact_peak']:.2e}",
            flush=True,
        )
    best_off, best_m = results[0]
    print(
        f"\n[scan] BEST --pick-object-align-offset {best_off[0]:.4f} {best_off[1]:.4f} {best_off[2]:.4f}",
        flush=True,
    )

    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
