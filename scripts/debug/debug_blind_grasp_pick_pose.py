#!/usr/bin/env python3
"""Debug: apply Pick01_03 keyframe joints and print object vs wrist distance."""

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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--object-init-index", type=int, default=0)
    p.add_argument("--keyframe-name", type=str, default="Pick01_03")
    p.add_argument("--pick-wrist-z-above-object", type=float, default=0.13)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app

    import ViTacLab.tasks  # noqa: F401
    from task_entries_single import resolve_env_cfg_entries
    from scipy.spatial.transform import Rotation as R
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import VideoTeleopControl

    def _T(pos, euler):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_euler("xyz", np.asarray(euler, dtype=np.float64), degrees=False).as_matrix()
        T[:3, 3] = np.asarray(pos, dtype=np.float64)
        return T

    kf_path = ROOT / "scripts/data_collection/full_trajectory/pose_keyframes/UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json"
    doc = json.loads(kf_path.read_text(encoding="utf-8"))
    kf = next(k for k in doc["keyframes"] if k["name"] == args.keyframe_name)

    env_e, cfg_e, _ = resolve_env_cfg_entries(task="blind_grasp")
    EnvCls = getattr(importlib.import_module(env_e.rsplit(":", 1)[0]), env_e.rsplit(":", 1)[1])
    CfgCls = getattr(importlib.import_module(cfg_e.rsplit(":", 1)[0]), cfg_e.rsplit(":", 1)[1])
    cfg = CfgCls()
    cfg.scene.num_envs = 1
    cfg.object_init_choice = int(args.object_init_index)
    env = EnvCls(cfg)
    env._use_rl_control = False
    env.reset()
    z = torch.zeros(1, env.num_actions, device=env.device)
    for _ in range(80):
        env.step(z)

    obj_env = kf.get("object_pos_env")
    if obj_env:
        pos_env = torch.tensor(obj_env, device=env.device, dtype=torch.float32).view(1, 3)
        pos_w = pos_env + env.scene.env_origins[0:1]
        state = env.object.data.root_state_w[0:1].clone()
        state[0, :3] = pos_w[0]
        state[0, 7:] = 0.0
        env.object.write_root_state_to_sim(state)
        for _ in range(20):
            env.step(z)

    sh = shadowhand_joint_names()
    jn = list(env.robot.joint_names)
    hand_j = np.asarray(kf["hand_joint_pos_shadow_order"], dtype=np.float64)[:24]
    marker_pos = np.asarray(kf["marker_pos_w"], dtype=np.float64)[:3]
    marker_euler = np.asarray(kf["marker_euler_xyz"], dtype=np.float64)[:3]
    obj_env_now = (env.object.data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu().numpy()
    rec_obj = np.asarray(kf.get("object_pos_env", obj_env_now), dtype=np.float64).ravel()[:3]
    wrist_pos = obj_env_now + (marker_pos - rec_obj)
    wrist_pos[2] = float(obj_env_now[2]) + float(args.pick_wrist_z_above_object)
    control = VideoTeleopControl(T_world_arm_base=_T(np.zeros(3), np.zeros(3)))
    tgt = control.compute(wrist_pos, marker_euler, hand_j)
    print(f"IK wrist target env: {wrist_pos} (z_above_obj={args.pick_wrist_z_above_object})")
    if tgt is None:
        print("[WARN] VideoTeleopControl.compute returned None — using keyframe arm_joint_pos")
        arm_j = np.asarray(kf["arm_joint_pos"], dtype=np.float64)
    else:
        arm_j = np.asarray(tgt.arm_joints, dtype=np.float64)
        print(f"IK arm joints: {arm_j}")

    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_ids = [i for i, n in enumerate(jn) if re.match(arm_expr, n)]
    hand_ids = [i for i, n in enumerate(jn) if re.match(hand_expr, n)]

    def hand_for_name(name, hv):
        for si, sn in enumerate(sh):
            if sn in name or name.endswith(sn):
                return float(hv[si])
        return 0.0

    arm_j = np.asarray(arm_j, dtype=np.float64).ravel()
    hand_j = np.asarray(hand_j, dtype=np.float64)[:24]
    full = np.zeros(env.robot.num_joints, dtype=np.float64)
    for i, idx in enumerate(arm_ids):
        full[idx] = arm_j[i]
    for idx in hand_ids:
        full[idx] = hand_for_name(jn[idx], hand_j)
    act = full[np.array(env.actuated_dof_indices, dtype=np.int64)]
    env.apply_joint_targets(torch.tensor(act, dtype=torch.float32, device=env.device).unsqueeze(0))
    for _ in range(60):
        env.step(z)

    cur = np.array([float(env.robot.data.joint_pos[0, i].item()) for i in arm_ids])
    print(f"  arm target:           {arm_j}")
    print(f"  arm current:          {cur}")
    print(f"  arm |err|:            {np.max(np.abs(cur-arm_j)):.4f}")

    obj = (env.object.data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu().numpy()
    ee_body = env.robot.find_bodies("wrist_3_link")[0][0]
    wrist = (env.robot.data.body_pos_w[0, ee_body] - env.scene.env_origins[0]).detach().cpu().numpy()
    print(f"keyframe: {args.keyframe_name} object_init_index={args.object_init_index}")
    print(f"  object env pos:     {obj}")
    print(f"  wrist env pos:      {wrist}")
    print(f"  wrist - object:     {wrist - obj}")
    print(f"  recorded object:    {kf.get('object_pos_env')}")
    print(f"  recorded marker:    {kf.get('marker_pos_w')}")
    for tip_name in ("ffdistal", "mfdistal", "thdistal", "palm"):
        try:
            bi = env.robot.find_bodies(tip_name)[0][0]
            tip = (env.robot.data.body_pos_w[0, bi] - env.scene.env_origins[0]).detach().cpu().numpy()
            print(f"  {tip_name:10s} env:   {tip}  dist_obj={float(np.linalg.norm(tip - obj)):.4f}")
        except Exception:
            pass
    # Physics settle without snap (PD targets only)
    env._use_rl_control = False
    env.apply_joint_targets(torch.tensor(act, dtype=torch.float32, device=env.device).unsqueeze(0))
    z0 = float(obj[2])
    zmax = z0
    for _ in range(120):
        env.step(z)
        z_obj = float((env.object.data.root_pos_w[0] - env.scene.env_origins[0])[2].item())
        zmax = max(zmax, z_obj)
    print(f"  after 120 PD steps: z={z_obj:.4f} zmax={zmax:.4f} (init z={z0:.4f})")
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
