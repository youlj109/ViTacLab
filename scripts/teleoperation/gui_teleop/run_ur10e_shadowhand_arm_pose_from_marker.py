#!/usr/bin/env python3
"""Tune UR10e arm joint targets from a visual marker pose + IK (no random arm actions).

Loads the same task presets as ``scripts/debug/run_ur10e_shadowhand_single.py``. Spawns a small
``VisualCuboid`` under ``/World/Debug/ArmIkTarget``. **Move/orient this prim in the
viewport** (e.g. with the move/rotate gizmo); each step reads its **world** pose,
runs the same pipeline as video teleop via ``VideoTeleopControl.compute``, and steps
the env.

Use this to find a good arm pose, then copy the printed joint dict into your cfg.

Examples (Isaac Sim python):

    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_shadowhand_arm_pose_from_marker.py \\
        --task pickup --num_envs 1 --enable_cameras

    # Custom initial marker pose (world frame, meters / rad euler xyz)
    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_shadowhand_arm_pose_from_marker.py \\
        --task pour --marker-pos 0.75 0.0 0.35 --marker-euler 0.0 0.78 0.0
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher


# Same presets as scripts/debug/run_ur10e_shadowhand_single.py
_TASK_PRESETS: dict[str, dict[str, str]] = {
    "pour": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
    },
    "pickup": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg",
    },
    "inhand": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandEnvCfg",
    },
}

MARKER_PRIM_PATH = "/World/Debug/ArmIkTarget"


def _load_symbol(entry: str) -> Any:
    if ":" not in entry:
        raise ValueError(f"Invalid entry '{entry}'. Expected 'module.path:SymbolName'.")
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _make_T(pos_xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T


def _quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Isaac / XFormPrim orientation is usually w,x,y,z."""
    q = np.asarray(quat, dtype=np.float64).ravel()
    if q.size != 4:
        return np.zeros(3, dtype=np.float64)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_euler("xyz")


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _repo_root() -> Path:
    """ViTacLab repo root (contains ``source/``)."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR10e arm IK from a visual marker (same task presets as run_ur10e_shadowhand_single.py).",
    )
    p.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="pickup", help="Preset task.")
    p.add_argument("--env", type=str, default="", help="Env entry module:Class (overrides --task).")
    p.add_argument("--cfg", type=str, default="", help="Cfg entry module:Class (overrides --task).")
    p.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    p.add_argument("--fps", type=float, default=30.0, help="Simulation loop target FPS.")
    p.add_argument(
        "--marker-pos",
        type=float,
        nargs=3,
        default=(0.65, 0.12, 0.42),
        metavar=("X", "Y", "Z"),
        help="Initial marker position in world frame (m).",
    )
    p.add_argument(
        "--marker-euler",
        type=float,
        nargs=3,
        default=(0.0, 2.2, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Initial marker orientation euler xyz in world frame (rad).",
    )
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="VideoTeleopControl T_world_arm_base translation.")
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="VideoTeleopControl T_world_arm_base rotation (rad).")
    p.add_argument(
        "--hand-joints",
        choices=["zeros", "sim"],
        default="sim",
        help="Hand vector passed to IK helper: zeros(24) or current sim hand (mapped by name).",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print arm joint dict every N steps (0 = disable periodic print).",
    )
    p.add_argument(
        "--print-on-change",
        action="store_true",
        help="Also print when IK arm joints change (thresholded).",
    )
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = run until close).")
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401

    repo_root = _repo_root()
    source_dir = repo_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from video_teleop.core.video_teleop_control import ArmHandTargets, VideoTeleopControl
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names

    preset = _TASK_PRESETS[str(args.task)]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"

    print(f"[INFO] Creating {EnvCls.__name__} (num_envs={cfg.scene.num_envs}) ...")
    env = EnvCls(cfg)
    robot = env.robot
    joint_names = list(robot.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    sh_names = shadowhand_joint_names()

    env.reset()
    action_dim = env.num_actions
    print(f"[INFO] action_dim={action_dim}, actuated_dof_indices count={len(env.actuated_dof_indices)}")
    print(f"[INFO] Move prim '{MARKER_PRIM_PATH}' in the viewport; arm IK follows its world pose.")
    if len(env.actuated_dof_indices) < len(arm_indices) + len(hand_indices):
        print(
            "[WARN] This env does not actuate all arm+hand DOFs (e.g. inhand is hand-only). "
            "Printed arm joints are still valid for copying into robot_cfg.init_state; "
            "env.step only applies the actuated subset.",
        )

    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.prims import XFormPrim

    VisualCuboid(
        prim_path=MARKER_PRIM_PATH,
        size=0.04,
        position=np.array(args.marker_pos, dtype=np.float64),
        visible=True,
        color=np.array([1.0, 0.2, 0.8]),
    )
    marker_xf = XFormPrim(prim_paths_expr=MARKER_PRIM_PATH, name="ArmIkTarget", usd=True)
    T0 = _make_T(np.array(args.marker_pos, dtype=np.float64), np.array(args.marker_euler, dtype=np.float64))
    pos0 = T0[:3, 3]
    quat_wxyz = R.from_matrix(T0[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_wxyz[3], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2]], dtype=np.float32)
    marker_xf.set_world_poses(
        positions=torch.tensor([pos0], dtype=torch.float32, device="cpu"),
        orientations=torch.tensor([quat_wxyz], dtype=torch.float32, device="cpu"),
    )

    T_world_arm_base = _make_T(np.array(args.arm_base_pos, dtype=np.float64), np.array(args.arm_base_euler, dtype=np.float64))
    control = VideoTeleopControl(T_world_arm_base=T_world_arm_base)

    def _hand_joints_shadow_from_sim() -> np.ndarray:
        out = np.zeros(24, dtype=np.float64)
        jpos = robot.data.joint_pos[0].detach().cpu().numpy()
        for sh_i, sh_name in enumerate(sh_names):
            for idx in hand_indices:
                n = joint_names[idx]
                if sh_name in n or n.endswith(sh_name):
                    out[sh_i] = float(jpos[idx])
                    break
        return out

    def _hand_joint_for_robot_name(name: str, hand_joints: np.ndarray) -> float:
        for sh_idx, sh_name in enumerate(sh_names):
            if sh_name in name or name.endswith(sh_name):
                return float(hand_joints[sh_idx])
        return 0.0

    def _build_action(arm_joints: np.ndarray, hand_joints: np.ndarray) -> torch.Tensor:
        full_dof = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_indices):
            if i < len(arm_joints):
                full_dof[idx] = arm_joints[i]
        for idx in hand_indices:
            full_dof[idx] = _hand_joint_for_robot_name(joint_names[idx], hand_joints)
        actuated = full_dof[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lower = env.robot_dof_lower_limits[0, env.actuated_dof_indices].cpu().numpy()
        upper = env.robot_dof_upper_limits[0, env.actuated_dof_indices].cpu().numpy()
        eps = 1e-6
        scale = np.where(upper - lower > eps, 2.0 * (actuated - lower) / (upper - lower) - 1.0, 0.0)
        return torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)

    def _print_arm_cfg_block(arm_j: np.ndarray) -> None:
        print("[INFO] Arm joint_pos snippet for ArticulationCfg.init_state (rad):")
        print('    joint_pos={')
        for i, idx in enumerate(arm_indices):
            if i < len(arm_j):
                print(f'        "{joint_names[idx]}": {float(arm_j[i]):.16f},')
        print("    },")

    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0
    last_arm_print: Optional[np.ndarray] = None

    while simulation_app.is_running():
        t0 = time.time()
        step += 1

        pos_t, ori_t = marker_xf.get_world_poses()
        pos = _to_numpy(pos_t[0]).ravel()[:3]
        quat_wxyz = _to_numpy(ori_t[0]).ravel()[:4]
        wrist_euler = _quat_wxyz_to_euler_xyz(quat_wxyz)

        if args.hand_joints == "zeros":
            hvec = np.zeros(24, dtype=np.float64)
        else:
            hvec = _hand_joints_shadow_from_sim()

        targets: Optional[ArmHandTargets] = control.compute(pos, wrist_euler, hvec)
        if targets is None:
            if step % 60 == 0:
                print("[WARN] IK failed for current marker pose.")
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        else:
            actions = _build_action(targets.arm_joints, targets.hand_joints)
            if args.print_every > 0 and step % int(args.print_every) == 0:
                _print_arm_cfg_block(targets.arm_joints)
            if args.print_on_change:
                if last_arm_print is None or np.max(np.abs(targets.arm_joints - last_arm_print)) > 0.02:
                    _print_arm_cfg_block(targets.arm_joints)
                    last_arm_print = targets.arm_joints.copy()

        if env.num_envs > 1:
            actions = actions.expand(env.num_envs, -1).clone()
        env.step(actions)

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

        if args.max_steps > 0 and step >= int(args.max_steps):
            break

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
