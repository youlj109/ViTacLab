#!/usr/bin/env python3
"""
Video teleoperation for UR10e+ShadowHand single-arm task.

Receives teleop data via IPC, applies same pose transforms as run_video_teleop_receiver,
solves IK for arm joints, and controls the simulation.

Usage:
    ./python.sh scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py \\
        --task pour --zmq-address ipc:///tmp/shadowhand_teleop_video.ipc \\
        --hand-mode left

    # With wrist pose visualization (VisualCuboid markers):
    ./python.sh scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py \\
        --task pour --enable-visualization --hand-mode left

    # In-hand cube (policy is hand-only in env; teleop drives hand joints, arm stays at task default pose)
    ./python.sh scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py \\
        --task inhand --enable-visualization --hand-mode left
"""

from __future__ import annotations

import argparse
from typing import Optional
import importlib
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher


def _make_T(pos_xyz, euler_xyz) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.array(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.array(pos_xyz, dtype=np.float64)
    return T


def _T_to_pose_wxyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = T[:3, 3].astype(np.float64)
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat().astype(np.float64)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return t, quat_wxyz


def _load_symbol(entry: str):
    if ":" not in entry:
        raise ValueError(f"Invalid entry '{entry}'. Expected 'module.path:SymbolName'.")
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


_TASK_PRESETS = {
    "pour": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video teleop for UR10e+ShadowHand single-arm task")
    parser.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="pour", help="Task preset")
    parser.add_argument("--env", type=str, default="", help="Env entry: module:Class")
    parser.add_argument("--cfg", type=str, default="", help="Cfg entry: module:Class")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (left=0, right=1)")
    parser.add_argument("--zmq-address", type=str, default="ipc:///tmp/shadowhand_teleop_video.ipc", help="ZMQ address")
    parser.add_argument("--hand-mode", type=str, default="left", choices=["left", "right", "both"])
    parser.add_argument("--task-fps", type=float, default=20.0, help="Target control FPS")
    parser.add_argument(
        "--enable-visualization",
        action="store_true",
        help="Enable 3D visualization (VisualCuboid markers for wrist pose)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print IK solution (arm joints) for debugging",
    )


    parser.add_argument("--tag0-world-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--tag0-world-euler", type=float, nargs=3, default=(0.0, 3.141592653589793, 1.5707963267948966), metavar=("RX", "RY", "RZ"))
    parser.add_argument("--tag1-hand-pos", type=float, nargs=3, default=(0.0, 0.0, 0.08), metavar=("X", "Y", "Z"))
    parser.add_argument("--tag1-hand-euler", type=float, nargs=3, default=(0.0, -1.5707963267948966, 1.5707963267948966), metavar=("RX", "RY", "RZ"))
    parser.add_argument("--flip-axis", type=str, default="none", choices=["none", "x", "y", "z"])
    parser.add_argument("--flip-where", type=str, default="tag1_hand", choices=["tag1_hand", "world_tag0", "both"])

    parser.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="T_world_arm_base translation")
    parser.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="T_world_arm_base rotation (rad)")
    parser.add_argument(
        "--pos-scale",
        type=float,
        nargs=3,
        default=(4, 1, 2),
        metavar=("X", "Y", "Z"),
        help="Position scale factors for xyz (default: 1 1 1)",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.enable_cameras = True  # pour/pickup/inhand envs spawn cameras (tactile where configured, third-person)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401

    project_root = Path(__file__).resolve().parents[4]
    source_dir = project_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from video_teleop.core.video_teleop_receiver import VideoTeleopReceiver
    from video_teleop.core.video_teleop_control import VideoTeleopControl, ArmHandTargets
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names

    if args.enable_visualization:
        from isaacsim.core.api.objects import VisualCuboid
        from isaacsim.core.prims import XFormPrim

        left_prim_path = "/World/Teleop/LeftHand"
        right_prim_path = "/World/Teleop/RightHand"
        VisualCuboid(
            prim_path=left_prim_path,
            size=0.03,
            position=np.array([0.0, 0.0, 0.0]),
            visible=(args.hand_mode in ["left", "both"]),
            color=np.array([0.0, 1.0, 0.0]),
        )
        VisualCuboid(
            prim_path=right_prim_path,
            size=0.03,
            position=np.array([0.0, 0.0, 0.0]),
            visible=(args.hand_mode in ["right", "both"]),
            color=np.array([1.0, 0.0, 0.0]),
        )
        visualizers = {
            "left": XFormPrim(prim_paths_expr=left_prim_path, name="LeftHand", usd=True),
            "right": XFormPrim(prim_paths_expr=right_prim_path, name="RightHand", usd=True),
        }
    else:
        visualizers = {}

    preset = _TASK_PRESETS[str(args.task)]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"

    print(f"[INFO] Creating {EnvCls.__name__} (num_envs={cfg.scene.num_envs})...")
    env = EnvCls(cfg)
    action_dim = env.num_actions
    print(f"[INFO] Action dim: {action_dim}")

    env.reset()

    receiver = VideoTeleopReceiver(
        zmq_address=args.zmq_address,
        print_rate_hz=0.0,
        enable_print=False,
    )
    receiver.start()

    T_tag0_world = _make_T(args.tag0_world_pos, args.tag0_world_euler)
    T_tag1_hand = _make_T(args.tag1_hand_pos, args.tag1_hand_euler)
    T_world_tag0 = np.linalg.inv(T_tag0_world)

    if args.flip_axis == "none":
        T_flip = np.eye(4, dtype=np.float64)
    elif args.flip_axis == "x":
        T_flip = np.eye(4, dtype=np.float64)
        T_flip[:3, :3] = np.diag([-1.0, 1.0, 1.0])
    elif args.flip_axis == "y":
        T_flip = np.eye(4, dtype=np.float64)
        T_flip[:3, :3] = np.diag([1.0, -1.0, 1.0])
    else:
        T_flip = np.eye(4, dtype=np.float64)
        T_flip[:3, :3] = np.diag([1.0, 1.0, -1.0])

    if args.flip_where in ("tag1_hand", "both"):
        T_tag1_hand = T_flip @ T_tag1_hand
    if args.flip_where in ("world_tag0", "both"):
        T_world_tag0 = T_world_tag0 @ T_flip

    T_world_arm_base = _make_T(args.arm_base_pos, args.arm_base_euler)
    pos_scale = np.array(args.pos_scale, dtype=np.float64)
    control = VideoTeleopControl(T_world_arm_base=T_world_arm_base)

    robot = env.robot
    joint_names = robot.joint_names
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    sh_names = shadowhand_joint_names()

    if args.debug:
        hand_lower = env.robot_dof_lower_limits[0, hand_indices].cpu().numpy()
        hand_upper = env.robot_dof_upper_limits[0, hand_indices].cpu().numpy()
        print("[INFO] ShadowHand joint limits (rad):")
        for i, idx in enumerate(hand_indices):
            print(f"  {joint_names[idx]}: [{hand_lower[i]:.4f}, {hand_upper[i]:.4f}]")

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
        actuated = full_dof[env.actuated_dof_indices]
        lower = env.robot_dof_lower_limits[0, env.actuated_dof_indices].cpu().numpy()
        upper = env.robot_dof_upper_limits[0, env.actuated_dof_indices].cpu().numpy()
        eps = 1e-6
        scale = np.where(upper - lower > eps, 2.0 * (actuated - lower) / (upper - lower) - 1.0, 0.0)
        return torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)

    last_left: Optional[ArmHandTargets] = None
    last_right: Optional[ArmHandTargets] = None
    last_left_wrist: Optional[tuple[np.ndarray, np.ndarray]] = None  # (pos, quat_wxyz)
    last_right_wrist: Optional[tuple[np.ndarray, np.ndarray]] = None
    target_dt = 1.0 / max(1e-3, float(args.task_fps))
    last_debug_time = 0.0
    debug_interval = 0.5  # seconds

    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down...")
        receiver.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[INFO] Teleop control started. Ctrl+C to stop.")
    if args.enable_visualization:
        print("[INFO] Visualization: enabled (VisualCuboid markers)")
    if args.debug:
        print("[INFO] Debug: enabled (IK arm_joints printed every 0.5s)")
    step = 0

    while simulation_app.is_running():
        t0 = time.time()
        step += 1
        message = receiver.get_latest_message()

        actions_list = []
        for env_idx in range(env.num_envs):
            side = "left" if env_idx == 0 else "right"
            if args.hand_mode == "left" and side != "left":
                targets = last_right
            elif args.hand_mode == "right" and side != "right":
                targets = last_left
            elif message is None:
                targets = last_left if side == "left" else last_right
            else:
                hand_key = "left_hand" if side == "left" else "right_hand"
                h = message.get(hand_key, {})
                if not h.get("detected", False):
                    targets = last_left if side == "left" else last_right
                else:
                    pos = h.get("robot_frame", {}).get("wrist_position")
                    ori = h.get("robot_frame", {}).get("wrist_orientation")
                    joints = h.get("hand_joints", {}).get("joint_angles")
                    if pos is None or ori is None or joints is None:
                        targets = last_left if side == "left" else last_right
                    else:
                        T_tag0_tag1 = _make_T(pos, ori)
                        T_world_tag1 = T_world_tag0 @ T_tag0_tag1
                        T_world_hand = T_world_tag1 @ T_tag1_hand
                        T_world_hand[:3, 3] *= pos_scale
                        wrist_pos = T_world_hand[:3, 3]
                        wrist_ori = R.from_matrix(T_world_hand[:3, :3]).as_euler("xyz")
                        joints = np.array(joints, dtype=np.float64)
                        targets = control.compute(wrist_pos, wrist_ori, joints)
                        if args.debug and time.time() - last_debug_time >= debug_interval:
                            if targets is not None:
                                print(f"[DEBUG] IK {side}: arm_joints={targets.arm_joints.tolist()}")
                            else:
                                print(f"[DEBUG] IK {side}: failed (using fallback)")
                            last_debug_time = time.time()
                        pos_world, quat_wxyz = _T_to_pose_wxyz(T_world_hand)
                        if side == "left":
                            last_left = targets
                            last_left_wrist = (pos_world, quat_wxyz)
                        else:
                            last_right = targets
                            last_right_wrist = (pos_world, quat_wxyz)

            if targets is None:
                targets = last_left if side == "left" else last_right
            if targets is None:
                targets = ArmHandTargets(
                    arm_joints=np.zeros(6, dtype=np.float64),
                    hand_joints=np.zeros(24, dtype=np.float64),
                )

            act = _build_action(targets.arm_joints, targets.hand_joints)
            actions_list.append(act)

        if actions_list:
            actions = torch.cat(actions_list, dim=0)
        else:
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)

        if args.enable_visualization and visualizers:
            if args.hand_mode in ["left", "both"] and last_left_wrist is not None:
                pos, quat = last_left_wrist
                visualizers["left"].set_world_poses(
                    positions=torch.tensor([pos], device="cpu", dtype=torch.float32),
                    orientations=torch.tensor([[quat[0], quat[1], quat[2], quat[3]]], device="cpu", dtype=torch.float32),
                )
            if args.hand_mode in ["right", "both"] and last_right_wrist is not None:
                pos, quat = last_right_wrist
                visualizers["right"].set_world_poses(
                    positions=torch.tensor([pos], device="cpu", dtype=torch.float32),
                    orientations=torch.tensor([[quat[0], quat[1], quat[2], quat[3]]], device="cpu", dtype=torch.float32),
                )

        env.step(actions)

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    receiver.stop()
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
