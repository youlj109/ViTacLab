#!/usr/bin/env python3
"""Map desired **palm** pose in world frame to UR10e **arm** joint angles (ikpy via VideoTeleopControl).

The IK chain ends at ``wrist_3_link``. You specify:

1. Palm pose in **world**: position + orientation (euler xyz or quaternion wxyz).
2. Fixed **wrist → palm** extrinsic ``T_wrist_palm``: pose of the palm frame **expressed in
   wrist_3 (tool) frame**, i.e. ``T_world_palm = T_world_wrist @ T_wrist_palm``.

Then ``T_world_wrist = T_world_palm @ inv(T_wrist_palm)``, and the script solves IK for the
six arm joints (same as ``video_teleop_control.VideoTeleopControl``).

This matches the idea behind video teleop's ``T_tag1_hand``: composed on the right of the
tracked frame to get the frame that IK actually targets.

**No Isaac Sim required** — run with system Python if ``ikpy`` / ``scipy`` are installed.

Examples::

    # Palm at (0.65, 0.12, 0.42) m, euler xyz (rad); palm 8cm along wrist +z from wrist_3
    python scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py \\
        --palm-pos 0.65 0.12 0.42 --palm-euler 0.0 2.2 0.0

    # Same orientation as quaternion w x y z (world)
    python scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py \\
        --palm-pos 0.65 0.12 0.42 --palm-quat-wxyz 1 0 0 0

    # Use the same default offset as run_video_teleop_ur10e_shadowhand_single (--tag1-hand-*)
    python scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py \\
        --palm-pos 0.65 0.12 0.42 --palm-euler 0.0 2.2 0.0 --offset-preset video-teleop-tag1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import math

import numpy as np
from scipy.spatial.transform import Rotation as SciR


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _make_T(pos_xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = SciR.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T


def _T_from_quat_wxyz(pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).ravel()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = SciR.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64).ravel()[:3]
    return T


def _euler_xyz_from_T(T: np.ndarray) -> np.ndarray:
    return SciR.from_matrix(T[:3, :3]).as_euler("xyz")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Palm world pose + wrist–palm offset → UR10 arm joint angles.")
    p.add_argument("--palm-pos", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="Palm origin in world (m).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--palm-euler", type=float, nargs=3, metavar=("RX", "RY", "RZ"), help="Palm orientation world euler xyz (rad). Use --degrees for degrees.")
    g.add_argument(
        "--palm-quat-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="Palm orientation quaternion (w,x,y,z) in world.",
    )
    p.add_argument("--degrees", action="store_true", help="Interpret --palm-euler as degrees.")

    p.add_argument(
        "--wrist-to-palm-pos",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.08),
        metavar=("X", "Y", "Z"),
        help="Translation part of T_wrist_palm: palm origin in wrist_3 frame (m). Default: 8cm along wrist z.",
    )
    p.add_argument(
        "--wrist-to-palm-euler",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Rotation part of T_wrist_palm (euler xyz, rad). Palm axes relative to wrist_3.",
    )
    p.add_argument("--offset-degrees", action="store_true", help="Interpret --wrist-to-palm-euler as degrees.")
    p.add_argument(
        "--offset-preset",
        choices=["", "video-teleop-tag1"],
        default="",
        help="Override wrist–palm offset: 'video-teleop-tag1' matches run_video_teleop default --tag1-hand-pos/euler.",
    )

    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="T_world_arm_base translation (m).")
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="T_world_arm_base euler xyz (rad).")
    p.add_argument("--arm-base-degrees", action="store_true", help="Interpret --arm-base-euler as degrees.")

    p.add_argument(
        "--urdf",
        type=str,
        default="",
        help="Override UR10+hand URDF path (default: VideoTeleopControl built-in left urdf).",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON line: {\"shoulder_pan_joint\": ..., ...}.")
    p.add_argument("--quiet", action="store_true", help="Only print joint block / JSON, no extra text.")
    return p


# UR10e arm joint names in URDF / cfg order (matches ikpy active chain extraction order in VideoTeleopControl)
_UR10_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def main() -> int:
    args = _build_parser().parse_args()

    repo_root = _repo_root()
    source_dir = repo_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from video_teleop.core.video_teleop_control import VideoTeleopControl

    palm_pos = np.array(args.palm_pos, dtype=np.float64)
    if args.palm_quat_wxyz is not None:
        T_world_palm = _T_from_quat_wxyz(palm_pos, np.array(args.palm_quat_wxyz, dtype=np.float64))
    else:
        e = np.array(args.palm_euler, dtype=np.float64)
        if args.degrees:
            e = np.deg2rad(e)
        T_world_palm = _make_T(palm_pos, e)

    if args.offset_preset == "video-teleop-tag1":
        wpos = np.array([0.0, 0.0, 0.15], dtype=np.float64)
        weul = np.array([np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0], dtype=np.float64)
    else:
        wpos = np.array(args.wrist_to_palm_pos, dtype=np.float64)
        weul = np.array(args.wrist_to_palm_euler, dtype=np.float64)
        if args.offset_degrees:
            weul = np.deg2rad(weul)
    T_wrist_palm = _make_T(wpos, weul)

    T_world_wrist = T_world_palm @ np.linalg.inv(T_wrist_palm)
    wrist_pos = T_world_wrist[:3, 3]
    wrist_euler = _euler_xyz_from_T(T_world_wrist)

    ab_e = np.array(args.arm_base_euler, dtype=np.float64)
    if args.arm_base_degrees:
        ab_e = np.deg2rad(ab_e)
    T_world_arm_base = _make_T(np.array(args.arm_base_pos, dtype=np.float64), ab_e)

    urdf = str(args.urdf).strip() or None
    control = VideoTeleopControl(T_world_arm_base=T_world_arm_base, urdf_path=urdf)
    hand_zeros = np.zeros(24, dtype=np.float64)
    targets = control.compute(wrist_pos, wrist_euler, hand_zeros)

    if targets is None:
        print("IK failed for the given palm pose and offset.", file=sys.stderr)
        return 1

    arm = np.asarray(targets.arm_joints, dtype=np.float64).ravel()
    if arm.size < 6:
        print("IK returned fewer than 6 arm joints.", file=sys.stderr)
        return 1
    arm = arm[:6]

    joint_dict = {name: float(arm[i]) for i, name in enumerate(_UR10_ARM_JOINT_NAMES)}

    if not args.quiet:
        print("T_wrist_palm (palm in wrist_3 frame): translation", wpos.tolist(), "euler_xyz (rad)", weul.tolist())
        print("Wrist target (world): pos", wrist_pos.tolist(), "euler_xyz (rad)", wrist_euler.tolist())
        print("UR10e arm joint angles (rad):")
    if args.json:
        print(json.dumps(joint_dict, separators=(",", ":")))
    else:
        print("joint_pos={")
        for name in _UR10_ARM_JOINT_NAMES:
            print(f'    "{name}": {joint_dict[name]:.16f},')
        print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
