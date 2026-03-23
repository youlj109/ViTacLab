"""
Command-line entry point for video teleoperation receiver (Phase 2).

This script receives IPC messages and visualizes them (no robot control).

Usage:
    # Recommended (stable): visualize using robot_frame wrist pose
    python scripts/teleoperation/video_teleop/run_video_teleop_receiver.py   --zmq-address ipc:///tmp/shadowhand_teleop_video.ipc   --enable-visualization --hand-mode left --tag0-world-euler 0 3.141592653589793 1.5707963267948966 --tag1-hand-euler 0  3.141592653589793 1.5707963267948966
"""

from __future__ import annotations

import argparse
import sys
import signal
from pathlib import Path

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(
    description="Video teleoperation receiver (IPC receive → visualize, no robot control)"
)
parser.add_argument(
    "--zmq-address",
    type=str,
    default="ipc:///tmp/shadowhand_teleop_video.ipc",
    help="ZeroMQ address (IPC or TCP, e.g., 'tcp://127.0.0.1:5555')",
)
parser.add_argument(
    "--print-rate",
    type=float,
    default=1.0,
    help="Rate at which to print messages (Hz, 0 to disable)",
)
parser.add_argument(
    "--disable-print",
    action="store_true",
    help="Disable printing messages",
)
parser.add_argument(
    "--enable-visualization",
    action="store_true",
    help="Enable 3D visualization in Isaac Sim (VisualCuboid markers)",
)
parser.add_argument(
    "--hand-mode",
    type=str,
    default="both",
    choices=["left", "right", "both"],
    help="Which hand(s) to visualize in Isaac Sim: 'left', 'right', or 'both'",
)
parser.add_argument(
    "--tag0-world-pos",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="T_tag0_world translation (world -> tag0), meters",
)
parser.add_argument(
    "--tag0-world-euler",
    type=float,
    nargs=3,
    default=(0.0, 3.141592653589793, 1.5707963267948966),
    metavar=("RX", "RY", "RZ"),
    help="T_tag0_world rotation as Euler xyz (world -> tag0), radians",
)
parser.add_argument(
    "--tag1-hand-pos",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="T_tag1_hand translation (tag1 -> hand), meters",
)
parser.add_argument(
    "--tag1-hand-euler",
    type=float,
    nargs=3,
    default=(0.0, 3.141592653589793, 1.5707963267948966),
    metavar=("RX", "RY", "RZ"),
    help="T_tag1_hand rotation as Euler xyz (tag1 -> hand), radians",
)
parser.add_argument(
    "--flip-axis",
    type=str,
    default="none",
    choices=["none", "x", "y", "z"],
    help="Optional axis flip (mirror) as diag([-1,1,1]) etc.",
)
parser.add_argument(
    "--flip-where",
    type=str,
    default="tag1_hand",
    choices=["tag1_hand", "world_tag0", "both"],
    help="Where to apply T_flip (default: tag1_hand)",
)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Sim modules
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

# Add source directory to path for hand_teleop
project_root = Path(__file__).resolve().parents[3]
source_dir = project_root / "source"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from video_teleop.core.video_teleop_receiver import VideoTeleopReceiver


def _make_T(pos_xyz, euler_xyz) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.array(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.array(pos_xyz, dtype=np.float64)
    return T


def _T_to_pose_wxyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = T[:3, 3].astype(np.float64)
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat().astype(np.float64)  # x,y,z,w
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return t, quat_wxyz


def main():
    """Main function."""
    # Initialize receiver
    print("[INFO] Initializing VideoTeleopReceiver...")
    receiver = VideoTeleopReceiver(
        zmq_address=args_cli.zmq_address,
        print_rate_hz=args_cli.print_rate,
        enable_print=not args_cli.disable_print,
    )
    
    # Setup visualization if enabled
    visualizers = {}
    if args_cli.enable_visualization:
        print("[INFO] Setting up 3D visualization...")
        
        # Create visual markers for left and right hands
        left_prim_path = "/World/Teleop/LeftHand"
        right_prim_path = "/World/Teleop/RightHand"
        
        # Left hand marker (green)
        VisualCuboid(
            prim_path=left_prim_path,
            size=0.03,
            position=np.array([0.0, 0.0, 0.0]),
            visible=(args_cli.hand_mode in ["left", "both"]),
            color=np.array([0.0, 1.0, 0.0]),  # Green
        )
        
        # Right hand marker (red)
        VisualCuboid(
            prim_path=right_prim_path,
            size=0.03,
            position=np.array([0.0, 0.0, 0.0]),
            visible=(args_cli.hand_mode in ["right", "both"]),
            color=np.array([1.0, 0.0, 0.0]),  # Red
        )
        
        # Create XFormPrim views for updating positions
        visualizers["left"] = XFormPrim(prim_paths_expr=left_prim_path, name="LeftHand", usd=True)
        visualizers["right"] = XFormPrim(prim_paths_expr=right_prim_path, name="RightHand", usd=True)
        
        print("[INFO] Visualization markers created")
        print(
            f"[INFO] Visualization uses robot_frame (hand pose relative to ground/origin), hand_mode={args_cli.hand_mode}"
        )
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down...")
        receiver.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start receiver
    print("[INFO] Starting receiver...")
    print(f"[INFO] Listening on: {args_cli.zmq_address}")
    print(f"[INFO] Print rate: {args_cli.print_rate} Hz")
    print(f"[INFO] Visualization: {'enabled' if args_cli.enable_visualization else 'disabled'}")
    print("[INFO] Press Ctrl+C to stop\n")
    
    receiver.start()

    T_tag0_world = _make_T(args_cli.tag0_world_pos, args_cli.tag0_world_euler)  # world -> tag0
    T_tag1_hand = _make_T(args_cli.tag1_hand_pos, args_cli.tag1_hand_euler)  # tag1 -> hand
    T_world_tag0 = np.linalg.inv(T_tag0_world)  # tag0 -> world

    flip_axis = str(args_cli.flip_axis).lower()
    if flip_axis == "none":
        T_flip = np.eye(4, dtype=np.float64)
    elif flip_axis == "x":
        T_flip = np.eye(4, dtype=np.float64)
        T_flip[:3, :3] = np.diag([-1.0, 1.0, 1.0])
    elif flip_axis == "y":
        T_flip = np.eye(4, dtype=np.float64)
        T_flip[:3, :3] = np.diag([1.0, -1.0, 1.0])
    else:  # "z"
        T_flip = np.eye(4, dtype=np.float64)
        T_flip[:3, :3] = np.diag([1.0, 1.0, -1.0])

    if args_cli.flip_where in ("tag1_hand", "both"):
        T_tag1_hand = T_flip @ T_tag1_hand
    if args_cli.flip_where in ("world_tag0", "both"):
        T_world_tag0 = T_world_tag0 @ T_flip
    
    # Main loop
    last_update_time = 0.0
    update_period = 1.0 / 60.0  # 60 Hz update rate for visualization
    last_debug_time = 0.0
    debug_interval = 2.0  # Print debug info every 2 seconds
    
    try:
        while simulation_app.is_running():
            # Get latest message
            message = receiver.get_latest_message()
            
            # Update visualization if enabled and message available
            if args_cli.enable_visualization and message is not None:
                current_time = time.time()
                if current_time - last_update_time >= update_period:
                    # Debug: print position occasionally
                    if current_time - last_debug_time >= debug_interval:
                        left_hand = message.get("left_hand", {})
                        right_hand = message.get("right_hand", {})
                        left_pos_robot = left_hand.get("robot_frame", {}).get("wrist_position") if left_hand.get("detected", False) else None
                        right_pos_robot = right_hand.get("robot_frame", {}).get("wrist_position") if right_hand.get("detected", False) else None
                        print(f"[DEBUG] Hand positions - Robot frame (for reference only, may have large offsets): L={left_pos_robot}, R={right_pos_robot}")
                        last_debug_time = current_time
                    
                    # Update left hand marker
                    left_hand = message.get("left_hand", {})
                    if args_cli.hand_mode in ["left", "both"] and left_hand.get("detected", False):
                        left_pos_robot = left_hand.get("robot_frame", {}).get("wrist_position")
                        left_ori_robot = left_hand.get("robot_frame", {}).get("wrist_orientation")  # Euler xyz
                        if left_pos_robot is not None and left_ori_robot is not None:
                            T_tag0_tag1 = _make_T(left_pos_robot, left_ori_robot)
                            T_world_tag1 = T_world_tag0 @ T_tag0_tag1
                            T_world_hand = T_world_tag1 @ T_tag1_hand
                            pos_world, quat_world_wxyz = _T_to_pose_wxyz(T_world_hand)
                            pos_tensor = torch.tensor([pos_world], device="cpu", dtype=torch.float32)
                            quat_tensor = torch.tensor(
                                [[quat_world_wxyz[0], quat_world_wxyz[1], quat_world_wxyz[2], quat_world_wxyz[3]]],
                                device="cpu",
                                dtype=torch.float32,
                            )
                            visualizers["left"].set_world_poses(positions=pos_tensor, orientations=quat_tensor)
                            last_update_time = current_time

                    # Update right hand marker
                    right_hand = message.get("right_hand", {})
                    if args_cli.hand_mode in ["right", "both"] and right_hand.get("detected", False):
                        right_pos_robot = right_hand.get("robot_frame", {}).get("wrist_position")
                        right_ori_robot = right_hand.get("robot_frame", {}).get("wrist_orientation")  # Euler xyz
                        if right_pos_robot is not None and right_ori_robot is not None:
                            T_tag0_tag1 = _make_T(right_pos_robot, right_ori_robot)
                            T_world_tag1 = T_world_tag0 @ T_tag0_tag1
                            T_world_hand = T_world_tag1 @ T_tag1_hand
                            pos_world, quat_world_wxyz = _T_to_pose_wxyz(T_world_hand)
                            pos_tensor = torch.tensor([pos_world], device="cpu", dtype=torch.float32)
                            quat_tensor = torch.tensor(
                                [[quat_world_wxyz[0], quat_world_wxyz[1], quat_world_wxyz[2], quat_world_wxyz[3]]],
                                device="cpu",
                                dtype=torch.float32,
                            )
                            visualizers["right"].set_world_poses(positions=pos_tensor, orientations=quat_tensor)
                            last_update_time = current_time

                    last_update_time = current_time
            
            # Step simulation (required for Isaac Sim to update)
            simulation_app.update()
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        receiver.stop()
        print("[INFO] Cleanup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

