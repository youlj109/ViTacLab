"""
Command-line entry point for video teleoperation sender

Usage:
    python scripts/teleoperation/video_teleop/run_video_teleop_sender.py --camera 0 --hand-mode left
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
import time
from pathlib import Path

# Add source directory to path (hand_teleop is in source/)
project_root = Path(__file__).resolve().parents[3]
source_dir = project_root / "source"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from video_teleop.config_paths import default_camera_calibration_yaml, default_hand_calibration_yaml

# Threading/scheduling stability:
# The MediaPipe/TFLite/XNNPACK/OpenCV stack may create threads and sometimes triggers
# a hard abort in glibc when changing thread priority (pthread_tpp_change_priority).
# This happens more often when receiver (Isaac Sim) is also running (GPU/CPU contention).
# Limit thread pools and avoid real-time priority to reduce crashes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("XNNPACK_NUM_THREADS", "1")
# Avoid OpenCV/OpenMP from setting real-time thread priority
os.environ.setdefault("OPENCV_FOR_THREAD_PRIORITY", "0")

# Import cv2 for visualization window handling
try:
    import cv2
except ImportError:
    cv2 = None  # Will check later if visualization is enabled

from video_teleop.core.video_listener import VideoListener
from video_teleop.core.video_teleop_sender import VideoTeleopSender
from video_teleop.core.mediapipe_shadowhand import ShadowHandJoints


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video teleoperation sender (perception → IPC send)"
    )
    
    # Video source
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    
    # VideoListener parameters
    parser.add_argument(
        "--hand-mode",
        type=str,
        default="both",
        choices=["left", "right", "both"],
        help="Hand detection mode",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=default_camera_calibration_yaml(),
        help="Path to camera calibration YAML (default: scripts/teleoperation/video_teleop/config/camera_calibration.yaml)",
    )
    parser.add_argument(
        "--hand-calibration",
        type=str,
        default=default_hand_calibration_yaml(),
        help="Path to hand range calibration YAML (default: scripts/teleoperation/video_teleop/config/hand_calibration.yaml)",
    )
    
    # IPC parameters
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="ipc:///tmp/shadowhand_teleop_video.ipc",
        help="ZeroMQ address (IPC or TCP, e.g., 'tcp://127.0.0.1:5555')",
    )
    parser.add_argument(
        "--send-rate",
        type=float,
        default=30.0,
        help="Target send rate (Hz)",
    )
    
    # Debug options
    parser.add_argument(
        "--enable-landmarks",
        action="store_true",
        help="Include landmarks in messages (increases message size)",
    )
    parser.add_argument(
        "--disable-visualization",
        action="store_true",
        help="Disable OpenCV visualization windows",
    )

    return parser.parse_args()


def _print_joint_debug_info(video_listener: VideoListener, side: str):
    """
    Print debug information comparing raw MediaPipe joint values vs mapped values.
    
    Args:
        video_listener: VideoListener instance
        side: "left" or "right"
    """
    try:
        # Get current frame using VideoListener's get_frame method
        frame = video_listener.get_frame()
        if frame is None:
            return
        
        # Get raw joint values (before mapping)
        joints_raw = video_listener.shadowhand.infer_joints_raw(frame)
        if joints_raw is None:
            return
        
        # Get mapped joint values (after _linear_map)
        joints_mapped = video_listener.shadowhand.infer_joints(frame)
        if joints_mapped is None or len(joints_mapped) != 24:
            return
        
        # Get calibration ranges if available
        joint_names = ShadowHandJoints.names()
        ranges = video_listener.shadowhand._ranges
        
        # Print comparison for ALL joints
        print(f"\n[DEBUG] {side.capitalize()} hand joint values (raw vs mapped):")
        
        # Group joints by finger type for organized output
        finger_groups = {
            "Wrist": ["WRJ"],
            "Index (FF)": ["FFJ"],
            "Middle (MF)": ["MFJ"],
            "Ring (RF)": ["RFJ"],
            "Little (LF)": ["LFJ"],
            "Thumb (TH)": ["THJ"],
        }
        
        # Print all joints grouped by finger
        for group_name, prefixes in finger_groups.items():
            group_indices = []
            for i, name in enumerate(joint_names):
                if any(name.upper().startswith(prefix) for prefix in prefixes):
                    group_indices.append(i)
            
            if group_indices:
                print(f"  {group_name} joints:")
                for i in group_indices:
                    joint_name = joint_names[i]
                    raw_val = joints_raw[i]
                    mapped_val = joints_mapped[i]
                    
                    # Get calibration range
                    if joint_name in ranges:
                        h0, h1, r0, r1 = ranges[joint_name]
                        # Check if value is clamped
                        clamp_info = ""
                        if abs(mapped_val - r1) < 1e-4:
                            clamp_info = " [CLAMPED TO UPPER]"
                        elif abs(mapped_val - r0) < 1e-4:
                            clamp_info = " [CLAMPED TO LOWER]"
                        
                        print(f"    {joint_name}: raw={raw_val:.4f}, mapped={mapped_val:.4f}, "
                              f"human_range=[{h0:.4f}, {h1:.4f}], robot_range=[{r0:.4f}, {r1:.4f}]{clamp_info}")
                    else:
                        print(f"    {joint_name}: raw={raw_val:.4f}, mapped={mapped_val:.4f}")
        
        print()  # Empty line for readability
        
    except Exception as e:
        print(f"[WARNING] Failed to print joint debug info: {e}")


def main():
    """Main function."""
    args = parse_args()

    # Limit OpenCV threads before any cv2 usage (reduces pthread priority crashes when receiver runs)
    if cv2 is not None:
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

    # Determine video source (camera only)
    camera_index = args.camera

    # Initialize VideoListener
    print("[INFO] Initializing VideoListener...")
    try:
        # Pass calibration files separately
        # hand_calibration is for joint ranges, calibration_file is for camera calibration
        video_listener = VideoListener(
            camera_index=camera_index,
            hand_mode=args.hand_mode,
            calibration_file=args.hand_calibration,  # Hand calibration file (joint ranges)
            camera_calibration_file=args.calibration_file,  # Camera calibration file (camera_matrix)
            enable_visualization=not args.disable_visualization,
        )
        video_listener.start()
        print("[INFO] VideoListener started")
    except Exception as e:
        print(f"[ERROR] Failed to initialize VideoListener: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Initialize VideoTeleopSender
    print("[INFO] Initializing VideoTeleopSender...")
    try:
        sender = VideoTeleopSender(
            video_listener=video_listener,
            zmq_address=args.zmq_address,
            send_rate_hz=args.send_rate,
            enable_landmarks=args.enable_landmarks,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize VideoTeleopSender: {e}")
        import traceback
        traceback.print_exc()
        video_listener.stop()
        return 1
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down...")
        sender.stop()
        video_listener.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start sender
    print("[INFO] Starting sender...")
    print(f"[INFO] Sending to: {args.zmq_address}")
    print(f"[INFO] Send rate: {args.send_rate} Hz")
    print(f"[INFO] Visualization: {'enabled' if not args.disable_visualization else 'disabled'}")
    print("[INFO] Press Ctrl+C to stop")
    print("")
    
    try:
        sender.start()
        
        # Wait a moment for sender to initialize
        time.sleep(0.5)
        
        # Check if cv2 is available for visualization
        if not args.disable_visualization and cv2 is None:
            print("[WARNING] cv2 (opencv-python) not available. Visualization disabled.")
            print("[WARNING] Install with: pip install opencv-python")
        
        # Keep main thread alive and print stats
        # Also update visualization window in main thread
        last_print_time = time.time()
        print_interval = 5.0  # Print every 5 seconds
        last_viz_update = time.time()
        viz_update_interval = 1.0 / 30.0  # Update visualization at ~30 Hz
        
        while True:
            current_time = time.time()
            
            # Update visualization window (if enabled).
            # This is important: VideoListener's visualization only updates when get_pose() is called.
            should_update_viz = (
                (not args.disable_visualization)
                and (cv2 is not None)
                and (current_time - last_viz_update >= viz_update_interval)
            )
            if should_update_viz:
                try:
                    if args.hand_mode == "both":
                        video_listener.get_pose_both(update_visualization=True)
                    else:
                        video_listener.get_pose(args.hand_mode, update_visualization=True)
                except Exception as e:
                    print(f"[WARNING] Visualization update failed: {e}")
                last_viz_update = current_time

            if not args.disable_visualization and cv2 is not None:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("\n[INFO] ESC pressed, shutting down...")
                    break
            else:
                # If no visualization, just sleep.
                time.sleep(0.1)
            
            # Print stats periodically
            if current_time - last_print_time >= print_interval:
                # Check if hands are detected (this also triggers visualization update)
                if args.hand_mode == "both":
                    left_pose, right_pose = video_listener.get_pose_both(update_visualization=False)
                    left_detected = left_pose is not None and left_pose.get("hand_joint_pose") is not None
                    right_detected = right_pose is not None and right_pose.get("hand_joint_pose") is not None
                    print(f"[INFO] Stats: Sent {sender.sequence} messages | "
                          f"Left: {'✓' if left_detected else '✗'}, "
                          f"Right: {'✓' if right_detected else '✗'}")
                    
                    # Print joint debug info for both hands
                    if left_detected:
                        _print_joint_debug_info(video_listener, "left")
                    if right_detected:
                        _print_joint_debug_info(video_listener, "right")
                    
                    if not left_detected and not right_detected:
                        print("[WARNING] No hands detected. Make sure your hands are visible in the camera.")
                else:
                    pose = video_listener.get_pose(args.hand_mode, update_visualization=False)
                    detected = pose[2] is not None  # hand_joint_pose
                    print(f"[INFO] Stats: Sent {sender.sequence} messages | "
                          f"{args.hand_mode.capitalize()}: {'✓' if detected else '✗'}")
                    
                    # Print joint debug info
                    if detected:
                        _print_joint_debug_info(video_listener, args.hand_mode)
                    
                    if not detected:
                        print(f"[WARNING] {args.hand_mode.capitalize()} hand not detected. Make sure your hand is visible in the camera.")
                last_print_time = current_time
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sender.stop()
        video_listener.stop()
        print("[INFO] Cleanup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

