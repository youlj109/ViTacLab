#!/usr/bin/env python3
"""
Hand Range Calibration Script

This script helps calibrate MediaPipe output ranges to match ShadowHand joint limits
by recording two poses: open hand and closed fist.

Usage:
    python source/video_teleop/tools/calibrate_hand_ranges.py --camera 6 --side right \\
        --output scripts/teleoperation/video_teleop/config/hand_calibration.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

# Add source directory to path
# File at source/video_teleop/tools/ -> parents[3] = project root (ViTacLab)
project_root = Path(__file__).resolve().parents[3]
source_dir = project_root / "source"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

import mediapipe as mp
from video_teleop.config_paths import default_hand_calibration_yaml
from video_teleop.core.mediapipe_shadowhand import MediaPipeShadowHand, ShadowHandJoints


def collect_samples(
    video_listener: MediaPipeShadowHand,
    cap: cv2.VideoCapture,
    pose_name: str,
    num_samples: int = 30,
    sample_interval: int = 2,
    mp_drawing=None,
    mp_drawing_styles=None,
    mp_hands=None,
) -> Optional[np.ndarray]:
    """
    Collect joint angle samples for a specific pose.
    
    Args:
        video_listener: MediaPipeShadowHand instance
        cap: OpenCV VideoCapture instance
        pose_name: Name of the pose (e.g., "open", "closed")
        num_samples: Number of samples to collect
        sample_interval: Frames to skip between samples
    
    Returns:
        Array of shape (num_samples, 24) with joint angles, or None if hand not detected
    """
    
    print(f"\n[INFO] Collecting samples for '{pose_name}' pose...")
    print(f"[INFO] Please show your {pose_name} hand to the camera.")
    print(f"[INFO] Press SPACE to start collecting samples, ESC to cancel")
    
    # Wait for user to press SPACE to start collecting
    waiting_for_start = True
    while waiting_for_start:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            return None
        
        # Show frame with instructions
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Pose: {pose_name.upper()} - Ready to collect",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "Press SPACE to start collecting samples",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "Press ESC to cancel",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        
        # Check for hand detection and draw hand landmarks
        # Use infer_joints_raw to get MediaPipe's raw output (before mapping to robot ranges)
        # This ensures calibration records the actual MediaPipe values, not mapped values
        joints = video_listener.infer_joints_raw(frame)
        
        # Get hand landmarks for visualization
        if mp_hands and mp_drawing and hasattr(video_listener, '_hands') and video_listener._hands is not None:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process with MediaPipe hands (use the internal hands instance)
            results = video_listener._hands.process(frame_rgb)
            
            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                cv2.putText(
                    display_frame,
                    "Hand detected!",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "No hand detected",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            # Fallback to simple detection status
            if joints is not None and len(joints) == 24:
                cv2.putText(
                    display_frame,
                    "Hand detected!",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "No hand detected",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        
        cv2.imshow("Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] Calibration cancelled")
            return None
        elif key == 32:  # SPACE - start collecting
            waiting_for_start = False
            print(f"[INFO] Starting to collect samples for '{pose_name}' pose...")
    
    # Now start collecting samples
    samples = []
    frame_count = 0
    sample_count = 0
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Use infer_joints_raw to get MediaPipe's raw output (before mapping to robot ranges)
        # This ensures calibration records the actual MediaPipe values, not mapped values
        joints = video_listener.infer_joints_raw(frame)
        
        # Show frame with instructions
        display_frame = frame.copy()
        
        # Draw hand landmarks for visualization
        if mp_hands and mp_drawing and hasattr(video_listener, '_hands') and video_listener._hands is not None:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process with MediaPipe hands
            results = video_listener._hands.process(frame_rgb)
            
            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
        cv2.putText(
            display_frame,
            f"Pose: {pose_name.upper()} | Samples: {sample_count}/{num_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            "Press SPACE when ready, ESC to cancel",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        if joints is not None and len(joints) == 24:
            # Hand detected
            cv2.putText(
                display_frame,
                "Hand detected!",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            # Collect sample every sample_interval frames
            if frame_count % sample_interval == 0:
                samples.append(joints)
                sample_count += 1
        else:
            # No hand detected
            cv2.putText(
                display_frame,
                "No hand detected",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        
        cv2.imshow("Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] Calibration cancelled")
            return None
        elif key == 32:  # SPACE - start/continue sampling
            pass
        
        frame_count += 1
    
    if len(samples) == 0:
        print(f"[ERROR] No samples collected for '{pose_name}' pose")
        return None
    
    samples_array = np.array(samples)
    print(f"[INFO] Collected {len(samples)} samples for '{pose_name}' pose")
    print(f"[INFO] Joint angle ranges:")
    joint_names = ShadowHandJoints.names()
    for i, name in enumerate(joint_names):
        min_val = np.min(samples_array[:, i])
        max_val = np.max(samples_array[:, i])
        mean_val = np.mean(samples_array[:, i])
        print(f"  {name}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    return samples_array


def compute_calibrated_ranges(
    open_samples: np.ndarray,
    closed_samples: np.ndarray,
    robot_limits: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Compute calibrated joint ranges from open and closed hand samples.
    
    Args:
        open_samples: Array of shape (N, 24) with open hand joint angles
        closed_samples: Array of shape (M, 24) with closed hand joint angles
        robot_limits: Dict mapping joint names to (robot_min, robot_max) tuples
    
    Returns:
        Dict mapping joint names to (human_min, human_max, robot_min, robot_max) tuples
    """
    joint_names = ShadowHandJoints.names()
    calibrated_ranges = {}
    
    for i, joint_name in enumerate(joint_names):
        # Get human range from samples
        open_vals = open_samples[:, i]
        closed_vals = closed_samples[:, i]
        
        # For each joint, determine which pose corresponds to min/max
        # For flexion joints (FFJ3, FFJ2, FFJ1, etc.), closed should be max, open should be min
        # For spread joints (FFJ4, MFJ4, etc.), we need to check both
        open_min = np.min(open_vals)
        open_max = np.max(open_vals)
        closed_min = np.min(closed_vals)
        closed_max = np.max(closed_vals)
        
        # Determine human range
        # For flexion joints: open hand = min (extended), closed hand = max (flexed)
        # For spread joints: use the full range from both poses
        if "J4" in joint_name or "J5" in joint_name or ("J1" in joint_name and "TH" not in joint_name):
            # Spread joints or wrist joints: use full range
            human_min = min(open_min, closed_min)
            human_max = max(open_max, closed_max)
        else:
            # Flexion joints: open = min, closed = max
            human_min = min(open_min, closed_min)  # More extended
            human_max = max(open_max, closed_max)   # More flexed
        
        # Add safety margin to handle values outside calibration range
        # This prevents issues when MediaPipe outputs values slightly outside the recorded range
        range_span = human_max - human_min
        # if range_span > 0:
        #     # Add 10% margin on each side
        #     margin = range_span * 0.1
        #     human_min = human_min - margin
        #     human_max = human_max + margin
        # else:
        #     # If range is zero (e.g., joint didn't move during calibration), use a small default range
        #     # For flexion joints, assume [0, small_value] or [small_value, 0] depending on sign
        #     if human_min >= 0:
        #         human_min = max(0.0, human_min - 0.1)
        #         human_max = human_max + 0.1
        #     else:
        #         human_min = human_min - 0.1
        #         human_max = min(0.0, human_max + 0.1)
        
        # Get robot limits
        robot_min, robot_max = robot_limits.get(joint_name, (0.0, 1.0))
        
        calibrated_ranges[joint_name] = (human_min, human_max, robot_min, robot_max)
        
        print(f"  {joint_name}: human=[{human_min:.4f}, {human_max:.4f}], robot=[{robot_min:.4f}, {robot_max:.4f}]")
    
    return calibrated_ranges


def get_robot_limits() -> Dict[str, Tuple[float, float]]:
    """
    Get ShadowHand robot joint limits.
    These should match the actual robot limits in your simulation.
    """
    # Spread + thumb base (THJ4/THJ5) locked at mid (match mediapipe_shadowhand); WRJ unchanged
    return {
        "WRJ1": (-0.7, 0.7),
        "WRJ2": (-0.7, 0.7),
        "FFJ4": (0.0, 0.0),
        "FFJ3": (-0.2618, 1.5708),
        "FFJ2": (0.0, 1.5708),
        "FFJ1": (0.0, 1.5708),
        "MFJ4": (0.0, 0.0),
        "MFJ3": (-0.2618, 1.5708),
        "MFJ2": (0.0, 1.5708),
        "MFJ1": (0.0, 1.5708),
        "RFJ4": (0.0, 0.0),
        "RFJ3": (-0.2618, 1.5708),
        "RFJ2": (0.0, 1.5708),
        "RFJ1": (0.0, 1.5708),
        "LFJ5": (0.3927, 0.3927),
        "LFJ4": (0.0, 0.0),
        "LFJ3": (-0.2618, 1.5708),
        "LFJ2": (0.0, 1.5708),
        "LFJ1": (0.0, 1.5708),
        "THJ5": (0.0, 0.0),
        "THJ4": (0.61085, 0.61085),
        "THJ3": (-0.2094, 0.2094),
        "THJ2": (-0.6981, 0.6981),
        "THJ1": (-0.2618, 1.5708),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate hand joint ranges for teleoperation")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="right", help="Hand side to calibrate")
    parser.add_argument(
        "--output",
        type=str,
        default=default_hand_calibration_yaml(),
        help="Output calibration YAML path (default: scripts/teleoperation/video_teleop/config/hand_calibration.yaml)",
    )
    parser.add_argument("--num-samples", type=int, default=30, help="Number of samples to collect per pose")
    parser.add_argument("--sample-interval", type=int, default=2, help="Frames to skip between samples")
    
    args = parser.parse_args()
    
    # Initialize MediaPipe
    hand_mode = args.side
    video_listener = MediaPipeShadowHand(hand_mode=hand_mode)
    
    # Initialize MediaPipe drawing utilities for visualization
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera {args.camera}")
        return 1
    
    print("[INFO] Hand Range Calibration")
    print(f"[INFO] Camera: {args.camera}")
    print(f"[INFO] Hand side: {args.side}")
    print(f"[INFO] Output file: {args.output}")
    print("\n[INFO] This calibration will record two poses:")
    print("  1. Open hand (fingers extended)")
    print("  2. Closed fist (fingers curled)")
    print("\n[INFO] Press SPACE when ready, ESC to cancel")
    
    # Collect open hand samples
    open_samples = collect_samples(
        video_listener,
        cap,
        "open",
        num_samples=args.num_samples,
        sample_interval=args.sample_interval,
        mp_drawing=mp_drawing,
        mp_drawing_styles=mp_drawing_styles,
        mp_hands=mp_hands,
    )
    if open_samples is None:
        cap.release()
        cv2.destroyAllWindows()
        return 1
    
    # Collect closed fist samples
    closed_samples = collect_samples(
        video_listener,
        cap,
        "closed",
        num_samples=args.num_samples,
        sample_interval=args.sample_interval,
        mp_drawing=mp_drawing,
        mp_drawing_styles=mp_drawing_styles,
        mp_hands=mp_hands,
    )
    if closed_samples is None:
        cap.release()
        cv2.destroyAllWindows()
        return 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Compute calibrated ranges
    print("\n[INFO] Computing calibrated ranges...")
    robot_limits = get_robot_limits()
    calibrated_ranges = compute_calibrated_ranges(open_samples, closed_samples, robot_limits)
    
    # Save calibration to YAML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_data = {
        "hand_side": args.side,
        "num_open_samples": len(open_samples),
        "num_closed_samples": len(closed_samples),
        "joint_ranges": {},
    }
    
    for joint_name, (h_min, h_max, r_min, r_max) in calibrated_ranges.items():
        calibration_data["joint_ranges"][joint_name] = {
            "human_min": float(h_min),
            "human_max": float(h_max),
            "robot_min": float(r_min),
            "robot_max": float(r_max),
        }
    
    with open(output_path, "w") as f:
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n[INFO] Calibration saved to: {output_path}")
    print("[INFO] You can now use this calibration file with --calibration flag in teleoperation scripts")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

