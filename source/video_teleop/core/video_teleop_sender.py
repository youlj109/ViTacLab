"""
Video teleoperation sender - Phase 1.

This module wraps VideoListener and sends hand/arm pose data via ZeroMQ IPC.
No Isaac dependencies, no IK, no coordinate transformation (only perception → send).
"""

from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np
import zmq
import msgpack

from .video_listener import VideoListener


class VideoTeleopSender:
    """
    Video teleoperation sender that wraps VideoListener and sends pose data via IPC.
    
    Responsibilities:
    - Capture video frames
    - Process with VideoListener
    - Pack data into protocol format
    - Send via ZeroMQ PUB
    """
    
    def __init__(
        self,
        video_listener: VideoListener,
        *,
        zmq_address: str = "ipc:///tmp/shadowhand_teleop_video.ipc",
        send_rate_hz: float = 30.0,
        enable_landmarks: bool = False,
    ) -> None:
        """
        Initialize video teleoperation sender.
        
        Args:
            video_listener: Initialized VideoListener instance
            zmq_address: ZeroMQ address (IPC or TCP)
            send_rate_hz: Target send rate (Hz)
            enable_landmarks: Whether to include landmarks in messages (for debugging)
        """
        self.video_listener = video_listener
        self.zmq_address = zmq_address
        self.send_rate_hz = float(send_rate_hz)
        self.send_period = 1.0 / self.send_rate_hz if self.send_rate_hz > 0 else 0.0
        self.enable_landmarks = bool(enable_landmarks)
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.zmq_address)
        
        # State
        self.sequence = 0
        self.is_running = False
        self.send_thread: Optional[threading.Thread] = None
        
        # Give subscribers time to connect (ZeroMQ PUB/SUB slow joiner)
        time.sleep(0.1)
        
        print(f"[VideoTeleopSender] Initialized")
        print(f"  ZMQ address: {self.zmq_address}")
        print(f"  Send rate: {self.send_rate_hz} Hz")
        print(f"  Landmarks enabled: {self.enable_landmarks}")
    
    def _pack_calibration_data(self) -> dict:
        """
        Pack calibration data from VideoListener.
        
        Returns:
            Calibration data dict
        """
        # Get calibration state from VideoListener
        left_home = None
        right_home = None
        calibrated = False
        
        try:
            # Check if calibration is complete for both hands
            if hasattr(self.video_listener, 'home_wrist_pos_raw_left') and \
               self.video_listener.home_wrist_pos_raw_left is not None:
                left_home = self.video_listener.home_wrist_pos_raw_left.tolist() if \
                    hasattr(self.video_listener.home_wrist_pos_raw_left, 'tolist') else \
                    list(self.video_listener.home_wrist_pos_raw_left)
            
            if hasattr(self.video_listener, 'home_wrist_pos_raw_right') and \
               self.video_listener.home_wrist_pos_raw_right is not None:
                right_home = self.video_listener.home_wrist_pos_raw_right.tolist() if \
                    hasattr(self.video_listener.home_wrist_pos_raw_right, 'tolist') else \
                    list(self.video_listener.home_wrist_pos_raw_right)
            
            # Consider calibrated if at least one hand is calibrated
            calibrated = (left_home is not None) or (right_home is not None)
        except Exception as e:
            # Silently fail if calibration data is not available
            pass
        
        return {
            "left_home_wrist_pos": left_home,
            "right_home_wrist_pos": right_home,
            "calibrated": calibrated,
        }
    
    def _pack_hand_data(self, pose_dict: Optional[dict], side: str) -> dict:
        """
        Pack single hand data from VideoListener.get_pose_both() output.
        
        Args:
            pose_dict: Output from VideoListener.get_pose_both() for one hand
            side: "left" or "right"
            
        Returns:
            Packed hand data dict following protocol format
        """
        if pose_dict is None:
            return {
                "detected": False,
                "camera_frame": {
                    "wrist_position": None,
                    "wrist_orientation": None,
                    "palm_direction": None,
                    "palm_normal": None,
                    "palm_position": None,
                },
                "robot_frame": {
                    "wrist_position": None,
                    "wrist_orientation": None,
                    "wrist_quaternion": None,
                },
                "hand_joints": {
                    "joint_angles": None,
                    "joint_names": None,
                },
                "landmarks": {
                    "enabled": self.enable_landmarks,
                    "data": None,
                },
            }
        
        # Extract data from pose_dict
        hand_pose_raw = pose_dict.get("hand_pose_raw")  # (21, 3) landmarks
        arm_pose_raw = pose_dict.get("arm_pose_raw")     # (4, 3) arm points
        hand_joint_pose = pose_dict.get("hand_joint_pose")  # (24,) joint angles
        wrist_pos = pose_dict.get("wrist_pos")            # (3,) wrist position (robot frame)
        wrist_ori = pose_dict.get("wrist_ori")           # (3,) wrist orientation euler (robot frame)
        
        # Convert numpy arrays to lists
        def to_list_or_none(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr
        
        # Pack camera frame data
        camera_frame = {
            "wrist_position": None,
            "wrist_orientation": None,
            "palm_direction": None,
            "palm_normal": None,
            "palm_position": None,
        }
        
        if arm_pose_raw is not None and isinstance(arm_pose_raw, np.ndarray):
            # arm_pose_raw: (4, 3) = [direction, palm_normal, wrist_pos, palm_pos]
            if arm_pose_raw.shape[0] >= 4:
                camera_frame["palm_direction"] = to_list_or_none(arm_pose_raw[0])
                camera_frame["palm_normal"] = to_list_or_none(arm_pose_raw[1])
                camera_frame["wrist_position"] = to_list_or_none(arm_pose_raw[2])
                camera_frame["palm_position"] = to_list_or_none(arm_pose_raw[3])
        
        # Pack robot frame data
        robot_frame = {
            "wrist_position": to_list_or_none(wrist_pos),
            "wrist_orientation": to_list_or_none(wrist_ori),
            "wrist_quaternion": None,  # Will be computed from euler if needed
        }
        
        # Convert euler to quaternion if wrist_ori is available
        if wrist_ori is not None and isinstance(wrist_ori, np.ndarray) and len(wrist_ori) == 3:
            try:
                from scipy.spatial.transform import Rotation as R
                r = R.from_euler('xyz', wrist_ori, degrees=False)
                quat = r.as_quat()  # (x, y, z, w)
                # Convert to (w, x, y, z)
                robot_frame["wrist_quaternion"] = [quat[3], quat[0], quat[1], quat[2]]
            except Exception as e:
                print(f"[WARNING] Failed to convert euler to quaternion for {side} hand: {e}")
        
        # Pack hand joints
        hand_joints = {
            "joint_angles": to_list_or_none(hand_joint_pose),
            "joint_names": None,  # Optional, can be added if needed
        }
        
        # Pack landmarks (optional)
        landmarks = {
            "enabled": self.enable_landmarks,
            "data": to_list_or_none(hand_pose_raw) if self.enable_landmarks else None,
        }
        
        return {
            "detected": True,
            "camera_frame": camera_frame,
            "robot_frame": robot_frame,
            "hand_joints": hand_joints,
            "landmarks": landmarks,
        }
    
    def _pack_message(self, left_pose: Optional[dict], right_pose: Optional[dict]) -> dict:
        """
        Pack complete message following protocol format.
        
        Args:
            left_pose: Output from VideoListener.get_pose_both()[0]
            right_pose: Output from VideoListener.get_pose_both()[1]
            
        Returns:
            Complete message dict
        """
        timestamp = time.time()
        
        message = {
            "header": {
                "version": "1.0",
                "timestamp": timestamp,
                "sequence": self.sequence,
                "source": "video_listener",
            },
            "left_hand": self._pack_hand_data(left_pose, "left"),
            "right_hand": self._pack_hand_data(right_pose, "right"),
            "calibration": self._pack_calibration_data(),
        }
        
        self.sequence += 1
        return message
    
    def _send_loop(self) -> None:
        """Main send loop (runs in separate thread)."""
        print(f"[VideoTeleopSender] Send loop started (thread ID: {threading.get_ident()})")
        
        last_send_time = time.perf_counter()
        error_count = 0
        last_debug_time = time.perf_counter()
        loop_count = 0
        
        while self.is_running:
            loop_count += 1
            try:
                # Get pose data from VideoListener
                # Skip visualization update in send loop (main thread handles it)
                if self.video_listener.hand_mode == "both":
                    try:
                        left_pose, right_pose = self.video_listener.get_pose_both(update_visualization=False)
                    except Exception as e:
                        print(f"[ERROR] get_pose_both() failed in send loop: {e}")
                        import traceback
                        traceback.print_exc()
                        left_pose, right_pose = None, None
                else:
                    # Single hand mode - convert to both format
                    # Skip visualization update in send loop (main thread handles it)
                    side = self.video_listener._default_side
                    hand_pose_raw, arm_pose_raw, hand_joint_pose, wrist_pos, wrist_ori = \
                        self.video_listener.get_pose(side, update_visualization=False)
                    
                    if side == "left":
                        left_pose = {
                            "hand_pose_raw": hand_pose_raw,
                            "arm_pose_raw": arm_pose_raw,
                            "hand_joint_pose": hand_joint_pose,
                            "wrist_pos": wrist_pos,
                            "wrist_ori": wrist_ori,
                        } if hand_joint_pose is not None else None
                        right_pose = None
                    else:
                        left_pose = None
                        right_pose = {
                            "hand_pose_raw": hand_pose_raw,
                            "arm_pose_raw": arm_pose_raw,
                            "hand_joint_pose": hand_joint_pose,
                            "wrist_pos": wrist_pos,
                            "wrist_ori": wrist_ori,
                        } if hand_joint_pose is not None else None
                
                # Pack message (always send, even if no hands detected)
                message = self._pack_message(left_pose, right_pose)
                
                # Serialize and send
                packed = msgpack.packb(message, use_bin_type=True)
                self.socket.send(packed)
                
                # Note: sequence is incremented in _pack_message
                error_count = 0  # Reset error count on success
                
                # Debug output (every 5 seconds)
                current_debug_time = time.perf_counter()
                if current_debug_time - last_debug_time >= 5.0:
                    left_detected = left_pose is not None and left_pose.get("hand_joint_pose") is not None
                    right_detected = right_pose is not None and right_pose.get("hand_joint_pose") is not None
                    print(f"[DEBUG] Send loop: loop_count={loop_count}, seq={self.sequence}, "
                          f"L={'✓' if left_detected else '✗'}, R={'✓' if right_detected else '✗'}, "
                          f"left_pose={'not None' if left_pose is not None else 'None'}, "
                          f"right_pose={'not None' if right_pose is not None else 'None'}")
                    last_debug_time = current_debug_time
                
                # First loop debug
                if loop_count == 1:
                    print(f"[DEBUG] First loop iteration: seq={self.sequence}")
                
                # Rate limiting
                current_time = time.perf_counter()
                elapsed = current_time - last_send_time
                sleep_time = max(0.0, self.send_period - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_send_time = time.perf_counter()
                
            except Exception as e:
                error_count += 1
                if error_count <= 3 or error_count % 10 == 0:  # Print first 3 errors, then every 10th
                    print(f"[ERROR] Send loop error (count={error_count}): {e}")
                    if error_count <= 3:
                        import traceback
                        traceback.print_exc()
                time.sleep(0.1)  # Brief pause before retry
    
    def start(self) -> None:
        """Start sending loop."""
        if self.is_running:
            print("[WARNING] Sender already running")
            return
        
        self.is_running = True
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.send_thread.start()
        
        # Wait a moment to ensure thread started
        import time
        time.sleep(0.1)
        
        # Verify thread is alive
        if not self.send_thread.is_alive():
            print("[ERROR] Send thread failed to start!")
            self.is_running = False
        else:
            print("[VideoTeleopSender] Started")
    
    def stop(self) -> None:
        """Stop sending loop."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.send_thread is not None:
            self.send_thread.join(timeout=2.0)
        self.socket.close()
        self.context.term()
        print("[VideoTeleopSender] Stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        """Context manager exit."""
        self.stop()

