"""Video-based hand listener (MediaPipe + AprilTag wrist pose)."""

from __future__ import annotations

import threading
import yaml
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from video_teleop.config_paths import DEFAULT_CAMERA_CALIBRATION_YAML, DEFAULT_HAND_CALIBRATION_YAML

from .arm_pose_estimator import ArmPoseEstimator
from .mediapipe_shadowhand import MediaPipeShadowHand


class VideoListener:
    """Receives video frames and outputs hand/arm pose for teleop."""

    def __init__(
        self,
        *,
        camera_index: Optional[int] = None,
        image_width: int = 640,
        image_height: int = 480,
        calibration_file: Optional[str] = DEFAULT_HAND_CALIBRATION_YAML,
        camera_calibration_file: Optional[str] = DEFAULT_CAMERA_CALIBRATION_YAML,
        hand_mode: str = "both",
        force_position_based_handedness: bool = True,
        enable_visualization: bool = True,
        window_name: str = "Video Listener",
        hand_pose_window_name: str = "Hand Pose",
    ) -> None:
        """
        Initialize video listener.
        
        Args:
            camera_index: Camera device index
            image_width: Camera frame width
            image_height: Camera frame height
            calibration_file: Path to **hand** joint-range calibration YAML (optional).
            camera_calibration_file: Path to **camera** intrinsics YAML (optional).
            hand_mode: Hand selection mode ("left", "right", "both")
                      For coordinate transformation, side is derived from hand_mode.
            force_position_based_handedness: If True, force left/right hand detection based on relative wrist positions.
                      Leftmost wrist = Left hand, Rightmost wrist = Right hand.
                      This prevents handedness confusion and jumping. Default: True
            enable_visualization: If True, enables real-time visualization windows
            window_name: Name of the main visualization window
            hand_pose_window_name: Name of the virtual hand pose window
        """
        if camera_index is None:
            camera_index = 0
        self.camera_index = camera_index
        self.image_width = image_width
        self.image_height = image_height
        
        self.hand_mode = hand_mode.lower()
        if self.hand_mode not in ["left", "right", "both"]:
            raise ValueError(f"hand_mode must be 'left', 'right', or 'both', got '{hand_mode}'")
        
        if self.hand_mode == "both":
            self._default_side = "right"
        elif self.hand_mode in ["left", "right"]:
            self._default_side = self.hand_mode

        camera_calib_file = camera_calibration_file
        hand_calib_file = None
        
        if calibration_file:
            try:
                with open(calibration_file, "r") as f:
                    calib_data = yaml.safe_load(f)
                    if calib_data:
                        if "joint_ranges" in calib_data:
                            hand_calib_file = calibration_file
                        elif "camera_matrix" in calib_data:
                            if camera_calib_file is None:
                                camera_calib_file = calibration_file
            except Exception as e:
                print(f"[VideoListener] Warning: Failed to check calibration file type: {e}")

        self.arm_estimator = ArmPoseEstimator(
            image_width_px=image_width,
            image_height_px=image_height,
            calibration_file=camera_calib_file,
            use_ground_as_world_origin=True,
        )
        
        hand_calibration_ranges = None
        if hand_calib_file:
            try:
                with open(hand_calib_file, "r") as f:
                    calib_data = yaml.safe_load(f)
                    if calib_data and "joint_ranges" in calib_data:
                        joint_ranges = calib_data["joint_ranges"]
                        hand_calibration_ranges = {}
                        for joint_name, range_data in joint_ranges.items():
                            hand_calibration_ranges[joint_name] = (
                                range_data["human_min"],
                                range_data["human_max"],
                                range_data["robot_min"],
                                range_data["robot_max"],
                            )
                        print(f"[VideoListener] Loaded hand calibration from {hand_calib_file}")
            except Exception as e:
                print(f"[VideoListener] Warning: Failed to load hand calibration: {e}")
        
        self.shadowhand = MediaPipeShadowHand(
            max_num_hands=2,
            hand_mode=self.hand_mode,
            force_position_based_handedness=force_position_based_handedness,
            custom_joint_ranges=hand_calibration_ranges,
        )

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_running = False

        # Wrist orientation is derived directly from the tag-relative basis.
        # Keep this as identity to avoid extra fixed frame rotations.
        self.BASE_ROT_MAT_INV = np.eye(3, dtype=np.float32)
        
        self._enable_visualization = bool(enable_visualization)
        self._window_name = window_name
        self._hand_pose_window_name = hand_pose_window_name
        self._fps_counter = 0
        self._fps_last_time = time.perf_counter()
        self._fps_value = 0.0
        
        if self._enable_visualization:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self._hand_pose_window_name, cv2.WINDOW_NORMAL)

        self._last_arm_pose_fail_print_ts: float = 0.0
        self._arm_pose_fail_print_interval_s: float = 1.0

        self._viz_draw_apriltag_axes: bool = False

        # Lock to prevent concurrent get_pose from send thread and main thread (avoids segfaults)
        self._pose_lock = threading.Lock()

    def start(self) -> None:
        """Start camera capture."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self._is_running = True

    def stop(self) -> None:
        """Stop camera capture."""
        self._is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera."""
        if not self._is_running:
            return None

        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def get_pose(self, side: Optional[str] = None, update_visualization: Optional[bool] = None) -> Tuple[Optional[np.ndarray], ...]:
        """
        Get pose for a single hand.
        """
        with self._pose_lock:
            return self._get_pose_impl(side, update_visualization)

    def _get_pose_impl(self, side: Optional[str], update_visualization: Optional[bool]) -> Tuple[Optional[np.ndarray], ...]:
        """Internal get_pose implementation (must hold _pose_lock)."""
        if side is None:
            side = self._default_side

        should_update_viz = update_visualization if update_visualization is not None else self._enable_visualization

        frame = self.get_frame()
        if frame is None:
            return None, None, None, None, None

        # Use MediaPipeShadowHand to get landmarks and hand label (avoids duplicate processing)
        landmarks, hand_label = self.shadowhand.get_landmarks(frame)
        
        if landmarks is None or hand_label is None:
            # Visualization even when no hand detected
            self._viz_draw_apriltag_axes = False
            if should_update_viz:
                self._update_visualization(frame, None, False)
            return None, None, None, None, None
        
        # In both mode, adjust side based on detected hand to ensure correct coordinate transformation
        if self.hand_mode == "both" and hand_label:
            detected_side = hand_label.lower()
            # Override side parameter to match detected hand for correct coordinate transformation
            side = detected_side
        
        # AprilTag-based arm pose
        try:
            arm_pose = self.arm_estimator.estimate(
                landmarks,
                self.image_width,
                self.image_height,
                frame_bgr=frame,
                side=side,
            )
        except Exception as e:
            now = time.time()
            if now - self._last_arm_pose_fail_print_ts >= self._arm_pose_fail_print_interval_s:
                self._last_arm_pose_fail_print_ts = now
                print(f"Arm pose estimation failed: {e}")
            arm_pose = None

        # For hand_pose_raw, we use MediaPipe's normalized coordinate system
        hand_pose_raw = landmarks.copy()

        arm_pose_raw = None
        wrist_pos = None
        wrist_ori = None

        self._viz_draw_apriltag_axes = arm_pose is not None
        if arm_pose is not None:
            arm_pose_raw = arm_pose.as_arm_points()  # (4, 3): [direction, palm_normal, wrist_pos, palm_pos]
            direction = arm_pose_raw[0]
            palm_normal = arm_pose_raw[1]
            wrist_pos = arm_pose_raw[2]

            palm_normal /= np.linalg.norm(palm_normal)
            direction /= np.linalg.norm(direction)
            target_y = np.cross(palm_normal, direction)
            target_y /= np.linalg.norm(target_y)

            r = np.array([direction, target_y, palm_normal]).T @ self.BASE_ROT_MAT_INV
            wrist_ori = R.from_matrix(r).as_euler("xyz")

        try:
            hand_joint_pose = self.shadowhand.infer_joints(frame)
            if side == "left":
                reverse_list = [2, 6, 10, 15, 22]
                for joint in reverse_list:
                    hand_joint_pose[joint] = -hand_joint_pose[joint]
        except Exception as e:
            print(f"ShadowHand joint estimation failed: {e}")
            return None, None, None, None, None

        if should_update_viz:
            self._update_visualization(frame, hand_label, hand_joint_pose is not None)

        return hand_pose_raw, arm_pose_raw, np.array(hand_joint_pose), wrist_pos, wrist_ori

    def get_pose_both(self, update_visualization: Optional[bool] = None) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Get pose for both hands.
        """
        with self._pose_lock:
            return self._get_pose_both_impl(update_visualization)

    def _get_pose_both_impl(self, update_visualization: Optional[bool]) -> Tuple[Optional[dict], Optional[dict]]:
        """Internal get_pose_both implementation (must hold _pose_lock)."""
        if self.hand_mode != "both":
            raise ValueError("get_pose_both() only works when hand_mode == 'both'")

        # Determine if we should update visualization
        should_update_viz = update_visualization if update_visualization is not None else self._enable_visualization
        
        frame = self.get_frame()
        if frame is None:
            return None, None

        # Use MediaPipeShadowHand to get landmarks and joints for both hands (avoids duplicate processing)
        # get_landmarks_both calls infer_joints_both internally, so we get both landmarks and joints
        both_landmarks = self.shadowhand.get_landmarks_both(frame)
        both_joints = self.shadowhand.infer_joints_both(frame)  # Already processed by get_landmarks_both, but we need joints
        
        if both_landmarks is None:
            # Visualization even when no hands detected
            self._viz_draw_apriltag_axes = False
            if should_update_viz:
                self._update_visualization_both(frame, False, False)
            return None, None
        
        left_data = both_landmarks.get("left")
        right_data = both_landmarks.get("right")
        
        def process_single_hand(landmarks_data, joints_data, side):
            """Helper to process a single hand."""
            if landmarks_data is None:
                return None
            
            landmarks = landmarks_data["landmarks"]  # Already normalized (21, 3)
            hand_label = landmarks_data["label"]

            hand_pose_raw = landmarks.copy()
            arm_pose_raw = None
            wrist_pos = None
            wrist_ori = None

            try:
                arm_pose = self.arm_estimator.estimate(
                    landmarks,
                    self.image_width,
                    self.image_height,
                    frame_bgr=frame,
                    side=side,
                )
            except Exception as e:
                now = time.time()
                if now - self._last_arm_pose_fail_print_ts >= self._arm_pose_fail_print_interval_s:
                    self._last_arm_pose_fail_print_ts = now
                    print(f"Arm pose estimation failed for {side} hand: {e}")
                arm_pose = None

            if arm_pose is not None:
                arm_pose_raw = arm_pose.as_arm_points()  # (4, 3): [direction, palm_normal, wrist_pos, palm_pos]
                direction = arm_pose_raw[0]
                palm_normal = arm_pose_raw[1]
                wrist_pos = arm_pose_raw[2]

                palm_normal /= np.linalg.norm(palm_normal)
                direction /= np.linalg.norm(direction)
                target_y = np.cross(palm_normal, direction)
                target_y /= np.linalg.norm(target_y)

                r = np.array([direction, target_y, palm_normal]).T @ self.BASE_ROT_MAT_INV
                wrist_ori = R.from_matrix(r).as_euler("xyz")

            # Get ShadowHand joint angles from joints_data
            if joints_data is None:
                return None
            
            hand_joint_pose = np.array(joints_data)
            
            if side == "left":
                reverse_list = [2, 6, 10, 15, 22]
                for joint in reverse_list:
                    hand_joint_pose[joint] = -hand_joint_pose[joint]

            return {
                "hand_pose_raw": hand_pose_raw,
                "arm_pose_raw": arm_pose_raw,
                "hand_joint_pose": hand_joint_pose,
                "wrist_pos": wrist_pos,
                "wrist_ori": wrist_ori,
            }
        
        # Process left hand
        left_joints = both_joints.get("left") if both_joints else None
        left_result = process_single_hand(left_data, left_joints, "left")
        
        # Process right hand
        right_joints = both_joints.get("right") if both_joints else None
        right_result = process_single_hand(right_data, right_joints, "right")
        
        if should_update_viz:
            left_detected = left_result is not None and left_result["hand_joint_pose"] is not None
            right_detected = right_result is not None and right_result["hand_joint_pose"] is not None
            self._viz_draw_apriltag_axes = (
                (left_result is not None and left_result.get("arm_pose_raw") is not None)
                or (right_result is not None and right_result.get("arm_pose_raw") is not None)
            )
            self._update_visualization_both(frame, left_detected, right_detected)
        
        return left_result, right_result

    def _update_fps(self) -> float:
        """Update and return current FPS."""
        self._fps_counter += 1
        now = time.perf_counter()
        dt = now - self._fps_last_time
        if dt >= 0.5:
            self._fps_value = self._fps_counter / dt
            self._fps_counter = 0
            self._fps_last_time = now
        return self._fps_value

    def _update_visualization(self, frame: np.ndarray, hand_label: Optional[str], hand_detected: bool) -> None:
        """Update visualization windows for single hand mode."""
        vis_frame = self.shadowhand.annotate(frame.copy())
        if self._viz_draw_apriltag_axes:
            vis_frame = self.arm_estimator.draw_latest_apriltag_frames(vis_frame)

        fps = self._update_fps()

        if hand_detected:
            label = hand_label or "Unknown"
            status = f"HAND({label}) MODE({self.hand_mode})"
            color = (0, 255, 0)
        else:
            status = f"NO_HAND MODE({self.hand_mode})"
            color = (0, 0, 255)
        
        cv2.putText(
            vis_frame,
            f"{status}  FPS:{fps:5.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        
        cv2.imshow(self._window_name, vis_frame)

        hand_pose_img = self.shadowhand.render_hand_pose(width=640, height=480, scale=1.0)
        cv2.imshow(self._hand_pose_window_name, hand_pose_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.stop()

    def _update_visualization_both(self, frame: np.ndarray, left_detected: bool, right_detected: bool) -> None:
        """Update visualization windows for both hands mode."""
        # Annotate frame with hand landmarks
        vis_frame = self.shadowhand.annotate(frame.copy())
        if self._viz_draw_apriltag_axes:
            vis_frame = self.arm_estimator.draw_latest_apriltag_frames(vis_frame)
        
        # Update FPS
        fps = self._update_fps()
        
        # Status text
        left_status = "L" if left_detected else "-"
        right_status = "R" if right_detected else "-"
        status = f"HANDS(L:{left_status} R:{right_status}) MODE(both)"
        color = (0, 255, 0) if (left_detected or right_detected) else (0, 0, 255)
        
        cv2.putText(
            vis_frame,
            f"{status}  FPS:{fps:5.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        
        cv2.imshow(self._window_name, vis_frame)
        
        # Virtual hand pose window
        hand_pose_img = self.shadowhand.render_hand_pose(width=640, height=480, scale=1.0)
        cv2.imshow(self._hand_pose_window_name, hand_pose_img)
        
        # Handle key press (ESC to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.stop()

    def __enter__(self) -> "VideoListener":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit."""
        self.stop()
        # MediaPipe hands instance is managed by MediaPipeShadowHand
        if self._enable_visualization:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import os
    
    # Handle direct execution (when run as script, not as module)
    if __package__ is None:
        # Add parent directories to path to enable absolute imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # hand_teleop/
        project_root = os.path.dirname(parent_dir)  # source/
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Re-import dependencies with absolute imports
        # Note: VideoListener class is already defined above, so we can use it directly
        # But we need to ensure dependencies are importable
        try:
            from video_teleop.core.arm_pose_estimator import ArmPoseEstimator
            from video_teleop.core.mediapipe_shadowhand import MediaPipeShadowHand
        except ImportError:
            print("Error: Cannot import dependencies when running directly.")
            print("Please use: python -m video_teleop.core.video_listener")
            print("Or import in your code: from video_teleop.core.video_listener import VideoListener")
            sys.exit(1)
    
    # Test example
    print("Video Listener Test")
    print("Note: For proper usage, use as a module:")
    print("  python -m video_teleop.core.video_listener")
    print("  or import: from video_teleop.core.video_listener import VideoListener\n")
    
    from video_teleop.config_paths import default_camera_calibration_yaml, default_hand_calibration_yaml

    cam_yaml = default_camera_calibration_yaml()
    hand_yaml = default_hand_calibration_yaml()
    calib_arg = cam_yaml if os.path.isfile(cam_yaml) else None
    hand_arg = hand_yaml if os.path.isfile(hand_yaml) else None
    if calib_arg is None:
        print(f"Warning: camera calibration not found at {cam_yaml}. Using default camera intrinsics.")
    if hand_arg is None:
        print(f"Warning: hand calibration not found at {hand_yaml}. Using default joint mapping.")

    listener = VideoListener(
        camera_index=0,
        calibration_file=hand_arg,
        camera_calibration_file=calib_arg,
        hand_mode="both",
        enable_visualization=True,
    )
    listener.start()

    print("Press 'q' to quit")
    print("Using both mode - will get pose for both hands\n")

    try:
        while True:
            if listener.hand_mode == "both":
                # Use get_pose_both() for both mode
                left_pose, right_pose = listener.get_pose_both()
                if left_pose is not None:
                    print(f"LEFT: Wrist pos: {left_pose['wrist_pos']}, Joints (first 5): {left_pose['hand_joint_pose'][:5]}")
                if right_pose is not None:
                    print(f"RIGHT: Wrist pos: {right_pose['wrist_pos']}, Joints (first 5): {right_pose['hand_joint_pose'][:5]}")
            else:
                # Use get_pose() for single hand mode
                hand_pose_raw, arm_pose_raw, hand_joint_pose, wrist_pos, wrist_ori = listener.get_pose("right")
                if hand_joint_pose is not None:
                    print(f"Wrist pos: {wrist_pos}, Joints (first 5): {hand_joint_pose[:5]}")
            time.sleep(0.1)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        listener.stop()

