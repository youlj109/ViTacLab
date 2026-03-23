"""Estimate arm pose parameters from AprilTags (and provide optional visualization)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import time

import numpy as np


@dataclass
class ArmPoseParams:
    """Arm pose parameters similar to LeapMotion output."""

    direction: np.ndarray  # Palm direction vector (normalized, 3D)
    palm_normal: np.ndarray  # Palm normal vector (normalized, 3D, pointing outward from palm)
    wrist_position: np.ndarray  # Wrist translation in meters (active reference frame)
    palm_position: np.ndarray  # Palm center translation in meters (active reference frame)

    def as_arm_points(self) -> np.ndarray:
        """Return as (4, 3) array compatible with GarmentLab format: [direction, palm_normal, wrist_pos, palm_pos]."""
        return np.array([self.direction, self.palm_normal, self.wrist_position, self.palm_position])


class ArmPoseEstimator:
    """AprilTag wrist pose estimator (tag0-relative)."""

    HAND_LENGTH_AVG = 0.19  # ~19cm, from wrist to middle fingertip (approximate)

    def __init__(
        self,
        *,
        focal_length_px: Optional[float] = None,
        image_width_px: int = 640,
        image_height_px: int = 480,
        calibration_file: Optional[str] = None,
        # AprilTag configuration (optional; used when frame is provided to estimate()).
        apriltag_enabled: bool = True,
        apriltag_family: str = "tag36h11",
        world_tag_id: int = 0,
        left_wrist_tag_id: int = 1,
        right_wrist_tag_id: int = 2,
        apriltag_tag_size_m: float = 0.05,
        # Tag axes mapping to "direction" (palm forward) and "palm_normal".
        # AprilTag tag coordinate axes are: x right, y down, z forward (out of the tag).
        tag_direction_vector: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        tag_palm_normal_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        # The tag "z" points outward from the tag. In a typical setup where the tag faces
        # the camera (camera -> people), the hand "palm normal" (outward from palm) is often
        # the opposite direction, so we default to -1. Flip to +1 if needed.
        tag_palm_normal_sign: float = 1.0,
        use_ground_as_world_origin: bool = True,
    ) -> None:
        """
        Initialize the estimator.

        Args:
            focal_length_px: Camera focal length in pixels. If None, loaded from calibration_file or estimated.
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels
            calibration_file: Path to camera calibration YAML file (optional). If provided, loads camera_matrix.
        """
        # AprilTag state (lazily import apriltag).
        self.apriltag_enabled = bool(apriltag_enabled)
        self.world_tag_id = int(world_tag_id)
        self.left_wrist_tag_id = int(left_wrist_tag_id)
        self.right_wrist_tag_id = int(right_wrist_tag_id)
        self.apriltag_family = str(apriltag_family)
        self.apriltag_tag_size_m = float(apriltag_tag_size_m)
        self.use_ground_as_world_origin = bool(use_ground_as_world_origin)
        self._apriltag_available = False
        self._detector = None

        self._tag_direction_vector = np.array(tag_direction_vector, dtype=np.float32)
        self._tag_palm_normal_vector = np.array(tag_palm_normal_vector, dtype=np.float32)
        self._tag_palm_normal_sign = float(tag_palm_normal_sign)

        self._last_frame_id: Optional[int] = None
        self._last_detections_by_id: dict[int, object] = {}
        # Used to rate-limit "tag not detected" diagnostics.
        self._last_apriltag_fail_ts: float = 0.0
        self._apriltag_fail_detail_interval_s: float = 2.0
        
        # Load calibration if provided
        calibration_loaded = False
        camera_matrix = None
        if calibration_file is not None:
            try:
                import yaml
                with open(calibration_file, "r") as f:
                    calib_data = yaml.safe_load(f)
                if calib_data and "camera_matrix" in calib_data:
                    camera_matrix = np.array(calib_data["camera_matrix"])
                    # Optional: apriltag tag physical size (meters) for pose scale.
                    for key in ("apriltag_tag_size_m", "tag_size_m", "tag_size"):
                        if key in calib_data:
                            try:
                                self.apriltag_tag_size_m = float(calib_data[key])
                                break
                            except Exception:
                                pass
                    # Use average of fx and fy as focal length
                    self.focal_length_px = float((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0)
                    if image_width_px is None:
                        image_width_px = calib_data.get("image_width", 640)
                    if image_height_px is None:
                        image_height_px = calib_data.get("image_height", 480)
                    print(f"Loaded calibration from {calibration_file}")
                    print(f"  Focal length: {self.focal_length_px:.1f} px")
                    print(f"  Image size: {image_width_px}x{image_height_px}")
                    calibration_loaded = True
                else:
                    print(f"Warning: Calibration file {calibration_file} does not contain 'camera_matrix'. Falling back to estimation.")
            except ImportError:
                print("Warning: PyYAML not installed. Cannot load calibration file.")
            except Exception as e:
                print(f"Warning: Failed to load calibration file: {e}")
        
        # If calibration was not loaded, estimate focal length
        if not calibration_loaded:
            # Estimate focal length if not provided (typical camera FOV ~60 degrees)
            if focal_length_px is None:
                # Approximate: focal_length ≈ image_width / (2 * tan(FOV/2))
                # For FOV=60°, this gives focal_length ≈ image_width * 0.866
                self.focal_length_px = float(image_width_px * 0.866)  # Rough estimate
            else:
                self.focal_length_px = float(focal_length_px)
        
        self.image_width_px = int(image_width_px)
        self.image_height_px = int(image_height_px)

        # Setup AprilTag camera params if possible.
        self._camera_params: Optional[Tuple[float, float, float, float]] = None
        if camera_matrix is not None and camera_matrix.shape == (3, 3):
            fx = float(camera_matrix[0, 0])
            fy = float(camera_matrix[1, 1])
            cx = float(camera_matrix[0, 2])
            cy = float(camera_matrix[1, 2])
            self._camera_params = (fx, fy, cx, cy)
        else:
            # Best-effort fallback for apriltag pose estimation.
            fx = fy = float(self.focal_length_px)
            cx = float(self.image_width_px) / 2.0
            cy = float(self.image_height_px) / 2.0
            self._camera_params = (fx, fy, cx, cy)

        if self.apriltag_enabled:
            try:
                import apriltag  # type: ignore

                # apriltag Detector wrapper expects DetectorOptions via the `options=` arg.
                options = apriltag.DetectorOptions(families=self.apriltag_family)
                self._detector = apriltag.Detector(options=options)
                self._apriltag_available = True
            except Exception as e:
                self._apriltag_available = False
                print(
                    "[ArmPoseEstimator] Warning: apriltag not available; AprilTag-based wrist pose estimation will fail. "
                    f"Import/init error: {e}"
                )

    def _detect_apriltags(self, frame_bgr: np.ndarray) -> dict[int, object]:
        """Detect AprilTags and return latest detections keyed by tag_id."""
        if not self._apriltag_available or self._detector is None:
            return {}

        fid = id(frame_bgr)
        if fid == self._last_frame_id and self._last_detections_by_id:
            return self._last_detections_by_id

        import cv2  # OpenCV is required elsewhere in this pipeline

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        fx, fy, cx, cy = self._camera_params if self._camera_params is not None else (0, 0, 0, 0)
        camera_params = (fx, fy, cx, cy)

        # In this apriltag wrapper, `detect()` returns only homography/center/corners.
        # To get pose_R/pose_t we must call `detection_pose(...)` for each detection.
        detections = self._detector.detect(gray)

        from types import SimpleNamespace

        by_id: dict[int, object] = {}
        for det in detections:
            try:
                pose, init_error, final_error = self._detector.detection_pose(
                    det,
                    camera_params=camera_params,
                    tag_size=self.apriltag_tag_size_m,
                    z_sign=1,
                )
                pose_arr = np.array(pose, dtype=np.float32)
                R_cam_tag = pose_arr[:3, :3]
                t_cam_tag = pose_arr[:3, 3]

                by_id[int(det.tag_id)] = SimpleNamespace(
                    tag_id=int(det.tag_id),
                    pose_R=R_cam_tag,
                    pose_t=t_cam_tag,
                    init_error=init_error,
                    final_error=final_error,
                )
            except Exception:
                # Skip tags whose pose solve fails.
                continue

        self._last_frame_id = fid
        self._last_detections_by_id = by_id
        return by_id

    def _estimate_from_apriltag(
        self,
        *,
        frame_bgr: np.ndarray,
        side: str,
    ) -> Optional[ArmPoseParams]:
        """AprilTag-based estimation. Returns None if required tags aren't found."""
        if not self.apriltag_enabled or not self._apriltag_available:
            return None

        detections_by_id = self._detect_apriltags(frame_bgr)
        if not detections_by_id:
            return None

        wrist_tag_id = self.left_wrist_tag_id if side == "left" else self.right_wrist_tag_id
        if wrist_tag_id not in detections_by_id:
            return None

        d = detections_by_id[wrist_tag_id]
        if not hasattr(d, "pose_R") or not hasattr(d, "pose_t"):
            return None

        R_cam_wrist_tag = np.array(d.pose_R, dtype=np.float32).reshape(3, 3)
        t_cam_wrist_tag = np.array(d.pose_t, dtype=np.float32).reshape(3)

        if self.use_ground_as_world_origin:
            if self.world_tag_id not in detections_by_id:
                return None
            d_world = detections_by_id[self.world_tag_id]
            if not hasattr(d_world, "pose_R") or not hasattr(d_world, "pose_t"):
                return None

            R_cam_world_tag = np.array(d_world.pose_R, dtype=np.float32).reshape(3, 3)
            t_cam_world_tag = np.array(d_world.pose_t, dtype=np.float32).reshape(3)

            R_world_wrist_tag = R_cam_world_tag.T @ R_cam_wrist_tag
            t_world_wrist_tag = R_cam_world_tag.T @ (t_cam_wrist_tag - t_cam_world_tag)

            direction = self._normalize(R_world_wrist_tag @ self._tag_direction_vector)
            palm_normal = self._normalize(
                self._tag_palm_normal_sign * (R_world_wrist_tag @ self._tag_palm_normal_vector)
            )

            wrist_position = t_world_wrist_tag
            palm_position = wrist_position + direction * self.HAND_LENGTH_AVG
        else:
            direction = self._normalize(R_cam_wrist_tag @ self._tag_direction_vector)
            palm_normal = self._normalize(
                self._tag_palm_normal_sign * (R_cam_wrist_tag @ self._tag_palm_normal_vector)
            )
            wrist_position = t_cam_wrist_tag
            palm_position = wrist_position + direction * self.HAND_LENGTH_AVG

        return ArmPoseParams(
            direction=direction,
            palm_normal=palm_normal,
            wrist_position=wrist_position,
            palm_position=palm_position,
        )

    def draw_latest_apriltag_frames(
        self,
        frame_bgr: np.ndarray,
        *,
        tag_ids: Optional[set[int]] = None,
        axis_length_m: Optional[float] = None,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw the latest detected AprilTag axes onto `frame_bgr`.
        """
        if frame_bgr is None:
            return frame_bgr
        if not self._last_detections_by_id:
            return frame_bgr
        if not self._apriltag_available or self._camera_params is None:
            return frame_bgr

        import cv2

        fx, fy, cx, cy = self._camera_params
        if fx <= 1e-6 or fy <= 1e-6:
            return frame_bgr

        L = float(axis_length_m) if axis_length_m is not None else float(self.apriltag_tag_size_m) * 0.6
        L = max(1e-6, L)

        def project(p_cam: np.ndarray) -> Optional[tuple[int, int]]:
            z = float(p_cam[2])
            if z <= 1e-6:
                return None
            u = fx * float(p_cam[0]) / z + cx
            v = fy * float(p_cam[1]) / z + cy
            return (int(round(u)), int(round(v)))

        for tag_id, d in self._last_detections_by_id.items():
            if tag_ids is not None and tag_id not in tag_ids:
                continue
            if not hasattr(d, "pose_R") or not hasattr(d, "pose_t"):
                continue

            try:
                R_cam_tag = np.array(d.pose_R, dtype=np.float32).reshape(3, 3)
                t_cam_tag = np.array(d.pose_t, dtype=np.float32).reshape(3)
            except Exception:
                continue

            origin_cam = t_cam_tag
            x_cam = R_cam_tag @ np.array([L, 0.0, 0.0], dtype=np.float32) + t_cam_tag
            y_cam = R_cam_tag @ np.array([0.0, L, 0.0], dtype=np.float32) + t_cam_tag
            z_cam = R_cam_tag @ np.array([0.0, 0.0, L], dtype=np.float32) + t_cam_tag

            p0 = project(origin_cam)
            px = project(x_cam)
            py = project(y_cam)
            pz = project(z_cam)
            if p0 is None:
                continue

            if px is not None:
                cv2.line(frame_bgr, p0, px, (0, 0, 255), line_thickness)
            if py is not None:
                cv2.line(frame_bgr, p0, py, (0, 255, 0), line_thickness)
            if pz is not None:
                cv2.line(frame_bgr, p0, pz, (255, 0, 0), line_thickness)

            cv2.putText(
                frame_bgr,
                f"id={tag_id}",
                (p0[0] + 5, p0[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        return frame_bgr

    def _normalize(self, v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Normalize a vector."""
        n = float(np.linalg.norm(v))
        if n < eps:
            return np.zeros_like(v)
        return v / n

    def estimate(
        self,
        _landmarks: np.ndarray,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        *,
        frame_bgr: Optional[np.ndarray] = None,
        side: Optional[str] = None,
    ) -> ArmPoseParams:
        """
        Estimate arm pose parameters.

        AprilTag-only: requires `frame_bgr` and `side`.

        Args:
            image_width: Image width in pixels (if None, uses instance default)
            image_height: Image height in pixels (if None, uses instance default)
            _landmarks: Ignored (kept for API compatibility)

        Returns:
            ArmPoseParams with direction, palm_normal, wrist_position, palm_position
        """
        if frame_bgr is None or side is None:
            raise ValueError("AprilTag estimation requires both frame_bgr and side.")

        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side}")

        if not self.apriltag_enabled or not self._apriltag_available:
            raise RuntimeError(
                "[AprilTag] apriltag detector not available. "
                "Please install/enable the `apriltag` Python package."
            )

        detections_by_id = self._detect_apriltags(frame_bgr)
        wrist_tag_id = self.left_wrist_tag_id if side == "left" else self.right_wrist_tag_id

        if wrist_tag_id not in detections_by_id:
            detected_ids = sorted(detections_by_id.keys())
            now = time.time()
            # Rate-limit the detailed message to avoid flooding logs when tags leave view.
            if now - self._last_apriltag_fail_ts >= self._apriltag_fail_detail_interval_s:
                self._last_apriltag_fail_ts = now
                raise RuntimeError(
                    f"[AprilTag] Missing wrist tag id={wrist_tag_id} (side='{side}'). "
                    f"Detected ids={detected_ids}. "
                    f"Make sure the wrist tag is visible and `apriltag_tag_size_m={self.apriltag_tag_size_m}` "
                    "matches the real physical tag size."
                )
            raise RuntimeError(f"[AprilTag] Missing wrist tag id={wrist_tag_id} (side='{side}').")

        self.world_origin_camera = None
        april_pose = self._estimate_from_apriltag(frame_bgr=frame_bgr, side=side)
        if april_pose is None:
            raise RuntimeError("[AprilTag] Wrist tag detected, but pose estimation returned None unexpectedly.")
        return april_pose

    # Landmarks-based API removed.

