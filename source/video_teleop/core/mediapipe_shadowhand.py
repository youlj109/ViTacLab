"""
MediaPipe Hands (21 landmarks) -> ShadowHand 24-DoF joint targets (radians).

Design goals:
- Pure geometry (no training, no deep learning frameworks beyond MediaPipe).
- Low-latency, real-time friendly implementation.
- Clean, reusable API.

Notes on the mapping:
- Flexion angles are computed from 3-point joint geometry: angle(A-B-C) at B.
  For a straight finger, MediaPipe joint angle is near pi. We convert to flexion
  via: flex = pi - angle(A-B-C), so straight ~0 rad, curled -> positive.
- "Spread" (abduction/adduction) is estimated in the palm plane using a simple
  hand coordinate frame (wrist, index_mcp, pinky_mcp, middle_mcp).
- ShadowHand joint limits and exact kinematic definitions vary by simulator / URDF.
  This module provides a reasonable, stable default mapping for teleoperation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    c = _clamp(c, -1.0, 1.0)
    return float(math.acos(c))


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC (at point b) in radians."""
    return _angle_between(a - b, c - b)


def _signed_angle(u: np.ndarray, v: np.ndarray, axis: np.ndarray, eps: float = 1e-9) -> float:
    """
    Signed angle from u to v around 'axis' (right-hand rule), in radians.
    u, v, axis are 3D vectors.
    """
    au = _normalize(u, eps)
    av = _normalize(v, eps)
    ax = _normalize(axis, eps)
    if float(np.linalg.norm(au)) < eps or float(np.linalg.norm(av)) < eps or float(np.linalg.norm(ax)) < eps:
        return 0.0
    s = float(np.dot(ax, np.cross(au, av)))
    c = float(np.dot(au, av))
    return float(math.atan2(s, c))


def _project_to_plane(v: np.ndarray, n_unit: np.ndarray) -> np.ndarray:
    return v - n_unit * float(np.dot(v, n_unit))


def _linear_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """
    Map x from [x0,x1] to [y0,y1] linearly, with clamping.
    
    If x is outside [x0, x1], it is clamped to the nearest boundary:
    - If x < x0: maps to y0
    - If x > x1: maps to y1
    - If x0 == x1: clamps x to [y0, y1] range (degenerate case)
    """
    if abs(x1 - x0) < 1e-9:
        # Degenerate case: x0 == x1
        # Clamp x to [y0, y1] range
        return float(_clamp(y0 if x <= x0 else y1, min(y0, y1), max(y0, y1)))
    
    # Calculate normalized position
    t = (x - x0) / (x1 - x0)
    
    # Clamp to [0, 1] range
    # This handles cases where x < x0 (t < 0) or x > x1 (t > 1)
    t = _clamp(t, 0.0, 1.0)
    
    # Linear interpolation
    result = float(y0 + t * (y1 - y0))
    
    # Final clamp to ensure result is within [y0, y1] (handles floating point errors)
    return float(_clamp(result, min(y0, y1), max(y0, y1)))


def _detect_handedness_geometry(hand_landmarks) -> Optional[str]:
    """
    Detect left/right hand using geometry: cross product of hand flexion direction and thumb direction.
    
    Method:
    - Hand flexion direction: from wrist to middle finger MCP (palm direction)
    - Thumb direction: from thumb CMC to thumb MCP (thumb abduction direction)
    - Cross product: hand_dir × thumb_dir
    - In MediaPipe coordinate system (X: left->right, Y: top->bottom, Z: near->far):
      * Right hand: cross product Z component < 0 (thumb points right, hand points up)
      * Left hand: cross product Z component > 0 (thumb points left, hand points up)
    
    Returns:
        "Left" or "Right" if detection is confident, None otherwise
    """
    if not hand_landmarks or len(hand_landmarks.landmark) < 9:
        return None
    
    # Extract key landmarks
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    thumb_cmc = np.array([hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z])
    thumb_mcp = np.array([hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y, hand_landmarks.landmark[2].z])
    middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
    
    # Hand flexion direction: from wrist to middle finger MCP (palm direction, pointing upward)
    hand_dir = middle_mcp - wrist
    
    # Thumb direction: from thumb CMC to thumb MCP (thumb abduction direction)
    thumb_dir = thumb_mcp - thumb_cmc
    
    # Normalize vectors
    hand_dir_norm = _normalize(hand_dir)
    thumb_dir_norm = _normalize(thumb_dir)
    
    # Check if vectors are valid
    if float(np.linalg.norm(hand_dir_norm)) < 1e-6 or float(np.linalg.norm(thumb_dir_norm)) < 1e-6:
        return None
    
    # Compute cross product: hand_dir × thumb_dir
    cross = np.cross(hand_dir_norm, thumb_dir_norm)
    
    # The Z component of the cross product indicates handedness
    # In MediaPipe coordinate system:
    # - Right hand: thumb points right (positive X), hand points up (negative Y)
    #   cross product Z < 0
    # - Left hand: thumb points left (negative X), hand points up (negative Y)
    #   cross product Z > 0
    cross_z = float(cross[2])
    
    # Use a threshold to determine handedness
    # Positive Z -> Left hand, Negative Z -> Right hand
    threshold = 0.05  # Threshold to avoid noise near zero
    
    if cross_z > threshold:
        return "Left"
    elif cross_z < -threshold:
        return "Right"
    else:
        # Cross product Z is near zero, detection uncertain
        return None


class _LowPass:
    def __init__(self) -> None:
        self._y: Optional[float] = None

    def has_last(self) -> bool:
        return self._y is not None

    def last(self) -> float:
        if self._y is None:
            return 0.0
        return float(self._y)

    def filt(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = float(x)
            return float(self._y)
        a = _clamp(alpha, 0.0, 1.0)
        self._y = float(a * x + (1.0 - a) * self._y)
        return float(self._y)


class OneEuroFilter:
    """
    One Euro Filter for smooth-yet-responsive joint targets.

    Based on: "The One Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input
    in Interactive Systems" (Casucci et al.)
    """

    def __init__(self, *, min_cutoff: float = 2.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_last: Optional[float] = None

    @staticmethod
    def _alpha(cutoff_hz: float, freq_hz: float) -> float:
        cutoff_hz = max(1e-6, float(cutoff_hz))
        freq_hz = max(1e-6, float(freq_hz))
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        te = 1.0 / freq_hz
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: float) -> float:
        if self._t_last is None:
            self._t_last = float(t)
            self._x.filt(float(x), 1.0)
            self._dx.filt(0.0, 1.0)
            return float(x)

        dt = max(1e-6, float(t - self._t_last))
        freq = 1.0 / dt

        # Derivative of the signal.
        x_prev = self._x.last()
        dx = (float(x) - x_prev) * freq
        alpha_d = self._alpha(self.d_cutoff, freq)
        dx_hat = self._dx.filt(dx, alpha_d)

        # Adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, freq)
        x_hat = self._x.filt(float(x), alpha)

        self._t_last = float(t)
        return float(x_hat)


@dataclass(frozen=True)
class ShadowHandJoints:
    """
    ShadowHand 24-DoF joint order used by this project.

    This follows a common ShadowHand URDF convention:
    - Wrist: WRJ2, WRJ1
    - FF/MF/RF: J4 is spread (ab/ad), J3 MCP flex, J2 PIP flex, J1 DIP flex
    - Little: LFJ5 (metacarpal spread), LFJ4 (spread), LFJ3 MCP flex, LFJ2 PIP flex, LFJ1 DIP flex
    - Thumb: THJ5 opposition (proxy), THJ4 ab/ad (proxy), THJ3 MCP flex, THJ2 IP flex, THJ1 CMC flex (proxy)

    Indices:
    0-1  : WRJ2, WRJ1
    2-5  : FFJ4, FFJ3, FFJ2, FFJ1
    6-9  : MFJ4, MFJ3, MFJ2, MFJ1
    10-13: RFJ4, RFJ3, RFJ2, RFJ1
    14-18: LFJ5, LFJ4, LFJ3, LFJ2, LFJ1
    19-23: THJ5, THJ4, THJ3, THJ2, THJ1
    """

    N: int = 24

    @staticmethod
    def names() -> List[str]:
        return [
            "WRJ2",
            "WRJ1",
            "FFJ4",
            "FFJ3",
            "FFJ2",
            "FFJ1",
            "MFJ4",
            "MFJ3",
            "MFJ2",
            "MFJ1",
            "RFJ4",
            "RFJ3",
            "RFJ2",
            "RFJ1",
            "LFJ5",
            "LFJ4",
            "LFJ3",
            "LFJ2",
            "LFJ1",
            "THJ5",
            "THJ4",
            "THJ3",
            "THJ2",
            "THJ1",
        ]


class MediaPipeShadowHand:
    """
    MediaPipe Hands -> ShadowHand 24-DoF mapping (geometry + smoothing).

    Usage:
        teleop = MediaPipeShadowHand()
        joints = teleop(frame_bgr)  # List[24] of radians
    """

    def __init__(
        self,
        *,
        max_num_hands: int = 2,  # Allow up to 2 hands
        hand_mode: str = "left",  # "left", "right", "both", "auto"
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        one_euro_min_cutoff: float = 2.0,
        one_euro_beta: float = 0.02,
        one_euro_d_cutoff: float = 1.0,
        hold_last_on_no_hand: bool = True,
        handedness_confidence_threshold: float = 0.8,
        force_position_based_handedness: bool = True,  # Force left/right based on wrist X position
        left_hand_color: Tuple[int, int, int] = (255, 0, 0),  # BGR: Blue for left hand
        right_hand_color: Tuple[int, int, int] = (0, 255, 0),  # BGR: Green for right hand
        custom_joint_ranges: Optional[Dict[str, Tuple[float, float, float, float]]] = None,  # Custom calibration ranges
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._max_num_hands = int(max_num_hands)
        self._model_complexity = int(model_complexity)
        self._min_detection_confidence = float(min_detection_confidence)
        self._min_tracking_confidence = float(min_tracking_confidence)
        self._hands = None
        self._timestamp_error_count = 0
        self._max_timestamp_errors = 3  # Reset MediaPipe after N consecutive errors (lowered to prevent segfaults)
        self._reset_hands()
        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles

        self._hold_last = bool(hold_last_on_no_hand)
        self._hand_mode = hand_mode.lower()  # "left", "right", "both", "auto"
        if self._hand_mode not in ["left", "right", "both", "auto"]:
            raise ValueError(f"hand_mode must be 'left', 'right', 'both', or 'auto', got '{hand_mode}'")

        self._filters: List[OneEuroFilter] = [
            OneEuroFilter(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta, d_cutoff=one_euro_d_cutoff)
            for _ in range(ShadowHandJoints.N)
        ]

        self._last_joints: List[float] = [0.0] * ShadowHandJoints.N
        self._hand_present: bool = False
        self._last_landmarks_px: Optional[np.ndarray] = None  # (21,2) in pixels of last processed frame
        self._last_hand_label: Optional[str] = None  # "Left" or "Right"
        
        # For both mode: store both hands
        self._last_left_joints: List[float] = [0.0] * ShadowHandJoints.N
        self._last_right_joints: List[float] = [0.0] * ShadowHandJoints.N
        self._left_hand_present: bool = False
        self._right_hand_present: bool = False
        self._last_left_landmarks_px: Optional[np.ndarray] = None
        self._last_right_landmarks_px: Optional[np.ndarray] = None
        
        # Store normalized landmarks (MediaPipe format: x, y, z in [0, 1])
        self._last_landmarks_normalized: Optional[np.ndarray] = None  # (21, 3) for single hand mode
        self._last_left_landmarks_normalized: Optional[np.ndarray] = None  # (21, 3) for both mode
        self._last_right_landmarks_normalized: Optional[np.ndarray] = None  # (21, 3) for both mode
        
        # Hand tracking consistency (to reduce left/right confusion)
        self._left_hand_history: List[float] = []  # Store left hand center X positions
        self._right_hand_history: List[float] = []  # Store right hand center X positions
        self._handedness_confidence_threshold: float = float(handedness_confidence_threshold)  # Minimum confidence for handedness
        self._force_position_based_handedness: bool = bool(force_position_based_handedness)  # Force position-based handedness
        
        # Visualization colors (BGR format)
        self._left_hand_color: Tuple[int, int, int] = tuple(left_hand_color)
        self._right_hand_color: Tuple[int, int, int] = tuple(right_hand_color)
        
        # Separate filters for both hands in both mode
        if self._hand_mode == "both":
            self._left_filters: List[OneEuroFilter] = [
                OneEuroFilter(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta, d_cutoff=one_euro_d_cutoff)
                for _ in range(ShadowHandJoints.N)
            ]
            self._right_filters: List[OneEuroFilter] = [
                OneEuroFilter(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta, d_cutoff=one_euro_d_cutoff)
                for _ in range(ShadowHandJoints.N)
            ]
        else:
            self._left_filters = None
            self._right_filters = None

        # Per-joint (human_min, human_max, robot_min, robot_max). Angles already in radians.
        # These limits are conservative and chosen for stability across users.
        # Load joint ranges (can be customized via calibration)
        if custom_joint_ranges is not None:
            # Use custom ranges, but fill in missing joints with defaults
            default_ranges = self._default_joint_ranges()
            self._ranges = {}
            for joint_name in ShadowHandJoints.names():
                if joint_name in custom_joint_ranges:
                    self._ranges[joint_name] = custom_joint_ranges[joint_name]
                else:
                    self._ranges[joint_name] = default_ranges[joint_name]
        else:
            self._ranges = self._default_joint_ranges()

    def _reset_hands(self) -> None:
        """Reset MediaPipe Hands instance to recover from timestamp errors."""
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass  # Ignore errors during cleanup
        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_num_hands,
                model_complexity=self._model_complexity,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            self._timestamp_error_count = 0
        except Exception as e:
            print(f"[ERROR] Failed to reset MediaPipe Hands: {e}")
            raise
    
    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass  # Ignore errors during cleanup

    def __enter__(self) -> "MediaPipeShadowHand":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def hand_present(self) -> bool:
        return bool(self._hand_present)

    @property
    def last_landmarks_px(self) -> Optional[np.ndarray]:
        """Last detected landmarks in pixel coordinates, shape (21,2), or None."""
        return None if self._last_landmarks_px is None else self._last_landmarks_px.copy()
    
    @property
    def last_hand_label(self) -> Optional[str]:
        """Last detected hand label ('Left' or 'Right'), or None."""
        return self._last_hand_label
    
    @property
    def hand_mode(self) -> str:
        """Current hand mode: 'left', 'right', 'both', or 'auto'."""
        return self._hand_mode
    
    @property
    def left_hand_present(self) -> bool:
        """Whether left hand is detected (only meaningful in both mode)."""
        return bool(self._left_hand_present)
    
    @property
    def right_hand_present(self) -> bool:
        """Whether right hand is detected (only meaningful in both mode)."""
        return bool(self._right_hand_present)
    
    def get_landmarks(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get normalized landmarks (MediaPipe format) and hand label for the selected hand.
        
        Args:
            frame_bgr: BGR image frame
            
        Returns:
            (landmarks, hand_label) tuple where:
            - landmarks: (21, 3) numpy array with normalized coordinates (x, y, z in [0, 1])
            - hand_label: "Left" or "Right" or None if no hand detected
        """
        if self._hand_mode == "both":
            # In both mode, use get_landmarks_both
            both_result = self.get_landmarks_both(frame_bgr)
            if both_result is None:
                return None, None
            # Return right hand if available, otherwise left hand
            if both_result["right"] is not None:
                return both_result["right"]["landmarks"], both_result["right"]["label"]
            elif both_result["left"] is not None:
                return both_result["left"]["landmarks"], both_result["left"]["label"]
            return None, None
        
        # Single hand mode - process frame
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None, None
        
        # Process frame to get landmarks
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        try:
            results = self._hands.process(frame_rgb)
            # Reset error count on success
            self._timestamp_error_count = 0
        except Exception as e:
            # Handle MediaPipe timestamp errors (e.g., non-monotonic timestamps from camera)
            error_str = str(e).lower()
            if "timestamp" in error_str or "packet" in error_str or "calculator" in error_str:
                self._timestamp_error_count += 1
                # Log warning for first few errors
                if self._timestamp_error_count <= 3:
                    import warnings
                    warnings.warn(f"MediaPipe timestamp error (count={self._timestamp_error_count}): {e}", RuntimeWarning)
                
                # Reset MediaPipe instance immediately on first error to prevent segfaults
                # MediaPipe can enter an inconsistent state after timestamp errors, so we reset early
                if self._timestamp_error_count == 1:
                    try:
                        print(f"[WARNING] MediaPipe timestamp error detected, resetting MediaPipe Hands to prevent segfault...")
                        self._reset_hands()
                    except Exception as reset_error:
                        print(f"[ERROR] Failed to reset MediaPipe: {reset_error}")
                        # If reset fails, try to create a completely new instance
                        try:
                            if self._hands is not None:
                                try:
                                    self._hands.close()
                                except Exception:
                                    pass
                            self._hands = self._mp_hands.Hands(
                                static_image_mode=False,
                                max_num_hands=self._max_num_hands,
                                model_complexity=self._model_complexity,
                                min_detection_confidence=self._min_detection_confidence,
                                min_tracking_confidence=self._min_tracking_confidence,
                            )
                            self._timestamp_error_count = 0
                            print(f"[INFO] Successfully recreated MediaPipe Hands instance")
                        except Exception as recreate_error:
                            print(f"[ERROR] Failed to recreate MediaPipe Hands: {recreate_error}")
                
                # Also reset if we've accumulated too many errors (backup safety)
                elif self._timestamp_error_count >= self._max_timestamp_errors:
                    print(f"[WARNING] Too many timestamp errors ({self._timestamp_error_count}), resetting MediaPipe Hands...")
                    try:
                        self._reset_hands()
                    except Exception as reset_error:
                        print(f"[ERROR] Failed to reset MediaPipe: {reset_error}")
                
                self._last_landmarks_normalized = None
                return None, None
            else:
                # For non-timestamp errors, also try to reset if it might be a MediaPipe state issue
                print(f"[WARNING] MediaPipe error (non-timestamp): {e}")
                self._last_landmarks_normalized = None
                return None, None
        
        if not results.multi_hand_landmarks:
            self._last_landmarks_normalized = None
            return None, None
        
        # Select hand based on mode
        hand_landmarks, hand_label = self._select_hand(results)
        
        if hand_landmarks is None:
            self._last_landmarks_normalized = None
            return None, None
        
        # Extract normalized landmarks
        landmarks = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            landmarks[i, 0] = float(lm.x)
            landmarks[i, 1] = float(lm.y)
            landmarks[i, 2] = float(lm.z)
        
        self._last_landmarks_normalized = landmarks
        return landmarks, hand_label
    
    def get_landmarks_both(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        Get normalized landmarks for both hands simultaneously.
        Only works when hand_mode == "both".
        
        This method calls infer_joints_both() to process the frame and extract landmarks,
        avoiding duplicate MediaPipe processing.
        
        Args:
            frame_bgr: BGR image frame
            
        Returns:
            Dict with "left" and "right" keys, each containing:
            {
                "landmarks": (21, 3) numpy array with normalized coordinates,
                "label": "Left" or "Right"
            }
            or None if no hands detected. Missing hand will be None.
        """
        if self._hand_mode != "both":
            raise ValueError("get_landmarks_both() only works when hand_mode == 'both'")
        
        # Call infer_joints_both to process frame (this will populate _last_left/right_landmarks_normalized)
        # This avoids duplicate MediaPipe processing - the frame is only processed once
        _ = self.infer_joints_both(frame_bgr)  # Process frame and extract landmarks
        
        # Return stored normalized landmarks
        result = {"left": None, "right": None}
        
        if self._last_left_landmarks_normalized is not None:
            result["left"] = {
                "landmarks": self._last_left_landmarks_normalized.copy(),
                "label": "Left"
            }
        
        if self._last_right_landmarks_normalized is not None:
            result["right"] = {
                "landmarks": self._last_right_landmarks_normalized.copy(),
                "label": "Right"
            }
        
        # Return None if no hands detected
        if result["left"] is None and result["right"] is None:
            return None
        
        return result

    @staticmethod
    def _default_joint_ranges() -> dict:
        # ShadowHand joint limits depend on your exact hand model. These work well as defaults.
        # Format: name -> (human_min, human_max, robot_min, robot_max)
        # Spread ranges are symmetric; flexion ranges start at 0.
        spread_h = (-0.7, 0.7)
        # MediaPipe calculates flexion angles as: math.pi - _angle_at(...)
        # So the actual output range is [0, math.pi] ≈ [0, 3.14159]
        # We should match this range in human range to avoid clamping
        flex_mcp_h = (0.0, math.pi)  # [0, π] to match actual MediaPipe output
        flex_pip_h = (0.0, math.pi)  # [0, π] to match actual MediaPipe output
        flex_dip_h = (0.0, math.pi)  # [0, π] to match actual MediaPipe output
        thumb_abd_h = (-1.0, 1.0)
        thumb_flex_h = (0.0, math.pi)  # [0, π] to match actual MediaPipe output
        thumb_opp_h = (-1.0, 1.0)

        # Spread joints (J4 + LFJ5): robot fixed at mid / neutral (no left-right teleop)
        spread_mid = 0.0
        lf5_mid = 0.5 * 0.7854  # mid of [0, 0.7854]
        # Thumb base: THJ5 opposition + THJ4 abduction fixed at mid
        th5_mid = 0.0  # mid of [-1.0472, 1.0472]
        th4_mid = 0.5 * 1.2217  # mid of [0, 1.2217]
        # robot_min, robot_max aligned with Isaac Sim ShadowHand limits (WRJ unchanged)
        return {
            "WRJ1": (-0.7, 0.7, -0.7, 0.7),
            "WRJ2": (-0.7, 0.7, -0.7, 0.7),
            # Index
            "FFJ4": (*spread_h, spread_mid, spread_mid),
            "FFJ3": (*flex_mcp_h, -0.2618, 1.5708),
            "FFJ2": (*flex_pip_h, 0.0, 1.5708),
            "FFJ1": (*flex_dip_h, 0.0, 1.5708),
            # Middle
            "MFJ4": (*spread_h, spread_mid, spread_mid),
            "MFJ3": (*flex_mcp_h, -0.2618, 1.5708),
            "MFJ2": (*flex_pip_h, 0.0, 1.5708),
            "MFJ1": (*flex_dip_h, 0.0, 1.5708),
            # Ring
            "RFJ4": (*spread_h, spread_mid, spread_mid),
            "RFJ3": (*flex_mcp_h, -0.2618, 1.5708),
            "RFJ2": (*flex_pip_h, 0.0, 1.5708),
            "RFJ1": (*flex_dip_h, 0.0, 1.5708),
            # Little
            "LFJ5": (*spread_h, lf5_mid, lf5_mid),
            "LFJ4": (*spread_h, spread_mid, spread_mid),
            "LFJ3": (*flex_mcp_h, -0.2618, 1.5708),
            "LFJ2": (*flex_pip_h, 0.0, 1.5708),
            "LFJ1": (*flex_dip_h, 0.0, 1.5708),
            # Thumb
            "THJ5": (*thumb_opp_h, th5_mid, th5_mid),
            "THJ4": (*thumb_abd_h, th4_mid, th4_mid),
            "THJ3": (*thumb_flex_h, -0.2094, 0.2094),
            "THJ2": (*thumb_flex_h, -0.6981, 0.6981),
            "THJ1": (*thumb_flex_h, -0.2618, 1.5708),
        }

    def annotate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Return a copy of the frame with MediaPipe landmarks and skeleton connections drawn.
        In both mode, draws both hands with different colors.
        """
        out = frame_bgr.copy()
        
        # MediaPipe Hands skeleton connections (21 landmarks):
        connections = [
            # Thumb chain
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index chain
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle chain
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring chain
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky chain
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (1, 5), (5, 9), (9, 13), (13, 17),
        ]
        
        # Draw both hands if in both mode
        if self._hand_mode == "both":
            # Draw left hand
            if self._last_left_landmarks_px is not None:
                left_color = self._left_hand_color
                left_color_dark = tuple(int(c * 0.8) for c in left_color)
                for (i, j) in connections:
                    pt1 = tuple(self._last_left_landmarks_px[i].astype(int))
                    pt2 = tuple(self._last_left_landmarks_px[j].astype(int))
                    cv2.line(out, pt1, pt2, left_color, 2, lineType=cv2.LINE_AA)
                for (x, y) in self._last_left_landmarks_px.astype(int):
                    cv2.circle(out, (int(x), int(y)), 4, left_color, -1, lineType=cv2.LINE_AA)
                    cv2.circle(out, (int(x), int(y)), 6, left_color_dark, 1, lineType=cv2.LINE_AA)
            
            # Draw right hand
            if self._last_right_landmarks_px is not None:
                right_color = self._right_hand_color
                right_color_dark = tuple(int(c * 0.8) for c in right_color)
                for (i, j) in connections:
                    pt1 = tuple(self._last_right_landmarks_px[i].astype(int))
                    pt2 = tuple(self._last_right_landmarks_px[j].astype(int))
                    cv2.line(out, pt1, pt2, right_color, 2, lineType=cv2.LINE_AA)
                for (x, y) in self._last_right_landmarks_px.astype(int):
                    cv2.circle(out, (int(x), int(y)), 4, right_color, -1, lineType=cv2.LINE_AA)
                    cv2.circle(out, (int(x), int(y)), 6, right_color_dark, 1, lineType=cv2.LINE_AA)
        else:
            # Single hand mode (use right hand color as default)
            if self._last_landmarks_px is None:
                return out
            
            hand_color = self._right_hand_color
            hand_color_dark = tuple(int(c * 0.8) for c in hand_color)
            
            for (i, j) in connections:
                pt1 = tuple(self._last_landmarks_px[i].astype(int))
                pt2 = tuple(self._last_landmarks_px[j].astype(int))
                cv2.line(out, pt1, pt2, hand_color, 2, lineType=cv2.LINE_AA)

            for (x, y) in self._last_landmarks_px.astype(int):
                cv2.circle(out, (int(x), int(y)), 4, hand_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (int(x), int(y)), 6, hand_color_dark, 1, lineType=cv2.LINE_AA)

        return out

    def render_hand_pose(self, width: int = 640, height: int = 480, scale: float = 1.0) -> np.ndarray:
        """
        Render hand pose on a blank canvas (for separate visualization window).
        Returns a BGR image with the hand skeleton drawn.
        In both mode, renders both hands side by side.

        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            scale: Scale factor for hand size (1.0 = fit to canvas)
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Same connections as in annotate()
        connections = [
            # Thumb chain
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index chain
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle chain
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring chain
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky chain
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (1, 5), (5, 9), (9, 13), (13, 17),
        ]
        
        def render_single_hand(landmarks_px, center_x, center_y, color, label=""):
            """Helper to render a single hand at a specific position."""
            if landmarks_px is None:
                return
            
            landmarks = landmarks_px.copy()
            
            # Compute bounding box
            min_x, min_y = landmarks.min(axis=0)
            max_x, max_y = landmarks.max(axis=0)
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            
            # Scale and translate
            if bbox_w > 0 and bbox_h > 0:
                scale_factor = min(width / 2 / bbox_w, height / bbox_h) * 0.8 * scale
                landmarks_scaled = (landmarks - np.array([center_x, center_y])) * scale_factor + np.array([center_x, center_y])
            else:
                landmarks_scaled = landmarks
            
            # Draw connections
            for (i, j) in connections:
                pt1 = tuple(landmarks_scaled[i].astype(int))
                pt2 = tuple(landmarks_scaled[j].astype(int))
                cv2.line(canvas, pt1, pt2, color, 3, lineType=cv2.LINE_AA)
            
            # Draw landmarks
            for (x, y) in landmarks_scaled.astype(int):
                cv2.circle(canvas, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (int(x), int(y)), 7, tuple(int(c * 0.8) for c in color), 2, lineType=cv2.LINE_AA)
            
            # Draw label if provided
            if label:
                cv2.putText(canvas, label, (int(center_x) - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)
        
        if self._hand_mode == "both":
            # Render both hands side by side
            # Left hand on left side
            if self._last_left_landmarks_px is not None:
                left_center_x = (self._last_left_landmarks_px[:, 0].min() + self._last_left_landmarks_px[:, 0].max()) / 2
                left_center_y = (self._last_left_landmarks_px[:, 1].min() + self._last_left_landmarks_px[:, 1].max()) / 2
                left_canvas_x = width / 4
                left_canvas_y = height / 2
                # Offset landmarks to left side
                landmarks_left = self._last_left_landmarks_px.copy()
                landmarks_left[:, 0] += left_canvas_x - left_center_x
                landmarks_left[:, 1] += left_canvas_y - left_center_y
                render_single_hand(landmarks_left, left_canvas_x, left_canvas_y, self._left_hand_color, "LEFT")
            
            # Right hand on right side
            if self._last_right_landmarks_px is not None:
                right_center_x = (self._last_right_landmarks_px[:, 0].min() + self._last_right_landmarks_px[:, 0].max()) / 2
                right_center_y = (self._last_right_landmarks_px[:, 1].min() + self._last_right_landmarks_px[:, 1].max()) / 2
                right_canvas_x = 3 * width / 4
                right_canvas_y = height / 2
                # Offset landmarks to right side
                landmarks_right = self._last_right_landmarks_px.copy()
                landmarks_right[:, 0] += right_canvas_x - right_center_x
                landmarks_right[:, 1] += right_canvas_y - right_center_y
                render_single_hand(landmarks_right, right_canvas_x, right_canvas_y, self._right_hand_color, "RIGHT")
            
            # Add title
            cv2.putText(canvas, "Virtual Hand Pose (Both)", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        else:
            # Single hand mode
            if self._last_landmarks_px is None:
                return canvas
            
            landmarks = self._last_landmarks_px.copy()

            # Compute bounding box and center
            min_x, min_y = landmarks.min(axis=0)
            max_x, max_y = landmarks.max(axis=0)
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            # Scale and translate to center of canvas
            if bbox_w > 0 and bbox_h > 0:
                scale_factor = min(width / bbox_w, height / bbox_h) * 0.8 * scale
                landmarks_scaled = (landmarks - np.array([center_x, center_y])) * scale_factor + np.array([width / 2, height / 2])
            else:
                landmarks_scaled = landmarks

            # Draw connections (skeleton) - thicker and brighter for visibility
            hand_color = self._right_hand_color  # Default to right hand color for single hand mode
            hand_color_dark = tuple(int(c * 0.8) for c in hand_color)
            for (i, j) in connections:
                pt1 = tuple(landmarks_scaled[i].astype(int))
                pt2 = tuple(landmarks_scaled[j].astype(int))
                cv2.line(canvas, pt1, pt2, hand_color, 3, lineType=cv2.LINE_AA)

            # Draw landmarks (joints) - larger for visibility
            for (x, y) in landmarks_scaled.astype(int):
                cv2.circle(canvas, (int(x), int(y)), 5, hand_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (int(x), int(y)), 7, hand_color_dark, 2, lineType=cv2.LINE_AA)

            # Add title
            cv2.putText(
                canvas,
                "Virtual Hand Pose",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        return canvas

    def __call__(self, frame_bgr: np.ndarray) -> List[float]:
        return self.infer_joints(frame_bgr)

    def infer_joints(self, frame_bgr: np.ndarray) -> List[float]:
        """
        Estimate ShadowHand joint targets for the selected hand(s) in the frame.
        
        In 'both' mode, returns right hand joints (or left if only left is present).
        Use infer_joints_both() to get both hands simultaneously.
        
        Returns a 24-element list of radians.
        """
        if self._hand_mode == "both":
            # In both mode, process both hands but return right hand (or left if only left)
            both_result = self.infer_joints_both(frame_bgr)
            if both_result is None:
                return [0.0] * ShadowHandJoints.N
            # Return right hand if available, otherwise left hand
            if both_result["right"] is not None:
                return both_result["right"]
            elif both_result["left"] is not None:
                return both_result["left"]
            return [0.0] * ShadowHandJoints.N
        
        # Single hand mode (left, right, or auto)
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image frame with shape (H, W, 3)")

        h, w = frame_bgr.shape[:2]
        t = time.perf_counter()

        # MediaPipe expects RGB input.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        try:
            results = self._hands.process(frame_rgb)
            # Reset error count on success
            self._timestamp_error_count = 0
        except Exception as e:
            # Handle MediaPipe timestamp errors (e.g., non-monotonic timestamps from camera)
            # This can happen when camera timestamps are not strictly increasing
            error_str = str(e).lower()
            if "timestamp" in error_str or "packet" in error_str or "calculator" in error_str:
                self._timestamp_error_count += 1
                # Log warning for first few errors
                if self._timestamp_error_count <= 3:
                    import warnings
                    warnings.warn(f"MediaPipe timestamp error (count={self._timestamp_error_count}): {e}", RuntimeWarning)
                
                # Reset MediaPipe instance immediately on first error to prevent segfaults
                if self._timestamp_error_count == 1:
                    try:
                        print(f"[WARNING] MediaPipe timestamp error detected, resetting MediaPipe Hands to prevent segfault...")
                        self._reset_hands()
                    except Exception as reset_error:
                        print(f"[ERROR] Failed to reset MediaPipe: {reset_error}")
                        # If reset fails, try to create a completely new instance
                        try:
                            if self._hands is not None:
                                try:
                                    self._hands.close()
                                except Exception:
                                    pass
                            self._hands = self._mp_hands.Hands(
                                static_image_mode=False,
                                max_num_hands=self._max_num_hands,
                                model_complexity=self._model_complexity,
                                min_detection_confidence=self._min_detection_confidence,
                                min_tracking_confidence=self._min_tracking_confidence,
                            )
                            self._timestamp_error_count = 0
                            print(f"[INFO] Successfully recreated MediaPipe Hands instance")
                        except Exception as recreate_error:
                            print(f"[ERROR] Failed to recreate MediaPipe Hands: {recreate_error}")
                
                # Also reset if we've accumulated too many errors (backup safety)
                elif self._timestamp_error_count >= self._max_timestamp_errors:
                    print(f"[WARNING] Too many timestamp errors ({self._timestamp_error_count}), resetting MediaPipe Hands...")
                    try:
                        self._reset_hands()
                    except Exception as reset_error:
                        print(f"[ERROR] Failed to reset MediaPipe: {reset_error}")
                
                # Return last known state or empty results
                if self._hold_last and self._hand_present:
                    return list(self._last_joints)
                return [0.0] * ShadowHandJoints.N
            else:
                # For non-timestamp errors, also return gracefully to prevent segfaults
                print(f"[WARNING] MediaPipe error (non-timestamp) in infer_joints: {e}")
                if self._hold_last and self._hand_present:
                    return list(self._last_joints)
                return [0.0] * ShadowHandJoints.N

        if not results.multi_hand_landmarks:
            self._hand_present = False
            self._last_landmarks_px = None
            if self._hold_last:
                return list(self._last_joints)
            return [0.0] * ShadowHandJoints.N

        self._hand_present = True

        # Select hand based on mode
        hand_landmarks, hand_label = self._select_hand(results)
        
        if hand_landmarks is None:
            self._hand_present = False
            self._last_landmarks_px = None
            if self._hold_last:
                return list(self._last_joints)
            return [0.0] * ShadowHandJoints.N

        pts = np.zeros((21, 3), dtype=np.float32)
        pts_px = np.zeros((21, 2), dtype=np.float32)
        landmarks_norm = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            # Convert to a consistent "pixel-ish" 3D space for geometry:
            # x, y in pixels; z scaled by image width (as MediaPipe suggests).
            pts[i, 0] = float(lm.x * w)
            pts[i, 1] = float(lm.y * h)
            pts[i, 2] = float(lm.z * w)
            pts_px[i, 0] = float(lm.x * w)
            pts_px[i, 1] = float(lm.y * h)
            # Store normalized landmarks
            landmarks_norm[i, 0] = float(lm.x)
            landmarks_norm[i, 1] = float(lm.y)
            landmarks_norm[i, 2] = float(lm.z)
        self._last_landmarks_px = pts_px
        self._last_landmarks_normalized = landmarks_norm
        self._last_hand_label = hand_label

        joints_raw = self._landmarks_to_shadowhand(pts, hand_label=hand_label)

        # Smooth each joint with One Euro filtering.
        joints_smooth = []
        for i, x in enumerate(joints_raw):
            joints_smooth.append(self._filters[i](float(x), t))

        self._last_joints = list(joints_smooth)
        return list(joints_smooth)
    
    def infer_joints_both(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        Estimate ShadowHand joint targets for both hands simultaneously.
        Only works when hand_mode == "both".
        
        Returns:
            Dict with "left" and "right" keys, each containing a 24-element list of radians,
            or None if no hands detected. Missing hand will be None.
            Example: {"left": [0.1, 0.2, ...], "right": [0.3, 0.4, ...]}
        """
        if self._hand_mode != "both":
            raise ValueError("infer_joints_both() only works when hand_mode == 'both'")
        
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image frame with shape (H, W, 3)")

        h, w = frame_bgr.shape[:2]
        t = time.perf_counter()

        # MediaPipe expects RGB input.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        try:
            results = self._hands.process(frame_rgb)
            # Reset error count on success
            self._timestamp_error_count = 0
        except Exception as e:
            # Handle MediaPipe timestamp errors (e.g., non-monotonic timestamps from camera)
            error_str = str(e).lower()
            if "timestamp" in error_str or "packet" in error_str or "calculator" in error_str:
                self._timestamp_error_count += 1
                # Log warning for first few errors
                if self._timestamp_error_count <= 3:
                    import warnings
                    warnings.warn(f"MediaPipe timestamp error (count={self._timestamp_error_count}): {e}", RuntimeWarning)
                
                # Reset MediaPipe instance immediately on first error to prevent segfaults
                if self._timestamp_error_count == 1:
                    try:
                        print(f"[WARNING] MediaPipe timestamp error detected, resetting MediaPipe Hands to prevent segfault...")
                        self._reset_hands()
                    except Exception as reset_error:
                        print(f"[ERROR] Failed to reset MediaPipe: {reset_error}")
                        # If reset fails, try to create a completely new instance
                        try:
                            if self._hands is not None:
                                try:
                                    self._hands.close()
                                except Exception:
                                    pass
                            self._hands = self._mp_hands.Hands(
                                static_image_mode=False,
                                max_num_hands=self._max_num_hands,
                                model_complexity=self._model_complexity,
                                min_detection_confidence=self._min_detection_confidence,
                                min_tracking_confidence=self._min_tracking_confidence,
                            )
                            self._timestamp_error_count = 0
                            print(f"[INFO] Successfully recreated MediaPipe Hands instance")
                        except Exception as recreate_error:
                            print(f"[ERROR] Failed to recreate MediaPipe Hands: {recreate_error}")
                
                # Also reset if we've accumulated too many errors (backup safety)
                elif self._timestamp_error_count >= self._max_timestamp_errors:
                    print(f"[WARNING] Too many timestamp errors ({self._timestamp_error_count}), resetting MediaPipe Hands...")
                    try:
                        self._reset_hands()
                    except Exception as reset_error:
                        print(f"[ERROR] Failed to reset MediaPipe: {reset_error}")
                
                if self._hold_last:
                    return {"left": list(self._last_left_joints) if self._left_hand_present else None,
                            "right": list(self._last_right_joints) if self._right_hand_present else None}
                return {"left": None, "right": None}
            else:
                # For non-timestamp errors, also return gracefully to prevent segfaults
                print(f"[WARNING] MediaPipe error (non-timestamp) in get_pose_both: {e}")
                if self._hold_last:
                    return {"left": list(self._last_left_joints) if self._left_hand_present else None,
                            "right": list(self._last_right_joints) if self._right_hand_present else None}
                return {"left": None, "right": None}

        if not results.multi_hand_landmarks:
            self._hand_present = False
            self._left_hand_present = False
            self._right_hand_present = False
            self._last_landmarks_px = None
            self._last_left_landmarks_px = None
            self._last_right_landmarks_px = None
            if self._hold_last:
                return {"left": list(self._last_left_joints) if self._left_hand_present else None,
                        "right": list(self._last_right_joints) if self._right_hand_present else None}
            return {"left": None, "right": None}

        # Get hand labels with confidence
        hand_labels = []
        hand_confidences = []
        if results.multi_handedness:
            for hand in results.multi_handedness:
                classification = hand.classification[0]
                hand_labels.append(classification.label)  # "Left" or "Right"
                hand_confidences.append(classification.score)  # Confidence 0-1
        else:
            hand_labels = [None] * len(results.multi_hand_landmarks)
            hand_confidences = [0.0] * len(results.multi_hand_landmarks)
        
        # Calculate hand center positions for position-based disambiguation
        hand_centers = []
        for hand_landmarks in results.multi_hand_landmarks:
            center_x = sum(lm.x for lm in hand_landmarks.landmark) / 21.0
            hand_centers.append(center_x)
        
        # Disambiguate hands using position + handedness + history
        left_landmarks = None
        right_landmarks = None
        left_label = None
        right_label = None
        left_idx = None
        right_idx = None
        
        # First pass: use handedness with confidence check + geometry validation
        # For force_position_based_handedness: determine labels based on relative wrist positions
        if self._force_position_based_handedness and len(results.multi_hand_landmarks) == 2:
            # Compare relative wrist positions: leftmost = Left, rightmost = Right
            wrist_positions = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                wrist_x = hand_landmarks.landmark[0].x
                wrist_positions.append((i, wrist_x))
            # Sort by X position (left to right)
            wrist_positions.sort(key=lambda x: x[1])
            # Assign labels based on relative position
            for idx, (original_idx, _) in enumerate(wrist_positions):
                if idx == 0:  # Leftmost hand
                    hand_labels[original_idx] = "Left"
                else:  # Rightmost hand
                    hand_labels[original_idx] = "Right"
                hand_confidences[original_idx] = 1.0  # High confidence
        
        for i, (hand_landmarks, label, confidence, center_x) in enumerate(
            zip(results.multi_hand_landmarks, hand_labels, hand_confidences, hand_centers)
        ):
            if not self._force_position_based_handedness:
                # Try geometry-based detection first (more reliable)
                geometry_label = _detect_handedness_geometry(hand_landmarks)
                
                # Use MediaPipe label if confidence is high, otherwise use geometry
                if label and confidence >= self._handedness_confidence_threshold:
                    # Validate with geometry if available
                    if geometry_label and geometry_label.lower() != label.lower():
                        # Geometry disagrees with MediaPipe - prefer geometry for high confidence cases
                        if confidence < 0.8:  # Only override if MediaPipe confidence is not very high
                            label = geometry_label
                elif geometry_label:
                    # MediaPipe confidence is low, use geometry-based detection
                    label = geometry_label
                    confidence = 0.7  # Assign a medium confidence for geometry-based detection
            
            # Assign hand based on (possibly corrected) label
            if label:
                if label.lower() == "left":
                    if left_landmarks is None:
                        left_landmarks = hand_landmarks
                        left_label = label
                        left_idx = i
                elif label.lower() == "right":
                    if right_landmarks is None:
                        right_landmarks = hand_landmarks
                        right_label = label
                        right_idx = i
        
        # Second pass: if both hands detected but handedness uncertain, use geometry + position + history
        # Also handle case where only one hand is detected but we need to assign it correctly
        if len(results.multi_hand_landmarks) == 2:
            # Force position-based mode: assign based on wrist X position
            if self._force_position_based_handedness:
                # Sort hands by wrist X position
                hand_wrist_positions = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    wrist_x = hand_landmarks.landmark[0].x
                    hand_wrist_positions.append((i, wrist_x, hand_landmarks))
                # Sort by X position (left to right)
                hand_wrist_positions.sort(key=lambda x: x[1])
                
                # Leftmost hand = Left, Rightmost hand = Right
                if left_landmarks is None and len(hand_wrist_positions) > 0:
                    left_landmarks = hand_wrist_positions[0][2]
                    left_label = "Left"
                    left_idx = hand_wrist_positions[0][0]
                if right_landmarks is None and len(hand_wrist_positions) > 1:
                    right_landmarks = hand_wrist_positions[1][2]
                    right_label = "Right"
                    right_idx = hand_wrist_positions[1][0]
            # Try geometry-based detection for unassigned hands
            elif left_landmarks is None or right_landmarks is None:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if (i == 0 and left_landmarks is None and right_landmarks is None) or \
                       (i == 1 and (left_landmarks is None or right_landmarks is None)):
                        geometry_label = _detect_handedness_geometry(hand_landmarks)
                        if geometry_label:
                            if geometry_label.lower() == "left" and left_landmarks is None:
                                left_landmarks = hand_landmarks
                                left_label = geometry_label
                                left_idx = i
                            elif geometry_label.lower() == "right" and right_landmarks is None:
                                right_landmarks = hand_landmarks
                                right_label = geometry_label
                                right_idx = i
            
            # If we still have history, use it to disambiguate remaining unassigned hands
            if len(self._left_hand_history) > 0 and len(self._right_hand_history) > 0:
                avg_left_x = sum(self._left_hand_history[-10:]) / len(self._left_hand_history[-10:])
                avg_right_x = sum(self._right_hand_history[-10:]) / len(self._right_hand_history[-10:])
                
                # Determine which hand is which based on position relative to history
                if left_landmarks is None or right_landmarks is None:
                    # Re-assign based on position
                    if hand_centers[0] < hand_centers[1]:
                        # Hand 0 is more left
                        if avg_left_x < avg_right_x:  # History confirms left is on left
                            if left_landmarks is None:
                                left_landmarks = results.multi_hand_landmarks[0]
                                left_label = hand_labels[0] or "Left"
                                left_idx = 0
                            if right_landmarks is None:
                                right_landmarks = results.multi_hand_landmarks[1]
                                right_label = hand_labels[1] or "Right"
                                right_idx = 1
                        else:  # History suggests swap
                            if left_landmarks is None:
                                left_landmarks = results.multi_hand_landmarks[1]
                                left_label = hand_labels[1] or "Left"
                                left_idx = 1
                            if right_landmarks is None:
                                right_landmarks = results.multi_hand_landmarks[0]
                                right_label = hand_labels[0] or "Right"
                                right_idx = 0
                    else:
                        # Hand 1 is more left
                        if avg_left_x < avg_right_x:
                            if left_landmarks is None:
                                left_landmarks = results.multi_hand_landmarks[1]
                                left_label = hand_labels[1] or "Left"
                                left_idx = 1
                            if right_landmarks is None:
                                right_landmarks = results.multi_hand_landmarks[0]
                                right_label = hand_labels[0] or "Right"
                                right_idx = 0
                        else:
                            if left_landmarks is None:
                                left_landmarks = results.multi_hand_landmarks[0]
                                left_label = hand_labels[0] or "Left"
                                left_idx = 0
                            if right_landmarks is None:
                                right_landmarks = results.multi_hand_landmarks[1]
                                right_label = hand_labels[1] or "Right"
                                right_idx = 1
        elif len(results.multi_hand_landmarks) == 1:
            # Only one hand detected - try geometry first, then use position + history
            if left_landmarks is None and right_landmarks is None:
                # Force position-based mode: for single hand, use history if available
                if self._force_position_based_handedness:
                    # If we have history, compare with average positions
                    if len(self._left_hand_history) > 0 and len(self._right_hand_history) > 0:
                        avg_left_x = sum(self._left_hand_history[-10:]) / len(self._left_hand_history[-10:])
                        avg_right_x = sum(self._right_hand_history[-10:]) / len(self._right_hand_history[-10:])
                        current_x = hand_centers[0]
                        # Assign to the hand that's closer in position
                        if abs(current_x - avg_left_x) < abs(current_x - avg_right_x):
                            left_landmarks = results.multi_hand_landmarks[0]
                            left_label = "Left"
                            left_idx = 0
                        else:
                            right_landmarks = results.multi_hand_landmarks[0]
                            right_label = "Right"
                            right_idx = 0
                    else:
                        # No history available, cannot determine with single hand
                        # Skip assignment (will be handled by fallback logic)
                        pass
                else:
                    # Neither hand assigned yet, try geometry-based detection
                    geometry_label = _detect_handedness_geometry(results.multi_hand_landmarks[0])
                    if geometry_label:
                        if geometry_label.lower() == "left":
                            left_landmarks = results.multi_hand_landmarks[0]
                            left_label = geometry_label
                            left_idx = 0
                        elif geometry_label.lower() == "right":
                            right_landmarks = results.multi_hand_landmarks[0]
                            right_label = geometry_label
                            right_idx = 0
            
            # If still not assigned, use position + history to determine which hand it is
            if left_landmarks is None and right_landmarks is None:
                if len(self._left_hand_history) > 0 and len(self._right_hand_history) > 0:
                    avg_left_x = sum(self._left_hand_history[-10:]) / len(self._left_hand_history[-10:])
                    avg_right_x = sum(self._right_hand_history[-10:]) / len(self._right_hand_history[-10:])
                    current_x = hand_centers[0]
                    
                    # Assign to the hand that's closer in position
                    if abs(current_x - avg_left_x) < abs(current_x - avg_right_x):
                        if left_landmarks is None:
                            left_landmarks = results.multi_hand_landmarks[0]
                            left_label = hand_labels[0] or "Left"
                            left_idx = 0
                    else:
                        if right_landmarks is None:
                            right_landmarks = results.multi_hand_landmarks[0]
                            right_label = hand_labels[0] or "Right"
                            right_idx = 0
            else:
                # No history, use handedness or default to right
                if hand_labels[0] and hand_labels[0].lower() == "left":
                    if left_landmarks is None:
                        left_landmarks = results.multi_hand_landmarks[0]
                        left_label = hand_labels[0]
                        left_idx = 0
                elif hand_labels[0] and hand_labels[0].lower() == "right":
                    if right_landmarks is None:
                        right_landmarks = results.multi_hand_landmarks[0]
                        right_label = hand_labels[0]
                        right_idx = 0
                else:
                    # No handedness info, default to right
                    if right_landmarks is None:
                        right_landmarks = results.multi_hand_landmarks[0]
                        right_label = "Right"
                        right_idx = 0
        else:
            # No history yet, use simple position-based assignment (for 2 hands)
            if left_landmarks is None or right_landmarks is None:
                if hand_centers[0] < hand_centers[1]:
                    # Hand 0 is on left side
                    if left_landmarks is None:
                        left_landmarks = results.multi_hand_landmarks[0]
                        left_label = hand_labels[0] or "Left"
                        left_idx = 0
                    if right_landmarks is None:
                        right_landmarks = results.multi_hand_landmarks[1]
                        right_label = hand_labels[1] or "Right"
                        right_idx = 1
                else:
                    # Hand 1 is on left side
                    if left_landmarks is None:
                        left_landmarks = results.multi_hand_landmarks[1]
                        left_label = hand_labels[1] or "Left"
                        left_idx = 1
                    if right_landmarks is None:
                        right_landmarks = results.multi_hand_landmarks[0]
                        right_label = hand_labels[0] or "Right"
                        right_idx = 0
        
        # Update history with current positions
        if left_landmarks is not None:
            left_center_x = sum(lm.x for lm in left_landmarks.landmark) / 21.0
            self._left_hand_history.append(left_center_x)
            if len(self._left_hand_history) > 30:  # Keep last 30 frames
                self._left_hand_history.pop(0)
        
        if right_landmarks is not None:
            right_center_x = sum(lm.x for lm in right_landmarks.landmark) / 21.0
            self._right_hand_history.append(right_center_x)
            if len(self._right_hand_history) > 30:  # Keep last 30 frames
                self._right_hand_history.pop(0)
        
        result = {"left": None, "right": None}
        
        # Process left hand
        if left_landmarks is not None:
            self._left_hand_present = True
            pts = np.zeros((21, 3), dtype=np.float32)
            pts_px = np.zeros((21, 2), dtype=np.float32)
            landmarks_norm = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(left_landmarks.landmark):
                pts[i, 0] = float(lm.x * w)
                pts[i, 1] = float(lm.y * h)
                pts[i, 2] = float(lm.z * w)
                pts_px[i, 0] = float(lm.x * w)
                pts_px[i, 1] = float(lm.y * h)
                landmarks_norm[i, 0] = float(lm.x)
                landmarks_norm[i, 1] = float(lm.y)
                landmarks_norm[i, 2] = float(lm.z)
            self._last_left_landmarks_px = pts_px
            self._last_left_landmarks_normalized = landmarks_norm
            
            joints_raw = self._landmarks_to_shadowhand(pts, hand_label=left_label)
            joints_smooth = []
            for i, x in enumerate(joints_raw):
                joints_smooth.append(self._left_filters[i](float(x), t))
            self._last_left_joints = list(joints_smooth)
            result["left"] = list(joints_smooth)
        else:
            self._left_hand_present = False
            if self._hold_last:
                result["left"] = list(self._last_left_joints)
        
        # Process right hand
        if right_landmarks is not None:
            self._right_hand_present = True
            pts = np.zeros((21, 3), dtype=np.float32)
            pts_px = np.zeros((21, 2), dtype=np.float32)
            landmarks_norm = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(right_landmarks.landmark):
                pts[i, 0] = float(lm.x * w)
                pts[i, 1] = float(lm.y * h)
                pts[i, 2] = float(lm.z * w)
                pts_px[i, 0] = float(lm.x * w)
                pts_px[i, 1] = float(lm.y * h)
                landmarks_norm[i, 0] = float(lm.x)
                landmarks_norm[i, 1] = float(lm.y)
                landmarks_norm[i, 2] = float(lm.z)
            self._last_right_landmarks_px = pts_px
            self._last_right_landmarks_normalized = landmarks_norm
            
            joints_raw = self._landmarks_to_shadowhand(pts, hand_label=right_label)
            joints_smooth = []
            for i, x in enumerate(joints_raw):
                joints_smooth.append(self._right_filters[i](float(x), t))
            self._last_right_joints = list(joints_smooth)
            result["right"] = list(joints_smooth)
        else:
            self._right_hand_present = False
            if self._hold_last:
                result["right"] = list(self._last_right_joints)
        
        # Update main hand_present flag
        self._hand_present = self._left_hand_present or self._right_hand_present
        
        # Update main landmarks for backward compatibility (use right if available, else left)
        if self._last_right_landmarks_px is not None:
            self._last_landmarks_px = self._last_right_landmarks_px
            self._last_hand_label = "Right"
            self._last_landmarks_normalized = self._last_right_landmarks_normalized
        elif self._last_left_landmarks_px is not None:
            self._last_landmarks_px = self._last_left_landmarks_px
            self._last_hand_label = "Left"
            self._last_landmarks_normalized = self._last_left_landmarks_normalized
        else:
            self._last_landmarks_px = None
            self._last_hand_label = None
            self._last_landmarks_normalized = None
        
        return result

    def _select_hand(self, results) -> Tuple[Optional[any], Optional[str]]:
        """
        Select hand based on hand_mode with improved handedness detection.
        
        Returns:
            (hand_landmarks, hand_label) tuple
        """
        num_hands = len(results.multi_hand_landmarks)
        
        if num_hands == 0:
            return None, None
        
        # Get hand labels with confidence
        hand_labels = []
        hand_confidences = []
        if results.multi_handedness:
            for hand in results.multi_handedness:
                classification = hand.classification[0]
                hand_labels.append(classification.label)  # "Left" or "Right"
                hand_confidences.append(classification.score)  # Confidence 0-1
        else:
            hand_labels = [None] * num_hands
            hand_confidences = [0.0] * num_hands
        
        # Force position-based handedness: override labels based on relative wrist positions
        if self._force_position_based_handedness:
            if len(results.multi_hand_landmarks) == 2:
                # Compare relative wrist positions: leftmost = Left, rightmost = Right
                wrist_positions = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_positions.append((i, wrist_x))
                # Sort by X position (left to right)
                wrist_positions.sort(key=lambda x: x[1])
                # Assign labels based on relative position
                for idx, (original_idx, _) in enumerate(wrist_positions):
                    if idx == 0:  # Leftmost hand
                        hand_labels[original_idx] = "Left"
                    else:  # Rightmost hand
                        hand_labels[original_idx] = "Right"
                    hand_confidences[original_idx] = 1.0  # High confidence
            # For single hand, keep original label (cannot determine without relative position)
        
        # Select based on mode
        if self._hand_mode == "auto":
            # Use first hand (default behavior)
            return results.multi_hand_landmarks[0], hand_labels[0]
        
        elif self._hand_mode == "left":
            # If multiple hands, try to find left hand by label
            if num_hands > 1:
                for i, label in enumerate(hand_labels):
                    if label and label.lower() == "left":
                        return results.multi_hand_landmarks[i], hand_labels[i]
            
            # Single hand: directly assume it's the left hand (user's intent)
            # No verification - user sets left mode and uses left hand
            return results.multi_hand_landmarks[0], "Left"
        
        elif self._hand_mode == "right":
            # If multiple hands, try to find right hand by label
            if num_hands > 1:
                for i, label in enumerate(hand_labels):
                    if label and label.lower() == "right":
                        return results.multi_hand_landmarks[i], hand_labels[i]
            
            # Single hand: directly assume it's the right hand (user's intent)
            # No verification - user sets right mode and uses right hand
            return results.multi_hand_landmarks[0], "Right"
        
        elif self._hand_mode == "both":
            # In "both" mode, prefer right hand (or first hand if no right)
            for i, label in enumerate(hand_labels):
                if label and label.lower() == "right":
                    return results.multi_hand_landmarks[i], hand_labels[i]
            # Fallback to first hand
            return results.multi_hand_landmarks[0], hand_labels[0]
        
        return None, None

    def _landmarks_to_shadowhand(self, pts: np.ndarray, *, hand_label: Optional[str]) -> List[float]:
        # MediaPipe landmark indices (Hands):
        # 0 wrist
        # 1-4 thumb (CMC, MCP, IP, TIP)
        # 5-8 index (MCP, PIP, DIP, TIP)
        # 9-12 middle
        # 13-16 ring
        # 17-20 pinky
        wrist = pts[0]
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = pts[1], pts[2], pts[3], pts[4]
        index_mcp, index_pip, index_dip, index_tip = pts[5], pts[6], pts[7], pts[8]
        middle_mcp, middle_pip, middle_dip, middle_tip = pts[9], pts[10], pts[11], pts[12]
        ring_mcp, ring_pip, ring_dip, ring_tip = pts[13], pts[14], pts[15], pts[16]
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip = pts[17], pts[18], pts[19], pts[20]

        # Palm frame for spread estimation.
        palm_x = _normalize(index_mcp - pinky_mcp)
        palm_y = _normalize(middle_mcp - wrist)
        palm_n = _normalize(np.cross(palm_x, palm_y))

        # If the palm normal degenerates (rare), fall back to a stable axis.
        if float(np.linalg.norm(palm_n)) < 1e-6:
            palm_n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        def base_angle(mcp: np.ndarray) -> float:
            v = mcp - wrist
            v = _project_to_plane(v, palm_n)
            return _signed_angle(palm_y, v, palm_n)

        a_index = base_angle(index_mcp)
        a_middle = base_angle(middle_mcp)
        a_ring = base_angle(ring_mcp)
        a_pinky = base_angle(pinky_mcp)

        # Spread relative to the middle finger as a reference.
        ff_spread = a_index - a_middle
        mf_spread = 0.0  # reference
        rf_spread = a_ring - a_middle
        lf_spread = a_pinky - a_middle

        # Extra little-finger metacarpal spread (ShadowHand LFJ5).
        lfj5 = 0.5 * (a_pinky - a_ring)

        # Finger flexion (MCP, PIP, DIP).
        ff_mcp = math.pi - _angle_at(wrist, index_mcp, index_pip)
        ff_pip = math.pi - _angle_at(index_mcp, index_pip, index_dip)
        ff_dip = math.pi - _angle_at(index_pip, index_dip, index_tip)

        mf_mcp = math.pi - _angle_at(wrist, middle_mcp, middle_pip)
        mf_pip = math.pi - _angle_at(middle_mcp, middle_pip, middle_dip)
        mf_dip = math.pi - _angle_at(middle_pip, middle_dip, middle_tip)

        rf_mcp = math.pi - _angle_at(wrist, ring_mcp, ring_pip)
        rf_pip = math.pi - _angle_at(ring_mcp, ring_pip, ring_dip)
        rf_dip = math.pi - _angle_at(ring_pip, ring_dip, ring_tip)

        lf_mcp = math.pi - _angle_at(wrist, pinky_mcp, pinky_pip)
        lf_pip = math.pi - _angle_at(pinky_mcp, pinky_pip, pinky_dip)
        lf_dip = math.pi - _angle_at(pinky_pip, pinky_dip, pinky_tip)

        # Thumb (handled separately): abduction/opposition in palm plane + flexion joints.
        thumb_vec = thumb_mcp - thumb_cmc
        thumb_vec_plane = _project_to_plane(thumb_vec, palm_n)
        th_plane_angle = _signed_angle(palm_y, thumb_vec_plane, palm_n)

        # Heuristic sign normalization by handedness to keep "abduction positive" consistent.
        # This keeps the joint direction stable across left/right hands.
        if hand_label is not None and hand_label.lower().startswith("left"):
            th_plane_angle = -th_plane_angle
            ff_spread, rf_spread, lf_spread, lfj5 = -ff_spread, -rf_spread, -lf_spread, -lfj5

        # Thumb joint heuristics:
        # - THJ4: ab/ad proxy in palm plane
        # - THJ1: CMC flex proxy from wrist->CMC->MCP
        # - THJ3: MCP flex
        # - THJ2: IP flex
        # - THJ5: opposition proxy (we reuse the palm-plane thumb angle for stability)
        th_abd = th_plane_angle  # THJ4
        th_cmc_flex = math.pi - _angle_at(wrist, thumb_cmc, thumb_mcp)  # THJ1
        th_mcp_flex = math.pi - _angle_at(thumb_cmc, thumb_mcp, thumb_ip)  # THJ3
        th_ip_flex = math.pi - _angle_at(thumb_mcp, thumb_ip, thumb_tip)  # THJ2
        th_opp = th_plane_angle  # THJ5

        # Wrist joints: not reliably inferable from a cropped ROI without calibration.
        # Keep them at 0 for stability (most ShadowHand demos control wrist separately).
        wrj1 = 0.0
        wrj2 = 0.0

        # Assemble in ShadowHand joint order.
        joints_human = {
            "WRJ2": wrj2,
            "WRJ1": wrj1,
            # Index (FF)
            "FFJ4": ff_spread,
            "FFJ3": ff_mcp,
            "FFJ2": ff_pip,
            "FFJ1": ff_dip,
            # Middle (MF)
            "MFJ4": mf_spread,
            "MFJ3": mf_mcp,
            "MFJ2": mf_pip,
            "MFJ1": mf_dip,
            # Ring (RF)
            "RFJ4": rf_spread,
            "RFJ3": rf_mcp,
            "RFJ2": rf_pip,
            "RFJ1": rf_dip,
            # Little (LF)
            "LFJ5": lfj5,
            "LFJ4": lf_spread,
            "LFJ3": lf_mcp,
            "LFJ2": lf_pip,
            "LFJ1": lf_dip,
            # Thumb (TH)
            "THJ5": th_opp,
            "THJ4": th_abd,
            "THJ3": th_mcp_flex,
            "THJ2": th_ip_flex,
            "THJ1": th_cmc_flex,
        }

        # Remap to robot ranges with clamping.
        out: List[float] = []
        for name in ShadowHandJoints.names():
            x = float(joints_human[name])
            h0, h1, r0, r1 = self._ranges[name]
            out.append(_linear_map(x, h0, h1, r0, r1))
        return out
    
    def _landmarks_to_shadowhand_raw(self, pts: np.ndarray, *, hand_label: Optional[str]) -> Dict[str, float]:
        """
        Get MediaPipe's raw joint angles (before mapping to robot ranges).
        This is used for calibration to record the actual MediaPipe output values.
        
        Returns:
            Dict mapping joint names to raw MediaPipe output values (radians)
        """
        # Reuse the same calculation logic as _landmarks_to_shadowhand
        # but return the raw values before mapping
        wrist = pts[0]
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = pts[1], pts[2], pts[3], pts[4]
        index_mcp, index_pip, index_dip, index_tip = pts[5], pts[6], pts[7], pts[8]
        middle_mcp, middle_pip, middle_dip, middle_tip = pts[9], pts[10], pts[11], pts[12]
        ring_mcp, ring_pip, ring_dip, ring_tip = pts[13], pts[14], pts[15], pts[16]
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip = pts[17], pts[18], pts[19], pts[20]

        # Palm frame for spread estimation.
        palm_x = _normalize(index_mcp - pinky_mcp)
        palm_y = _normalize(middle_mcp - wrist)
        palm_n = _normalize(np.cross(palm_x, palm_y))

        if float(np.linalg.norm(palm_n)) < 1e-6:
            palm_n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        def base_angle(mcp: np.ndarray) -> float:
            v = mcp - wrist
            v = _project_to_plane(v, palm_n)
            return _signed_angle(palm_y, v, palm_n)

        a_index = base_angle(index_mcp)
        a_middle = base_angle(middle_mcp)
        a_ring = base_angle(ring_mcp)
        a_pinky = base_angle(pinky_mcp)

        ff_spread = a_index - a_middle
        mf_spread = 0.0
        rf_spread = a_ring - a_middle
        lf_spread = a_pinky - a_middle
        lfj5 = 0.5 * (a_pinky - a_ring)

        ff_mcp = math.pi - _angle_at(wrist, index_mcp, index_pip)
        ff_pip = math.pi - _angle_at(index_mcp, index_pip, index_dip)
        ff_dip = math.pi - _angle_at(index_pip, index_dip, index_tip)

        mf_mcp = math.pi - _angle_at(wrist, middle_mcp, middle_pip)
        mf_pip = math.pi - _angle_at(middle_mcp, middle_pip, middle_dip)
        mf_dip = math.pi - _angle_at(middle_pip, middle_dip, middle_tip)

        rf_mcp = math.pi - _angle_at(wrist, ring_mcp, ring_pip)
        rf_pip = math.pi - _angle_at(ring_mcp, ring_pip, ring_dip)
        rf_dip = math.pi - _angle_at(ring_pip, ring_dip, ring_tip)

        lf_mcp = math.pi - _angle_at(wrist, pinky_mcp, pinky_pip)
        lf_pip = math.pi - _angle_at(pinky_mcp, pinky_pip, pinky_dip)
        lf_dip = math.pi - _angle_at(pinky_pip, pinky_dip, pinky_tip)

        thumb_vec = thumb_mcp - thumb_cmc
        thumb_vec_plane = _project_to_plane(thumb_vec, palm_n)
        th_plane_angle = _signed_angle(palm_y, thumb_vec_plane, palm_n)

        if hand_label is not None and hand_label.lower().startswith("left"):
            th_plane_angle = -th_plane_angle
            ff_spread, rf_spread, lf_spread, lfj5 = -ff_spread, -rf_spread, -lf_spread, -lfj5

        th_abd = th_plane_angle
        th_cmc_flex = math.pi - _angle_at(wrist, thumb_cmc, thumb_mcp)
        th_mcp_flex = math.pi - _angle_at(thumb_cmc, thumb_mcp, thumb_ip)
        th_ip_flex = math.pi - _angle_at(thumb_mcp, thumb_ip, thumb_tip)
        th_opp = th_plane_angle

        wrj1 = 0.0
        wrj2 = 0.0

        return {
            "WRJ2": wrj2,
            "WRJ1": wrj1,
            "FFJ4": ff_spread,
            "FFJ3": ff_mcp,
            "FFJ2": ff_pip,
            "FFJ1": ff_dip,
            "MFJ4": mf_spread,
            "MFJ3": mf_mcp,
            "MFJ2": mf_pip,
            "MFJ1": mf_dip,
            "RFJ4": rf_spread,
            "RFJ3": rf_mcp,
            "RFJ2": rf_pip,
            "RFJ1": rf_dip,
            "LFJ5": lfj5,
            "LFJ4": lf_spread,
            "LFJ3": lf_mcp,
            "LFJ2": lf_pip,
            "LFJ1": lf_dip,
            "THJ5": th_opp,
            "THJ4": th_abd,
            "THJ3": th_mcp_flex,
            "THJ2": th_ip_flex,
            "THJ1": th_cmc_flex,
        }
    
    def infer_joints_raw(self, frame_bgr: np.ndarray) -> Optional[List[float]]:
        """
        Get MediaPipe's raw joint angles (before mapping to robot ranges) for calibration.
        Returns a 24-element list in ShadowHand joint order, or None if no hand detected.
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        try:
            results = self._hands.process(frame_rgb)
            # Reset error count on success
            self._timestamp_error_count = 0
        except Exception as e:
            # Handle MediaPipe timestamp errors gracefully
            error_str = str(e).lower()
            if "timestamp" in error_str or "packet" in error_str or "calculator" in error_str:
                self._timestamp_error_count += 1
                # Reset MediaPipe instance immediately on first error to prevent segfaults
                if self._timestamp_error_count == 1:
                    try:
                        print(f"[WARNING] MediaPipe timestamp error in infer_joints_raw, resetting MediaPipe Hands...")
                        self._reset_hands()
                    except Exception as reset_error:
                        print(f"[ERROR] Failed to reset MediaPipe in infer_joints_raw: {reset_error}")
            # Return None on any error to prevent segfaults
            return None

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks, hand_label = self._select_hand(results)
        if hand_landmarks is None:
            return None

        pts = np.zeros((21, 3), dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        for i, lm in enumerate(hand_landmarks.landmark):
            pts[i, 0] = float(lm.x * w)
            pts[i, 1] = float(lm.y * h)
            pts[i, 2] = float(lm.z * w)

        joints_raw_dict = self._landmarks_to_shadowhand_raw(pts, hand_label=hand_label)
        
        # Convert to list in ShadowHand joint order
        joint_names = ShadowHandJoints.names()
        return [joints_raw_dict[name] for name in joint_names]


