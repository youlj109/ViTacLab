"""
Video teleoperation control: wrist pose -> IK -> arm+hand joint targets.

Transforms teleop wrist pose to arm base frame, solves IK for UR10e,
and combines with hand joint targets from teleop.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class ArmHandTargets:
    """Arm (6) + hand (24) joint targets in radians."""

    arm_joints: np.ndarray  # (6,)
    hand_joints: np.ndarray  # (24,) ShadowHand order


class VideoTeleopControl:
    """
    Transforms teleop wrist pose to arm base frame and solves IK for UR10e.
    Hand joints are passed through from teleop (calibration applied upstream).
    """

    def __init__(
        self,
        *,
        urdf_path: Optional[str] = None,
        arm_base_link: str = "base_link",
        arm_ee_link: str = "wrist_3_link",
        T_world_arm_base: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            urdf_path: Path to UR10e+ShadowHand URDF. If None, uses default.
            arm_base_link: Base link for IK chain.
            arm_ee_link: End-effector link for IK (wrist before hand).
            T_world_arm_base: (4,4) transform from world to arm base. If None, identity.
        """
        if urdf_path is None:
            proj_root = Path(__file__).resolve().parents[3]
            urdf_path = str(
                proj_root / "source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_left_hand_glb.urdf"
            )
        self.urdf_path = Path(urdf_path)
        self.arm_base_link = arm_base_link
        self.arm_ee_link = arm_ee_link
        self.T_world_arm_base = (
            np.eye(4, dtype=np.float64) if T_world_arm_base is None else np.array(T_world_arm_base, dtype=np.float64)
        )
        self._chain = None
        # Elbow-up seed: shoulder_lift < 0 keeps elbow above table
        self._elbow_up_seed = np.array([0.0, -1.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_arm_joints: Optional[np.ndarray] = None

    def _ensure_chain(self) -> None:
        if self._chain is not None:
            return
        try:
            import ikpy.chain
            base_el = [
                "base_link",
                "base_link-base_link_inertia",
                "base_link_inertia",
                "shoulder_pan_joint",
                "shoulder_link",
                "shoulder_lift_joint",
                "upper_arm_link",
                "elbow_joint",
                "forearm_link",
                "wrist_1_joint",
                "wrist_1_link",
                "wrist_2_joint",
                "wrist_2_link",
                "wrist_3_joint",
                "wrist_3_link",
            ]
            # OriginLink(0) + base_link_inertia(1) + 6 revolute + wrist_3-flange(8) = 9 links
            # Only revolute joints (indices 2-7) are active; fixed links must be False
            active_mask = [False, False, True, True, True, True, True, True, False]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*fixed.*active_links_mask.*", module="ikpy.chain")
                self._chain = ikpy.chain.Chain.from_urdf_file(
                    str(self.urdf_path),
                    base_elements=base_el,
                    base_element_type="link",
                    active_links_mask=active_mask,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load IK chain from {self.urdf_path}: {e}") from e

    def _make_T(self, pos_xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
        T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
        return T

    def compute(
        self,
        wrist_pos_world: np.ndarray,
        wrist_ori_world: np.ndarray,
        hand_joints: np.ndarray,
    ) -> Optional[ArmHandTargets]:
        """
        Compute arm+hand joint targets from teleop pose and hand joints.

        Args:
            wrist_pos_world: (3,) wrist position in world frame (m).
            wrist_ori_world: (3,) wrist orientation Euler xyz in world frame (rad).
            hand_joints: (24,) hand joint angles in ShadowHand order (rad).

        Returns:
            ArmHandTargets or None if IK fails.
        """
        self._ensure_chain()
        T_world_hand = self._make_T(wrist_pos_world, wrist_ori_world)
        T_arm_base_hand = np.linalg.inv(self.T_world_arm_base) @ T_world_hand
        target_pos = T_arm_base_hand[:3, 3]
        target_rot = T_arm_base_hand[:3, :3]

        def _solve_ik(initial_arm: np.ndarray) -> Optional[np.ndarray]:
            full_init = self._chain.active_to_full(
                initial_arm.ravel()[:6],
                np.zeros(len(self._chain.links), dtype=np.float64),
            )
            try:
                ik_result = self._chain.inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_rot,
                    orientation_mode="all",
                    initial_position=full_init,
                )
                ik_result = np.asarray(ik_result, dtype=np.float64)
                active = self._chain.active_from_full(ik_result)
                out = np.array(active, dtype=np.float64).ravel()
                if out.size < 6:
                    out = np.pad(out, (0, 6 - out.size), constant_values=0.0)
                elif out.size > 6:
                    out = out[:6]
                return out
            except Exception:
                return None

        # Prefer elbow-up: shoulder_lift (index 1) < 0
        initial = self._last_arm_joints if self._last_arm_joints is not None else self._elbow_up_seed
        arm_joints = _solve_ik(initial)
        if arm_joints is not None and arm_joints.size >= 2 and arm_joints[1] > 0:
            # Elbow down, retry with elbow-up seed; only accept if retry gives elbow up
            arm_joints_retry = _solve_ik(self._elbow_up_seed)
            if arm_joints_retry is not None and arm_joints_retry.size >= 2 and arm_joints_retry[1] < 0:
                arm_joints = arm_joints_retry
        if arm_joints is not None:
            self._last_arm_joints = arm_joints.copy()
        else:
            return None

        hand_joints = np.asarray(hand_joints, dtype=np.float64).ravel()
        if hand_joints.size != 24:
            hand_joints = np.zeros(24, dtype=np.float64)

        return ArmHandTargets(arm_joints=arm_joints, hand_joints=hand_joints)
