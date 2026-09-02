"""Canonical paths for video teleop calibration YAML files.

Files live under ``scripts/teleoperation/video_teleop/config/`` at the repository root
(``camera_calibration.yaml``, ``hand_calibration.yaml``).
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """ViTacLab repository root (directory that contains ``source/``)."""
    return Path(__file__).resolve().parents[2]


def video_teleop_config_dir() -> Path:
    """Directory for camera / hand calibration YAML (versioned next to teleop launchers)."""
    return project_root() / "scripts" / "teleoperation" / "video_teleop" / "config"


def default_camera_calibration_yaml() -> str:
    return str(video_teleop_config_dir() / "camera_calibration.yaml")


def default_hand_calibration_yaml() -> str:
    return str(video_teleop_config_dir() / "hand_calibration.yaml")


# Resolved once at import (used as argparse / VideoListener defaults)
DEFAULT_CAMERA_CALIBRATION_YAML = default_camera_calibration_yaml()
DEFAULT_HAND_CALIBRATION_YAML = default_hand_calibration_yaml()
