"""
ShadowHand 24-DoF joint names (no mediapipe dependency).

Used by run_task scripts that run inside Isaac Sim, where mediapipe may not be installed.
"""

from __future__ import annotations

from typing import List


def shadowhand_joint_names() -> List[str]:
    """ShadowHand 24-DoF joint order (WRJ2, WRJ1, FFJ4, ..., THJ1)."""
    return [
        "WRJ2", "WRJ1",
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
    ]
