"""
List OpenCV-accessible camera indices (quick probe).

Usage (from ViTacLab repo root):

    conda activate video_teleoperator
    PYTHONPATH=source python source/video_teleop/tools/list_cameras.py --max-index 15
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe VideoCapture indices and report which open.")
    parser.add_argument(
        "--max-index",
        type=int,
        default=15,
        help="Try indices 0 .. max_index-1 (default: 15)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Request frame width when probing",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Request frame height when probing",
    )
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as e:
        raise SystemExit("opencv-python is required: pip install opencv-python") from e

    print(f"[INFO] Probing camera indices 0..{args.max_index - 1} (request {args.width}x{args.height})")
    found: list[int] = []
    for i in range(max(0, args.max_index)):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  OK  index={i}  frame_size={w}x{h}")
            found.append(i)
        else:
            print(f"  --- index={i} opened but read failed")

    if not found:
        print("[WARN] No cameras found. Check USB permissions, drivers, or try a higher --max-index.")
    else:
        print(f"[INFO] Use e.g. --camera {found[0]} with sender / calibration tools.")


if __name__ == "__main__":
    main()
