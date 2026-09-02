"""
Chessboard camera calibration (intrinsics + distortion), YAML output for video teleop.

Usage (from ViTacLab repo root):

    conda activate video_teleoperator
    PYTHONPATH=source python source/video_teleop/tools/camera_calibration.py \\
        --camera 0 --rows 5 --cols 7 --square-size 0.025 \\
        --output scripts/teleoperation/video_teleop/config/camera_calibration.yaml

Verify:

    PYTHONPATH=source python source/video_teleop/tools/camera_calibration.py \\
        --camera 0 --verify scripts/teleoperation/video_teleop/config/camera_calibration.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def _ensure_project_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    source_dir = project_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return project_root


def _object_points(rows: int, cols: int, square_m: float) -> np.ndarray:
    """rows/cols = inner corners; grid on z=0."""
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid.astype(np.float32) * float(square_m)
    return objp


def run_calibration(args: argparse.Namespace, project_root: Path) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    pattern_size = (args.cols, args.rows)  # (width, height) inner corners
    objp = _object_points(args.rows, args.cols, args.square_size)

    images_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print("[INFO] Space = capture image, q = finish & calibrate, ESC = quit without saving")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            cv2.drawChessboardCorners(display, pattern_size, corners, found)
        cv2.putText(
            display,
            f"captured={len(images_points)}  SPACE=add  q=calibrate",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("camera_calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Aborted.")
            return 1
        if key == ord(" ") and found:
            images_points.append(corners)
            print(f"[INFO] Stored sample {len(images_points)}")
        if key == ord("q") or key == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(images_points) < 5:
        print(f"[ERROR] Need at least 5 valid captures, got {len(images_points)}")
        return 1
    if image_size is None:
        print("[ERROR] No frame size")
        return 1

    obj_points = [objp] * len(images_points)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        images_points,
        image_size,
        None,
        None,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "checkerboard_rows": args.rows,
        "checkerboard_cols": args.cols,
        "square_size_m": float(args.square_size),
        "reprojection_error": float(ret),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Wrote {out_path}  RMS reprojection error={ret:.6f}")
    return 0


def run_verify(args: argparse.Namespace, project_root: Path) -> int:
    calib_path = Path(args.verify)
    if not calib_path.is_absolute():
        calib_path = project_root / calib_path
    if not calib_path.is_file():
        print(f"[ERROR] Missing file: {calib_path}")
        return 1
    with open(calib_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cam = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["distortion_coefficients"], dtype=np.float64)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        return 1
    print("[INFO] Verify mode: undistorted preview. q or ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        new_cam, _ = cv2.getOptimalNewCameraMatrix(cam, dist, (frame.shape[1], frame.shape[0]), 1.0)
        und = cv2.undistort(frame, cam, dist, None, new_cam)
        cv2.imshow("original", frame)
        cv2.imshow("undistorted", und)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera calibration for video teleop (chessboard)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--rows", type=int, default=5, help="Inner corner rows")
    parser.add_argument("--cols", type=int, default=7, help="Inner corner columns")
    parser.add_argument("--square-size", type=float, default=0.025, help="Square size in meters")
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/teleoperation/video_teleop/config/camera_calibration.yaml",
    )
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--verify", type=str, default="", help="Path to YAML; show undistort preview")
    args = parser.parse_args()

    project_root = _ensure_project_path()

    if args.verify:
        return run_verify(args, project_root)
    return run_calibration(args, project_root)


if __name__ == "__main__":
    raise SystemExit(main())
