#!/usr/bin/env python3
"""Render third-person demo video from replay record npz.

Expected input comes from:
  scripts/rsl_rl/full_tra/play_full_tra_single_v5.py --record-data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected frame shape (H,W,C), got {x.shape}")
    if x.shape[-1] > 3:
        x = x[..., :3]
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        mx = float(np.max(x)) if x.size > 0 else 0.0
        if mx <= 1.0:
            x = np.clip(x, 0.0, 1.0) * 255.0
        else:
            x = np.clip(x, 0.0, 255.0)
        x = x.astype(np.uint8)
    return x


def _pick_npz(record_dir: Path, episode_file: str) -> Path:
    if episode_file:
        p = Path(episode_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"episode file not found: {p}")
        return p
    cands = sorted(record_dir.glob("episode_*.npz"))
    if not cands:
        raise FileNotFoundError(f"no episode_*.npz found in {record_dir}")
    return cands[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render third-person video from replay npz record.")
    ap.add_argument("--record_dir", type=str, required=True)
    ap.add_argument("--episode_file", type=str, default="", help="Optional explicit episode npz path.")
    ap.add_argument("--output_video", type=str, required=True)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--title", type=str, default="Forge Replay Demo")
    ap.add_argument(
        "--focus_tail_frames",
        type=int,
        default=120,
        help="If >0, render only the last N frames (success segment focus).",
    )
    ap.add_argument(
        "--slowmo_tail_frames",
        type=int,
        default=60,
        help="Within selected clip, repeat the last N frames for slow-motion emphasis.",
    )
    ap.add_argument(
        "--slowmo_repeat",
        type=int,
        default=2,
        help="Repeat factor for slow-motion tail frames (>=1).",
    )
    args = ap.parse_args()

    record_dir = Path(args.record_dir).expanduser().resolve()
    npz_path = _pick_npz(record_dir, args.episode_file)
    out_path = Path(args.output_video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.load(npz_path, allow_pickle=False)
    if "third_person_camera" not in arr:
        raise KeyError(
            f"'third_person_camera' key not found in {npz_path}. "
            "Please rerun replay with --enable_cameras and record-data."
        )

    frames = arr["third_person_camera"]
    if frames.ndim != 4 and frames.ndim != 5:
        raise RuntimeError(f"Unexpected third_person_camera shape: {frames.shape}")
    if frames.ndim == 5:
        # (T, N, H, W, C) -> use env 0
        frames = frames[:, 0]

    total_all = int(frames.shape[0])
    if int(args.focus_tail_frames) > 0:
        n_tail = min(int(args.focus_tail_frames), total_all)
        start = total_all - n_tail
        frames = frames[start:]
    else:
        start = 0

    frame0 = _to_uint8_rgb(frames[0])
    h, w = int(frame0.shape[0]), int(frame0.shape[1])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {out_path}")

    total = int(frames.shape[0])
    slow_n = min(max(int(args.slowmo_tail_frames), 0), total)
    slow_repeat = max(int(args.slowmo_repeat), 1)
    for i in range(total):
        rgb = _to_uint8_rgb(frames[i])
        bgr = rgb[:, :, ::-1].copy()
        cv2.putText(bgr, args.title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            bgr,
            f"frame {i+1}/{total} (src {start+i+1}/{total_all})",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        repeat = slow_repeat if i >= (total - slow_n) else 1
        if repeat > 1:
            cv2.putText(
                bgr,
                "slow-motion (success segment)",
                (12, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )
        for _ in range(repeat):
            writer.write(bgr)

    writer.release()
    print(f"[DONE] replay demo saved: {out_path}")
    print(f"[INFO] source record: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
