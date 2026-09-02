#!/usr/bin/env python3
"""Create a presentation-style comparison video from loose/strict alignment videos.

This script does not rerun simulation. It composes two existing mp4 files side-by-side
with clear titles and an interpretation banner for easier demonstration to reviewers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _draw_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fit_frame(frame: np.ndarray, out_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == out_h:
        return frame
    out_w = max(1, int(round(float(w) * float(out_h) / float(h))))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose loose/strict videos into one presentation video.")
    parser.add_argument("--loose_video", type=str, required=True)
    parser.add_argument("--strict_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--title", type=str, default="ViTacSim Object Attribution Demo (interference_only)")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    loose_path = Path(args.loose_video).expanduser().resolve()
    strict_path = Path(args.strict_video).expanduser().resolve()
    out_path = Path(args.output_video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap_loose = cv2.VideoCapture(str(loose_path))
    cap_strict = cv2.VideoCapture(str(strict_path))
    if not cap_loose.isOpened():
        raise RuntimeError(f"Failed to open loose video: {loose_path}")
    if not cap_strict.isOpened():
        raise RuntimeError(f"Failed to open strict video: {strict_path}")

    ok_l, frame_l = cap_loose.read()
    ok_s, frame_s = cap_strict.read()
    if not ok_l or frame_l is None:
        raise RuntimeError(f"Loose video has no readable frame: {loose_path}")
    if not ok_s or frame_s is None:
        raise RuntimeError(f"Strict video has no readable frame: {strict_path}")

    body_h = max(frame_l.shape[0], frame_s.shape[0])
    top_bar_h = 72
    mid_caption_h = 40
    bottom_bar_h = 56

    frame_l = _fit_frame(frame_l, body_h)
    frame_s = _fit_frame(frame_s, body_h)
    body_w = frame_l.shape[1] + frame_s.shape[1]
    out_h = top_bar_h + mid_caption_h + body_h + bottom_bar_h
    out_w = body_w

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {out_path}")

    def compose(fl: np.ndarray, fs: np.ndarray) -> np.ndarray:
        fl = _fit_frame(fl, body_h)
        fs = _fit_frame(fs, body_h)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Top banner
        canvas[0:top_bar_h, :, :] = (28, 28, 28)
        _draw_text(canvas, args.title, (18, 30), 0.78, (255, 255, 255), 2)
        _draw_text(canvas, "Goal: show false activation without strict attribution vs suppression with strict attribution", (18, 58), 0.53, (210, 210, 210), 1)

        # Mid captions
        y0 = top_bar_h
        canvas[y0 : y0 + mid_caption_h, :, :] = (40, 40, 40)
        split_x = fl.shape[1]
        cv2.line(canvas, (split_x, y0), (split_x, y0 + mid_caption_h + body_h), (80, 80, 80), 2)
        _draw_text(canvas, "STRICT=0 (Loose attribution)  -> expected false activation", (16, y0 + 26), 0.55, (80, 180, 255), 2)
        _draw_text(canvas, "STRICT=1 (Strict attribution) -> expected suppression", (split_x + 16, y0 + 26), 0.55, (120, 230, 120), 2)

        # Body
        y1 = y0 + mid_caption_h
        canvas[y1 : y1 + body_h, 0 : fl.shape[1], :] = fl
        canvas[y1 : y1 + body_h, fl.shape[1] : fl.shape[1] + fs.shape[1], :] = fs

        # Bottom interpretation bar
        y2 = y1 + body_h
        canvas[y2 : y2 + bottom_bar_h, :, :] = (28, 28, 28)
        _draw_text(canvas, "Interpretation: when raw_depth_contact_rate>0 but physx target force=0, any non-zero sensor_fn indicates attribution error.", (16, y2 + 22), 0.5, (230, 230, 230), 1)
        _draw_text(canvas, "Pass signal: STRICT=1 keeps sensor_fn/false_rate near zero while contact from interference object still exists.", (16, y2 + 45), 0.5, (230, 230, 230), 1)
        return canvas

    writer.write(compose(frame_l, frame_s))
    while True:
        ok_l, frame_l = cap_loose.read()
        ok_s, frame_s = cap_strict.read()
        if (not ok_l) and (not ok_s):
            break
        if not ok_l:
            frame_l = np.zeros((body_h, max(1, frame_l.shape[1] if frame_l is not None else 640), 3), dtype=np.uint8)
        if not ok_s:
            frame_s = np.zeros((body_h, max(1, frame_s.shape[1] if frame_s is not None else 640), 3), dtype=np.uint8)
        writer.write(compose(frame_l, frame_s))

    cap_loose.release()
    cap_strict.release()
    writer.release()
    print(f"[DONE] demo video saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
