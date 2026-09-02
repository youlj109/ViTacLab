#!/usr/bin/env python3
"""Extract Xense ball-calibration frames from advisor mp4 for Taxim polycalib.

Selects:
  - 1 no-contact background (lowest center-patch temporal variance)
  - ~N diverse ball-indent frames (max |rgb-bg|, spread in center/radius)

Usage::

    python3 scripts/calibration/import_ball_calib_video.py
    python3 scripts/calibration/import_ball_calib_video.py --video logs/file-000.mp4 --num-ball 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import XENSE_LAB_HW, repo_root  # noqa: E402

# Xense datasheet: sensing area 17.5 x 29.5 mm @ 400x700.
XENSE_SENSING_MM = (17.5, 29.5)
XENSE_BALL_RADIUS_MM = 3.0  # 6 mm steel ball


@dataclass
class FrameScore:
    path: Path
    diff: float
    center_yx: tuple[float, float]
    radius_px: float


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(arr.astype(np.uint8)).save(path)
        return
    except ImportError:
        pass
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _extract_frames(mp4: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-vsync", "0", str(out_dir / "f%04d.png")],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob("f*.png"))


def _center_patch_std(rgb: np.ndarray, *, frac: float = 0.45) -> float:
    gray = rgb.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    rh, rw = int(h * frac), int(w * frac)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    patch = gray[y0 : y0 + rh, x0 : x0 + rw]
    return float(patch.std())


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _contact_blob(bg: np.ndarray, frame: np.ndarray) -> tuple[tuple[float, float], float, float]:
    """Return ((cy, cx), radius_px, diff_mean) from rgb diff blob."""
    import cv2

    diff = np.abs(frame.astype(np.float32) - bg.astype(np.float32)).mean(axis=2)
    diff_blur = cv2.GaussianBlur(diff, (9, 9), 0)
    peak = float(diff_blur.max())
    if peak < 1.0:
        h, w = diff.shape
        return (h / 2.0, w / 2.0), 0.0, float(diff.mean())

    thr = max(2.0, 0.12 * peak)
    mask = (diff_blur >= thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = diff.shape
        return (h / 2.0, w / 2.0), 0.0, float(diff.mean())

    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 20.0:
        h, w = diff.shape
        return (h / 2.0, w / 2.0), 0.0, float(diff.mean())

    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    return (float(cy), float(cx)), float(radius), float(diff.mean())


def _pick_bg_frame(frames: list[Path], *, skip: int) -> tuple[Path, dict]:
    scored: list[tuple[float, Path, float]] = []
    for p in frames[skip:]:
        rgb = _load_rgb(p)
        scored.append((_center_patch_std(rgb), p, float(rgb.mean())))
    scored.sort(key=lambda x: x[0])
    best_std, best_path, best_mean = scored[0]
    return best_path, {
        "frame_file": best_path.name,
        "center_std": best_std,
        "mean": best_mean,
        "skip_warmup": skip,
    }


def _select_diverse(scored: list[FrameScore], *, target: int) -> list[FrameScore]:
    if len(scored) <= target:
        return scored

    # Greedy farthest-point in normalized (cy, cx, radius) space.
    cy_vals = [s.center_yx[0] for s in scored]
    cx_vals = [s.center_yx[1] for s in scored]
    r_vals = [s.radius_px for s in scored]
    cy_rng = max(cy_vals) - min(cy_vals) + 1e-6
    cx_rng = max(cx_vals) - min(cx_vals) + 1e-6
    r_rng = max(r_vals) - min(r_vals) + 1e-6

    def _feat(s: FrameScore) -> np.ndarray:
        cy, cx = s.center_yx
        return np.array([cy / cy_rng, cx / cx_rng, s.radius_px / r_rng], dtype=np.float32)

    remaining = scored.copy()
    remaining.sort(key=lambda s: s.diff, reverse=True)
    selected = [remaining.pop(0)]
    feats = [_feat(selected[0])]

    while len(selected) < target and remaining:
        best_i = 0
        best_score = -1.0
        for i, cand in enumerate(remaining):
            f = _feat(cand)
            min_d = min(float(np.linalg.norm(f - sf)) for sf in feats)
            score = min_d + 0.05 * cand.diff
            if score > best_score:
                best_score = score
                best_i = i
        pick = remaining.pop(best_i)
        selected.append(pick)
        feats.append(_feat(pick))

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Import ball-calibration frames from advisor mp4.")
    parser.add_argument(
        "--video",
        type=str,
        default=str(repo_root() / "logs" / "file-000.mp4"),
        help="Advisor 6mm ball calibration mp4 (400x700).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/ball_calib_raw"),
    )
    parser.add_argument("--bg-warmup-skip", type=int, default=10)
    parser.add_argument("--num-ball", type=int, default=50)
    parser.add_argument("--min-diff", type=float, default=2.0, help="Min mean |rgb-bg| to count as contact.")
    args = parser.parse_args()

    _require_ffmpeg()
    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_w, out_h = XENSE_LAB_HW

    if not video.is_file():
        print(f"[ERR] video not found: {video}", file=sys.stderr)
        return 1

    bg_dir = out_dir / "bg"
    ball_dir = out_dir / "ball"
    if bg_dir.exists():
        shutil.rmtree(bg_dir)
    if ball_dir.exists():
        shutil.rmtree(ball_dir)
    bg_dir.mkdir(parents=True)
    ball_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="vitac_ball_") as tmp:
        frames = _extract_frames(video, Path(tmp) / "frames")
        if not frames:
            print("[ERR] no frames decoded", file=sys.stderr)
            return 1

        bg_path, bg_meta = _pick_bg_frame(frames, skip=int(args.bg_warmup_skip))
        bg_rgb = _load_rgb(bg_path)
        if bg_rgb.shape[1] != out_w or bg_rgb.shape[0] != out_h:
            print(f"[WARN] frame size {bg_rgb.shape[1]}x{bg_rgb.shape[0]} != expected {out_w}x{out_h}")

        _save_rgb(bg_rgb, bg_dir / "no_contact.png")

        scored: list[FrameScore] = []
        for p in frames:
            if p == bg_path:
                continue
            rgb = _load_rgb(p)
            diff = _mean_abs_diff(rgb, bg_rgb)
            if diff < float(args.min_diff):
                continue
            center_yx, radius_px, _ = _contact_blob(bg_rgb, rgb)
            if radius_px < 3.0:
                continue
            scored.append(FrameScore(path=p, diff=diff, center_yx=center_yx, radius_px=radius_px))

        if len(scored) < 10:
            print(f"[ERR] only {len(scored)} contact frames (need >=10)", file=sys.stderr)
            return 1

        selected = _select_diverse(scored, target=int(args.num_ball))
        ball_records: list[dict] = []
        for i, item in enumerate(selected):
            dst = ball_dir / f"{i:03d}.png"
            _save_rgb(_load_rgb(item.path), dst)
            ball_records.append(
                {
                    "file": dst.name,
                    "source_frame": item.path.name,
                    "diff": item.diff,
                    "center_yx": [item.center_yx[0], item.center_yx[1]],
                    "radius_px": item.radius_px,
                }
            )

    mm_per_px_w = XENSE_SENSING_MM[0] / out_w
    mm_per_px_h = XENSE_SENSING_MM[1] / out_h
    meta = {
        "source_video": str(video),
        "output_size_wh": [out_w, out_h],
        "sensing_area_mm": list(XENSE_SENSING_MM),
        "ball_radius_mm": XENSE_BALL_RADIUS_MM,
        "mm_per_pixel": {"width": mm_per_px_w, "height": mm_per_px_h, "mean": (mm_per_px_w + mm_per_px_h) / 2.0},
        "bg": bg_meta,
        "num_contact_candidates": len(scored),
        "num_ball_selected": len(selected),
        "ball_frames": ball_records,
    }
    meta_path = out_dir / "import_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] bg -> {bg_dir / 'no_contact.png'}")
    print(f"[OK] ball frames -> {ball_dir} ({len(selected)} images)")
    print(f"[OK] metadata -> {meta_path}")
    print("")
    print("[NEXT] python3 scripts/calibration/build_xense_polycalib.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
