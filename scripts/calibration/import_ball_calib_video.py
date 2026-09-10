#!/usr/bin/env python3
"""Extract Xense ball-calibration frames from advisor mp4 for Taxim polycalib.

Selects:
  - 1 no-contact background (lowest center-patch temporal variance)
  - ~N diverse ball-indent frames (max |rgb-bg|, spread in center/radius)

Usage::

    python3 scripts/calibration/import_ball_calib_video.py
    python3 scripts/calibration/import_ball_calib_video.py --video data/calibration/file-000.mp4 --num-ball 50
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
    chroma_peak: float


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
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(mp4), "-vsync", "0", str(out_dir / "f%04d.png")],
            check=True,
            capture_output=True,
        )
    else:
        import cv2

        capture = cv2.VideoCapture(str(mp4))
        if not capture.isOpened():
            raise RuntimeError(f"Could not decode video with OpenCV: {mp4}")
        index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            cv2.imwrite(str(out_dir / f"f{index:04d}.png"), frame_bgr)
            index += 1
        capture.release()
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


def _chromatic_response(bg: np.ndarray, frame: np.ndarray) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    bg_f = bg.astype(np.float32)
    delta_chroma = (frame_f - frame_f.mean(axis=2, keepdims=True)) - (
        bg_f - bg_f.mean(axis=2, keepdims=True)
    )
    response = np.sqrt(np.sum(delta_chroma * delta_chroma, axis=2))
    import cv2

    return cv2.GaussianBlur(response, (0, 0), 5.0)


def _contact_blob(
    bg: np.ndarray,
    frame: np.ndarray,
    *,
    max_radius_px: float | None = None,
) -> tuple[tuple[float, float], float, float]:
    """Return a physically bounded ball-contact circle from chromatic response.

    Xense contact illumination is bipolar (cyan on one side and red on the
    other), while exposure drift and moving printed markers dominate a naive
    absolute RGB difference.  Detect the channel-centered response instead,
    join the two illumination lobes, and use an area-equivalent circle rather
    than the enclosing circle of a diffuse halo.
    """
    import cv2

    frame_f = frame.astype(np.float32)
    bg_f = bg.astype(np.float32)
    response = _chromatic_response(bg, frame)
    peak = float(response.max())
    if peak < 1.0:
        h, w = response.shape
        return (h / 2.0, w / 2.0), 0.0, float(np.abs(frame_f - bg_f).mean())

    # A lower chromatic threshold keeps both opposed illumination lobes. Using
    # only the strongest lobe can move the center by 20--35 px on asymmetric
    # frames (for example ball frame 037).
    threshold = max(float(np.percentile(response, 90.0)), 0.30 * peak)
    mask = (response >= threshold).astype(np.uint8)
    if max_radius_px is None:
        max_radius_px = 0.18 * min(response.shape)
    close_size = max(9, int(round(0.75 * float(max_radius_px))))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        h, w = response.shape
        return (h / 2.0, w / 2.0), 0.0, float(np.abs(frame_f - bg_f).mean())

    scores = [float(response[labels == label].sum()) for label in range(1, count)]
    label = 1 + int(np.argmax(scores))
    area = int(stats[label, cv2.CC_STAT_AREA])
    if area < 20:
        h, w = response.shape
        return (h / 2.0, w / 2.0), 0.0, float(np.abs(frame_f - bg_f).mean())

    cx, cy = centers[label]
    radius = float(np.sqrt(area / np.pi))

    # The binary component centroid is biased toward whichever illumination
    # lobe is brighter. Estimate the geometric center as the midpoint between
    # the opposed red and cyan lobes when both are confidently present.
    delta_rgb = frame_f - bg_f
    red_axis = delta_rgb[:, :, 0] - 0.5 * (delta_rgb[:, :, 1] + delta_rgb[:, :, 2])
    red_axis = cv2.GaussianBlur(red_axis, (0, 0), 4.0)
    region = labels == label
    region_values = red_axis[region]
    median = float(np.median(region_values))
    spread = max(float(np.percentile(np.abs(region_values - median), 75.0)), 1.0)
    positive = np.where(region, np.maximum(red_axis - (median + 0.15 * spread), 0.0), 0.0)
    negative = np.where(region, np.maximum((median - 0.15 * spread) - red_axis, 0.0), 0.0)

    def _weighted_center(weights: np.ndarray) -> tuple[float, float] | None:
        total = float(weights.sum())
        if total <= 1.0e-6:
            return None
        yy, xx = np.mgrid[: weights.shape[0], : weights.shape[1]]
        return float((weights * xx).sum() / total), float((weights * yy).sum() / total)

    positive_center = _weighted_center(positive)
    negative_center = _weighted_center(negative)
    if positive_center is not None and negative_center is not None:
        separation = float(np.linalg.norm(np.subtract(positive_center, negative_center)))
        midpoint = 0.5 * (np.asarray(positive_center) + np.asarray(negative_center))
        midpoint_shift = float(np.linalg.norm(midpoint - np.asarray((cx, cy))))
        if 0.25 * radius <= separation <= 2.25 * radius and midpoint_shift <= 0.35 * radius:
            cx, cy = float(midpoint[0]), float(midpoint[1])

    # A projected spherical contact cannot extend past the ball's equator.
    # Keep a small margin so the singular 90-degree rim never enters a bin.
    radius = min(radius, 0.97 * float(max_radius_px))
    return (float(cy), float(cx)), radius, float(np.abs(frame_f - bg_f).mean())


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
        default=str(repo_root() / "data" / "calibration" / "file-000.mp4"),
        help="Advisor 6mm ball calibration mp4 (400x700).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/ball_calib_raw"),
    )
    parser.add_argument("--bg-warmup-skip", type=int, default=10)
    parser.add_argument(
        "--reference-bg",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/advisor_processed/bg.jpg"),
        help="Explicit true no-contact RGB frame; defaults to advisor_processed/bg.jpg.",
    )
    parser.add_argument("--num-ball", type=int, default=50)
    parser.add_argument(
        "--num-validation",
        type=int,
        default=50,
        help="Additional temporally separated contact frames reserved from fitting.",
    )
    parser.add_argument(
        "--validation-temporal-gap",
        type=int,
        default=3,
        help="Exclude validation frames within this many video frames of any training frame.",
    )
    parser.add_argument("--min-diff", type=float, default=2.0, help="Min mean |rgb-bg| to count as contact.")
    parser.add_argument(
        "--min-chroma-peak",
        type=float,
        default=35.0,
        help="Minimum blurred chromatic-response peak; rejects no-contact/exposure-only frames.",
    )
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_w, out_h = XENSE_LAB_HW

    if not video.is_file():
        print(f"[ERR] video not found: {video}", file=sys.stderr)
        return 1

    bg_dir = out_dir / "bg"
    ball_dir = out_dir / "ball"
    validation_dir = out_dir.parent / f"{out_dir.name}_validation"
    validation_ball_dir = validation_dir / "ball"
    if bg_dir.exists():
        shutil.rmtree(bg_dir)
    if ball_dir.exists():
        shutil.rmtree(ball_dir)
    if validation_ball_dir.exists():
        shutil.rmtree(validation_ball_dir)
    bg_dir.mkdir(parents=True)
    ball_dir.mkdir(parents=True)
    validation_ball_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="vitac_ball_") as tmp:
        frames = _extract_frames(video, Path(tmp) / "frames")
        if not frames:
            print("[ERR] no frames decoded", file=sys.stderr)
            return 1

        if args.reference_bg:
            bg_path = Path(args.reference_bg).expanduser().resolve()
            if not bg_path.is_file():
                print(f"[ERR] reference background not found: {bg_path}", file=sys.stderr)
                return 1
            bg_rgb = _load_rgb(bg_path)
            bg_meta = {
                "source": "explicit_reference",
                "path": str(bg_path),
                "mean": float(bg_rgb.mean()),
            }
        else:
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
            chroma_peak = float(_chromatic_response(bg_rgb, rgb).max())
            if chroma_peak < float(args.min_chroma_peak):
                continue
            max_radius_px = XENSE_BALL_RADIUS_MM / (XENSE_SENSING_MM[0] / out_w)
            center_yx, radius_px, _ = _contact_blob(
                bg_rgb,
                rgb,
                max_radius_px=max_radius_px,
            )
            if radius_px < 3.0:
                continue
            cy, cx = center_yx
            if (
                cx - radius_px < 0
                or cx + radius_px >= out_w
                or cy - radius_px < 0
                or cy + radius_px >= out_h
            ):
                continue
            scored.append(
                FrameScore(
                    path=p,
                    diff=diff,
                    center_yx=center_yx,
                    radius_px=radius_px,
                    chroma_peak=chroma_peak,
                )
            )

        if len(scored) < 10:
            print(f"[ERR] only {len(scored)} contact frames (need >=10)", file=sys.stderr)
            return 1

        selected = _select_diverse(scored, target=int(args.num_ball))
        selected_indices = {int(item.path.stem.lstrip("f")) for item in selected}
        temporal_gap = max(int(args.validation_temporal_gap), 0)
        validation_candidates = [
            item
            for item in scored
            if all(
                abs(int(item.path.stem.lstrip("f")) - train_index) > temporal_gap
                for train_index in selected_indices
            )
        ]
        validation_selected = _select_diverse(
            validation_candidates,
            target=int(args.num_validation),
        )
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
                    "chroma_peak": item.chroma_peak,
                }
            )
        validation_records: list[dict] = []
        for i, item in enumerate(validation_selected):
            dst = validation_ball_dir / f"{i:03d}.png"
            _save_rgb(_load_rgb(item.path), dst)
            validation_records.append(
                {
                    "file": dst.name,
                    "source_frame": item.path.name,
                    "diff": item.diff,
                    "center_yx": [item.center_yx[0], item.center_yx[1]],
                    "radius_px": item.radius_px,
                    "chroma_peak": item.chroma_peak,
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
        "num_validation_selected": len(validation_selected),
        "validation_temporal_gap": temporal_gap,
        "validation_dir": str(validation_dir),
        "validation_frames": validation_records,
    }
    meta_path = out_dir / "import_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] bg -> {bg_dir / 'no_contact.png'}")
    print(f"[OK] ball frames -> {ball_dir} ({len(selected)} images)")
    print(
        f"[OK] held-out ball frames -> {validation_ball_dir} "
        f"({len(validation_selected)} images, temporal gap>{temporal_gap})"
    )
    print(f"[OK] metadata -> {meta_path}")
    print("")
    print("[NEXT] python3 scripts/calibration/build_xense_polycalib.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
