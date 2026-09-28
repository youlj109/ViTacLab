#!/usr/bin/env python3
"""Export mp4 replays from play_single / play_record canonical NPZ files.

Outputs (per episode):
- ``*_third_person.mp4``
- ``*_tactile_rgb.mp4``  (5 fingertips concatenated horizontally)
- ``*_tactile_force_field.mp4``
- ``*_demo_panel.mp4``   (third-person + tactile RGB + force field, or RGB + force field if no camera)

No Isaac Sim dependency — only numpy, imageio, opencv-python.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Literal

import cv2
import imageio
import numpy as np


def compute_tactile_shear_image(
    tactile_normal_force: np.ndarray,
    tactile_shear_force: np.ndarray,
    normal_force_threshold: float = 0.00008,
    shear_force_threshold: float = 0.0005,
    resolution: int = 30,
) -> np.ndarray:
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]
    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)
    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            sx = float(tactile_shear_force[row, col, 0])
            sy = float(tactile_shear_force[row, col, 1])
            nf_v = float(np.asarray(tactile_normal_force[row, col]).reshape(-1)[0])
            loc1_x = float(loc0_x) + sx / shear_force_threshold * resolution
            loc1_y = float(loc0_y) + sy / shear_force_threshold * resolution
            color = (
                0.0,
                float(max(0.0, 1.0 - nf_v / normal_force_threshold)),
                float(min(1.0, nf_v / normal_force_threshold)),
            )
            cv2.arrowedLine(
                imgs_tactile,
                (int(loc0_y), int(loc0_x)),
                (int(loc1_y), int(loc1_x)),
                color,
                6,
                tipLength=0.4,
            )
    return imgs_tactile


def render_activation_heatmap(
    activation_hw: np.ndarray,
    *,
    cell_px: int = 12,
) -> np.ndarray:
    """Colormap activation in [0,1] to RGB (used when TacSL force arrays are empty)."""
    act = np.asarray(activation_hw, dtype=np.float32)
    if act.ndim == 3 and act.shape[-1] == 1:
        act = act[..., 0]
    scale = float(np.percentile(act, 99.0)) + 1e-6
    n = np.clip(act / scale, 0.0, 1.0)
    r = n
    g = 0.25 * n
    b = 1.0 - 0.75 * n
    img = np.stack([r, g, b], axis=-1)
    nrows, ncols = n.shape[:2]
    up = max(1, int(cell_px))
    img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.resize(img_u8, (ncols * up, nrows * up), interpolation=cv2.INTER_NEAREST)


def tactile_rgb_to_activation_grid(
    rgb_t: np.ndarray,
    baseline_t: np.ndarray | None,
    *,
    grid_hw: tuple[int, int] = (20, 25),
) -> np.ndarray:
    """Downsample GelSight RGB change to a (H,W) activation map per sensor."""
    rgb = _ensure_uint8_rgb(rgb_t).astype(np.float32)
    base = _ensure_uint8_rgb(baseline_t if baseline_t is not None else rgb_t).astype(np.float32)
    diff = np.abs(rgb - base)
    if diff.ndim == 3 and diff.shape[-1] == 3:
        diff = diff.mean(axis=-1)
    gh, gw = grid_hw
    h, w = diff.shape[:2]
    patch_h = max(1, h // gh)
    patch_w = max(1, w // gw)
    grid = np.zeros((gh, gw), dtype=np.float32)
    for r in range(gh):
        r0 = min(h - 1, r * patch_h)
        r1 = min(h, (r + 1) * patch_h)
        for c in range(gw):
            c0 = min(w - 1, c * patch_w)
            c1 = min(w, (c + 1) * patch_w)
            grid[r, c] = float(diff[r0:r1, c0:c1].mean())
    return grid


def frame_tactile_rgb_activation(
    rgb_t: np.ndarray,
    baseline_t: np.ndarray | None,
    *,
    dual_arm: bool | None = None,
    ff_cell_px: int = 12,
    grid_hw: tuple[int, int] = (20, 25),
) -> np.ndarray:
    """Fallback force-field strip from tactile RGB diff (when normal/shear forces are all zero)."""
    rgb = np.asarray(rgb_t)
    base = np.asarray(baseline_t) if baseline_t is not None else None
    n_sensors = int(rgb.shape[0])
    if dual_arm is None:
        dual_arm = n_sensors == 10

    def _one(sensor_idx: int) -> np.ndarray:
        act = tactile_rgb_to_activation_grid(rgb[sensor_idx], base[sensor_idx] if base is not None else None, grid_hw=grid_hw)
        return render_activation_heatmap(act, cell_px=ff_cell_px)

    if dual_arm and n_sensors == 10:
        left_row = np.concatenate([_one(i) for i in range(5)], axis=1)
        right_row = np.concatenate([_one(i) for i in range(5, 10)], axis=1)
        target_w = max(left_row.shape[1], right_row.shape[1])
        return np.vstack([_resize_to_width(left_row, target_w), _resize_to_width(right_row, target_w)])

    pieces = [_one(i) for i in range(n_sensors)]
    return np.concatenate(pieces, axis=1)


def render_force_field_heatmap(
    nf_hw: np.ndarray,
    sf_hw2: np.ndarray,
    *,
    cell_px: int = 12,
    eps: float = 1e-6,
) -> np.ndarray:
    """Smooth heatmap renderer (percentile scaling, no per-frame threshold tuning)."""
    nf = np.asarray(nf_hw, dtype=np.float32)
    if nf.ndim == 3 and nf.shape[-1] == 1:
        nf = nf[..., 0]
    sf = np.asarray(sf_hw2, dtype=np.float32)
    nf_scale = float(np.percentile(np.abs(nf), 99.0)) + eps
    sf_scale = float(np.percentile(np.linalg.norm(sf, axis=-1), 99.0)) + eps
    n = np.clip(nf / nf_scale, 0.0, 1.0)
    sx = np.clip(sf[..., 0] / sf_scale, -1.0, 1.0)
    sy = np.clip(sf[..., 1] / sf_scale, -1.0, 1.0)
    r = 0.5 + 0.5 * sx
    g = 0.5 + 0.5 * sy
    b = n
    img = np.stack([r, g, b], axis=-1)
    img = img * (0.3 + 0.7 * n[..., None])
    nrows, ncols = nf.shape[:2]
    up = max(1, int(cell_px))
    img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.resize(img_u8, (ncols * up, nrows * up), interpolation=cv2.INTER_NEAREST)


def concat_tactile_rgb(tactile_rgb: np.ndarray, *, dual_arm: bool | None = None) -> np.ndarray:
    """Concatenate fingertip RGB. Dual-arm NPZ (10 sensors): left row + right row."""
    x = np.asarray(tactile_rgb)
    if x.ndim != 4:
        return np.concatenate(tactile_rgb, axis=1)
    n_sensors = int(x.shape[0])
    if dual_arm is None:
        dual_arm = n_sensors == 10
    if dual_arm and n_sensors == 10:
        left_row = np.concatenate([x[i] for i in range(5)], axis=1)
        right_row = np.concatenate([x[i] for i in range(5, 10)], axis=1)
        target_w = max(left_row.shape[1], right_row.shape[1])
        return np.vstack([_resize_to_width(left_row, target_w), _resize_to_width(right_row, target_w)])
    if n_sensors > 1:
        return np.concatenate([x[i] for i in range(n_sensors)], axis=1)
    return np.concatenate(tactile_rgb, axis=1)


def enhance_tactile_rgb_display(
    rgb: np.ndarray,
    *,
    baseline: np.ndarray | None,
    mode: Literal["raw", "diff", "diff_boost"],
    boost: float,
) -> np.ndarray:
    x = _ensure_uint8_rgb(rgb).astype(np.float32)
    if mode == "raw":
        return x.astype(np.uint8)
    ref = _ensure_uint8_rgb(baseline if baseline is not None else rgb).astype(np.float32)
    diff = np.abs(x - ref)
    if mode == "diff_boost":
        diff = np.clip(diff * float(boost), 0.0, 255.0)
    if diff.ndim == 3 and diff.shape[-1] == 3:
        diff = diff.mean(axis=-1, keepdims=True)
        diff = np.repeat(diff, 3, axis=-1)
    return diff.astype(np.uint8)


def _squeeze_normal(nf: np.ndarray) -> np.ndarray:
    nf = np.asarray(nf, dtype=np.float32)
    if nf.ndim == 5 and nf.shape[-1] == 1:
        nf = nf[..., 0]
    if nf.ndim == 3 and nf.shape[-1] == 1:
        nf = nf[..., 0]
    return nf


def _resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def _align_even(value: int) -> int:
    return max(2, int(value) // 2 * 2)


def _fit_for_video(img: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Downscale large composite frames so ffmpeg stays stable."""
    x = _ensure_uint8_rgb(img)
    h, w = x.shape[:2]
    if max_width <= 0 and max_height <= 0:
        return x
    scale = 1.0
    if max_width > 0 and w > max_width:
        scale = min(scale, max_width / float(w))
    if max_height > 0 and h > max_height:
        scale = min(scale, max_height / float(h))
    if scale >= 1.0:
        out = x
    else:
        new_w = max(2, int(round(w * scale)))
        new_h = max(2, int(round(h * scale)))
        out = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ew, eh = _align_even(out.shape[1]), _align_even(out.shape[0])
    if (ew, eh) == (out.shape[1], out.shape[0]):
        return out
    return cv2.resize(out, (ew, eh), interpolation=cv2.INTER_AREA)


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x.astype(np.float32), 0.0, 255.0 if x.max() > 1.0 else 1.0)
        if x.max() <= 1.0:
            x = x * 255.0
        x = x.astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    return x


def _auto_force_thresholds(
    nf: np.ndarray,
    sf: np.ndarray,
    *,
    normal_thr: float,
    shear_thr: float,
    auto_scale: bool,
) -> tuple[float, float]:
    if not auto_scale:
        return normal_thr, shear_thr
    nf2 = _squeeze_normal(nf)
    peak_n = max(float(np.percentile(np.abs(nf2), 99.0)), 1e-9)
    peak_s = max(float(np.percentile(np.linalg.norm(sf, axis=-1), 99.0)), 1e-9)
    return peak_n, peak_s


def frame_force_field(
    nf_t: np.ndarray,
    sf_t: np.ndarray,
    normal_thr: float,
    shear_thr: float,
    resolution: int,
    *,
    ff_mode: Literal["arrows", "heatmap"],
    ff_cell_px: int = 12,
    dual_arm: bool | None = None,
) -> np.ndarray:
    nf = _squeeze_normal(nf_t)
    sf = np.asarray(sf_t, dtype=np.float32)
    n_sensors = int(nf.shape[0])
    if dual_arm is None:
        dual_arm = n_sensors == 10

    def _render_one(finger_idx: int) -> np.ndarray:
        if ff_mode == "heatmap":
            return render_force_field_heatmap(nf[finger_idx], sf[finger_idx], cell_px=ff_cell_px)
        peak_n = max(float(np.abs(nf[finger_idx]).max()), normal_thr, 1e-9)
        peak_s = max(float(np.linalg.norm(sf[finger_idx], axis=-1).max()), shear_thr, 1e-9)
        img = compute_tactile_shear_image(
            nf[finger_idx],
            sf[finger_idx],
            normal_force_threshold=peak_n,
            shear_force_threshold=peak_s,
            resolution=resolution,
        )
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    if dual_arm and n_sensors == 10:
        left_pieces = [_render_one(i) for i in range(5)]
        right_pieces = [_render_one(i) for i in range(5, 10)]
        left_row = np.concatenate(left_pieces, axis=1)
        right_row = np.concatenate(right_pieces, axis=1)
        target_w = max(left_row.shape[1], right_row.shape[1])
        return np.vstack([_resize_to_width(left_row, target_w), _resize_to_width(right_row, target_w)])

    pieces = [_render_one(f) for f in range(n_sensors)]
    return np.concatenate(pieces, axis=1)


def _resize_to_min_height(img: np.ndarray, min_h: int) -> np.ndarray:
    x = _ensure_uint8_rgb(img)
    h, w = x.shape[:2]
    if h >= min_h:
        return x
    scale = min_h / float(max(1, h))
    new_w = max(2, int(round(w * scale)))
    return cv2.resize(x, (new_w, min_h), interpolation=cv2.INTER_AREA)


def make_tactile_only_panel(
    tactile: np.ndarray,
    ff: np.ndarray,
    *,
    panel_max_width: int,
    panel_max_height: int,
    min_tactile_h: int = 220,
    min_ff_h: int = 220,
) -> np.ndarray:
    """Stack GelSight RGB (hue/diff) over force-field when no third-person camera is recorded."""
    tactile = _ensure_uint8_rgb(tactile)
    ff = _ensure_uint8_rgb(ff)
    target_w = max(tactile.shape[1], ff.shape[1])
    if panel_max_width > 0:
        target_w = min(target_w, panel_max_width)
    panel = np.vstack(
        [
            _resize_to_min_height(_resize_to_width(tactile, target_w), min_tactile_h),
            _resize_to_min_height(_resize_to_width(ff, target_w), min_ff_h),
        ]
    )
    return _fit_for_video(panel, panel_max_width, panel_max_height)


def make_panel(
    third: np.ndarray,
    tactile: np.ndarray,
    ff: np.ndarray,
    *,
    panel_max_width: int,
    panel_max_height: int,
    min_third_h: int = 360,
    min_tactile_h: int = 180,
    min_ff_h: int = 180,
) -> np.ndarray:
    third = _ensure_uint8_rgb(third)
    tactile = _ensure_uint8_rgb(tactile)
    ff = _ensure_uint8_rgb(ff)
    target_w = max(third.shape[1], tactile.shape[1], ff.shape[1])
    if panel_max_width > 0:
        target_w = min(target_w, panel_max_width)
    panel = np.vstack(
        [
            _resize_to_min_height(_resize_to_width(third, target_w), min_third_h),
            _resize_to_min_height(_resize_to_width(tactile, target_w), min_tactile_h),
            _resize_to_min_height(_resize_to_width(ff, target_w), min_ff_h),
        ]
    )
    return _fit_for_video(panel, panel_max_width, panel_max_height)


def _find_npz(path: str) -> list[str]:
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p) and p.endswith(".npz"):
        return [p]
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.npz")))
        if files:
            return files
    raise FileNotFoundError(f"No .npz found under: {path}")


def export_episode(
    npz_path: str,
    out_dir: str,
    *,
    fps: float,
    normal_thr: float,
    shear_thr: float,
    ff_resolution: int,
    make_panel_video: bool,
    auto_scale: bool,
    ff_mode: Literal["arrows", "heatmap"],
    rgb_mode: Literal["raw", "diff", "diff_boost"],
    rgb_boost: float,
    panel_max_width: int,
    panel_max_height: int,
    ff_cell_px: int,
) -> dict[str, str]:
    data = np.load(npz_path, allow_pickle=True)
    if "joint_pos" not in data.files:
        raise ValueError(f"{npz_path}: missing joint_pos")
    n_frames = int(data["joint_pos"].shape[0])

    third_person = data["third_person_camera"] if "third_person_camera" in data.files else None
    twist = data["twist_camera"] if "twist_camera" in data.files else None
    if third_person is not None and twist is not None:
        print("[INFO] panel top rows: third_person (overview) + twist_camera (cap close-up).")
    elif twist is not None:
        print("[INFO] panel top row: twist_camera only.")
    elif third_person is not None:
        print("[INFO] panel top row: third_person only.")
    tactile_rgb = data["tactile_rgb_image"] if "tactile_rgb_image" in data.files else None
    has_ff = "tactile_normal_force" in data.files and "tactile_shear_force" in data.files
    nf = data["tactile_normal_force"] if has_ff else None
    sf = data["tactile_shear_force"] if has_ff else None

    if has_ff and nf is not None and sf is not None:
        normal_thr, shear_thr = _auto_force_thresholds(
            nf, sf, normal_thr=normal_thr, shear_thr=shear_thr, auto_scale=auto_scale
        )
        force_peak = float(np.max(np.abs(_squeeze_normal(nf))))
        use_rgb_ff_fallback = force_peak <= 1e-12 and tactile_rgb is not None
        if use_rgb_ff_fallback:
            print("[INFO] tactile_normal_force is all zero; using tactile RGB activation heatmap for force-field strip.")
        else:
            print(f"[INFO] force-field scale: normal_thr={normal_thr:.6g}, shear_thr={shear_thr:.6g}, mode={ff_mode}")
    else:
        use_rgb_ff_fallback = False

    rgb_baseline = tactile_rgb[0] if tactile_rgb is not None else None
    dual_arm = tactile_rgb is not None and np.asarray(tactile_rgb).ndim >= 2 and int(np.asarray(tactile_rgb).shape[1]) == 10

    stem = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    outputs: dict[str, str] = {}
    writers: dict[str, Any] = {}

    def _writer(key: str, suffix: str):
        if key not in writers:
            path = os.path.join(out_dir, f"{stem}_{suffix}.mp4")
            writers[key] = (
                imageio.get_writer(path, fps=fps, macro_block_size=1),
                path,
            )
        return writers[key][0]

    for i in range(n_frames):
        top_row = None
        if third_person is not None and twist is not None:
            tp = _ensure_uint8_rgb(third_person[i])
            tw = _ensure_uint8_rgb(twist[i])
            target_w = max(tp.shape[1], tw.shape[1])
            top_row = np.vstack([_resize_to_width(tp, target_w), _resize_to_width(tw, target_w)])
        elif twist is not None:
            top_row = _ensure_uint8_rgb(twist[i])
        elif third_person is not None:
            top_row = _ensure_uint8_rgb(third_person[i])

        if top_row is not None:
            _writer("third", "third_person").append_data(top_row)
        tactile_frame = None
        if tactile_rgb is not None:
            tactile_frame = enhance_tactile_rgb_display(
                concat_tactile_rgb(tactile_rgb[i], dual_arm=dual_arm),
                baseline=concat_tactile_rgb(rgb_baseline, dual_arm=dual_arm) if rgb_baseline is not None else None,
                mode=rgb_mode,
                boost=rgb_boost,
            )
            _writer("tactile", "tactile_rgb").append_data(tactile_frame)
        if has_ff and nf is not None and sf is not None:
            if use_rgb_ff_fallback and tactile_rgb is not None:
                ff_frame = frame_tactile_rgb_activation(
                    tactile_rgb[i],
                    rgb_baseline,
                    dual_arm=dual_arm,
                    ff_cell_px=ff_cell_px,
                )
            else:
                ff_frame = frame_force_field(
                    nf[i],
                    sf[i],
                    normal_thr,
                    shear_thr,
                    ff_resolution,
                    ff_mode=ff_mode,
                    ff_cell_px=ff_cell_px,
                    dual_arm=dual_arm,
                )
            _writer("ff", "tactile_force_field").append_data(ff_frame)
            if make_panel_video and tactile_frame is not None:
                if top_row is not None:
                    panel = make_panel(
                        top_row,
                        tactile_frame,
                        ff_frame,
                        panel_max_width=panel_max_width,
                        panel_max_height=panel_max_height,
                        min_third_h=280,
                    )
                else:
                    panel = make_tactile_only_panel(
                        tactile_frame,
                        ff_frame,
                        panel_max_width=panel_max_width,
                        panel_max_height=panel_max_height,
                    )
                _writer("panel", "demo_panel").append_data(panel)

    for key, (w, path) in writers.items():
        w.close()
        outputs[key] = path
        print(f"[OK] {key}: {path} ({n_frames} frames @ {fps} fps)")

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize play_record NPZ as mp4.")
    parser.add_argument("path", nargs="?", default="play_records", help="NPZ file or directory.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/demo_data_collection_replay/videos",
        help="Output directory for mp4 files.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument(
        "--normal_thr",
        type=float,
        default=0.00008,
        help="Arrow-mode normal threshold; ignored when --auto_scale (default) picks episode percentiles.",
    )
    parser.add_argument("--shear_thr", type=float, default=0.0005)
    parser.add_argument("--ff_resolution", type=int, default=30)
    parser.add_argument(
        "--ff_mode",
        choices=("arrows", "heatmap"),
        default="heatmap",
        help="Force-field renderer. heatmap is smoother for V2 corrected forces.",
    )
    parser.add_argument(
        "--rgb_mode",
        choices=("raw", "diff", "diff_boost"),
        default="diff_boost",
        help="Tactile RGB display. diff_boost amplifies small GelSight changes vs frame-0 baseline.",
    )
    parser.add_argument("--rgb_boost", type=float, default=8.0, help="Multiplier for --rgb_mode diff_boost.")
    parser.add_argument(
        "--no_auto_scale",
        action="store_true",
        help="Keep legacy fixed thresholds (only for old SDF-scale force fields).",
    )
    parser.add_argument("--no_panel", action="store_true", help="Skip combined demo_panel.mp4 export.")
    parser.add_argument(
        "--panel_max_width",
        type=int,
        default=1920,
        help="Max width for demo_panel.mp4 (prevents ffmpeg broken pipe on huge frames).",
    )
    parser.add_argument(
        "--panel_max_height",
        type=int,
        default=2400,
        help="Max height for demo_panel.mp4.",
    )
    parser.add_argument(
        "--ff_cell_px",
        type=int,
        default=12,
        help="Heatmap upsample pixels per force-grid cell.",
    )
    args = parser.parse_args()

    files = _find_npz(args.path)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    for f in files:
        print(f"[INFO] exporting {f}")
        export_episode(
            f,
            out_dir,
            fps=args.fps,
            normal_thr=args.normal_thr,
            shear_thr=args.shear_thr,
            ff_resolution=args.ff_resolution,
            make_panel_video=not args.no_panel,
            auto_scale=not args.no_auto_scale,
            ff_mode=args.ff_mode,
            rgb_mode=args.rgb_mode,
            rgb_boost=args.rgb_boost,
            panel_max_width=args.panel_max_width,
            panel_max_height=args.panel_max_height,
            ff_cell_px=args.ff_cell_px,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
