#!/usr/bin/env python3
"""Replay and visualize recorded UR10e+ShadowHand data.

This script loads per-step record files created by:
  scripts/debug/run_ur10e_shadowhand_single.py --record_path ...

It can visualize:
- TacSL tactile RGB images (5 fingertips)
- Tactile force-field (FF) image rendered from normal/shear arrays
- Third-person camera RGB

Supported formats: .pt (torch.save) and .npz (flattened keys).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
from typing import Any

import numpy as np
import torch


TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if x.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if x.max() <= 1.0:
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _render_tactile_ff_rgb(nf: np.ndarray, sf: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Render tactile normal/shear arrays into an RGB image (no Isaac Sim dependency).

    Args:
        nf: (H, W) normal force
        sf: (H, W, 2) shear force (x, y)
    Returns:
        (H, W, 3) uint8 image in [0, 255]
    """
    nf = np.asarray(nf, dtype=np.float32)
    sf = np.asarray(sf, dtype=np.float32)
    if nf.ndim != 2 or sf.ndim != 3 or sf.shape[-1] != 2:
        raise ValueError(f"Invalid shapes for ff render: nf={nf.shape}, sf={sf.shape}")

    # Robust scaling: normalize by high percentile to reduce outliers.
    nf_scale = np.percentile(np.abs(nf), 99.0) + eps
    sf_scale = np.percentile(np.linalg.norm(sf, axis=-1), 99.0) + eps

    n = np.clip(nf / nf_scale, 0.0, 1.0)
    sx = np.clip(sf[..., 0] / sf_scale, -1.0, 1.0)
    sy = np.clip(sf[..., 1] / sf_scale, -1.0, 1.0)

    # Map shear direction to R/G, normal to B (and also brighten overall).
    r = 0.5 + 0.5 * sx
    g = 0.5 + 0.5 * sy
    b = n
    img = np.stack([r, g, b], axis=-1)

    # Modulate by normal to emphasize contact regions.
    img = img * (0.3 + 0.7 * n[..., None])
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_numpy(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _npz_get(npz: dict[str, Any], key: str) -> Any | None:
    # saved keys are like "robot/joint_pos"
    return npz.get(key, None)


def _load_record(path: str) -> dict[str, Any]:
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=False)
        # return as a plain dict[str, np.ndarray]
        return {k: data[k] for k in data.files}
    raise ValueError(f"Unsupported record extension for: {path}")


def _extract_step_from_name(path: str) -> int:
    m = re.search(r"_step_(\d+)\.", os.path.basename(path))
    if m:
        return int(m.group(1))
    return -1


def _collect_files(inputs: list[str]) -> list[str]:
    files: list[str] = []
    for inp in inputs:
        p = os.path.expanduser(inp)
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.pt")))
            files.extend(glob.glob(os.path.join(p, "*.npz")))
        else:
            files.extend(glob.glob(p))
    files = [f for f in files if f.endswith((".pt", ".npz"))]
    files.sort(key=lambda x: (_extract_step_from_name(x), x))
    return files


def _get_tactile_entry(rec: dict[str, Any], sensor_name: str, field: str) -> Any | None:
    # .pt format: rec["sensors"]["tactile"][sensor_name][field]
    # .npz format: key like "sensors/tactile/tactile_sensor_ff/rgb"
    if "sensors" in rec and isinstance(rec.get("sensors"), dict):
        tactile = rec["sensors"].get("tactile", None)
        if isinstance(tactile, dict) and sensor_name in tactile:
            return tactile[sensor_name].get(field, None)
    k = f"sensors/tactile/{sensor_name}/{field}"
    return _npz_get(rec, k)


def _get_third_person_rgb(rec: dict[str, Any]) -> Any | None:
    if "sensors" in rec and isinstance(rec.get("sensors"), dict):
        cam = rec["sensors"].get("third_person_camera", None)
        if isinstance(cam, dict):
            return cam.get("rgb", None)
    return _npz_get(rec, "sensors/third_person_camera/rgb")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize recorded tactile + third-person camera data.")
    p.add_argument(
        "inputs",
        nargs="+",
        help="Record directory, glob, or file(s). Example: /tmp/pickup_rec/ or /tmp/pickup_rec/run1_step_*.pt",
    )
    p.add_argument("--fps", type=float, default=20.0, help="Playback FPS (default: 20).")
    p.add_argument("--index", type=int, default=0, help="Start file index (default: 0).")
    p.add_argument("--max_frames", type=int, default=0, help="If >0, stop after N frames.")
    p.add_argument("--show_rgb", action="store_true", help="Show tactile RGB images.")
    p.add_argument("--show_ff", action="store_true", help="Show tactile force-field images.")
    p.add_argument("--show_third", action="store_true", help="Show third-person RGB image.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    files = _collect_files(list(args.inputs))
    if not files:
        raise FileNotFoundError(f"No record files found from inputs: {args.inputs}")

    # default: show all if nothing specified
    if not (args.show_rgb or args.show_ff or args.show_third):
        args.show_rgb = True
        args.show_ff = True
        args.show_third = True

    # NOTE: Do NOT import isaaclab_contrib/isaacsim here. This script should be usable
    # outside Isaac Sim for offline replay. We use a local renderer instead.
    compute_tactile_shear_image = _render_tactile_ff_rgb if args.show_ff else None

    import matplotlib

    matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
    import matplotlib.pyplot as plt

    # Create figure layout
    nrows = 2 if (args.show_rgb and args.show_ff) else 1
    ncols = 5
    has_grid = args.show_rgb or args.show_ff
    has_third = args.show_third
    if has_grid and has_third:
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=[1, 1, 1, 1, 1, 2.0])
        axes_grid = []
        for r in range(nrows):
            row_axes = [fig.add_subplot(gs[r, c]) for c in range(ncols)]
            axes_grid.append(row_axes)
        # unify to 2D array for (row, col) indexing
        axes_grid = np.asarray(axes_grid, dtype=object)
        ax_third = fig.add_subplot(gs[:, -1])
    elif has_grid:
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6 if nrows == 2 else 3))
        axes_grid = axes if isinstance(axes, np.ndarray) else np.asarray([axes])
        axes_grid = axes_grid.reshape(nrows, ncols)
        ax_third = None
    else:
        fig, ax_third = plt.subplots(1, 1, figsize=(8, 6))
        axes_grid = None

    # Placeholder artists
    rgb_ims: list[Any] = []
    ff_ims: list[Any] = []
    third_im: Any | None = None

    if axes_grid is not None:
        if args.show_rgb:
            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                ax = axes_grid[0, i] if (args.show_rgb and args.show_ff) else axes_grid[0, i]
                im = ax.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
                ax.set_title(name.replace("tactile_sensor_", "").upper() + " RGB")
                ax.axis("off")
                rgb_ims.append(im)
        if args.show_ff:
            row = 1 if (args.show_rgb and args.show_ff) else 0
            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                ax = axes_grid[row, i]
                im = ax.imshow(np.zeros((20 * 30, 25 * 30, 3), dtype=np.uint8))
                ax.set_title(name.replace("tactile_sensor_", "").upper() + " FF")
                ax.axis("off")
                ff_ims.append(im)

    if ax_third is not None and args.show_third:
        third_im = ax_third.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        ax_third.set_title("Third-person RGB")
        ax_third.axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    plt.pause(0.1)

    start_idx = max(0, min(int(args.index), len(files) - 1))
    max_frames = int(args.max_frames)
    target_dt = 1.0 / max(1e-3, float(args.fps))

    print(f"Found {len(files)} record files. Starting at index {start_idx}.")
    for j, path in enumerate(files[start_idx:], start=0):
        t0 = time.time()
        rec = _load_record(path)

        # tactile per-finger
        if axes_grid is not None:
            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if args.show_rgb and i < len(rgb_ims):
                    rgb = _get_tactile_entry(rec, name, "rgb")
                    rgb_np = _to_numpy(rgb)
                    if rgb_np is not None and rgb_np.ndim >= 3:
                        # recorded with env_ids=[k] => first dim is 1
                        img = rgb_np[0] if rgb_np.ndim == 4 else rgb_np
                        rgb_ims[i].set_data(_img_to_uint8(img))
                if args.show_ff and i < len(ff_ims) and compute_tactile_shear_image is not None:
                    nf = _get_tactile_entry(rec, name, "normal_force")
                    sf = _get_tactile_entry(rec, name, "shear_force")
                    nf_np = _to_numpy(nf)
                    sf_np = _to_numpy(sf)
                    if nf_np is not None and sf_np is not None:
                        nf0 = nf_np[0]
                        sf0 = sf_np[0]
                        # infer grid shape from array length
                        p = int(nf0.shape[-1])
                        nrows_guess, ncols_guess = 20, 25
                        if p != nrows_guess * ncols_guess:
                            # try to make a near-square guess (fallback)
                            nrows_guess = int(np.sqrt(p))
                            ncols_guess = max(1, p // max(1, nrows_guess))
                        nf_img = nf0.reshape(nrows_guess, ncols_guess)
                        sf_img = sf0.reshape(nrows_guess, ncols_guess, 2)
                        ff = compute_tactile_shear_image(nf_img, sf_img)
                        ff_ims[i].set_data(_img_to_uint8(ff))

        # third-person camera
        if third_im is not None and args.show_third:
            cam_rgb = _get_third_person_rgb(rec)
            cam_np = _to_numpy(cam_rgb)
            if cam_np is not None and cam_np.ndim >= 3:
                img = cam_np[0] if cam_np.ndim == 4 else cam_np
                third_im.set_data(_img_to_uint8(img))

        fig.suptitle(os.path.basename(path), fontsize=10)
        fig.canvas.draw_idle()
        plt.pause(0.001)

        if max_frames > 0 and (j + 1) >= max_frames:
            break

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    plt.close("all")


if __name__ == "__main__":
    main()

