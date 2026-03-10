#!/usr/bin/env python3
"""View camera and tactile images from saved play_record data frame-by-frame.

Supports current play_record.py format:
  - .pt: list of episode dicts
  - .h5: groups episode_0, episode_1, ...
  - .npz: episode_0_*, episode_1_*, ...
Use --episode to choose which episode to view. Tactile images (tactile_rgb_image) are
shown as left/right sensors (flat 460800 format) or as tactile_0, tactile_1, ... (T,N,H,W,3).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

# Tactile image layout (forge_env_cfg: 2 sensors * 240*320*3)
TACTILE_H, TACTILE_W = 240, 320
TACTILE_NUM_SENSORS = 2
TACTILE_FLAT_PER_SENSOR = TACTILE_H * TACTILE_W * 3  # 230400
TACTILE_FLAT_TOTAL = TACTILE_NUM_SENSORS * TACTILE_FLAT_PER_SENSOR  # 460800


def _get_array(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v)


def _episode_dict_to_flat(ep: dict) -> dict[str, np.ndarray]:
    """Convert one episode dict to flat dict of numpy arrays."""
    out = {}
    for k, v in ep.items():
        if k == "camera_rgb" and isinstance(v, dict):
            for cam_key, arr in v.items():
                out[cam_key] = _get_array(arr)
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            out[k] = _get_array(v)
    return out


def load_episode_data(path: str, episode_idx: int = 0) -> tuple[dict[str, np.ndarray], int]:
    """Load one episode from .pt / .h5 / .npz (play_record format). Returns (flat dict for that episode, num_episodes)."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            num_ep = len(raw)
            idx = min(episode_idx, num_ep - 1)
            return _episode_dict_to_flat(raw[idx]), num_ep
        if isinstance(raw, dict):
            return _episode_dict_to_flat(raw), 1
        raise ValueError(f"Unknown .pt structure: {type(raw)}")

    if ext == ".h5":
        import h5py
        with h5py.File(path, "r") as f:
            episode_groups = sorted([k for k in f.keys() if re.match(r"episode_\d+", k)])
            if episode_groups:
                num_ep = len(episode_groups)
                idx = min(episode_idx, num_ep - 1)
                grp = f[episode_groups[idx]]
                data = {k: np.array(grp[k][()]) for k in grp.keys()}
                return data, num_ep
            data = {k: np.array(f[k][()]) for k in f.keys()}
            return data, 1

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as z:
            files = list(z.files)
            all_arrays = {k: np.array(z[k]) for k in files}
        ep_pattern = re.compile(r"^episode_(\d+)_(.+)$")
        by_episode = defaultdict(dict)
        for key in files:
            m = ep_pattern.match(key)
            if m:
                ep_idx, name = int(m.group(1)), m.group(2)
                by_episode[ep_idx][name] = all_arrays[key]
            else:
                by_episode[0][key] = all_arrays[key]
        if not by_episode:
            raise ValueError("No valid keys in .npz")
        max_ep = max(by_episode.keys())
        episodes = [by_episode[i] for i in range(max_ep + 1) if i in by_episode]
        num_ep = len(episodes)
        idx = min(episode_idx, num_ep - 1)
        return episodes[idx], num_ep

    raise ValueError(f"Unsupported extension: {ext}")


def get_image_keys(data: dict[str, np.ndarray]) -> list[str]:
    """Return keys that are viewport, camera, or tactile image data, in display order."""
    viewport = [k for k in ["viewport_rgb"] if k in data]
    camera = sorted(k for k in data.keys() if k.startswith("camera_rgb_"))
    tactile = [k for k in ["tactile_rgb_image"] if k in data]
    return viewport + camera + tactile


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype in (np.float32, np.float64):
        return (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
    return np.clip(img.astype(np.float64), 0, 255).astype(np.uint8)


def frame_to_display_images(
    data: dict[str, np.ndarray],
    image_keys: list[str],
    frame_idx: int,
    env_idx: int = 0,
) -> list[tuple[str, np.ndarray]]:
    """
    For a given frame and env index, return list of (title, rgb image) for display.
    Handles play_record format: tactile_rgb_image as (T, 460800) or (T, N, H, W, 3).
    """
    result = []
    for key in image_keys:
        arr = data[key]
        if arr.ndim < 2:
            continue
        T = arr.shape[0]
        if frame_idx < 0 or frame_idx >= T:
            continue

        if key == "tactile_rgb_image":
            # play_record saves flattened: (T, 460800) -> left 230400 + right 230400
            if arr.ndim == 2 and arr.shape[1] == TACTILE_FLAT_TOTAL:
                flat = arr[frame_idx]
                left = flat[:TACTILE_FLAT_PER_SENSOR].reshape(TACTILE_H, TACTILE_W, 3)
                right = flat[TACTILE_FLAT_PER_SENSOR:].reshape(TACTILE_H, TACTILE_W, 3)
                result.append(("tactile_left", _to_uint8(left)))
                result.append(("tactile_right", _to_uint8(right)))
                continue
            # (T, num_envs, 460800)
            if arr.ndim == 3 and arr.shape[2] == TACTILE_FLAT_TOTAL:
                n_env = arr.shape[1]
                e = min(env_idx, n_env - 1)
                flat = arr[frame_idx, e]
                left = flat[:TACTILE_FLAT_PER_SENSOR].reshape(TACTILE_H, TACTILE_W, 3)
                right = flat[TACTILE_FLAT_PER_SENSOR:].reshape(TACTILE_H, TACTILE_W, 3)
                result.append(("tactile_left", _to_uint8(left)))
                result.append(("tactile_right", _to_uint8(right)))
                continue
            # (T, N, H, W, 3) unflattened format (no env dim)
            if arr.ndim == 5:
                n_sensors = arr.shape[1]
                for s in range(n_sensors):
                    img = arr[frame_idx, s]
                    result.append((f"tactile_{s}", _to_uint8(img)))
                continue
            # (T, H, W, 3) single tactile sensor
            if arr.ndim == 4:
                result.append(("tactile", _to_uint8(arr[frame_idx])))
                continue
            continue

        # Camera / viewport: (T, H, W, 3) or (T, num_envs, H, W, 3)
        if arr.ndim == 5:
            n_env = arr.shape[1]
            e = min(env_idx, n_env - 1)
            img = arr[frame_idx, e]
        elif arr.ndim == 4:
            img = arr[frame_idx]
        else:
            continue
        img = _to_uint8(img)
        name = "viewport (Isaac Sim window)" if key == "viewport_rgb" else key.replace("camera_rgb_", "")
        result.append((name, img))
    return result


def run_matplotlib_viewer(
    data: dict[str, np.ndarray],
    path: str,
    episode_idx: int = 0,
    num_episodes: int = 1,
    env_idx: int = 0,
) -> None:
    """Interactive frame-by-frame viewer using matplotlib (slider + keyboard)."""
    import matplotlib
    matplotlib.use("TkAgg" if "DISPLAY" in os.environ else "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # Mutable viewer state (episode hot-switch supported)
    image_keys: list[str] = []
    num_frames = 0
    num_envs = 1
    frame_idx = 0
    current_env_idx = env_idx

    def compute_episode_stats(ep_data: dict[str, np.ndarray], keys: list[str]) -> tuple[int, int]:
        if not keys:
            return 0, 1
        first_key = keys[0]
        arr = ep_data[first_key]
        n_frames = int(arr.shape[0]) if arr.ndim >= 1 else 0
        n_envs = 1
        if first_key != "tactile_rgb_image":
            n_envs = int(arr.shape[1]) if arr.ndim == 5 else 1
        elif first_key == "tactile_rgb_image" and arr.ndim == 3:
            n_envs = int(arr.shape[1])
        return n_frames, max(1, n_envs)

    fig = plt.figure(figsize=(10, 7))
    axs_flat: list = []
    ax_slider = None
    slider: Slider | None = None

    def get_current_images():
        return frame_to_display_images(data, image_keys, frame_idx, current_env_idx)

    def draw():
        nonlocal axs_flat
        images = get_current_images()
        # Hide all by default; show used axes.
        for ax in axs_flat:
            ax.set_visible(False)
        for i, (title, img) in enumerate(images):
            if i >= len(axs_flat):
                break
            ax = axs_flat[i]
            ax.set_visible(True)
            ax.clear()
            ep_label = f" ep{episode_idx}/{num_episodes-1}" if num_episodes > 1 else ""
            h, w = img.shape[:2]
            ax.imshow(img, aspect="equal")
            ax.set_title(f"{title} ({w}×{h})  [frame {frame_idx}/{max(0, num_frames-1)}{ep_label}]")
            ax.axis("off")

        title = f"Play data: {os.path.basename(path)}"
        if num_episodes > 1:
            title += f"  (episode {episode_idx} of {num_episodes})"
        fig.suptitle(title, fontsize=10)
        fig.canvas.draw_idle()

    def ensure_layout(n_plots: int):
        """Ensure figure has enough axes for current episode."""
        nonlocal axs_flat, ax_slider, slider
        n_plots = max(1, int(n_plots))
        need_rebuild = len(axs_flat) < n_plots or slider is None
        if not need_rebuild:
            return

        fig.clf()
        ncol = min(3, n_plots)
        nrow = (n_plots + ncol - 1) // ncol
        axs = fig.subplots(nrow, ncol)
        axs_flat = np.atleast_1d(axs).flatten().tolist()
        plt.subplots_adjust(bottom=0.12, top=0.92)

        ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
        slider = Slider(ax_slider, "Frame", 0, max(0, num_frames - 1), valinit=0, valstep=1)

        def on_slider(_val):
            nonlocal frame_idx
            if slider is None:
                return
            frame_idx = int(slider.val)
            draw()

        slider.on_changed(on_slider)

    def update_slider_range():
        """Update slider limits to match current episode frame count."""
        if slider is None:
            return
        vmax = max(0, num_frames - 1)
        slider.valmin = 0
        slider.valmax = vmax
        # Keep current slider axis in sync.
        slider.ax.set_xlim(0, vmax if vmax > 0 else 1)

    def load_episode(new_episode_idx: int):
        """Load new episode in-place and refresh view."""
        nonlocal data, episode_idx, image_keys, num_frames, num_envs, frame_idx, current_env_idx
        if num_episodes <= 1:
            return
        new_episode_idx = int(np.clip(new_episode_idx, 0, num_episodes - 1))
        if new_episode_idx == episode_idx:
            return

        data, _ = load_episode_data(path, new_episode_idx)
        episode_idx = new_episode_idx
        image_keys = get_image_keys(data)
        if not image_keys:
            print("No camera or tactile_rgb_image keys found in this episode.")
            return

        num_frames, num_envs = compute_episode_stats(data, image_keys)
        frame_idx = 0
        current_env_idx = int(np.clip(current_env_idx, 0, max(0, num_envs - 1)))

        images0 = frame_to_display_images(data, image_keys, 0, current_env_idx)
        ensure_layout(len(images0))
        update_slider_range()
        if slider is not None:
            slider.set_val(0)
        draw()

    # Initialize first episode
    image_keys = get_image_keys(data)
    if not image_keys:
        print("No camera or tactile_rgb_image keys found in this file.")
        return
    num_frames, num_envs = compute_episode_stats(data, image_keys)
    images0 = frame_to_display_images(data, image_keys, 0, current_env_idx)
    ensure_layout(len(images0))

    def on_key(event):
        nonlocal frame_idx, current_env_idx
        k = event.key
        if k == "right" or k == "n":
            frame_idx = min(max(0, num_frames - 1), frame_idx + 1)
        elif k == "left" or k == "p":
            frame_idx = max(0, frame_idx - 1)
        elif k == "up" and num_envs > 1:
            current_env_idx = min(num_envs - 1, current_env_idx + 1)
        elif k == "down" and num_envs > 1:
            current_env_idx = max(0, current_env_idx - 1)
        elif (k in ("]", "pagedown")) and num_episodes > 1:
            load_episode(episode_idx + 1)
            return
        elif (k in ("[", "pageup")) and num_episodes > 1:
            load_episode(episode_idx - 1)
            return
        elif k == "home" and num_episodes > 1:
            load_episode(0)
            return
        elif k == "end" and num_episodes > 1:
            load_episode(num_episodes - 1)
            return
        else:
            return

        if slider is not None:
            slider.set_val(frame_idx)
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update_slider_range()
    draw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="View camera and tactile images from play_record data (supports multi-episode .pt/.h5/.npz).",
    )
    parser.add_argument("path", type=str, help="Path to saved .pt, .h5, or .npz file.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to view (default 0).")
    parser.add_argument("--env", type=int, default=0, help="Environment index when data has multiple envs (default 0).")
    args = parser.parse_args()
    path = os.path.abspath(args.path)

    try:
        data, num_episodes = load_episode_data(path, args.episode)
    except FileNotFoundError:
        print(f"Could not find: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

    image_keys = get_image_keys(data)
    if not image_keys:
        print("No camera or tactile_rgb_image keys in this file. Keys present:", list(data.keys()))
        sys.exit(1)

    print(f"Loaded {path}")
    print(f"Image keys: {image_keys}")
    for k in image_keys:
        arr = data[k]
        if hasattr(arr, "shape") and arr.ndim >= 2:
            print(f"  {k}: shape {arr.shape}")
    if num_episodes > 1:
        print(f"Viewing episode {args.episode} of {num_episodes} (use --episode N to switch).")
    if num_episodes > 1:
        print("Keyboard: Left/Right (or P/N) = prev/next frame; [ / ] (or PageUp/PageDown) = prev/next episode; Home/End = first/last episode.")
    else:
        print("Keyboard: Left/Right (or P/N) = previous/next frame. Close window to exit.")
    run_matplotlib_viewer(data, path, episode_idx=args.episode, num_episodes=num_episodes, env_idx=args.env)


if __name__ == "__main__":
    main()
