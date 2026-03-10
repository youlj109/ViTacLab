#!/usr/bin/env python3
"""Check saved play_record data: list keys, shapes, dtypes, validate format, and show data categories.

Supports the current play_record.py format:
  - .pt: list of episode dicts (each dict has actions, tactile_*, camera_rgb_*, etc. with shape (T_ep, ...))
  - .h5: groups episode_0, episode_1, ... with same per-episode structure
  - .npz: keys episode_0_actions, episode_0_tactile_rgb_image, ... (one episode per index)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

# 定义数据分类
OPTIONAL_POLICY_KEYS = {"obs", "rewards", "dones"}
MANDATORY_KEYS = {"actions"}
TACTILE_KEYS = {"tactile_normal_force", "tactile_shear_force", "tactile_rgb_image"}
CAMERA_PREFIX = "camera_rgb_"


def _get_array(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v)


def _episode_dict_to_flat(ep: dict) -> dict[str, np.ndarray]:
    """Convert one episode dict (tensors or arrays) to flat dict of numpy arrays."""
    out = {}
    for k, v in ep.items():
        if k == "camera_rgb" and isinstance(v, dict):
            for cam_key, arr in v.items():
                out[cam_key] = _get_array(arr)
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            out[k] = _get_array(v)
    return out


def _load_pt(path: str) -> tuple[list[dict[str, np.ndarray]], bool]:
    """Load .pt file. Returns (list of episode flat dicts, True if multi-episode format)."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        # play_record format: list of episode dicts
        episodes = [_episode_dict_to_flat(ep) for ep in raw]
        return episodes, True
    if isinstance(raw, dict):
        # legacy single flat dict
        return [_episode_dict_to_flat(raw)], False
    return [], False


def _load_h5(path: str) -> tuple[list[dict[str, np.ndarray]], bool]:
    """Load .h5 file. Returns (list of episode flat dicts, True if multi-episode format)."""
    import h5py
    with h5py.File(path, "r") as f:
        episode_groups = sorted([k for k in f.keys() if re.match(r"episode_\d+", k)])
        if episode_groups:
            episodes = []
            for g in episode_groups:
                grp = f[g]
                episodes.append({k: np.array(grp[k][()]) for k in grp.keys()})
            return episodes, True
        # flat keys at root
        flat = {k: np.array(f[k][()]) for k in f.keys()}
        return [flat], False


def _load_npz(path: str) -> tuple[list[dict[str, np.ndarray]], bool]:
    """Load .npz file. Returns (list of episode flat dicts, True if multi-episode format)."""
    with np.load(path, allow_pickle=True) as z:
        files = list(z.files)
        data = {k: np.array(z[k]) for k in files}
    ep_pattern = re.compile(r"^episode_(\d+)_(.+)$")
    by_episode = defaultdict(dict)
    for key in files:
        m = ep_pattern.match(key)
        if m:
            ep_idx, name = int(m.group(1)), m.group(2)
            by_episode[ep_idx][name] = data[key]
        else:
            by_episode[0][key] = data[key]
    if not by_episode:
        return [], False
    max_ep = max(by_episode.keys())
    episodes = [by_episode[i] for i in range(max_ep + 1) if i in by_episode]
    multi = len(episodes) > 1 or any(ep_pattern.match(k) for k in files)
    return episodes, multi


def load_play_record_file(path: str) -> tuple[list[dict[str, np.ndarray]], bool]:
    """Load file; return (list of episode dicts, is_multi_episode)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        return _load_pt(path)
    if ext == ".h5":
        return _load_h5(path)
    if ext == ".npz":
        return _load_npz(path)
    return [], False


def _categorize_keys(keys: set[str]) -> dict[str, list[str]]:
    categories = defaultdict(list)
    for k in sorted(keys):
        if k in MANDATORY_KEYS or k in OPTIONAL_POLICY_KEYS:
            categories["policy"].append(k)
        elif k in TACTILE_KEYS:
            categories["tactile"].append(k)
        elif k.startswith(CAMERA_PREFIX):
            categories["camera"].append(k)
        else:
            categories["other"].append(k)
    return dict(categories)


def _check_format(data: dict[str, np.ndarray]) -> list[str]:
    issues = []
    t_values = []
    for k, arr in data.items():
        if arr.dtype.kind == "O":
            issues.append(f"  [INVALID] '{k}': dtype is object.")
        if arr.ndim == 0:
            issues.append(f"  [INVALID] '{k}': 0-d array.")
        if arr.ndim >= 1:
            t_values.append((k, arr.shape[0]))
    if not t_values:
        return issues
    t_ref = t_values[0][1]
    for k, t in t_values:
        if t != t_ref:
            issues.append(f"  [MISMATCH] '{k}': dim0={t}, expected T={t_ref}.")
    return issues


def _print_report(path: str, data: dict[str, np.ndarray], file_type: str, episode_label: str = "") -> None:
    title = f"{file_type}: {path}"
    if episode_label:
        title += f"  ({episode_label})"
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    if not data:
        print("  (empty data)")
        return

    categories = _categorize_keys(set(data.keys()))
    print("\n[ 数据种类 Data categories ]")
    for cat, keys in categories.items():
        print(f"  {cat}: {keys}")

    print("\n[ 详细形状 Per-key shape & dtype ]")
    for k in sorted(data.keys()):
        arr = data[k]
        print(f"  {k:30} | shape={str(arr.shape):20} | dtype={arr.dtype}")

    print("\n[ 格式校验 ]")
    issues = _check_format(data)
    if not issues:
        print("  OK: 数据对齐，格式正确。")
    else:
        for msg in issues:
            print(msg)

    present = set(data.keys())
    saved_optional = present.intersection(OPTIONAL_POLICY_KEYS)
    print(f"\n  [info] 当前已保存的可选策略数据: {sorted(list(saved_optional)) if saved_optional else '无'}")


def main():
    parser = argparse.ArgumentParser(description="Check play_record data (supports multi-episode .pt / .h5 / .npz).")
    parser.add_argument("path", type=str, help="Path to file or directory.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to report when file has multiple (default 0).")
    parser.add_argument("--all", action="store_true", help="Print report for every episode (default: only --episode).")
    args = parser.parse_args()

    paths = [os.path.join(args.path, f) for f in os.listdir(args.path)] if os.path.isdir(args.path) else [args.path]
    for p in paths:
        if not p.endswith((".pt", ".h5", ".npz")):
            continue
        try:
            episodes, multi = load_play_record_file(p)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue
        if not episodes:
            print(f"Could not load or unknown format: {p}")
            continue

        ext = os.path.splitext(p)[1]
        if multi:
            print(f"\n[ {p} ]  play_record 多回合格式: {len(episodes)} episodes")
        if args.all:
            for i, data in enumerate(episodes):
                _print_report(p, data, ext, episode_label=f"episode_{i}")
        else:
            idx = min(args.episode, len(episodes) - 1)
            _print_report(p, episodes[idx], ext, episode_label=f"episode_{idx}" if multi else "")


if __name__ == "__main__":
    main()
