#!/usr/bin/env python3
"""Check saved play_record data: list keys, shapes, dtypes, validate format, and show data categories."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch


# Expected key categories produced by play_record.py
POLICY_KEYS = {"obs", "actions", "rewards", "dones"}
TACTILE_KEYS = {"tactile_normal_force", "tactile_shear_force", "tactile_rgb_image"}
CAMERA_PREFIX = "camera_rgb_"


def _get_array(v) -> np.ndarray:
    """Convert tensor or array to numpy for unified handling."""
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v)


def _load_as_dict(path: str) -> dict[str, np.ndarray] | None:
    """Load .pt / .h5 / .npz into a single dict of name -> numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            return None
        return {k: _get_array(v) for k, v in data.items() if isinstance(v, (torch.Tensor, np.ndarray))}
    if ext == ".h5":
        try:
            import h5py
        except ImportError:
            return None
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k][()]) for k in f.keys()}
    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: np.array(data[k]) for k in data.files}
    return None


def _categorize_keys(keys: set[str]) -> dict[str, list[str]]:
    """Group keys into policy / viewport / tactile / camera / other."""
    categories = defaultdict(list)
    for k in sorted(keys):
        if k in POLICY_KEYS:
            categories["policy"].append(k)
        elif k == "viewport_rgb":
            categories["viewport"].append(k)
        elif k in TACTILE_KEYS:
            categories["tactile"].append(k)
        elif k.startswith(CAMERA_PREFIX):
            categories["camera"].append(k)
        else:
            categories["other"].append(k)
    return dict(categories)


def _format_value_range(arr: np.ndarray) -> str:
    """Summarize value range for stats line."""
    if arr.size == 0:
        return "size=0"
    if arr.dtype.kind == "O":
        return "dtype=object (invalid)"
    if np.issubdtype(arr.dtype, np.number):
        return f"min={np.min(arr):.6g}, max={np.max(arr):.6g}"
    return ""


def _infer_image_info(arr: np.ndarray) -> str:
    """If shape looks like (T, H, W, C) or (T, flat), add a short note."""
    if arr.ndim < 2:
        return ""
    t = arr.shape[0]
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        return f"  [image: T={t}, H={arr.shape[1]}, W={arr.shape[2]}, C={arr.shape[3]}]"
    if arr.ndim == 2:
        return f"  [flattened image: T={t}, dim={arr.shape[1]}]"
    return ""


def _check_format(data: dict[str, np.ndarray]) -> list[str]:
    """Validate format: same T on dim 0, no object dtype, no 0-d. Returns list of issue messages."""
    issues = []
    t_values = []
    for k, arr in data.items():
        if arr.dtype.kind == "O":
            issues.append(f"  [INVALID] '{k}': dtype is object (should be numeric or image dtype).")
        if arr.ndim == 0:
            issues.append(f"  [INVALID] '{k}': 0-d array (should have shape (T, ...)).")
        if arr.ndim >= 1:
            t_values.append((k, arr.shape[0]))
    if not t_values:
        return issues
    t_ref = t_values[0][1]
    for k, t in t_values:
        if t != t_ref:
            issues.append(f"  [MISMATCH] '{k}': first dim={t}, expected T={t_ref} (all keys should have same num_steps).")
    return issues


def _print_report(path: str, data: dict[str, np.ndarray], file_type: str) -> None:
    """Print data categories, per-key info, and format validation."""
    print(f"\n{'='*60}")
    print(f"  {file_type}: {path}")
    print("=" * 60)

    if not data:
        print("  (no dict or empty)")
        return

    categories = _categorize_keys(set(data.keys()))
    print("\n[ 数据种类 Data categories ]")
    for cat, keys in categories.items():
        print(f"  {cat}: {keys}")

    print("\n[ 各键形状与类型 Per-key shape & dtype ]")
    for k in sorted(data.keys()):
        arr = data[k]
        line = f"  {k}: shape={arr.shape}, dtype={arr.dtype}"
        print(line)
        stats = _format_value_range(arr)
        if stats:
            print(f"       {stats}")
        img_note = _infer_image_info(arr)
        if img_note:
            print(img_note)

    print("\n[ 格式校验 Format validation ]")
    issues = _check_format(data)
    if not issues:
        print("  OK: 所有键首维一致，无非 object/0-d 数组。")
    else:
        for msg in issues:
            print(msg)

    # Expected keys hint
    expected = POLICY_KEYS
    missing = expected - set(data.keys())
    if missing:
        print(f"\n  [hint] 缺少 policy 键 missing policy keys: {sorted(missing)}")
    else:
        print(f"\n  [hint] policy 键齐全 (obs, actions, rewards, dones).")


def check_pt(path: str) -> None:
    """Inspect a .pt file and run unified report."""
    data = _load_as_dict(path)
    if data is None:
        print(f"\n--- .pt: {path} ---")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  Root type: {type(raw)} (not a dict of tensors)")
        return
    _print_report(path, data, ".pt")


def check_h5(path: str) -> None:
    """Inspect a .h5 file and run unified report."""
    if not path.endswith(".h5"):
        return
    try:
        import h5py
    except ImportError:
        print(f"\n--- .h5: {path} ---")
        print("  (install h5py to inspect .h5)")
        return
    data = _load_as_dict(path)
    if data is None:
        print(f"\n--- .h5: {path} ---")
        print("  (could not load as dict of arrays)")
        return
    _print_report(path, data, ".h5")


def check_npz(path: str) -> None:
    """Inspect a .npz file and run unified report."""
    if not path.endswith(".npz"):
        return
    data = _load_as_dict(path)
    if data is None:
        print(f"\n--- .npz: {path} ---")
        print("  (could not load as dict of arrays)")
        return
    _print_report(path, data, ".npz")


def main():
    parser = argparse.ArgumentParser(
        description="Check saved play_record data: list keys, shapes, dtypes, and validate format."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to .pt, .h5, .npz file or directory containing them.",
    )
    args = parser.parse_args()
    path = os.path.abspath(args.path)

    if os.path.isfile(path):
        if path.endswith(".pt"):
            check_pt(path)
        elif path.endswith(".h5"):
            check_h5(path)
        elif path.endswith(".npz"):
            check_npz(path)
        else:
            print("Unknown extension. Use .pt, .h5, or .npz")
            sys.exit(1)
    elif os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            p = os.path.join(path, name)
            if not os.path.isfile(p):
                continue
            if name.endswith(".pt"):
                check_pt(p)
            elif name.endswith(".h5"):
                check_h5(p)
            elif name.endswith(".npz"):
                check_npz(p)
    else:
        print(f"Not a file or directory: {path}")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
