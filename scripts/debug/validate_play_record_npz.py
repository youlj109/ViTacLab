#!/usr/bin/env python3
"""Validate canonical play_single / play_record NPZ schema."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REQUIRED_KEYS = (
    "joint_pos",
    "tactile_pos",
    "tactile_normal_force",
    "tactile_shear_force",
    "tactile_rgb_image",
    "third_person_camera",
    "third_person_camera_pos",
)

TACTILE_ONLY_REQUIRED_KEYS = (
    "joint_pos",
    "tactile_normal_force",
    "tactile_shear_force",
    "tactile_rgb_image",
)

TACTILE_ONLY_FORBIDDEN_KEYS = (
    "tactile_pos",
    "ee_pos_env",
    "ee_quat_env",
    "object_pos",
    "object_rot",
    "third_person_camera_pos",
    "third_person_camera_depth",
    "twist_camera",
    "twist_camera_pos",
    "twist_camera_depth",
)
DEMO_ALLOWED_CAMERA_KEYS = frozenset({"third_person_camera"})

OPTIONAL_KEYS = ("action",)


def _find_npz(path: str) -> list[str]:
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p) and p.endswith(".npz"):
        return [p]
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.npz")))
        if files:
            return files
    raise FileNotFoundError(f"No .npz found under: {path}")


def validate_npz(path: str, *, profile: str = "canonical") -> dict:
    if profile not in ("canonical", "tactile_only"):
        raise ValueError(f"Unknown profile: {profile}")
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        data = {}
        errors = []
        for key in sorted(keys):
            try:
                data[key] = archive[key]
            except ValueError as exc:
                errors.append(f"{key}: {exc}")
    if profile == "tactile_only":
        required = TACTILE_ONLY_REQUIRED_KEYS
        forbidden = [k for k in TACTILE_ONLY_FORBIDDEN_KEYS if k in keys]
        forbidden.extend(
            k
            for k in sorted(keys)
            if (k.endswith("_camera") or k.endswith("_camera_pos") or k.endswith("_camera_depth"))
            and k not in DEMO_ALLOWED_CAMERA_KEYS
        )
        forbidden = sorted(set(forbidden))
    else:
        required = REQUIRED_KEYS
        forbidden = []
    missing = [k for k in required if k not in keys]
    result = {
        "path": path,
        "ok": False,
        "missing": missing,
        "forbidden": forbidden,
        "keys": sorted(keys),
        "frames": None,
        "shapes": {},
        "profile": profile,
        "errors": errors,
    }
    if "joint_pos" in data and data["joint_pos"].ndim > 0:
        result["frames"] = int(data["joint_pos"].shape[0])
    specs = {
        "joint_pos": (2, None),
        "joint_vel": (2, None),
        "action": (2, None),
        "tactile_pos": (3, (7,)),
        "tactile_normal_force": (5, (1,)),
        "tactile_shear_force": (5, (2,)),
        "tactile_rgb_image": (5, (3, 4)),
    }
    for key, arr in data.items():
        result["shapes"][key] = tuple(arr.shape)
        spec = specs.get(key)
        if key.endswith("_camera"):
            spec = (4, (3, 4))
        elif key.endswith("_camera_pos"):
            spec = (3, (7,))
        if spec is None:
            continue  # Metadata is not a time-aligned policy channel.
        ndim, channels = spec
        if arr.ndim != ndim or any(size == 0 for size in arr.shape):
            errors.append(f"{key}: expected {ndim} nonempty dimensions, got {arr.shape}")
            continue
        if channels is not None and arr.shape[-1] not in channels:
            errors.append(f"{key}: expected final dimension in {channels}, got {arr.shape}")
        if key.endswith("_camera_pos") and arr.shape[1] != 1:
            errors.append(f"{key}: expected one camera pose, got {arr.shape}")
        if result["frames"] is not None and arr.shape[0] != result["frames"]:
            errors.append(f"{key}: has {arr.shape[0]} frames; expected {result['frames']}")
        if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
            errors.append(f"{key}: must contain finite numeric values")
    normal = data.get("tactile_normal_force")
    shear = data.get("tactile_shear_force")
    if normal is not None and shear is not None and normal.shape[:-1] != shear.shape[:-1]:
        errors.append(f"normal/shear grids differ: {normal.shape} vs {shear.shape}")
    sensor_counts = {
        key: arr.shape[1]
        for key, arr in data.items()
        if key.startswith("tactile_") and key in specs and arr.ndim >= 2
    }
    if len(set(sensor_counts.values())) > 1:
        errors.append(f"tactile sensor counts differ: {sensor_counts}")
    result["ok"] = not missing and not forbidden and not errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate play_record NPZ files.")
    parser.add_argument(
        "path",
        nargs="?",
        default="play_records",
        help="NPZ file or directory (default: play_records).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write validation summary JSON.",
    )
    parser.add_argument(
        "--profile",
        choices=("canonical", "tactile_only"),
        default="canonical",
        help="tactile_only forbids geometry/vision policy channels; a third-person RGB demo camera is allowed.",
    )
    args = parser.parse_args()

    files = _find_npz(args.path)
    results = [validate_npz(f, profile=args.profile) for f in files]
    all_ok = all(r["ok"] for r in results)

    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        print(f"[{status}] {r['path']}")
        print(f"  frames={r['frames']} keys={len(r['keys'])}")
        if r["missing"]:
            print(f"  missing required: {r['missing']}")
        if r.get("forbidden"):
            print(f"  forbidden geometry/vision keys: {r['forbidden']}")
        for error in r["errors"]:
            print(f"  invalid: {error}")
        for k, shape in r["shapes"].items():
            print(f"  - {k}: {shape}")

    if args.json_out:
        out = os.path.abspath(os.path.expanduser(args.json_out))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"all_ok": all_ok, "results": results}, f, indent=2)
        print(f"[INFO] wrote {out}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
