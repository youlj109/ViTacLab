"""Convert canonical ViTacLab episode NPZ files into policy-training Zarr data.

This is the single maintained dataset conversion entry for both camera-only
Diffusion Policy and multimodal ViTacDP.  Input episodes are produced by the
collectors under ``scripts/data_collection``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Direct CLI execution puts policy/ (not the repository root) on sys.path.
# Canonical pose conversion imports the maintained policy rotation helper.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing episode NPZ files.")
    parser.add_argument("output_zarr", type=Path, help="Destination Zarr directory.")
    parser.add_argument(
        "--policy",
        choices=("Diffusion_Policy", "ViTacDP"),
        required=True,
        help="Output schema: camera-only Diffusion Policy or multimodal ViTacDP.",
    )
    parser.add_argument(
        "--episode-glob",
        default="episode*.npz",
        help="Glob used to find episodes below input_dir (default: episode*.npz).",
    )
    parser.add_argument("--max-episodes", type=int, default=0, help="Maximum episodes to convert; 0 uses all.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame before transition alignment.")
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=None,
        help="Ordered NPZ camera keys. Default auto-selects third_person_camera then twist_camera.",
    )
    parser.add_argument("--joint-key", default="joint_pos", help="NPZ key used for policy state.")
    parser.add_argument(
        "--action-source",
        choices=("next_joint", "recorded_action"),
        default="next_joint",
        help="Use next-frame joint targets or the NPZ action field as training actions.",
    )
    parser.add_argument("--action-key", default="action", help="NPZ key used with --action-source recorded_action.")
    parser.add_argument(
        "--tactile-type",
        choices=("rgb", "force"),
        default="rgb",
        help="ViTacDP tactile representation; ignored for Diffusion_Policy.",
    )
    parser.add_argument(
        "--profile",
        choices=("canonical", "tactile_only"),
        default="canonical",
        help="canonical keeps cameras and tactile poses; tactile_only uses joint + tactile arrays only.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output_zarr if it already exists.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate episode keys/shapes and print the resolved schema without writing Zarr.",
    )
    return parser


def _pose_to_6d(pose: np.ndarray) -> np.ndarray:
    from policy.ViTacDP.vitacdp_core.model.common.rotation_transformer import RotationTransformer

    pose = np.asarray(pose)
    if pose.shape[-1] != 7:
        raise ValueError(f"Pose must end in xyz+quaternion (7 values), got {pose.shape}.")
    transformer = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")
    rotation = transformer.forward(pose[..., 3:7]).astype(np.float32, copy=False)
    return np.concatenate((pose[..., :3].astype(np.float32, copy=False), rotation), axis=-1)


def _camera_keys(data, requested: list[str] | None, policy: str, profile: str) -> list[str]:
    if profile == "tactile_only":
        return []
    if requested:
        keys = list(requested)
    else:
        keys = [key for key in ("third_person_camera", "twist_camera") if key in data]
    if not keys:
        raise KeyError("No camera keys found; pass --camera-keys explicitly.")
    if policy == "Diffusion_Policy" and len(keys) > 2:
        raise ValueError("Diffusion_Policy supports at most head and twist cameras.")
    for key in keys:
        if key not in data:
            raise KeyError(f"Missing requested camera key: {key}")
        if data[key].ndim != 4 or data[key].shape[-1] not in (3, 4):
            raise ValueError(f"Camera {key} must be (T,H,W,3/4), got {data[key].shape}.")
    return keys


def _episode_payload(path: Path, args, expected_camera_keys: list[str] | None):
    with np.load(path, allow_pickle=False) as data:
        frames = None

        def read(key: str, ndim: int, channels: tuple[int, ...] | None = None):
            nonlocal frames
            value = np.asarray(data[key])
            if value.ndim != ndim or any(size == 0 for size in value.shape):
                raise ValueError(f"{path.name} key {key!r} needs {ndim} nonempty dimensions, got {value.shape}.")
            if channels is not None and value.shape[-1] not in channels:
                raise ValueError(f"{path.name} key {key!r} needs final dimension in {channels}, got {value.shape}.")
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"{path.name} key {key!r} must contain finite numeric values.")
            if frames is None:
                frames = len(value)
            elif len(value) != frames:
                raise ValueError(f"{path.name} key {key!r} has {len(value)} frames; expected {frames} before stride.")
            return value[:: args.stride]

        keys = _camera_keys(data, args.camera_keys or expected_camera_keys, args.policy, args.profile)
        state = read(args.joint_key, 2)
        cameras = [read(key, 4, (3, 4))[..., :3] for key in keys]
        lengths = [len(state), *(len(value) for value in cameras)]

        tactile = tactile_pos = camera_pos = None
        if args.policy == "ViTacDP":
            if args.profile != "tactile_only":
                pose_keys = [f"{key}_pos" for key in keys]
                missing_pose = [key for key in pose_keys if key not in data]
                if missing_pose:
                    raise KeyError(f"{path.name} lacks camera pose keys required by ViTacDP: {missing_pose}")
                camera_poses = [read(key, 3, (7,)) for key in pose_keys]
                if any(pose.shape[1] != 1 for pose in camera_poses):
                    raise ValueError(f"{path.name} each camera pose must be (T,1,7).")
                camera_pos = np.concatenate(camera_poses, axis=1)
                camera_pos = _pose_to_6d(camera_pos)
                if "tactile_pos" not in data:
                    raise KeyError(f"{path.name} lacks tactile_pos required by ViTacDP.")
                tactile_pos = _pose_to_6d(read("tactile_pos", 3, (7,)))
            if args.tactile_type == "rgb":
                if "tactile_rgb_image" not in data:
                    raise KeyError(f"{path.name} lacks tactile_rgb_image.")
                tactile = read("tactile_rgb_image", 5, (3, 4))[..., :3]
            else:
                required = ("tactile_normal_force", "tactile_shear_force")
                missing = [key for key in required if key not in data]
                if missing:
                    raise KeyError(f"{path.name} lacks tactile force keys: {missing}")
                normal = read(required[0], 5, (1,))
                shear = read(required[1], 5, (2,))
                if normal.shape[:-1] != shear.shape[:-1]:
                    raise ValueError(f"{path.name} normal/shear grids differ: {normal.shape} vs {shear.shape}.")
                tactile = np.concatenate((normal, shear), axis=-1).astype(np.float32, copy=False)
            if tactile_pos is not None and tactile_pos.shape[1] != tactile.shape[1]:
                raise ValueError(f"{path.name} tactile pose and image/force sensor counts differ.")
            lengths.append(len(tactile))
            if camera_pos is not None:
                lengths.append(len(camera_pos))
            if tactile_pos is not None:
                lengths.append(len(tactile_pos))

        if args.action_source == "recorded_action":
            if args.action_key not in data:
                raise KeyError(f"{path.name} lacks action key {args.action_key!r}.")
            recorded_action = read(args.action_key, 2)
            lengths.append(len(recorded_action))
        else:
            recorded_action = None

        if len(set(lengths)) != 1:
            raise ValueError(f"{path.name} has inconsistent time lengths: {lengths}")
        if lengths[0] < 2:
            raise ValueError(f"{path.name} must contain at least two frames.")

        state_out = state[:-1].astype(np.float32, copy=False)
        if args.action_source == "next_joint":
            action_out = state[1:].astype(np.float32, copy=False)
        else:
            action_out = recorded_action[:-1].astype(np.float32, copy=False)

        payload = {
            "state": state_out,
            "action": action_out,
        }
        if args.policy == "Diffusion_Policy":
            if args.profile == "tactile_only":
                raise ValueError("Diffusion_Policy cannot use --profile tactile_only; it is camera-only.")
            payload["head_camera"] = cameras[0][:-1].astype(np.uint8, copy=False)
            if len(cameras) > 1:
                payload["twist_camera"] = cameras[1][:-1].astype(np.uint8, copy=False)
        else:
            payload["tac_rgb" if args.tactile_type == "rgb" else "tac_force"] = tactile[:-1]
            if args.profile != "tactile_only":
                payload["camera"] = np.stack([camera[:-1] for camera in cameras], axis=1).astype(np.uint8, copy=False)
                payload["camera_pos"] = camera_pos[:-1].astype(np.float32, copy=False)
                payload["tac_pos"] = tactile_pos[:-1].astype(np.float32, copy=False)
        for key, value in payload.items():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                first_bad = tuple(int(index) for index in np.argwhere(~np.isfinite(value))[0])
                raise ValueError(f"{path.name} key {key!r} contains NaN/Inf at index {first_bad}.")
        return keys, payload


def _append_dataset(group, key: str, value: np.ndarray, compressor, chunk_length: int):
    if key not in group:
        group.create_dataset(
            key,
            shape=(0, *value.shape[1:]),
            chunks=(min(chunk_length, max(1, len(value))), *value.shape[1:]),
            dtype=value.dtype,
            compressor=compressor,
            overwrite=True,
        )
    dataset = group[key]
    if dataset.shape[1:] != value.shape[1:] or dataset.dtype != value.dtype:
        raise ValueError(
            f"Dataset {key} changed shape/dtype: existing {dataset.shape[1:]}/{dataset.dtype}, "
            f"new {value.shape[1:]}/{value.dtype}."
        )
    dataset.append(value)


def main() -> None:
    args = build_parser().parse_args()
    if args.policy == "Diffusion_Policy" and args.profile == "tactile_only":
        raise SystemExit("Diffusion_Policy cannot use --profile tactile_only; it is camera-only.")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output_zarr.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    episodes = sorted(input_dir.rglob(args.episode_glob))
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise SystemExit(f"No episodes matched {args.episode_glob!r} under {input_dir}")

    resolved_cameras = None
    resolved_shapes = None
    resolved_dtypes = None
    episode_ends = []
    total = 0
    # Validate every episode before touching an existing output directory.
    for path in episodes:
        camera_keys, payload = _episode_payload(path, args, resolved_cameras)
        payload_shapes = {key: list(value.shape[1:]) for key, value in payload.items()}
        payload_dtypes = {key: str(value.dtype) for key, value in payload.items()}
        if resolved_cameras is None:
            resolved_cameras = camera_keys
            resolved_shapes = payload_shapes
            resolved_dtypes = payload_dtypes
        elif camera_keys != resolved_cameras:
            raise ValueError(f"Camera order changed in {path.name}: {camera_keys} != {resolved_cameras}")
        elif payload_shapes != resolved_shapes:
            raise ValueError(f"Schema shape changed in {path.name}: {payload_shapes} != {resolved_shapes}")
        elif payload_dtypes != resolved_dtypes:
            raise ValueError(f"Schema dtype changed in {path.name}: {payload_dtypes} != {resolved_dtypes}")
        length = len(payload["state"])
        total += length
        episode_ends.append(total)

    schema = {
        "policy": args.policy,
        "profile": args.profile,
        "episodes": len(episodes),
        "transitions": total,
        "camera_keys": resolved_cameras,
        "tactile_type": args.tactile_type if args.policy == "ViTacDP" else None,
        "action_source": args.action_source,
        "shapes": resolved_shapes,
    }
    if not args.validate_only:
        # An output ancestor would delete the source episodes with --overwrite.
        if input_dir == output or output in input_dir.parents:
            raise SystemExit(f"Output must not be the input directory or its ancestor: {output}")
        import zarr
        from numcodecs import Blosc

        if output.exists():
            if not args.overwrite:
                raise SystemExit(f"Output exists; pass --overwrite to replace it: {output}")
            shutil.rmtree(output)
        root = zarr.group(str(output))
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        for path in episodes:
            _, payload = _episode_payload(path, args, resolved_cameras)
            for key, value in payload.items():
                _append_dataset(data_group, key, value, compressor, chunk_length=256)
        meta_group.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64), overwrite=True)
        root.attrs["vitaclab_schema"] = json.dumps(schema, sort_keys=True)
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    if not args.validate_only:
        print(f"Wrote {args.profile} policy dataset: {output}")


if __name__ == "__main__":
    main()
