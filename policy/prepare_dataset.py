"""Convert canonical ViTacLab episode NPZ files into policy-training Zarr data.

This is the single maintained dataset conversion entry for both camera-only
Diffusion Policy and multimodal ViTacDP.  Input episodes are produced by the
collectors under ``scripts/data_collection``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


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


def _camera_keys(data, requested: list[str] | None, policy: str) -> list[str]:
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
        keys = _camera_keys(data, args.camera_keys or expected_camera_keys, args.policy)
        state = np.asarray(data[args.joint_key])[:: args.stride]
        cameras = [np.asarray(data[key])[:: args.stride, ..., :3] for key in keys]
        lengths = [len(state), *(len(value) for value in cameras)]

        tactile = tactile_pos = camera_pos = None
        if args.policy == "ViTacDP":
            pose_keys = [f"{key}_pos" for key in keys]
            missing_pose = [key for key in pose_keys if key not in data]
            if missing_pose:
                raise KeyError(f"{path.name} lacks camera pose keys required by ViTacDP: {missing_pose}")
            camera_pos = np.concatenate([np.asarray(data[key])[:: args.stride] for key in pose_keys], axis=1)
            camera_pos = _pose_to_6d(camera_pos)
            if "tactile_pos" not in data:
                raise KeyError(f"{path.name} lacks tactile_pos required by ViTacDP.")
            tactile_pos = _pose_to_6d(np.asarray(data["tactile_pos"])[:: args.stride])
            if args.tactile_type == "rgb":
                if "tactile_rgb_image" not in data:
                    raise KeyError(f"{path.name} lacks tactile_rgb_image.")
                tactile = np.asarray(data["tactile_rgb_image"])[:: args.stride, ..., :3]
            else:
                required = ("tactile_normal_force", "tactile_shear_force")
                missing = [key for key in required if key not in data]
                if missing:
                    raise KeyError(f"{path.name} lacks tactile force keys: {missing}")
                normal = np.asarray(data[required[0]])[:: args.stride]
                shear = np.asarray(data[required[1]])[:: args.stride]
                tactile = np.concatenate((normal, shear), axis=-1).astype(np.float32, copy=False)
            lengths.extend((len(camera_pos), len(tactile), len(tactile_pos)))

        if args.action_source == "recorded_action":
            if args.action_key not in data:
                raise KeyError(f"{path.name} lacks action key {args.action_key!r}.")
            recorded_action = np.asarray(data[args.action_key])[:: args.stride]
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
            payload["head_camera"] = cameras[0][:-1].astype(np.uint8, copy=False)
            if len(cameras) > 1:
                payload["twist_camera"] = cameras[1][:-1].astype(np.uint8, copy=False)
        else:
            payload["camera"] = np.stack([camera[:-1] for camera in cameras], axis=1).astype(np.uint8, copy=False)
            payload["camera_pos"] = camera_pos[:-1].astype(np.float32, copy=False)
            payload["tac_rgb" if args.tactile_type == "rgb" else "tac_force"] = tactile[:-1]
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
    episode_ends = []
    total = 0
    # Validate every episode before touching an existing output directory.
    for path in episodes:
        camera_keys, payload = _episode_payload(path, args, resolved_cameras)
        payload_shapes = {key: list(value.shape[1:]) for key, value in payload.items()}
        if resolved_cameras is None:
            resolved_cameras = camera_keys
            resolved_shapes = payload_shapes
        elif camera_keys != resolved_cameras:
            raise ValueError(f"Camera order changed in {path.name}: {camera_keys} != {resolved_cameras}")
        elif payload_shapes != resolved_shapes:
            raise ValueError(f"Schema shape changed in {path.name}: {payload_shapes} != {resolved_shapes}")
        length = len(payload["state"])
        total += length
        episode_ends.append(total)

    schema = {
        "policy": args.policy,
        "episodes": len(episodes),
        "transitions": total,
        "camera_keys": resolved_cameras,
        "tactile_type": args.tactile_type if args.policy == "ViTacDP" else None,
        "action_source": args.action_source,
        "shapes": resolved_shapes,
    }
    if not args.validate_only:
        if output.exists():
            if not args.overwrite:
                raise SystemExit(f"Output exists; pass --overwrite to replace it: {output}")
            shutil.rmtree(output)
        import zarr
        from numcodecs import Blosc

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
        print(f"Wrote canonical policy dataset: {output}")


if __name__ == "__main__":
    main()
