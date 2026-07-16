"""Shared high-fidelity scene helpers for ``full_tra`` record scripts.

Record scripts set optional fields on the env cfg before ``Env(cfg)`` is constructed.
Environments that call :func:`~ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env.spawn_high_fidelity_scene_if_enabled`
during ``_setup_scene`` will spawn the decorative USD under each env.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_HIGH_FIDELITY_SCENE_USD = (
    "source/ViTacLab/ViTacLab/assets/data/Scene/kitchen/kitchen_6/kitchen6.usd"
)
DEFAULT_PICKUP_HIGH_FIDELITY_OBJECT_USD = (
    "source/ViTacLab/ViTacLab/assets/data/Objects/cosmos_assets/1_object_A/apple/apple.usd"
)
DEFAULT_HIGH_FIDELITY_SCENE_PRIM_PATH = "/World/envs/env_.*/HighFidelityScene"


def add_high_fidelity_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags shared by single / dual / simple_gripper record scripts."""
    parser.add_argument(
        "--enable-high-fidelity-scene",
        action="store_true",
        help=(
            "Spawn a high-fidelity background USD in each env (kitchen6 by default). "
            "Requires the env to call spawn_high_fidelity_scene_if_enabled in _setup_scene."
        ),
    )
    parser.add_argument(
        "--high-fidelity-scene-usd",
        type=str,
        default="",
        help=f"Override scene USD (default: {DEFAULT_HIGH_FIDELITY_SCENE_USD}).",
    )
    parser.add_argument(
        "--high-fidelity-scene-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Scale for the high-fidelity scene USD.",
    )
    parser.add_argument(
        "--high-fidelity-scene-translation",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="Translation (m) of high-fidelity scene under each env (align large room USDs to the table).",
    )
    parser.add_argument(
        "--high-fidelity-scene-euler",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Euler xyz (rad) for high-fidelity scene orientation (converted to wxyz quat).",
    )
    parser.add_argument(
        "--high-fidelity-object-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("OX", "OY", "OZ"),
        help="Scale override for task objects when supported (pickup object, dual hole, etc.).",
    )
    parser.add_argument(
        "--high-fidelity-held-usd",
        type=str,
        default="",
        help="simple_gripper / Factory forge only: override cfg.task.held_asset spawn usd_path.",
    )


def _resolve_repo_usd_path(path: str) -> str:
    """Return absolute path when ``path`` is relative to repo cwd."""
    raw = str(path or "").strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    for candidate in (Path.cwd() / p,):
        if candidate.is_file():
            return str(candidate.resolve())
    cur = Path.cwd()
    for _ in range(12):
        if (cur / "source" / "ViTacLab").is_dir() and (cur / p).is_file():
            return str((cur / p).resolve())
        if cur.parent == cur:
            break
        cur = cur.parent
    return raw


def _euler_xyz_to_quat_wxyz(euler_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation as R

    q_xyzw = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_quat()
    return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _is_pickup_task(*, preset_key: str | None, env_entry: str, cfg_entry: str) -> bool:
    return (
        str(preset_key) == "pickup"
        or "simple_dexhand.hand_pickup" in str(env_entry)
        or "simple_dexhand.hand_pickup" in str(cfg_entry)
    )


def apply_high_fidelity_cfg(
    cfg: Any,
    args: argparse.Namespace,
    *,
    preset_key: str | None,
    env_entry: str,
    cfg_entry: str,
) -> None:
    """Set high-fidelity fields on ``cfg`` from parsed CLI (no-op if flag is off)."""
    if not bool(getattr(args, "enable_high_fidelity_scene", False)):
        return
    if not hasattr(cfg, "scene"):
        print("[WARN] --enable-high-fidelity-scene: cfg has no scene; skipping.")
        return

    scene_usd = str(getattr(args, "high_fidelity_scene_usd", "") or "").strip()
    if not scene_usd:
        scene_usd = DEFAULT_HIGH_FIDELITY_SCENE_USD
    scene_usd = _resolve_repo_usd_path(scene_usd)

    scene_scale = tuple(float(v) for v in getattr(args, "high_fidelity_scene_scale", (1.0, 1.0, 1.0)))
    scene_translation = tuple(float(v) for v in getattr(args, "high_fidelity_scene_translation", (0.0, 0.0, 0.0)))
    scene_euler = tuple(float(v) for v in getattr(args, "high_fidelity_scene_euler", (0.0, 0.0, 0.0)))
    scene_orientation = _euler_xyz_to_quat_wxyz(scene_euler)
    object_scale = tuple(float(v) for v in getattr(args, "high_fidelity_object_scale", (1.0, 1.0, 1.0)))

    setattr(cfg, "enable_high_fidelity_scene", True)
    setattr(cfg, "high_fidelity_scene_usd_path", scene_usd)
    setattr(cfg, "high_fidelity_scene_prim_path", DEFAULT_HIGH_FIDELITY_SCENE_PRIM_PATH)
    setattr(cfg, "high_fidelity_scene_scale", scene_scale)
    setattr(cfg, "high_fidelity_scene_translation", scene_translation)
    setattr(cfg, "high_fidelity_scene_orientation", scene_orientation)
    print("[INFO] Enabled high-fidelity scene via --enable-high-fidelity-scene.")
    print(f"[INFO] cfg.high_fidelity_scene_usd_path={scene_usd}")
    print(f"[INFO] cfg.high_fidelity_scene_scale={scene_scale}")
    print(f"[INFO] cfg.high_fidelity_scene_translation={scene_translation}")
    print(f"[INFO] cfg.high_fidelity_scene_orientation(wxyz)={scene_orientation}")

    if _is_pickup_task(preset_key=preset_key, env_entry=env_entry, cfg_entry=cfg_entry):
        if hasattr(cfg, "object_cfg") and hasattr(cfg.object_cfg, "spawn"):
            setattr(cfg.object_cfg.spawn, "usd_path", DEFAULT_PICKUP_HIGH_FIDELITY_OBJECT_USD)
            if hasattr(cfg.object_cfg.spawn, "scale"):
                setattr(cfg.object_cfg.spawn, "scale", object_scale)
            if hasattr(cfg, "object_scale"):
                setattr(cfg, "object_scale", object_scale)
            print(
                f"[INFO] pickup object usd -> {getattr(cfg.object_cfg.spawn, 'usd_path', None)}, "
                f"scale={getattr(cfg.object_cfg.spawn, 'scale', None)}"
            )

    if object_scale != (1.0, 1.0, 1.0) and hasattr(cfg, "hole_cfg") and hasattr(cfg.hole_cfg, "spawn"):
        if hasattr(cfg.hole_cfg.spawn, "scale"):
            setattr(cfg.hole_cfg.spawn, "scale", object_scale)
        if hasattr(cfg, "hole_scale"):
            setattr(cfg, "hole_scale", object_scale)
        print(f"[INFO] dual hole scale -> {object_scale}")

    held_usd = str(getattr(args, "high_fidelity_held_usd", "") or "").strip()
    task = getattr(cfg, "task", None)
    if held_usd and task is not None:
        held_asset = getattr(task, "held_asset", None)
        if held_asset is not None and hasattr(held_asset, "spawn"):
            spawn = held_asset.spawn
            if hasattr(spawn, "usd_path"):
                setattr(spawn, "usd_path", held_usd)
            if hasattr(spawn, "scale"):
                setattr(spawn, "scale", object_scale)
            print(f"[INFO] forge held_asset usd -> {held_usd}, scale={object_scale}")
