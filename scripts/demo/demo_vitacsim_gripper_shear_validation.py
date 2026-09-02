#!/usr/bin/env python3
"""Gripper shear-force ViTacSim validation (main advisor plan).

Franka parallel GelSight R15 grasps validation W100, then applies lateral EE motion
(+X in Factory fixed frame) to generate tangential load. Compare tacsl vs vitacsim.

Usage::

    ../IsaacLab/isaaclab.sh -p scripts/demo/demo_vitacsim_gripper_shear_validation.py \\
        --headless --enable_cameras --device cuda:0 \\
        --shear-action 0.5 --sensor-mode vitacsim
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

_OUTPUT_SCHEMA = "sf_gripper_v1"

parser = argparse.ArgumentParser(description="ViTacSim gripper shear validation (Franka + W100).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--weight-id", type=str, default="W100", choices=("W200", "W100", "W050", "W020", "W010"))
parser.add_argument("--sensor-mode", type=str, choices=("tacsl", "vitacsim"), default="vitacsim")
parser.add_argument(
    "--shear-action",
    type=float,
    default=0.5,
    help="Normalized Forge +X action in [-1,1] (maps to pos_action_bounds).",
)
parser.add_argument("--settle-steps", type=int, default=30, help="Steps with zero action after grasp.")
parser.add_argument("--shear-steps", type=int, default=80, help="Steps holding lateral shear action.")
parser.add_argument("--record-steps", type=int, default=40)
parser.add_argument(
    "--out-dir",
    type=str,
    default="logs/vitacsim_validation/shear_force/gripper",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--no-marker",
    action="store_true",
    help="Disable FOTS marker overlay on Taxim RGB (default: GelSight markers enabled).",
)
parser.add_argument(
    "--marker-pattern",
    type=str,
    default="gelsight",
    choices=("gelsight", "xense"),
    help="Marker layout when marker overlay is enabled.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ViTacLab.tasks  # noqa: F401

from ViTacLab.tasks.direct.vitacsim_validation.gripper_shear_env import GripperShearValidationEnv
from ViTacLab.tasks.direct.vitacsim_validation.gripper_shear_env_cfg import build_gripper_shear_env_cfg
from ViTacLab.tasks.direct.vitacsim_validation.weight_spec import WEIGHT_MASS_KG


def _shear_tag(action: float) -> str:
    s = f"{action:.3f}".rstrip("0").rstrip(".")
    return f"S{s.replace('-', 'm')}"


def _save_rgb(path: Path, rgb_u8: torch.Tensor) -> None:
    try:
        from PIL import Image
    except ImportError:
        np.save(path.with_suffix(".npy"), rgb_u8.detach().cpu().numpy())
        return
    arr = rgb_u8.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 3:
        Image.fromarray(arr.astype(np.uint8)).save(path)


def _sensor_arrays(sensor) -> tuple[np.ndarray | None, np.ndarray | None, torch.Tensor | None, torch.Tensor | None]:
    data = sensor.data
    nf = getattr(data, "tactile_normal_force", None)
    sf = getattr(data, "tactile_shear_force", None)
    rgb = getattr(data, "tactile_rgb_image_corrected", None)
    if rgb is None:
        rgb = getattr(data, "tactile_rgb_image", None)
    nf_np = nf[0].detach().cpu().numpy() if nf is not None else None
    sf_np = sf[0].detach().cpu().numpy() if sf is not None else None
    rgb_t = rgb[0].detach().cpu() if rgb is not None else None
    rgb_depth = getattr(data, "tactile_rgb_image", None)
    rgb_depth_t = rgb_depth[0].detach().cpu() if rgb_depth is not None else None
    return nf_np, sf_np, rgb_t, rgb_depth_t


def _peak_stats(nf: np.ndarray | None, sf: np.ndarray | None) -> dict[str, float]:
    out = {"fn_peak_max": 0.0, "ft_peak_max": 0.0, "ft_field_sum": 0.0}
    if nf is not None:
        out["fn_peak_max"] = float(np.nan_to_num(np.abs(nf), nan=0.0).max())
    if sf is not None:
        mag = np.linalg.norm(np.nan_to_num(sf, nan=0.0), axis=-1)
        out["ft_peak_max"] = float(mag.max())
        out["ft_field_sum"] = float(mag.sum())
    return out


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    mass_kg = float(WEIGHT_MASS_KG[args_cli.weight_id])
    nominal_fn = mass_kg * 9.81
    shear = float(np.clip(args_cli.shear_action, -1.0, 1.0))
    tag = _shear_tag(shear)

    print(f"[INFO] gripper shear weight={args_cli.weight_id} Fn_nom≈{nominal_fn:.3f} N")
    print(f"[INFO] shear_action={shear:.3f} sensor_mode={args_cli.sensor_mode}")
    marker_on = not bool(args_cli.no_marker)
    print(
        f"[INFO] marker enabled={marker_on} pattern="
        f"{args_cli.marker_pattern if marker_on else 'none'}"
    )

    cfg = build_gripper_shear_env_cfg(
        weight_id=args_cli.weight_id,
        sensor_mode=args_cli.sensor_mode,
        enable_marker=marker_on,
        marker_pattern=args_cli.marker_pattern,
    )
    cfg.scene.num_envs = int(args_cli.num_envs)
    env = GripperShearValidationEnv(cfg, render_mode=None)

    env.reset()
    device = env.device
    zero = torch.zeros(env.num_envs, env.cfg.action_space, device=device)
    shear_action = zero.clone()
    shear_action[:, 0] = shear
    shear_action[:, 6] = -1.0

    for _ in range(int(args_cli.settle_steps)):
        env.step(zero)

    for _ in range(int(args_cli.shear_steps)):
        env.step(shear_action)

    fn_l = fn_r = sf_l = sf_r = None
    rgb_l = rgb_r = None
    stats_l = stats_r = {"fn_peak_max": 0.0, "ft_peak_max": 0.0, "ft_field_sum": 0.0}

    for _ in range(int(args_cli.record_steps)):
        env.step(shear_action)
        if "tactile_sensor_left" in env.scene.sensors:
            nf, sf, rgb, _ = _sensor_arrays(env.scene["tactile_sensor_left"])
            fn_l, sf_l, rgb_l = nf, sf, rgb
            stats_l = _peak_stats(nf, sf)
        if "tactile_sensor_right" in env.scene.sensors:
            nf, sf, rgb, _ = _sensor_arrays(env.scene["tactile_sensor_right"])
            fn_r, sf_r, rgb_r = nf, sf, rgb
            stats_r = _peak_stats(nf, sf)

    out_dir = (
        Path(args_cli.out_dir).expanduser().resolve()
        / tag
        / args_cli.weight_id
        / args_cli.sensor_mode
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for side, nf, sf, rgb in (("left", fn_l, sf_l, rgb_l), ("right", fn_r, sf_r, rgb_r)):
        if nf is not None:
            np.save(out_dir / f"tactile_normal_force_{side}.npy", nf)
        if sf is not None:
            np.save(out_dir / f"tactile_shear_force_{side}.npy", sf)
        if rgb is not None:
            _save_rgb(out_dir / f"tactile_rgb_{side}.png", rgb)

    summary = {
        "output_schema": _OUTPUT_SCHEMA,
        "experiment": "gripper_lateral_ee_shear",
        "weight_id": args_cli.weight_id,
        "mass_kg": mass_kg,
        "nominal_fn_n": nominal_fn,
        "shear_action_x": shear,
        "shear_action_tag": tag,
        "sensor_mode": args_cli.sensor_mode,
        "enable_marker_simulation": marker_on,
        "marker_pattern": args_cli.marker_pattern if marker_on else "none",
        "fn_peak_max_left": stats_l["fn_peak_max"],
        "fn_peak_max_right": stats_r["fn_peak_max"],
        "ft_peak_max_left": stats_l["ft_peak_max"],
        "ft_peak_max_right": stats_r["ft_peak_max"],
        "ft_peak_max": max(stats_l["ft_peak_max"], stats_r["ft_peak_max"]),
        "ft_field_sum_left": stats_l["ft_field_sum"],
        "ft_field_sum_right": stats_r["ft_field_sum"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[INFO] ft_peak L/R={stats_l['ft_peak_max']:.4f}/{stats_r['ft_peak_max']:.4f} "
        f"fn_peak L/R={stats_l['fn_peak_max']:.4f}/{stats_r['fn_peak_max']:.4f}"
    )
    print(f"[INFO] saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = int(main())
    except KeyboardInterrupt:
        exit_code = 130
    os._exit(exit_code)
