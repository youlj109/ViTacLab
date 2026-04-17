#!/usr/bin/env python3
"""Run GelSight short-finger pretraining tasks (mass / friction) with a simple loop."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher

_TASK_PRESETS: dict[str, dict[str, str]] = {
    "mass": {
        "env": (
            "ViTacLab.tasks.direct.pretraining.mass_pretrain.gelsight_mass_pretrain_env:"
            "GelsightFingerMassPretrainEnv"
        ),
        "cfg": (
            "ViTacLab.tasks.direct.pretraining.mass_pretrain.gelsight_mass_pretrain_env_cfg:"
            "GelsightFingerMassPretrainEnvCfg"
        ),
    },
    "friction": {
        "env": (
            "ViTacLab.tasks.direct.pretraining.friction_pretrain.gelsight_friction_pretrain_env:"
            "GelsightFingerFrictionPretrainEnv"
        ),
        "cfg": (
            "ViTacLab.tasks.direct.pretraining.friction_pretrain.gelsight_friction_pretrain_env_cfg:"
            "GelsightFingerFrictionPretrainEnvCfg"
        ),
    },
}


def _repo_root() -> Path:
    """ViTacLab repo root (contains ``source/``)."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _load_symbol(entry: str) -> Any:
    if ":" not in entry:
        raise ValueError(f"Invalid entry '{entry}'. Expected 'module.path:SymbolName'.")
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _entries_from_gym_registry(task_id: str) -> tuple[str, str]:
    import gymnasium as gym

    tid = task_id.split(":")[-1].strip()
    spec = gym.spec(tid)
    ep = spec.entry_point
    env_entry = f"{ep.__module__}:{ep.__name__}" if callable(ep) else str(ep)
    cfg_ep = (spec.kwargs or {}).get("env_cfg_entry_point")
    if not cfg_ep:
        raise ValueError(f"Registry task {tid!r} has no env_cfg_entry_point in spec.kwargs.")
    return env_entry, cfg_ep


def _resolve_env_cfg_entries(task: str) -> tuple[str, str]:
    if task in _TASK_PRESETS:
        preset = _TASK_PRESETS[task]
        return preset["env"], preset["cfg"]
    return _entries_from_gym_registry(task)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run GelSight short-finger pretraining tasks.")
    p.add_argument(
        "--task",
        type=str,
        default="mass",
        help="Preset task key (mass|friction) or any registered Gym ID.",
    )
    p.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    p.add_argument("--fps", type=float, default=30.0, help="Loop target FPS.")
    p.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = run until close).")
    p.add_argument("--print_every", type=int, default=120, help="Print extras/log every N steps (0 = off).")
    p.add_argument(
        "--show_rgb",
        action="store_true",
        help="Matplotlib: live GelSight tactile RGB (implies --enable_cameras).",
    )
    p.add_argument(
        "--show_ff",
        action="store_true",
        help="Matplotlib: live tactile force-field RGB from normal/shear (implies --enable_cameras).",
    )
    p.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Env index for tactile display when num_envs > 1 (default: 0).",
    )
    # Friction / mass pretrain (cfg 含 plot_xyz_force_live 时生效)。
    p.add_argument(
        "--plot-force-xyz",
        action="store_true",
        help="Live matplotlib: TacSL patch mean curves vs t (friction: optional applied source).",
    )
    p.add_argument(
        "--plot-force-source",
        type=str,
        choices=("applied", "tactile"),
        default=None,
        help="Friction: applied or tactile; mass: only tactile (applied ignored).",
    )
    p.add_argument(
        "--plot-force-max-points",
        type=int,
        default=None,
        help="Rolling buffer length for --plot-force-xyz (default: cfg).",
    )
    p.add_argument(
        "--plot-force-update-interval",
        type=int,
        default=None,
        help="Redraw plot every N env steps (default: cfg).",
    )
    p.add_argument(
        "--print-force-mean-every",
        type=int,
        default=None,
        help="Print debug line every N steps: friction=forces; mass=mass+tactile (0=off).",
    )
    AppLauncher.add_app_launcher_args(p)
    return p


def _to_numpy(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


# Default cell size in :func:`compute_tactile_shear_image` (``visuotactile_render.py``).
_TACTILE_SHEAR_VIZ_RESOLUTION = 30

_ff_cv2 = None
_ff_compute_tactile_shear_image = None


def _tactile_shear_image_rgb_uint8(nf_hw: np.ndarray, sf_hw2: np.ndarray) -> np.ndarray:
    """Arrow-field FF image; OpenCV draws BGR, ``imshow`` uses RGB (see ``tacsl_sensor_gelsight_finger_short``)."""

    global _ff_cv2, _ff_compute_tactile_shear_image
    if _ff_compute_tactile_shear_image is None:
        import cv2

        from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image

        _ff_cv2 = cv2
        _ff_compute_tactile_shear_image = compute_tactile_shear_image

    img_bgr = _ff_compute_tactile_shear_image(nf_hw, sf_hw2)
    u8 = (np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _ff_cv2.cvtColor(u8, _ff_cv2.COLOR_BGR2RGB)


def main() -> int:
    args = _build_arg_parser().parse_args()
    if getattr(args, "show_rgb", False) or getattr(args, "show_ff", False):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401

    repo_root = _repo_root()
    source_dir = repo_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    env_entry, cfg_entry = _resolve_env_cfg_entries(args.task)
    print(f"[INFO] env={env_entry}\n[INFO] cfg={cfg_entry}")
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    _enable_cams = bool(getattr(args, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    setattr(cfg, "enable_cameras", _enable_cams)
    print(f"[INFO] cfg.enable_cameras={getattr(cfg, 'enable_cameras', None)}")

    if hasattr(cfg, "plot_xyz_force_live"):
        _plot_cli = False
        if getattr(args, "plot_force_xyz", False):
            cfg.plot_xyz_force_live = True
            _plot_cli = True
        if getattr(args, "plot_force_source", None) is not None and hasattr(cfg, "plot_xyz_force_live_source"):
            cfg.plot_xyz_force_live_source = str(args.plot_force_source)
            _plot_cli = True
        if getattr(args, "plot_force_max_points", None) is not None:
            cfg.plot_xyz_force_live_max_points = max(32, int(args.plot_force_max_points))
            _plot_cli = True
        if getattr(args, "plot_force_update_interval", None) is not None:
            cfg.plot_xyz_force_live_update_interval = max(1, int(args.plot_force_update_interval))
            _plot_cli = True
        if getattr(args, "print_force_mean_every", None) is not None and hasattr(cfg, "print_xyz_force_mean_interval"):
            cfg.print_xyz_force_mean_interval = max(0, int(args.print_force_mean_every))
            _plot_cli = True
        if _plot_cli:
            print(
                "[INFO] pretrain plot (CLI): "
                f"plot_xyz_force_live={getattr(cfg, 'plot_xyz_force_live', None)} "
                f"source={getattr(cfg, 'plot_xyz_force_live_source', None)} "
                f"print_interval={getattr(cfg, 'print_xyz_force_mean_interval', None)}"
            )

    print(f"[INFO] Creating {EnvCls.__name__} (num_envs={cfg.scene.num_envs}) ...")
    env = EnvCls(cfg)

    from ViTacLab.tasks.direct.pretraining.gelsight_finger_pretrain_base_env import TACTILE_SENSOR_NAME

    fig = None
    ax_rgb = None
    ax_ff = None
    im_rgb = None
    im_ff = None
    nrows, ncols = 20, 25
    env_idx = max(0, min(int(getattr(args, "env_index", 0)), env.num_envs - 1))

    if args.show_rgb or args.show_ff:
        import matplotlib

        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        if args.show_rgb and args.show_ff:
            fig, (ax_rgb, ax_ff) = plt.subplots(2, 1, figsize=(10, 7))
        elif args.show_rgb:
            fig, ax_rgb = plt.subplots(1, 1, figsize=(10, 4))
        else:
            fig, ax_ff = plt.subplots(1, 1, figsize=(10, 4))

        if TACTILE_SENSOR_NAME in env.scene.sensors:
            try:
                nrows, ncols = env.scene[TACTILE_SENSOR_NAME].cfg.tactile_array_size
            except Exception:
                pass

        if args.show_rgb and ax_rgb is not None:
            z = np.zeros((240, 320, 3), dtype=np.uint8)
            im_rgb = ax_rgb.imshow(z)
            ax_rgb.set_title("GelSight RGB")
            ax_rgb.axis("off")
        if args.show_ff and ax_ff is not None:
            zf = np.zeros(
                (max(1, nrows) * _TACTILE_SHEAR_VIZ_RESOLUTION, max(1, ncols) * _TACTILE_SHEAR_VIZ_RESOLUTION, 3),
                dtype=np.uint8,
            )
            im_ff = ax_ff.imshow(zf)
            ax_ff.set_title("Tactile FF (compute_tactile_shear_image)")
            ax_ff.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.05)
        print("[INFO] Tactile viewer: ensure --enable_cameras (forced when using --show_rgb / --show_ff).")

    env.reset()

    actions = torch.zeros_like(env.actions)
    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0

    while simulation_app.is_running():
        t0 = time.time()
        step += 1

        obs, rew, terminated, truncated, extras = env.step(actions)

        if fig is not None and TACTILE_SENSOR_NAME in env.scene.sensors:
            import matplotlib.pyplot as plt

            data = env.scene[TACTILE_SENSOR_NAME].data
            if args.show_rgb and im_rgb is not None:
                img = getattr(data, "tactile_rgb_image", None)
                if img is not None and img.ndim == 4:
                    e = min(env_idx, img.shape[0] - 1)
                    im_rgb.set_data(_img_to_uint8(img[e].detach().cpu().numpy()))
            if args.show_ff and im_ff is not None:
                nf = getattr(data, "tactile_normal_force", None)
                sf = getattr(data, "tactile_shear_force", None)
                if nf is not None and sf is not None:
                    e = min(env_idx, nf.shape[0] - 1)
                    nf_flat = nf[e].detach().cpu().numpy().reshape(-1)
                    sf_flat = sf[e].detach().cpu().numpy().reshape(-1, 2)
                    p = int(nf_flat.shape[0])
                    nr, nc = nrows, ncols
                    if p != nr * nc:
                        nr = int(np.sqrt(p))
                        nc = max(1, p // max(1, nr))
                    nf_img = nf_flat.reshape(nr, nc)
                    sf_img = sf_flat.reshape(nr, nc, 2)
                    im_ff.set_data(_tactile_shear_image_rgb_uint8(nf_img, sf_img))
            fig.canvas.draw_idle()
            plt.pause(0.001)

        if args.print_every > 0 and step % int(args.print_every) == 0:
            log = extras.get("log", {})
            if log:
                env0 = {k: _to_numpy(v)[0] if hasattr(v, "__len__") else v for k, v in log.items()}
                print(f"[INFO] step={step} log(env0)={env0}")

        if args.max_steps > 0 and step >= int(args.max_steps):
            break

        dt = time.time() - t0
        if dt < target_dt:
            time.sleep(target_dt - dt)

    simulation_app.close()
    env.close()
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
