#!/usr/bin/env python3
"""Unified UR10e + ShadowHand *single* debug runner (tactile viewer).

This script can run different Direct env tasks by only switching the env/cfg entry strings.

Examples (inside Isaac Sim python):

    # Pour task
    ./python.sh scripts/debug/run_ur10e_shadowhand_single.py --task pour --num_envs 1 --show_rgb --enable_cameras

    # Pickup task
    ./python.sh scripts/debug/run_ur10e_shadowhand_single.py --task pickup --num_envs 1 --show_rgb --show_ff --random_actions --enable_cameras

    # Fully custom (module:Class)
    ./python.sh scripts/debug/run_ur10e_shadowhand_single.py \\
        --env ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv \\
        --cfg ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg \\
        --num_envs 1 --show_rgb --enable_cameras
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher


TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)


_TASK_PRESETS: dict[str, dict[str, str]] = {
    "pour": {
        "env": "ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
    },
    "pickup": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg",
    },
}


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _ff_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def _load_symbol(entry: str) -> Any:
    """Load `module.path:SymbolName`."""
    if ":" not in entry:
        raise ValueError(f"Invalid entry '{entry}'. Expected 'module.path:SymbolName'.")
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _flatten_for_npz(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested record into a 1-level dict for npz saving."""
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            out.update(_flatten_for_npz(v, key))
        return out
    if torch.is_tensor(obj):
        out[prefix] = obj.detach().cpu().numpy()
        return out
    if isinstance(obj, (int, float, bool, str)):
        out[prefix] = np.asarray(obj)
        return out
    # skip unsupported types (e.g., None)
    return out


def _resolve_record_paths(record_path: str, fmt: str) -> tuple[str, str]:
    """Return (output_dir, file_prefix)."""
    rp = os.path.expanduser(record_path)
    # If a directory is provided, keep it; else treat as prefix under its parent.
    if rp.endswith(os.sep) or os.path.isdir(rp):
        out_dir = rp.rstrip(os.sep)
        prefix = "record"
    else:
        out_dir = os.path.dirname(rp) or "."
        base = os.path.basename(rp)
        if base.endswith(f".{fmt}"):
            base = base[: -(len(fmt) + 1)]
        prefix = base or "record"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, prefix


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UR10e+ShadowHand single env and show tactile images.")
    parser.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="pour", help="Preset task.")
    parser.add_argument("--env", type=str, default="", help="Env entry: module:Class (overrides --task).")
    parser.add_argument("--cfg", type=str, default="", help="Cfg entry: module:Class (overrides --task).")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    parser.add_argument("--env_index", type=int, default=0, help="Env index to visualize (default: 0).")
    parser.add_argument("--fps", type=float, default=20.0, help="Target display FPS (default: 20).")
    parser.add_argument("--max_steps", type=int, default=0, help="If >0, stop after N steps.")
    parser.add_argument("--random_actions", action="store_true", help="Apply random actions instead of zeros.")
    parser.add_argument("--show_rgb", action="store_true", help="Show tactile RGB images.")
    parser.add_argument("--show_ff", action="store_true", help="Show tactile force-field images (if enabled).")
    parser.add_argument(
        "--record_path",
        type=str,
        default="",
        help="If set, save env._get_record() every step to this directory or file prefix.",
    )
    parser.add_argument("--record_format", choices=["pt", "npz"], default="pt", help="Record file format.")
    parser.add_argument("--record_every", type=int, default=1, help="Record every N steps (default: 1).")
    parser.add_argument("--record_env_index", type=int, default=-1, help="Env index to record (default: env_index).")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import tasks after app launch (gym registrations, etc.)
    import ViTacLab.tasks  # noqa: F401

    preset = _TASK_PRESETS[str(args.task)]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]

    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    # Matplotlib setup
    fig = None
    ax_rgb = None
    ax_ff = None
    rgb_ims: list = []
    ff_ims: list = []
    nrows, ncols = 20, 25

    if args.show_rgb or args.show_ff:
        import matplotlib

        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        if args.show_rgb and args.show_ff:
            fig, axes = plt.subplots(2, 5, figsize=(16, 6))
            ax_rgb = axes[0]
            ax_ff = axes[1]
        elif args.show_rgb:
            fig, ax_rgb = plt.subplots(1, 5, figsize=(16, 3))
        elif args.show_ff:
            fig, ax_ff = plt.subplots(1, 5, figsize=(16, 3))

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"

    print(f"Creating {EnvCls.__name__} (device={cfg.device}, num_envs={cfg.scene.num_envs}) ...")
    env = EnvCls(cfg)
    action_dim = env.num_actions
    print(f"Action dim: {action_dim}, Obs dim: {cfg.observation_space}")

    obs, _ = env.reset()
    policy_obs = obs.get("policy")
    if policy_obs is not None:
        print(f"Reset ok. policy obs shape: {tuple(policy_obs.shape)}")

    # Initialize tactile sensors' nominal render (for camera tactile)
    for name in TACTILE_SENSOR_NAMES:
        if name in env.scene.sensors:
            try:
                env.scene[name].get_initial_render()
            except Exception:
                pass

    # Determine tactile array size for FF from sensor cfg
    if args.show_ff and fig is not None:
        for name in TACTILE_SENSOR_NAMES:
            if name in env.scene.sensors:
                try:
                    nrows, ncols = env.scene[name].cfg.tactile_array_size
                except Exception:
                    pass
                break

    # Create plot artists
    if fig is not None:
        import matplotlib.pyplot as plt

        if args.show_rgb and ax_rgb is not None:
            zero_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
            axes_rgb = ax_rgb if isinstance(ax_rgb, np.ndarray) else [ax_rgb]
            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if i >= len(axes_rgb):
                    break
                title = name.replace("tactile_sensor_", "").upper()
                im = axes_rgb[i].imshow(zero_rgb)
                axes_rgb[i].set_title(f"{title} RGB")
                axes_rgb[i].axis("off")
                rgb_ims.append(im)

        if args.show_ff and ax_ff is not None:
            zero_ff = np.zeros((nrows * 30, ncols * 30, 3), dtype=np.uint8)
            axes_ff = ax_ff if isinstance(ax_ff, np.ndarray) else [ax_ff]
            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if i >= len(axes_ff):
                    break
                title = name.replace("tactile_sensor_", "").upper()
                im = axes_ff[i].imshow(zero_ff)
                axes_ff[i].set_title(f"{title} FF")
                axes_ff[i].axis("off")
                ff_ims.append(im)

        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)

    # Optional FF renderer
    compute_tactile_shear_image = None
    if args.show_ff:
        from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image as _cts

        compute_tactile_shear_image = _cts

    target_dt = 1.0 / max(1e-3, float(args.fps))
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))
    rec_env_idx = env_idx if int(args.record_env_index) < 0 else max(0, min(int(args.record_env_index), env.num_envs - 1))
    step = 0
    do_record = bool(str(args.record_path).strip())
    record_dir = ""
    record_prefix = ""
    if do_record:
        record_dir, record_prefix = _resolve_record_paths(str(args.record_path).strip(), str(args.record_format))
        if not hasattr(env, "_get_record"):
            raise AttributeError(f"{type(env).__name__} has no _get_record(). Please implement it in the env.")
        print(f"Recording enabled: dir='{record_dir}', prefix='{record_prefix}', fmt={args.record_format}, every={args.record_every}, env={rec_env_idx}")

    print("Environment created. Starting viewer (Ctrl+C to stop).")
    while simulation_app.is_running():
        t0 = time.time()
        step += 1

        if args.random_actions:
            actions = 0.3 * (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0)
        else:
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)

        env.step(actions)

        # Record
        if do_record and int(args.record_every) > 0 and (step % int(args.record_every) == 0):
            rec = env._get_record(env_ids=[rec_env_idx])
            fname = os.path.join(record_dir, f"{record_prefix}_step_{step:06d}.{args.record_format}")
            if str(args.record_format) == "pt":
                torch.save(rec, fname)
            else:
                flat = _flatten_for_npz(rec)
                np.savez_compressed(fname, **flat)

        # Update plots
        if fig is not None and (rgb_ims or ff_ims):
            import matplotlib.pyplot as plt

            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if name not in env.scene.sensors:
                    continue
                data = env.scene[name].data

                # RGB
                if args.show_rgb and rgb_ims and i < len(rgb_ims):
                    img = getattr(data, "tactile_rgb_image", None)
                    if img is not None and img.ndim == 4:
                        e = min(env_idx, img.shape[0] - 1)
                        rgb_ims[i].set_data(_img_to_uint8(img[e].detach().cpu().numpy()))

                # FF
                if args.show_ff and ff_ims and i < len(ff_ims) and compute_tactile_shear_image is not None:
                    nf = getattr(data, "tactile_normal_force", None)
                    sf = getattr(data, "tactile_shear_force", None)
                    if nf is not None and sf is not None:
                        e = min(env_idx, nf.shape[0] - 1)
                        nf_np = nf[e].view(nrows, ncols).detach().cpu().numpy()
                        sf_np = sf[e].view(nrows, ncols, 2).detach().cpu().numpy()
                        ff_ims[i].set_data(_ff_to_uint8(compute_tactile_shear_image(nf_np, sf_np)))

            fig.canvas.draw_idle()
            plt.pause(0.001)

        if args.max_steps > 0 and step >= int(args.max_steps):
            break

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    env.close()
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    simulation_app.close()


if __name__ == "__main__":
    main()

