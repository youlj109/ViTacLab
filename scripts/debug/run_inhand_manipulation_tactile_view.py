#!/usr/bin/env python3
"""Run InHandManipulationEnv (Shadow Hand) and visualize TacSL tactile outputs.

Usage (inside Isaac Sim python):

    ./python.sh scripts/debug/run_inhand_manipulation_tactile_view.py --enable_cameras

Or (if your PYTHONPATH is set up):

    python scripts/debug/run_inhand_manipulation_tactile_view.py --enable_cameras
"""

from __future__ import annotations

import argparse
import time

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


def _ff_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert force-field viz image in [0,1] float to uint8."""
    if img.dtype != np.uint8:
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run InHandManipulationEnv and show TacSL tactile force-field images.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs (default: 1).")
    parser.add_argument("--env-index", type=int, default=0, help="Env index to visualize (default: 0).")
    parser.add_argument("--fps", type=float, default=20.0, help="Target display FPS (default: 20).")
    parser.add_argument(
        "--random_actions",
        action="store_true",
        help="If set, apply small random actions instead of zeros.",
    )
    parser.add_argument(
        "--show-rgb",
        action="store_true",
        help="If set, also enable & display tactile RGB (heavier).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Launch Isaac Sim app first.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Lazily import matplotlib after app launch.
    import matplotlib

    matplotlib.use("TkAgg" if "DISPLAY" in __import__("os").environ else "Agg")
    import matplotlib.pyplot as plt

    # Import task modules only after Isaac app is running.
    import isaaclab_tasks  # noqa: F401
    import ViTacLab.tasks  # noqa: F401

    from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image
    from ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv
    from ViTacLab.tasks.direct.simple_dexhand.shadow_hand.shadow_hand_env_cfg import ShadowHandTactileEnvCfg

    cfg = ShadowHandTactileEnvCfg()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"

    # Optional: enable tactile RGB on all five sensors (default cfg disables it).
    if args.show_rgb:
        for name in TACTILE_SENSOR_NAMES:
            if hasattr(cfg.scene, name):
                getattr(cfg.scene, name).enable_camera_tactile = True

    print(f"Creating InHandManipulationEnv (ShadowHand) on device={cfg.device} with num_envs={cfg.scene.num_envs} ...")
    env = InHandManipulationEnv(cfg)

    try:
        # Initial reset.
        env.reset()

        # Initialize nominal tactile render for camera-based tactile sensing (RGB).
        # Similar to ForgeEnv: get_initial_render() must be called once after reset
        # so that VisuoTactileSensor has a baseline depth image.
        if args.show_rgb:
            for name in TACTILE_SENSOR_NAMES:
                if name in env.scene.sensors:
                    sensor = env.scene[name]
                    if getattr(sensor.cfg, "enable_camera_tactile", False):
                        try:
                            sensor.get_initial_render()
                        except Exception:
                            # Keep viewer robust even if one sensor fails to initialize RGB.
                            pass

        print("Environment created. Starting tactile viewer (Ctrl+C to stop).")

        # Infer tactile array shape from first available sensor.
        nrows, ncols = 20, 25
        for name in TACTILE_SENSOR_NAMES:
            if name in env.scene.sensors:
                nrows, ncols = env.scene[name].cfg.tactile_array_size  # type: ignore[index]
                break

        # Layout:
        # - If show_rgb: 2 rows (RGB + force-field), 5 columns (ff/lf/mf/rf/th)
        # - Else: 1 row, 5 columns (force-field)
        if args.show_rgb:
            fig, axes = plt.subplots(2, 5, figsize=(16, 6))
            ax_rgb = axes[0]
            ax_ff = axes[1]
        else:
            fig, ax_ff = plt.subplots(1, 5, figsize=(16, 3))
            ax_rgb = None
        plt.tight_layout()

        # Initialize images.
        zero_ff = np.zeros((nrows * 30, ncols * 30, 3), dtype=np.uint8)
        ff_ims = []
        rgb_ims = []

        for i, name in enumerate(TACTILE_SENSOR_NAMES):
            title = name.replace("tactile_sensor_", "").upper()
            im = ax_ff[i].imshow(zero_ff)
            ax_ff[i].set_title(f"{title} FF")
            ax_ff[i].axis("off")
            ff_ims.append(im)

            if args.show_rgb and ax_rgb is not None:
                zero_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
                imr = ax_rgb[i].imshow(zero_rgb)
                ax_rgb[i].set_title(f"{title} RGB")
                ax_rgb[i].axis("off")
                rgb_ims.append(imr)

        fig.canvas.draw()
        plt.pause(0.1)

        target_dt = 1.0 / max(1e-3, float(args.fps))
        env_idx = max(0, int(args.env_index))

        while simulation_app.is_running():
            t0 = time.time()

            action_dim = int(getattr(env.cfg, "action_space", 0) or 0)
            if action_dim <= 0:
                # fallback: infer from action tensor shape expected by _pre_physics_step
                action_dim = 20

            if args.random_actions:
                actions = 0.1 * (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0)
            else:
                actions = torch.zeros(env.num_envs, action_dim, device=env.device)

            env.step(actions)
            print(f"actions: {actions}")

            # Update plots from sensor data.
            try:
                for i, name in enumerate(TACTILE_SENSOR_NAMES):
                    if name not in env.scene.sensors:
                        continue
                    data = env.scene[name].data  # type: ignore[index]

                    nf = getattr(data, "tactile_normal_force", None)
                    sf = getattr(data, "tactile_shear_force", None)
                    if nf is not None and sf is not None:
                        e = min(env_idx, nf.shape[0] - 1)
                        nf_np = nf[e].view(nrows, ncols).detach().cpu().numpy()
                        sf_np = sf[e].view(nrows, ncols, 2).detach().cpu().numpy()
                        ff_img = compute_tactile_shear_image(nf_np, sf_np)
                        ff_ims[i].set_data(_ff_to_uint8(ff_img))

                    if args.show_rgb:
                        rgb = getattr(data, "tactile_rgb_image", None)
                        if rgb is not None and rgb.ndim == 4 and i < len(rgb_ims):
                            e = min(env_idx, rgb.shape[0] - 1)
                            rgb_ims[i].set_data(rgb[e].detach().cpu().numpy().astype(np.uint8))

                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                pass

            elapsed = time.time() - t0
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing environment and viewer...")
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            plt.close("all")
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()

