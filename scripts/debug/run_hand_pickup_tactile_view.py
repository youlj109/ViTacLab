#!/usr/bin/env python3
"""Run UR10e + ShadowHand pickup env and visualize TacSL tactile images.

Usage (inside Isaac Sim python):

    ./python.sh scripts/debug/run_hand_pickup_tactile_view.py --num_envs 1 --show_rgb --show_ff --enable_cameras

You can also enable random actions:

    ./python.sh scripts/debug/run_hand_pickup_tactile_view.py --num_envs 1 --show_rgb --show_ff --random_actions --enable_cameras
"""

from __future__ import annotations

import argparse
import os
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


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert tactile RGB image to uint8 [0,255] with simple normalization."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _ff_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert force-field viz image in [0,1] float to uint8."""
    if img.dtype != np.uint8:
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UR10e + ShadowHand pickup env and show TacSL tactile images.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    parser.add_argument("--env_index", type=int, default=0, help="Env index to visualize (default: 0).")
    parser.add_argument("--fps", type=float, default=20.0, help="Target display FPS (default: 20).")
    parser.add_argument(
        "--random_actions",
        action="store_true",
        help="Apply small random actions instead of zeros.",
    )
    parser.add_argument(
        "--show_rgb",
        action="store_true",
        help="Show tactile RGB images.",
    )
    parser.add_argument(
        "--show_ff",
        action="store_true",
        help="Show tactile force-field images.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import task after app launch
    import ViTacLab.tasks  # noqa: F401

    from ViTacLab.tasks.direct.simple_dexhand.hand_pickup import (
        UR10eShadowHandPickupEnv,
        UR10eShadowHandPickupEnvCfg,
    )
    from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import (
        compute_tactile_shear_image,
    )

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

        # Layout:
        # - if both RGB + FF: 2 rows x 5 cols
        # - if only one:       1 row x 5 cols
        if args.show_rgb and args.show_ff:
            fig, axes = plt.subplots(2, 5, figsize=(16, 6))
            ax_rgb = axes[0]
            ax_ff = axes[1]
        elif args.show_rgb:
            fig, ax_rgb = plt.subplots(1, 5, figsize=(16, 3))
        elif args.show_ff:
            fig, ax_ff = plt.subplots(1, 5, figsize=(16, 3))

    # Build env config
    cfg = UR10eShadowHandPickupEnvCfg()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"

    print(
        f"Creating UR10eShadowHandPickupEnv (device={cfg.device}, num_envs={cfg.scene.num_envs}) ..."
    )
    env = UR10eShadowHandPickupEnv(cfg)

    action_dim = env.num_actions
    print(f"Action dim: {action_dim}, Obs dim: {cfg.observation_space}")

    # Initialize env and nominal tactile render
    obs, _ = env.reset()
    policy_obs = obs.get("policy")
    if policy_obs is not None:
        print(f"Reset ok. policy obs shape: {tuple(policy_obs.shape)}")

    # Initialize tactile sensors' nominal render (for camera tactile)
    for name in TACTILE_SENSOR_NAMES:
        if name in env.scene.sensors:
            sensor = env.scene[name]
            try:
                sensor.get_initial_render()
            except Exception:
                pass

    # After env creation, determine tactile array size from first sensor
    if args.show_ff and fig is not None:
        import matplotlib.pyplot as plt

        for name in TACTILE_SENSOR_NAMES:
            if name in env.scene.sensors:
                nrows, ncols = env.scene[name].cfg.tactile_array_size
                break

        zero_ff = np.zeros((nrows * 30, ncols * 30, 3), dtype=np.uint8)

        if ax_ff is not None:
            if isinstance(ax_ff, np.ndarray):
                axes_ff = ax_ff
            else:
                axes_ff = [ax_ff]

            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if i >= len(axes_ff):
                    break
                title = name.replace("tactile_sensor_", "").upper()
                im = axes_ff[i].imshow(zero_ff)
                axes_ff[i].set_title(f"{title} FF")
                axes_ff[i].axis("off")
                ff_ims.append(im)

    if args.show_rgb and fig is not None:
        import matplotlib.pyplot as plt

        zero_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        if ax_rgb is not None:
            if isinstance(ax_rgb, np.ndarray):
                axes_rgb = ax_rgb
            else:
                axes_rgb = [ax_rgb]
            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if i >= len(axes_rgb):
                    break
                title = name.replace("tactile_sensor_", "").upper()
                im = axes_rgb[i].imshow(zero_rgb)
                axes_rgb[i].set_title(f"{title} RGB")
                axes_rgb[i].axis("off")
                rgb_ims.append(im)

    if fig is not None:
        import matplotlib.pyplot as plt

        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)

    target_dt = 1.0 / max(1e-3, float(args.fps))
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))

    print("Environment created. Starting tactile viewer (Ctrl+C to stop).")

    while simulation_app.is_running():
        t0 = time.time()

        if args.random_actions:
            actions = 0.05 * (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0)
        else:
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)

        obs, rewards, terminated, truncated, infos = env.step(actions)
        dones = terminated | truncated

        # Update tactile plots
        if fig is not None and (rgb_ims or ff_ims):
            import matplotlib.pyplot as plt

            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if name not in env.scene.sensors:
                    continue
                data = env.scene[name].data

                # RGB tactile image
                if rgb_ims and i < len(rgb_ims) and args.show_rgb:
                    img = getattr(data, "tactile_rgb_image", None)
                    if img is not None and img.ndim == 4:
                        e = min(env_idx, img.shape[0] - 1)
                        img_np = img[e].detach().cpu().numpy()
                        rgb_ims[i].set_data(_img_to_uint8(img_np))

                # Force-field image
                if ff_ims and i < len(ff_ims) and args.show_ff:
                    nf = getattr(data, "tactile_normal_force", None)
                    sf = getattr(data, "tactile_shear_force", None)
                    if nf is not None and sf is not None:
                        e = min(env_idx, nf.shape[0] - 1)
                        nf_np = nf[e].view(nrows, ncols).detach().cpu().numpy()
                        sf_np = sf[e].view(nrows, ncols, 2).detach().cpu().numpy()
                        ff_img = compute_tactile_shear_image(nf_np, sf_np)
                        ff_ims[i].set_data(_ff_to_uint8(ff_img))

            fig.canvas.draw_idle()
            plt.pause(0.001)

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    # simple cleanup after loop exits
    env.close()
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    simulation_app.close()


if __name__ == "__main__":
    main()

