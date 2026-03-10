#!/usr/bin/env python3
"""Run Isaac-Forge-PegInsert-Direct-v0 and display tactile RGB images in real time.

Usage (from repo root, inside Isaac Sim Python):

    ./python.sh scripts/debug/run_forge_peg_tactile_view.py

You can also use the system Python if your PYTHONPATH already includes Isaac Lab and this project:

    python scripts/debug/run_forge_peg_tactile_view.py
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np
import torch
from isaaclab.app import AppLauncher


TACTILE_H, TACTILE_W = 240, 320
TACTILE_NUM_SENSORS = 2
TACTILE_FLAT_PER_SENSOR = TACTILE_H * TACTILE_W * 3  # 230400
TACTILE_FLAT_TOTAL = TACTILE_NUM_SENSORS * TACTILE_FLAT_PER_SENSOR  # 460800


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float or int image to uint8 [0, 255]."""
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(img, 0.0, 255.0).astype(np.uint8)
    return np.clip(img.astype(np.float64), 0.0, 255.0).astype(np.uint8)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser (including Isaac AppLauncher args, like train/play scripts)."""
    parser = argparse.ArgumentParser(
        description="Run Isaac-Forge-PegInsert-Direct-v0 and show tactile RGB images in real time.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target display FPS (controls sleep between steps; default: 20).",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="If set, apply small random actions instead of zeros.",
    )
    parser.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Index of env to visualize when num-envs > 1 (default: 0).",
    )
    # Follow scripts/rsl_rl/train.py & play.py: append AppLauncher CLI args
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Launch Isaac Sim app first (same pattern as train.py / play.py).
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Lazily import matplotlib after app launch (avoids backend issues).
    import matplotlib

    matplotlib.use("TkAgg" if "DISPLAY" in __import__("os").environ else "Agg")
    import matplotlib.pyplot as plt

    # Import task modules only after Isaac app is running, to avoid omni import issues.
    import isaaclab_tasks  # noqa: F401
    import ViTacLab.tasks  # noqa: F401
    from ViTacLab.tasks.direct.simple_gripper.forge_env import ForgeEnv
    from ViTacLab.tasks.direct.simple_gripper.forge_env_cfg import ForgeTaskPegInsertCfg

    # Build environment configuration for the PegInsert task.
    cfg = ForgeTaskPegInsertCfg()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    # Use full observation mode so that tactile_* are included in observations as well.
    cfg.obs_mode = "full"
    # Use AppLauncher-provided device (same flag name as train/play scripts).
    device = getattr(args, "device", None) or "cuda:0"
    cfg.device = device

    print(f"Creating ForgeEnv for PegInsert on device={cfg.device} with num_envs={cfg.scene.num_envs} ...")
    env = ForgeEnv(cfg)

    try:
        # Initial reset (DirectRLEnv style: returns obs dict and extras).
        obs, extras = env.reset()
        print("Environment created. Starting real-time tactile viewer (Ctrl+C to stop).")

        # Prepare Matplotlib figure with two subplots (left/right tactile RGB).
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 4))
        plt.tight_layout()

        # Initialize with zeros until first data arrives.
        zero_img = np.zeros((TACTILE_H, TACTILE_W, 3), dtype=np.uint8)
        im_left = ax_left.imshow(zero_img)
        ax_left.set_title("Tactile Left")
        ax_left.axis("off")

        im_right = ax_right.imshow(zero_img)
        ax_right.set_title("Tactile Right")
        ax_right.axis("off")

        fig.canvas.draw()
        plt.pause(0.1)

        target_dt = 1.0 / max(1e-3, float(args.fps))
        env_idx = max(0, int(args.env_index))

        # Main simulation and visualization loop.
        while simulation_app.is_running():
            t0 = time.time()

            # Build actions tensor on the correct device (zero or small random, does not matter for visualization).
            action_dim = int(env.cfg.action_space)
            if args.random_actions:
                actions = 0.05 * (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0)
            else:
                actions = torch.zeros(env.num_envs, action_dim, device=env.device)

            # Step environment.
            obs, rew, terminated, truncated, extras = env.step(actions)

            # Read tactile RGB images directly from TacSL sensor data (similar to tacsl_sensor.py).
            try:
                # Left sensor
                if "tactile_sensor_left" in env.scene.sensors:
                    data_left = env.scene["tactile_sensor_left"].data  # type: ignore[index]
                    img_left = getattr(data_left, "tactile_rgb_image", None)
                else:
                    img_left = None

                # Right sensor
                if "tactile_sensor_right" in env.scene.sensors:
                    data_right = env.scene["tactile_sensor_right"].data  # type: ignore[index]
                    img_right = getattr(data_right, "tactile_rgb_image", None)
                else:
                    img_right = None

                # Update left image
                if img_left is not None and img_left.ndim == 4:
                    e = min(env_idx, img_left.shape[0] - 1)
                    arr_l = img_left[e].detach().cpu().numpy()  # (H, W, 3)
                    im_left.set_data(_to_uint8(arr_l))

                # Update right image
                if img_right is not None and img_right.ndim == 4:
                    e = min(env_idx, img_right.shape[0] - 1)
                    arr_r = img_right[e].detach().cpu().numpy()  # (H, W, 3)
                    im_right.set_data(_to_uint8(arr_r))

                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                # Keep simulation robust even if visualization fails for a few frames.
                pass

            # Simple FPS control.
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
        plt.close("all")
        # Close Isaac Sim app
        simulation_app.close()


if __name__ == "__main__":
    main()

