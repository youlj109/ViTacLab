#!/usr/bin/env python3
"""Run UR10e + ShadowHand deformable cup pour environment for debugging.

Usage (inside Isaac Sim):

    ./python.sh scripts/debug/run_ur10e_shadowhand_pour_env.py --num_envs 1

With tactile force-field visualization:

    ./python.sh scripts/debug/run_ur10e_shadowhand_pour_env.py --num_envs 1 --show_tactile

With random actions:

    ./python.sh scripts/debug/run_ur10e_shadowhand_pour_env.py --num_envs 1 --random_actions
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


def _ff_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert force-field viz image in [0,1] float to uint8."""
    if img.dtype != np.uint8:
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert tactile RGB image to uint8 [0,255] with simple normalization."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    # If already in [0,1], scale up; otherwise clip to [0,255].
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UR10e + ShadowHand deformable cup pour env for debugging.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    parser.add_argument("--env_index", type=int, default=0, help="Env index to visualize (default: 0).")
    parser.add_argument("--max_steps", type=int, default=500, help="Max simulation steps (default: 500).")
    parser.add_argument("--fps", type=float, default=30.0, help="Target step rate (default: 30).")
    parser.add_argument(
        "--random_actions",
        action="store_true",
        help="Apply small random actions instead of zeros.",
    )
    parser.add_argument(
        "--show_tactile",
        action="store_true",
        help="Show TacSL tactile force-field images in a matplotlib window.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import task after app launch
    import ViTacLab.tasks  # noqa: F401

    from ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env import (
        UR10eShadowHandPourEnv,
    )
    from ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env_cfg import (
        UR10eShadowHandPourEnvCfg,
    )

    # Build config: few envs for debug
    cfg = UR10eShadowHandPourEnvCfg()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"

    print(
        f"Creating UR10eShadowHandPourEnv (device={cfg.device}, num_envs={cfg.scene.num_envs}) ..."
    )
    env = UR10eShadowHandPourEnv(cfg)

    action_dim = env.num_actions
    print(f"Action dim: {action_dim}, Obs dim: {cfg.observation_space}")

    # Optional tactile matplotlib window (show tactile RGB images)
    fig = None
    ax_rgb = None
    rgb_ims = []
    if args.show_tactile:
        import matplotlib
        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        fig, ax_rgb = plt.subplots(1, 5, figsize=(16, 3))
        zero_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        for i, name in enumerate(TACTILE_SENSOR_NAMES):
            title = name.replace("tactile_sensor_", "").upper()
            im = ax_rgb[i].imshow(zero_rgb)
            ax_rgb[i].set_title(f"{title} RGB")
            ax_rgb[i].axis("off")
            rgb_ims.append(im)
        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)

    # 初始化环境
    obs, _ = env.reset()
    policy_obs = obs.get("policy")
    if policy_obs is not None:
        print(f"Reset ok. policy obs shape: {tuple(policy_obs.shape)}")

    # 对所有 TacSL 触觉传感器先调用一次 get_initial_render，建立 nominal tactile
    # （VisuoTactileSensor 的相机触觉在第一次使用前必须先做这一帧基线渲染）
    for name in TACTILE_SENSOR_NAMES:
        if name in env.scene.sensors:
            sensor = env.scene[name]
            sensor.get_initial_render()

    target_dt = 1.0 / max(1e-3, float(args.fps))
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))
    step = 0

    # 主循环：只要窗口还在就一直跑
    while simulation_app.is_running():
        t0 = time.time()

        if args.random_actions:
            actions = 0.05 * (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0)
        else:
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)

        obs, rewards, terminated, truncated, infos = env.step(actions)
        step += 1
        dones = terminated | truncated

        # 每一步都打印，方便确认循环是否在跑
        r = rewards.mean().item()
        d = dones.sum().item()
        print(f"Step {step}: reward_mean={r:.4f}, dones={d}/{env.num_envs}")

    # 更新触觉可视化（RGB 图像）
    if args.show_tactile and fig is not None and ax_rgb is not None and len(rgb_ims) == 5:
        import matplotlib.pyplot as plt
        for i, name in enumerate(TACTILE_SENSOR_NAMES):
            if name not in env.scene.sensors:
                continue
            data = env.scene[name].data
            img = getattr(data, "tactile_rgb_image", None)
            if img is not None and img.ndim == 4 and i < len(rgb_ims):
                e = min(env_idx, img.shape[0] - 1)
                img_np = img[e].detach().cpu().numpy()  # (H, W, 3)
                rgb_ims[i].set_data(_img_to_uint8(img_np))
        fig.canvas.draw_idle()
        plt.pause(0.001)

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    print(f"Done after {step} steps.")

    # 收尾清理（让异常直接抛出，方便调试）
    env.close()
    if args.show_tactile:
        import matplotlib.pyplot as plt
        plt.close("all")
    simulation_app.close()


if __name__ == "__main__":
    main()
