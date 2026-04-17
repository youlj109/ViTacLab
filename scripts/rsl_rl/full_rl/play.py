# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
from datetime import datetime
import traceback
import gymnasium as gym
import numpy as np
import torch

# Shared helpers with ``ik_rl`` live under ``scripts/rsl_rl/ik_rl/utils``
_RSL_RL_ROOT = os.path.dirname(os.path.abspath(__file__))
_IK_UTILS = os.path.abspath(os.path.join(_RSL_RL_ROOT, "..", "ik_rl", "utils"))
if _IK_UTILS not in sys.path:
    sys.path.insert(0, _IK_UTILS)

from isaaclab.app import AppLauncher
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--show_rgb",
    action="store_true",
    default=False,
    help="Show TacSL tactile RGB in matplotlib (implies --enable_cameras for ViTacLab tasks).",
)
parser.add_argument(
    "--show_ff",
    action="store_true",
    default=False,
    help="Show TacSL tactile force-field (compute_tactile_shear_image arrows) in matplotlib.",
)
parser.add_argument(
    "--fps",
    type=float,
    default=20.0,
    help="Target display FPS when using --show_rgb / --show_ff (default: 20).",
)
parser.add_argument(
    "--env_index",
    type=int,
    default=0,
    help="Which sub-environment to visualize for tactile panels (default: 0).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video or show TacSL tactile streams
if args_cli.video or args_cli.show_rgb or args_cli.show_ff:
    args_cli.enable_cameras = True

args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ViTacLab.tasks  # noqa: F401
from ViTacLab.utils.vitaclab_marl_rsl import multi_agent_to_single_agent

from isaaclab.utils.math import (
    transform_points,
    unproject_depth,
)

def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Batched farthest point sampling (FPS).

    Args:
        xyz: Point coordinates of shape (B, N, 3).
        npoint: Number of points to sample.

    Returns:
        Tensor of shape (B, npoint, 3). If N < npoint, pads by repeating the last point.
    """
    B, N, _ = xyz.shape
    if N <= npoint:
        if N < npoint:
            pad = npoint - N
            xyz = torch.cat([xyz, xyz[:, -1:, :].expand(B, pad, -1)], dim=1)
        return xyz
    device = xyz.device
    dtype = xyz.dtype
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=dtype)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)
    for j in range(npoint):
        centroids[:, j] = farthest
        center = xyz[batch_idx, farthest].unsqueeze(1)
        dist = torch.sum((xyz - center) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    ii = batch_idx.unsqueeze(1).expand(B, npoint)
    return xyz[ii, centroids]


def _append_env_frame_to_buffer(buf, obs_record):
    for k, v in obs_record.items():
        if k not in buf:
            buf[k] = []
        buf[k].append(v.numpy())


def _apply_farthest_point_sample(env, env_idx, buf):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # World -> env-local: same convention as root_pos_w - env_origins (parallel envs share comparable xyz).
    env_origin = env.scene.env_origins[env_idx].to(device=device, dtype=torch.float32)

    def depth_to_pointcloud(depth, camera_intrinsic, camera_pos, camera_quat):
        depth_nhw = torch.from_numpy(depth.squeeze(-1)).to(device=device)
        pts_cam = unproject_depth(depth_nhw, camera_intrinsic, is_ortho=True)
        pts_w = transform_points(pts_cam, pos=camera_pos, quat=camera_quat)
        pts_env = pts_w - env_origin.unsqueeze(0)
        pts_env = _farthest_point_sample(pts_env.unsqueeze(0), 2048).squeeze(0).cpu().numpy()
        return pts_env

    ret_buf = dict()
    for k, v in buf.items():
        print(k, v[0].shape)
        if "_depth" in k:
            depth = buf[k]
            camera_name = k.replace("_depth", "")
            camera_intrinsic = env.scene[camera_name].data.intrinsic_matrices[env_idx]
            camera_pos = env.scene[camera_name].data.pos_w[env_idx]
            camera_quat = env.scene[camera_name].data.quat_w_ros[env_idx]
            pointcloud_name = camera_name + "_pointcloud_env"
            ret_buf[pointcloud_name] = [depth_to_pointcloud(depth_i, camera_intrinsic, camera_pos, camera_quat) for depth_i in depth]
            # 判断里面有没有nan
            if any(np.isnan(vi).any() for vi in ret_buf[pointcloud_name]):
                print(f"nan in {k}")
                print([np.isnan(vi).any() for vi in ret_buf[pointcloud_name]])
        else:
            ret_buf[k] = buf[k]
    return ret_buf

def _episode_buffer_to_npz_kwargs(env, env_idx, buf):
    buf = _apply_farthest_point_sample(env, env_idx, buf)
    # Skip optional keys that were never appended (e.g. no depth/pointcloud in record).
    return {k: np.stack(v, axis=0) for k, v in buf.items() if len(v) > 0}
    

def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


# Cell size in :func:`compute_tactile_shear_image` (``visuotactile_render.py``).
_TACTILE_SHEAR_VIZ_RESOLUTION = 30

_ff_cv2 = None
_ff_compute_tactile_shear_image = None


def _tactile_shear_image_rgb_uint8(nf_hw: np.ndarray, sf_hw2: np.ndarray) -> np.ndarray:
    """Force-field image from ``compute_tactile_shear_image``; OpenCV draws BGR, ``imshow`` uses RGB."""
    global _ff_cv2, _ff_compute_tactile_shear_image
    if _ff_compute_tactile_shear_image is None:
        import cv2

        from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image

        _ff_cv2 = cv2
        _ff_compute_tactile_shear_image = compute_tactile_shear_image

    img_bgr = _ff_compute_tactile_shear_image(nf_hw, sf_hw2)
    u8 = (np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _ff_cv2.cvtColor(u8, _ff_cv2.COLOR_BGR2RGB)


def _unwrap_to_isaac_env(env: object) -> object:
    """Unwrap ``RslRlVecEnvWrapper`` / gym wrappers / MARL conversion to an env with ``scene``."""
    cur: Any = env
    seen: set[int] = set()
    for _ in range(20):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "scene") and hasattr(cur.scene, "sensors"):
            return cur
        nxt = getattr(cur, "unwrapped", None)
        if nxt is None:
            nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise RuntimeError(
        "Could not unwrap to an Isaac Lab env with scene.sensors (needed for --show_rgb/--show_ff). "
        "Ensure --enable_cameras and a TacSL scene (e.g. UR10e + Shadow Hand)."
    )


def _collect_tactile_sensor_keys(scene_env: object) -> list[str]:
    sensors = getattr(scene_env.scene, "sensors", None)
    if sensors is None:
        return []
    return sorted(k for k in sensors.keys() if "tactile_sensor" in k)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # ForgeEnv (and similar) gate tactile + third_person_camera on cfg.enable_cameras, not only AppLauncher.
    # save_data reads obs["record"] which requires those sensors — mirror train.py injection.
    _enable_cams = bool(getattr(args_cli, "enable_cameras", False)) or bool(
        int(os.environ.get("ENABLE_CAMERAS", "0"))
    )
    if getattr(args_cli, "save_data", False):
        _enable_cams = True
    setattr(env_cfg, "enable_cameras", _enable_cams)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    env.unwrapped._use_rl_control = True

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    # 多环境：从 obs 或 env 获取 num_envs
    obs_policy = obs.get("policy", obs) if isinstance(obs, dict) else obs
    num_envs = obs_policy.shape[0] if hasattr(obs_policy, "shape") else getattr(env.unwrapped, "num_envs", 1)
    device = env.unwrapped.device

    # 每个环境独立的 episode 缓冲；所有 env 共享同一个 global episode 步数
    current_ep_buffers = [dict() for _ in range(num_envs)]
    step_in_episode = 0  # global episode step counter (since last env.reset())
    # 当前 global episode 中，各 env 是否已经保存过一条成功轨迹（避免重复保存）
    success_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)

    episodes_collected = 0
    total_timestep = 0
    max_steps_per_episode = args_cli.max_steps if args_cli.max_steps > 0 else None

    print(f"[INFO] Starting simulation. num_envs={num_envs}. Target: {args_cli.num_episodes} successful trajectories.")
    if max_steps_per_episode is not None:
        print(f"[INFO] Max steps per trajectory: {max_steps_per_episode} (trajectory discarded if no success by then).")
    run_until_target = args_cli.save_data and args_cli.num_episodes > 0
    if run_until_target:
        print("[INFO] Data recording: only successful trajectories are saved; run until target count or app closed.")
    else:
        is_running = simulation_app.is_running()
        print(f"[INFO] simulation_app.is_running() = {is_running} (loop depends on it).")

    # 保存收集到的多回合数据
    if args_cli.save_data:
        seed_val = agent_cfg.seed if agent_cfg.seed is not None else -1
        if args_cli.data_path is not None:
            data_dir = os.path.abspath(args_cli.data_path)
        else:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("data", "rsl_rl", task_name + "_" + str(seed_val), current_time))
        os.makedirs(data_dir, exist_ok=True)
        
    if args_cli.save_data:
        for i in range(num_envs):
            _append_env_frame_to_buffer(
                current_ep_buffers[i],
                obs["record"][i],
            )

    while True:
        if not simulation_app.is_running():
            if run_until_target and (episodes_collected > 0 or total_timestep > 0):
                print("[INFO] App no longer running; stopping and saving collected data.")
            elif not run_until_target:
                print("[INFO] App no longer running; exiting.")
            break
        if run_until_target and episodes_collected >= args_cli.num_episodes:
            break

    dt = env.unwrapped.step_dt

    fig = None
    rgb_ims: list = []
    ff_ims: list = []
    nrows, ncols = 20, 25
    tactile_keys: list[str] = []
    scene_env: object | None = None
    render_ff = None
    env_idx = max(0, min(int(args_cli.env_index), env.num_envs - 1))

    if args_cli.show_rgb or args_cli.show_ff:
        scene_env = _unwrap_to_isaac_env(env)
        tactile_keys = _collect_tactile_sensor_keys(scene_env)
        if not tactile_keys:
            sk = sorted(scene_env.scene.sensors.keys()) if getattr(scene_env.scene, "sensors", None) else []
            print(
                "[WARN] --show_rgb/--show_ff: no tactile_sensor_* in scene.sensors; viewer disabled.\n"
                "       Dual-arm ViTacLab: use scene cfg class UR10eDualShadowHandTacSLSceneCfg "
                "(not ...DirectSceneCfg) so GelSight/TacSL sensors are spawned when enable_cameras is set."
            )
            if sk:
                tail = " ..." if len(sk) > 25 else ""
                print(f"       Registered sensor keys ({len(sk)}): {sk[:25]}{tail}")
        else:
            import matplotlib

            matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
            import matplotlib.pyplot as plt

            if args_cli.show_ff:
                for name in tactile_keys:
                    if name in scene_env.scene.sensors:
                        try:
                            nrows, ncols = scene_env.scene[name].cfg.tactile_array_size
                        except Exception:
                            pass
                        break

            n_sensors = len(tactile_keys)
            w = max(12.0, 3.0 * n_sensors)
            if args_cli.show_rgb and args_cli.show_ff:
                fig, axes = plt.subplots(2, n_sensors, figsize=(w, 6))
                if n_sensors == 1:
                    ax_rgb = [axes[0]]
                    ax_ff = [axes[1]]
                else:
                    ax_rgb = [axes[0, j] for j in range(n_sensors)]
                    ax_ff = [axes[1, j] for j in range(n_sensors)]
            elif args_cli.show_rgb:
                fig, axr = plt.subplots(1, n_sensors, figsize=(w, 3))
                ax_rgb = [axr] if n_sensors == 1 else [axr[j] for j in range(n_sensors)]
                ax_ff = []
            else:
                fig, axf = plt.subplots(1, n_sensors, figsize=(w, 3))
                ax_rgb = []
                ax_ff = [axf] if n_sensors == 1 else [axf[j] for j in range(n_sensors)]

            zero_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
            for i, name in enumerate(tactile_keys):
                title = name.replace("tactile_sensor_", "").replace("_", " ").upper()
                if args_cli.show_rgb and i < len(ax_rgb):
                    im = ax_rgb[i].imshow(zero_rgb)
                    ax_rgb[i].set_title(f"{title} RGB")
                    ax_rgb[i].axis("off")
                    rgb_ims.append(im)
                if args_cli.show_ff and i < len(ax_ff):
                    zero_ff = np.zeros(
                        (max(1, nrows) * _TACTILE_SHEAR_VIZ_RESOLUTION, max(1, ncols) * _TACTILE_SHEAR_VIZ_RESOLUTION, 3),
                        dtype=np.uint8,
                    )
                    im = ax_ff[i].imshow(zero_ff)
                    ax_ff[i].set_title(f"{title} FF")
                    ax_ff[i].axis("off")
                    ff_ims.append(im)

            plt.tight_layout()
            fig.canvas.draw()
            plt.pause(0.1)
            render_ff = _tactile_shear_image_rgb_uint8 if args_cli.show_ff else None
            print(
                f"[INFO] Tactile viewer: {len(tactile_keys)} sensor(s), env_index={env_idx}, fps={float(args_cli.fps):.1f}"
            )

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            obs, _, dones, infos = env.step(actions)

        # 多环境：dones 展平为 (num_envs,)
        dones_flat = dones.flatten() if isinstance(dones, torch.Tensor) else torch.tensor(dones, device=device).flatten()
        if dones_flat.numel() != num_envs:
            dones_flat = dones_flat.expand(num_envs)

        if args_cli.save_data:
            for i in range(num_envs):
                _append_env_frame_to_buffer(
                    current_ep_buffers[i],
                    obs["record"][i],
                )

        total_timestep += 1
        step_in_episode += 1

        # 1) 成功：立即保存当前 buffer，但不 reset 环境；一个 global episode 中每个 env 只保存一次
        if args_cli.save_data:
            success_per_env = infos["curr_success_per_env"]
            for i in range(num_envs):
                if success_per_env[i].item() and not success_saved[i].item():
                    ep_data_torch = current_ep_buffers[i]
                    np.savez_compressed(
                        os.path.join(data_dir, f"episode_{episodes_collected}.npz"),
                        **_episode_buffer_to_npz_kwargs(env.unwrapped, i, ep_data_torch),
                    )
                    episodes_collected += 1
                    success_saved[i] = True
                    print(f"[INFO] Env {i} success at global step {step_in_episode}; saved episode {episodes_collected} / {args_cli.num_episodes}")
            if episodes_collected >= args_cli.num_episodes:
                break

        # 2) global max_steps：此时才调用 env.reset()，并整体清空缓冲
        if max_steps_per_episode is not None and step_in_episode >= max_steps_per_episode:
            if args_cli.save_data:
                for i in range(num_envs):
                    if not success_saved[i].item():
                        # ep_data_torch = current_ep_buffers[i]
                        # np.savez_compressed(
                        #     os.path.join(data_dir, f"episode_{episodes_collected}.npz"),
                        #     **_episode_buffer_to_npz_kwargs(env.unwrapped, i, ep_data_torch),
                        # )
                        # episodes_collected += 1
                        print(f"[INFO] Env {i} discarded (max_steps={max_steps_per_episode} without success).")
            current_ep_buffers = [dict() for _ in range(num_envs)]
            success_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)
            step_in_episode = 0

            # reset policy 隐状态与环境本身
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                with torch.inference_mode():
                    policy_nn.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
            with torch.inference_mode():
                obs, _ = env.reset()

            if args_cli.save_data and episodes_collected >= args_cli.num_episodes:
                break

        if fig is not None and (rgb_ims or ff_ims) and scene_env is not None and tactile_keys:
            import matplotlib.pyplot as plt

            for i, name in enumerate(tactile_keys):
                if name not in scene_env.scene.sensors:
                    continue
                data = scene_env.scene[name].data

                if args_cli.show_rgb and rgb_ims and i < len(rgb_ims):
                    img = getattr(data, "tactile_rgb_image", None)
                    if img is not None and img.ndim == 4:
                        e = min(env_idx, img.shape[0] - 1)
                        rgb_ims[i].set_data(_img_to_uint8(img[e].detach().cpu().numpy()))

                if args_cli.show_ff and ff_ims and i < len(ff_ims) and render_ff is not None:
                    nf = getattr(data, "tactile_normal_force", None)
                    
                    sf = getattr(data, "tactile_shear_force", None)
                    if nf is not None and sf is not None:
                        e = min(env_idx, nf.shape[0] - 1)
                        nf_flat = nf[e].detach().cpu().numpy().reshape(-1)
                        sf_flat = sf[e].detach().cpu().numpy().reshape(-1, 2)
                        p = int(nf_flat.shape[0])
                        nrows_guess, ncols_guess = nrows, ncols
                        if p != nrows_guess * ncols_guess:
                            nrows_guess = int(np.sqrt(p))
                            ncols_guess = max(1, p // max(1, nrows_guess))
                        nf_img = nf_flat.reshape(nrows_guess, ncols_guess)
                        sf_img = sf_flat.reshape(nrows_guess, ncols_guess, 2)
                        ff_ims[i].set_data(render_ff(nf_img, sf_img))

            fig.canvas.draw_idle()
            plt.pause(0.001)

        elapsed = time.time() - start_time
        if args_cli.show_rgb or args_cli.show_ff:
            view_dt = 1.0 / max(1e-3, float(args_cli.fps))
            sleep_v = view_dt - elapsed
            if sleep_v > 0:
                time.sleep(sleep_v)
        elif args_cli.real_time:
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()