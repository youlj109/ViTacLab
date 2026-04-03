from __future__ import annotations
import argparse
from mimetypes import init
import os
import sys
import time
import traceback
from datetime import datetime
import math
import gymnasium as gym
import numpy as np
import torch
import imageio

from isaaclab.app import AppLauncher
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# cli_args lives under scripts/rsl_rl/ik_rl/utils (same as full_rl/play.py)
_POLICY_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IK_RL_UTILS = os.path.abspath(os.path.join(_POLICY_SCRIPT_DIR, "..", "rsl_rl", "ik_rl", "utils"))
if _IK_RL_UTILS not in sys.path:
    sys.path.insert(0, _IK_RL_UTILS)

import cli_args  # isort: skip
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play and record data with an RSL-RL agent.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--data_num", type=int, default=200, help="Number of data to use for the policy.")
    parser.add_argument("--checkpoint_num", type=int, default=1000, help="Number of checkpoint to use for the policy.")
    parser.add_argument("--policy_name", type=str, default="ViTacDP", help="Name of the policy.")
    parser.add_argument("--version", type=str, default=None, help="Version of the policy.")
    
    parser.add_argument(
        "--max_steps", type=int, default=100,
        help="When save_data: max steps per trajectory; if reached without success, trajectory is discarded. num_episodes counts only successful trajectories. 0 = no step limit (only success or env done ends trajectory).",
    )

    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_arg_parser()
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import ViTacLab.tasks  # noqa: F401

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

def _apply_farthest_point_sample(buf):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in buf.items():
        if "pointcloud" in k:
            buf[k] = _farthest_point_sample(v.to(device=device), 2048).cpu()
    return buf

def init_episode_buffers():
    buffers = {
        "joint_pos": [],
        "tactile_normal_force": [],
        "tactile_shear_force": [],
        "tactile_rgb_image": [],
        "third_person_camera": []
    }
    return buffers

def _append_env_frame_to_buffer(
    buf,
    obs_record,
):
    buf["joint_pos"].append(obs_record["joint_pos"].numpy())
    buf["tactile_normal_force"].append(obs_record["tactile_normal_force"].numpy())
    buf["tactile_shear_force"].append(obs_record["tactile_shear_force"].numpy())
    buf["tactile_rgb_image"].append(obs_record["tactile_rgb_image"].numpy())
    buf["third_person_camera"].append(obs_record["third_person_camera"].numpy())
    
def save_result(save_dir, episode_count, current_ep_buffers, success_episode):
    for i in range(len(current_ep_buffers)):
        idx = episode_count + i
        video_writer = imageio.get_writer(os.path.join(save_dir, f"episode_{idx}.mp4"), fps=20)
        for frame in current_ep_buffers[i]["third_person_camera"]:
            video_writer.append_data(frame)
        video_writer.close()
    with open(os.path.join(save_dir, f"all_success.txt"), "a") as f:
        for success in success_episode:
            f.write(f"{success}\n")
        

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    save_dir = os.path.join("data", "validation", task_name, args_cli.policy_name + "_" + args_cli.version if args_cli.version else args_cli.policy_name, str(args_cli.data_num) + "_" + str(args_cli.checkpoint_num))
    start_episode_index = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    else:
        saved_episodes = [int(file.split("_")[-1].split(".")[0]) for file in os.listdir(save_dir) if file.endswith(".mp4")]
        if len(saved_episodes) == 0:
            start_episode_index = 0
        else:
            start_episode_index = max(saved_episodes) + 1
    print(f"[INFO] Start episode index: {start_episode_index}")
    
    # ForgeEnv (and similar) gate tactile + third_person_camera on cfg.enable_cameras, not only AppLauncher.
    # save_data reads obs["record"] which requires those sensors — mirror train.py injection.
    _enable_cams = bool(getattr(args_cli, "enable_cameras", False)) or bool(
        int(os.environ.get("ENABLE_CAMERAS", "0"))
    )
    if getattr(args_cli, "save_data", False):
        _enable_cams = True
    setattr(env_cfg, "enable_cameras", _enable_cams)

    # 延长内置 episode 时限，避免 DirectRLEnv 因 timeout 自动 reset（Factory/Forge 由 episode_length_s 推导步数上限）。
    _play_policy_env_steps_cap = 1_000_000
    if hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg, "decimation"):
        _step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
        env_cfg.episode_length_s = float(_play_policy_env_steps_cap) * _step_dt
    elif hasattr(env_cfg, "max_episode_length"):
        env_cfg.max_episode_length = int(_play_policy_env_steps_cap)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    base_env = env.unwrapped
    base_env._use_rl_control = False
    base_env.set_stiffness_damping(True)
    dt = base_env.step_dt
    obs = env.get_observations()
    # Align joint position target with current state to avoid first-frame jerk.
    if hasattr(base_env, "apply_joint_targets") and "record" in obs and "joint_pos" in obs["record"]:
        base_env.apply_joint_targets(obs["record"]["joint_pos"].to(device=base_env.device))

    # 多环境：从 obs 或 env 获取 num_envs
    obs_policy = obs.get("policy", obs) if isinstance(obs, dict) else obs
    num_envs = obs_policy.shape[0] if hasattr(obs_policy, "shape") else getattr(base_env, "num_envs", 1)
    device = base_env.device

    # 模仿学习：DP 输出关节角，直接下发给 robot；用哑 action 推 env.step
    action_dim = getattr(base_env.cfg, "action_space", 7)
    dummy_actions = torch.zeros(num_envs, action_dim, device=device, dtype=torch.float32)

    import importlib

    Policy_Encapsulation = getattr(importlib.import_module(f"policy.{args_cli.policy_name}.deploy_policy"), "Encapsulation")
    init_dict = {
        "task_name": task_name + "_" + args_cli.version if args_cli.version else task_name,
        "data_num": args_cli.data_num,
        "checkpoint_num": args_cli.checkpoint_num,
        "checkpoint_path": os.path.join("policy", "ImplicitRDP", "data", "outputs", "2026.03.15", "15.42.48_train_reactive_diffusion_transformer_image_insert_v2_policy_bigger_acp", "checkpoints", "latest.ckpt")
    }
    policy = Policy_Encapsulation(init_dict, num_envs)
    
    current_ep_buffers = [init_episode_buffers() for _ in range(num_envs)]

    # 每个环境独立的 episode 缓冲；所有 env 共享同一个 global episode 步数
    step_in_episode = 0  # global episode step counter (since last env.reset())
    
    success_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
    max_steps_per_episode = args_cli.max_steps

    success_count = 0
    episode_count = start_episode_index
    
    for i in range(num_envs):
        _append_env_frame_to_buffer(
            current_ep_buffers[i],
            obs["record"][i],
        )

    while True:
        if not simulation_app.is_running():
            break

        start_time = time.time()
        
        with torch.inference_mode():
            joint_pos_cmd = torch.tensor(policy.get_action(_apply_farthest_point_sample(obs["record"])), device=device)
            base_env.apply_joint_targets(joint_pos_cmd)
            obs, rewards, dones, infos = env.step(dummy_actions)
            obs, rewards, dones, infos = env.step(dummy_actions)
            obs, rewards, dones, infos = env.step(dummy_actions)
            obs, rewards, dones, infos = env.step(dummy_actions)
            obs, rewards, dones, infos = env.step(dummy_actions)

            # 多环境：dones 展平为 (num_envs,)
            dones_flat = dones.flatten() if isinstance(dones, torch.Tensor) else torch.tensor(dones, device=device).flatten()
            if dones_flat.numel() != num_envs:
                dones_flat = dones_flat.expand(num_envs)

            step_in_episode += 1
            
            for i in range(num_envs):
                _append_env_frame_to_buffer(
                    current_ep_buffers[i],
                    obs["record"][i],
                )

            # 1) 成功：立即保存当前 buffer，但不 reset 环境；一个 global episode 中每个 env 只保存一次
            success_per_env = infos["curr_success_per_env"]
            for i in range(num_envs):
                if success_per_env[i].item() and not success_episode[i]:
                    # 任务成功
                    print(f"[INFO] Env {i} success at global step {step_in_episode}")
                    success_count += 1
                    success_episode[i] = True
                
            # 2) global max_steps：此时才调用 env.reset()，并整体清空缓冲
            if max_steps_per_episode is not None and step_in_episode >= max_steps_per_episode:
                print(f"[INFO] Episode {episode_count} ended")
                save_result(save_dir, episode_count, current_ep_buffers, success_episode)
                step_in_episode = 0
                episode_count += 1

                break

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    
    print(f"[INFO] Success count: {success_count}")
    print(f"[INFO] Episode count: {episode_count}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()