from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import h5py

from isaaclab.app import AppLauncher
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import cli_args  # isort: skip

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play and record data with an RSL-RL agent.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

    # data recording options
    parser.add_argument("--save_data", action="store_true", default=False, help="If set, save play trajectory data to disk.")
    parser.add_argument("--data_path", type=str, default=None, help="Directory to save recorded data.")
    
    # 新增：控制回合数的参数
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to record when save_data is enabled. Forces num_envs=1.")
    
    parser.add_argument(
        "--max_steps", type=int, default=0,
        help="Maximum steps per episode. When reached, the current episode is truncated and saved, then a new episode starts. 0 = no limit (run until env returns done).",
    )
    parser.add_argument("--save_h5", action="store_true", default=False, help="Additionally save data as an HDF5 (.h5) file.")
    parser.add_argument("--save_npz", action="store_true", default=False, help="Additionally save data as a NumPy .npz file.")
    
    # 可选数据保存开关
    parser.add_argument("--save_obs", action="store_true", default=False, help="Whether to save observations.")
    parser.add_argument("--save_rewards", action="store_true", default=False, help="Whether to save rewards.")
    parser.add_argument("--save_dones", action="store_true", default=False, help="Whether to save dones.")

    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_arg_parser()
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video or args_cli.save_data:
    args_cli.enable_cameras = True

# 核心修改点 1：开启保存数据时，强制设定 num_envs=1 以便按回合(episode)收集
if args_cli.save_data:
    print(f"[INFO] Data recording enabled. Forcing num_envs=1 to record {args_cli.num_episodes} episodes.")
    args_cli.num_envs = 1

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
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import ViTacLab.tasks  # noqa: F401
    

# 辅助函数：初始化单个 episode 的缓冲区
def init_episode_buffers(args):
    buffers = {
        "actions": [],
        "tactile_normal_force": [],
        "tactile_shear_force": [],
        "tactile_rgb_image": [],
        "camera_rgb": {}
    }
    if args.save_obs: buffers["obs"] = []
    if args.save_rewards: buffers["rewards"] = []
    if args.save_dones: buffers["dones"] = []
    return buffers


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_record"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    # 核心修改点 2：建立全局 episodes 列表和当前 episode 缓冲
    all_episodes_data = []
    current_ep_buffers = init_episode_buffers(args_cli)
    
    episodes_collected = 0
    total_timestep = 0
    step_in_episode = 0  # 当前 episode 已跑步数，用于 max_steps（每 episode 上限）
    max_steps_per_episode = args_cli.max_steps if args_cli.max_steps > 0 else None

    print(f"[INFO] Starting simulation. Target: {args_cli.num_episodes} episodes.")
    if max_steps_per_episode is not None:
        print(f"[INFO] Max steps per episode: {max_steps_per_episode} (episode will be truncated and saved when reached).")
    # 当 save_data 时以「达到目标回合数」为主条件
    run_until_target = args_cli.save_data and args_cli.num_episodes > 0
    if run_until_target:
        print("[INFO] Data recording mode: loop will run until target episodes (or until app is closed).")
    else:
        is_running = simulation_app.is_running()
        print(f"[INFO] simulation_app.is_running() = {is_running} (loop depends on it).")

    while True:
        if not simulation_app.is_running():
            if run_until_target and (episodes_collected > 0 or total_timestep > 0):
                print("[INFO] App no longer running; stopping and saving collected data.")
            elif not run_until_target:
                print("[INFO] App no longer running; exiting.")
            break
        if run_until_target and episodes_collected >= args_cli.num_episodes:
            break
        if args_cli.video and total_timestep >= args_cli.video_length:
            break

        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

        if args_cli.save_data:
            # 去除 num_envs=1 的 batch 维度(squeeze(0))，彻底避免后续形状污染
            current_ep_buffers["actions"].append(actions.detach().cpu().squeeze(0))

            if args_cli.save_obs:
                obs_tensor = obs["policy"] if (hasattr(obs, "__getitem__") and "policy" in obs) else obs
                current_ep_buffers["obs"].append(obs_tensor.detach().cpu().squeeze(0))
            if args_cli.save_rewards:
                current_ep_buffers["rewards"].append(rewards.detach().cpu().squeeze(0))
            if args_cli.save_dones:
                current_ep_buffers["dones"].append(dones.detach().cpu().squeeze(0))

            base_env = env.unwrapped
            if hasattr(base_env, "tactile_normal_force"):
                current_ep_buffers["tactile_normal_force"].append(base_env.tactile_normal_force.detach().cpu().squeeze(0))
            if hasattr(base_env, "tactile_shear_force"):
                current_ep_buffers["tactile_shear_force"].append(base_env.tactile_shear_force.detach().cpu().squeeze(0))
            # 触觉图像保持原始空间结构 (num_sensors, H, W, 3)，不保存展平后的最后一维
            if hasattr(base_env, "tactile_rgb_image"):
                t = base_env.tactile_rgb_image.detach().cpu().squeeze(0)
                if t.dim() == 1 and hasattr(base_env, "tactile_image_height") and hasattr(base_env, "tactile_image_width"):
                    H = getattr(base_env, "tactile_image_height")
                    W = getattr(base_env, "tactile_image_width")
                    per = H * W * 3
                    n_sensors = t.numel() // per
                    imgs = [t[i * per : (i + 1) * per].reshape(H, W, 3) for i in range(n_sensors)]
                    current_ep_buffers["tactile_rgb_image"].append(torch.stack(imgs, dim=0))
                else:
                    current_ep_buffers["tactile_rgb_image"].append(t)

            if hasattr(base_env, "scene") and getattr(base_env.scene, "sensors", None):
                for cam_name in base_env.scene.sensors:
                    try:
                        cam = base_env.scene[cam_name]
                        if not hasattr(cam, "data") or not getattr(cam.data, "output", None): continue
                        out = cam.data.output
                        if "rgb" not in out: continue
                        rgb = out["rgb"]
                        if rgb is not None and rgb.numel() > 0:
                            key = f"camera_rgb_{cam_name}"
                            if key not in current_ep_buffers["camera_rgb"]:
                                current_ep_buffers["camera_rgb"][key] = []
                            current_ep_buffers["camera_rgb"][key].append(rgb.detach().cpu().squeeze(0))
                    except Exception:
                        pass

        total_timestep += 1
        step_in_episode += 1

        # 判断回合是否结束：env 返回 done，或达到本 episode 的 max_steps（截断）
        env_done = dones.item() if dones.dim() == 0 else dones.any().item()
        truncated = max_steps_per_episode is not None and step_in_episode >= max_steps_per_episode
        episode_ended = env_done or truncated
        if truncated and args_cli.save_data:
            print(f"[INFO] Episode truncated at max_steps={max_steps_per_episode} steps.")
        if args_cli.save_data and episode_ended:
            # 将当前 list 组装成 tensor
            ep_data_torch = {}
            for k, v in current_ep_buffers.items():
                if k == "camera_rgb":
                    for cam_key, cam_buf in v.items():
                        if len(cam_buf) > 0: ep_data_torch[cam_key] = torch.stack(cam_buf, dim=0)
                elif len(v) > 0:
                    ep_data_torch[k] = torch.stack(v, dim=0)
            
            all_episodes_data.append(ep_data_torch)
            episodes_collected += 1
            print(f"[INFO] Collected episode {episodes_collected} / {args_cli.num_episodes}")

            # 重置当前缓冲区和本 episode 步数
            current_ep_buffers = init_episode_buffers(args_cli)
            step_in_episode = 0

            # 若因 max_steps 截断（env 未返回 done），需主动 reset 以开始新 episode（须在 inference_mode 内，否则 env 内原地写会报错）
            if truncated:
                with torch.inference_mode():
                    obs, _ = env.reset()
            if episodes_collected >= args_cli.num_episodes:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 若开启了保存但未收集到任何数据，提示可能原因
    if args_cli.save_data and len(all_episodes_data) == 0:
        msg = (
            "[WARN] save_data was set but no episodes were collected, so nothing was saved. "
            "Often this is because simulation_app.is_running() was False from the start. "
            "If you are on a machine without display (e.g. SSH), try running with: --headless"
        )
        print(msg)

    # 保存收集到的多回合数据
    if args_cli.save_data and len(all_episodes_data) > 0:
        if args_cli.data_path is not None:
            data_dir = os.path.abspath(args_cli.data_path)
        else:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("data", "rsl_rl", task_name, current_time))
        os.makedirs(data_dir, exist_ok=True)

        seed_val = agent_cfg.seed if agent_cfg.seed is not None else -1
        base_filename = f"{train_task_name}_play_record_seed{seed_val}_eps{episodes_collected}"

        # 1. 保存为 .pt (List of Dicts)
        pt_path = os.path.join(data_dir, base_filename + ".pt")
        torch.save(all_episodes_data, pt_path)
        print(f"[INFO] Saved {episodes_collected} episodes to: {pt_path}")

        # 2. 保存为 .h5 (利用 Group 层级结构存储多回合)
        if args_cli.save_h5:
            h5_path = os.path.join(data_dir, base_filename + ".h5")
            with h5py.File(h5_path, "w") as f:
                for i, ep_data in enumerate(all_episodes_data):
                    ep_group = f.create_group(f"episode_{i}")
                    for k, v in ep_data.items():
                        arr = v.numpy()
                        ep_group.create_dataset(k, data=arr, compression="gzip", dtype=arr.dtype)
            print(f"[INFO] Saved HDF5 to: {h5_path}")

        # 3. 保存为 .npz (展开格式: episode_0_actions, episode_1_actions)
        if args_cli.save_npz:
            npz_dict = {}
            for i, ep_data in enumerate(all_episodes_data):
                for k, v in ep_data.items():
                    npz_dict[f"episode_{i}_{k}"] = v.numpy()
            npz_path = os.path.join(data_dir, base_filename + ".npz")
            np.savez_compressed(npz_path, **npz_dict)
            print(f"[INFO] Saved NPZ to: {npz_path}")

    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()