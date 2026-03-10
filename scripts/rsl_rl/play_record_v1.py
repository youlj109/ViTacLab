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

    # data recording options
    parser.add_argument(
        "--save_data",
        action="store_true",
        default=False,
        help="If set, save play trajectory data (including tactile) to disk.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Directory to save recorded data. Defaults to 'data/rsl_rl/<TASK_NAME>/<current_time>'.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum number of environment steps to run. 0 means run until the app is closed.",
    )
    parser.add_argument(
        "--save_h5",
        action="store_true",
        default=False,
        help="Additionally save data as an HDF5 (.h5) file.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        default=False,
        help="Additionally save data as a NumPy .npz file.",
    )
    
    # 可选数据保存开关 (默认不保存，如有需要请在命令中添加)
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

# always enable cameras if we want video or save_data (tactile sensors need camera rendering)
if args_cli.video or args_cli.save_data:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
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
    

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and optionally record trajectory data."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_record"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play_record.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # 必须保存的数据缓冲区
    actions_buf = []
    tactile_normal_force_buf = []
    tactile_shear_force_buf = []
    tactile_rgb_image_buf = []
    camera_rgb_buffers = {}

    # 可选数据缓冲区 (仅在参数开启时启用)
    obs_buf = [] if args_cli.save_obs else None
    rewards_buf = [] if args_cli.save_rewards else None
    dones_buf = [] if args_cli.save_dones else None

    max_steps = args_cli.max_steps if args_cli.max_steps > 0 else None

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

        # record data if requested
        if args_cli.save_data:
            # 1. 记录必须保存的 action
            actions_buf.append(actions.detach().cpu())

            # 2. 按需记录可选数据
            if obs_buf is not None:
                obs_tensor = obs["policy"] if (hasattr(obs, "__getitem__") and "policy" in obs) else obs
                if not isinstance(obs_tensor, torch.Tensor):
                    raise RuntimeError("Observation is not a tensor; cannot save.")
                obs_buf.append(obs_tensor.detach().cpu())
            
            if rewards_buf is not None:
                rewards_buf.append(rewards.detach().cpu())
            
            if dones_buf is not None:
                dones_buf.append(dones.detach().cpu())

            # 3. 记录触觉和相机数据 (默认必须项)
            base_env = env.unwrapped
            if hasattr(base_env, "tactile_normal_force"):
                tactile_normal_force_buf.append(base_env.tactile_normal_force.detach().cpu())
            if hasattr(base_env, "tactile_shear_force"):
                tactile_shear_force_buf.append(base_env.tactile_shear_force.detach().cpu())
            if hasattr(base_env, "tactile_rgb_image"):
                tactile_rgb_image_buf.append(base_env.tactile_rgb_image.detach().cpu())

            if hasattr(base_env, "scene") and getattr(base_env.scene, "sensors", None):
                for cam_name in base_env.scene.sensors:
                    try:
                        cam = base_env.scene[cam_name]
                        if not hasattr(cam, "data") or not getattr(cam.data, "output", None):
                            continue
                        out = cam.data.output
                        if "rgb" not in out:
                            continue
                        rgb = out["rgb"]
                        if rgb is not None and rgb.numel() > 0:
                            key = f"camera_rgb_{cam_name}"
                            if key not in camera_rgb_buffers:
                                camera_rgb_buffers[key] = []
                            camera_rgb_buffers[key].append(rgb.detach().cpu())
                    except Exception:
                        pass

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if max_steps is not None and timestep >= max_steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # save recorded data
    if args_cli.save_data and len(actions_buf) > 0:
        if args_cli.data_path is not None:
            data_dir = os.path.abspath(args_cli.data_path)
        else:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("data", "rsl_rl", task_name, current_time))
        os.makedirs(data_dir, exist_ok=True)

        # 构建 torch tensor 字典，只放入已启用的数据
        data_torch = {"actions": torch.stack(actions_buf, dim=0)}
        if obs_buf: data_torch["obs"] = torch.stack(obs_buf, dim=0)
        if rewards_buf: data_torch["rewards"] = torch.stack(rewards_buf, dim=0)
        if dones_buf: data_torch["dones"] = torch.stack(dones_buf, dim=0)
        
        if len(tactile_normal_force_buf) > 0:
            data_torch["tactile_normal_force"] = torch.stack(tactile_normal_force_buf, dim=0)
        if len(tactile_shear_force_buf) > 0:
            data_torch["tactile_shear_force"] = torch.stack(tactile_shear_force_buf, dim=0)
        if len(tactile_rgb_image_buf) > 0:
            data_torch["tactile_rgb_image"] = torch.stack(tactile_rgb_image_buf, dim=0)
        for cam_key, buf in camera_rgb_buffers.items():
            if len(buf) > 0:
                data_torch[cam_key] = torch.stack(buf, dim=0)

        # 转换为 numpy 格式
        data_numpy = {}
        for k, v in data_torch.items():
            arr = v.cpu().numpy()
            data_numpy[k] = np.array(arr, copy=True)

        seed_val = agent_cfg.seed if agent_cfg.seed is not None else -1
        num_envs = getattr(env.unwrapped, "num_envs", None) or env_cfg.scene.num_envs
        base_filename = f"{train_task_name}_play_record_seed{seed_val}_envs{num_envs}_steps{timestep}"

        # 始终保存 .pt
        pt_path = os.path.join(data_dir, base_filename + ".pt")
        torch.save(data_torch, pt_path)
        print(f"[INFO] Saved play data to: {pt_path}")

        # 可选保存 .h5
        if args_cli.save_h5:
            h5_path = os.path.join(data_dir, base_filename + ".h5")
            with h5py.File(h5_path, "w") as f:
                for k, v in data_numpy.items():
                    f.create_dataset(k, data=v, compression="gzip", dtype=v.dtype)
            print(f"[INFO] Saved play data to: {h5_path}")

        # 可选保存 .npz
        if args_cli.save_npz:
            npz_path = os.path.join(data_dir, base_filename + ".npz")
            np.savez_compressed(npz_path, **data_numpy)
            print(f"[INFO] Saved play data to: {npz_path}")

    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()