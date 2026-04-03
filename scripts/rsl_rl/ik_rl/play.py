# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play an IK-RL checkpoint and optionally save ``obs["record"]`` trajectories."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import os
import platform
import sys
import time
import traceback
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from packaging import version

_IK_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_IK_UTILS = os.path.join(_IK_RL_DIR, "utils")
if _IK_UTILS not in sys.path:
    sys.path.insert(0, _IK_UTILS)

from isaaclab.app import AppLauncher
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import cli_args  # isort: skip
from ik_rl_load_config import (  # isort: skip
    apply_sys_argv_ik_yaml_defaults,
    default_pickup_ik_yaml_path,
    resolve_ik_config_path,
    warn_if_task_mismatch_with_ik_yaml,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play IK-RL policy with full_rl-compatible CLI.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

    parser.add_argument("--save_data", action="store_true", default=False, help="If set, save play trajectory data to disk.")
    parser.add_argument("--data_path", type=str, default=None, help="Directory to save recorded data.")
    parser.add_argument("--num_episodes", type=int, default=50, help="Target number of successful trajectories to save.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Max steps per trajectory while saving data. 0 means no step cap.",
    )

    # Keep IK-specific knobs and defaults.
    parser.add_argument("--trajectory", type=str, default="object:150:0,goal:-1:0")
    parser.add_argument("--object-to-palm-offset", type=float, nargs=3, default=(0.0, 0.0, 0.05), metavar=("OX", "OY", "OZ"))
    parser.add_argument("--palm-in-wrist-pos", type=float, nargs=3, default=(0.0, 0.0, 0.35), metavar=("PX", "PY", "PZ"))
    parser.add_argument(
        "--palm-in-wrist-euler",
        type=float,
        nargs=3,
        default=(np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0),
        metavar=("RX", "RY", "RZ"),
    )
    parser.add_argument("--palm-orient", type=str, choices=("fixed", "pickup_down"), default="pickup_down")
    parser.add_argument("--palm-normal-local", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--palm-yaw-offset", type=float, default=0.0)
    parser.add_argument("--world-down", type=float, nargs=3, default=(0.0, 0.0, -1.0))
    parser.add_argument("--palm-euler", type=float, nargs=3, default=(0.0, 2.2, 0.0))
    parser.add_argument("--palm-euler-in-anchor", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--ee-body", type=str, default="wrist_3_link")
    parser.add_argument("--ik-method", type=str, choices=("pinv", "svd", "trans", "dls"), default="dls")
    parser.add_argument("--ik-lambda", type=float, default=None)
    parser.add_argument(
        "--ik-config",
        type=str,
        default=None,
        help="YAML with task + palm/IK/trajectory (configs/ik_rl_pickup.yaml).",
    )

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_arg_parser()
apply_sys_argv_ik_yaml_defaults(parser)
args_cli, hydra_args = parser.parse_known_args()

_cfg_path = resolve_ik_config_path(sys.argv, default_pickup_ik_yaml_path())
if _cfg_path is not None:
    print(f"[INFO] IK palm/trajectory defaults merged from YAML: {_cfg_path}")
warn_if_task_mismatch_with_ik_yaml(_cfg_path, args_cli.task)

if getattr(args_cli, "save_data", False):
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import ViTacLab.tasks  # noqa: F401, E402

from ik_rl_hand_vec_env import (  # noqa: E402
    ArmIkHandActionExpander,
    IkHandRslRlVecEnvWrapper,
    IkRlHandArmCfg,
    parse_trajectory_phases,
)

from isaaclab.utils.math import (
    transform_points,
    unproject_depth,
)


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    print(xyz.shape)
    batch, n_points, _ = xyz.shape
    if n_points <= npoint:
        if n_points < npoint:
            pad = npoint - n_points
            xyz = torch.cat([xyz, xyz[:, -1:, :].expand(batch, pad, -1)], dim=1)
        return xyz
    device = xyz.device
    dtype = xyz.dtype
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch, n_points), 1e10, device=device, dtype=dtype)
    farthest = torch.randint(0, n_points, (batch,), dtype=torch.long, device=device)
    batch_idx = torch.arange(batch, device=device)
    for j in range(npoint):
        centroids[:, j] = farthest
        center = xyz[batch_idx, farthest].unsqueeze(1)
        dist = torch.sum((xyz - center) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    ii = batch_idx.unsqueeze(1).expand(batch, npoint)
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    _enable_cams = bool(getattr(args_cli, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    if getattr(args_cli, "save_data", False):
        _enable_cams = True
    setattr(env_cfg, "enable_cameras", _enable_cams)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base: DirectRLEnv = env.unwrapped
    env.reset()

    traj = parse_trajectory_phases(args_cli.trajectory)
    ik_cfg = IkRlHandArmCfg(
        object_to_palm_offset=tuple(args_cli.object_to_palm_offset),
        palm_in_wrist_pos=tuple(args_cli.palm_in_wrist_pos),
        palm_in_wrist_euler_xyz=tuple(args_cli.palm_in_wrist_euler),
        palm_orientation_mode=args_cli.palm_orient,
        palm_euler_xyz=tuple(args_cli.palm_euler),
        palm_normal_in_palm_frame=tuple(args_cli.palm_normal_local),
        world_down=tuple(args_cli.world_down),
        palm_yaw_offset_rad=float(args_cli.palm_yaw_offset),
        palm_euler_in_anchor_frame=tuple(args_cli.palm_euler_in_anchor),
        trajectory=traj,
        ee_body_name=str(args_cli.ee_body),
        ik_method=args_cli.ik_method,
        ik_lambda=args_cli.ik_lambda,
    )
    expander = ArmIkHandActionExpander(base, ik_cfg)
    wrapped = IkHandRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions, expander=expander)
    wrapped.unwrapped._use_rl_control = True

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = wrapped.unwrapped.step_dt
    obs = wrapped.get_observations()
    obs_policy = obs.get("policy", obs) if isinstance(obs, dict) else obs
    num_envs = obs_policy.shape[0] if hasattr(obs_policy, "shape") else getattr(wrapped.unwrapped, "num_envs", 1)
    device = wrapped.unwrapped.device

    current_ep_buffers = [dict() for _ in range(num_envs)]
    step_in_episode = 0
    success_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes_collected = 0
    max_steps_per_episode = args_cli.max_steps if args_cli.max_steps > 0 else None
    run_until_target = args_cli.save_data and args_cli.num_episodes > 0
    print(f"[INFO] Starting simulation. num_envs={num_envs}. Target: {args_cli.num_episodes} successful trajectories.")
    if max_steps_per_episode is not None:
        print(f"[INFO] Max steps per trajectory: {max_steps_per_episode} (trajectory discarded if no success by then).")

    if args_cli.save_data:
        seed_val = agent_cfg.seed if agent_cfg.seed is not None else -1
        if args_cli.data_path is not None:
            data_dir = os.path.abspath(args_cli.data_path)
        else:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("data", "rsl_rl", task_name + "_" + str(seed_val), current_time))
        os.makedirs(data_dir, exist_ok=True)
        for i in range(num_envs):
            _append_env_frame_to_buffer(current_ep_buffers[i], obs["record"][i])

    while True:
        if not simulation_app.is_running():
            break
        if run_until_target and episodes_collected >= args_cli.num_episodes:
            break

        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, infos = wrapped.step(actions)

        if policy_nn is not None and hasattr(policy_nn, "reset"):
            with torch.inference_mode():
                policy_nn.reset(dones)

        if args_cli.save_data:
            for i in range(num_envs):
                _append_env_frame_to_buffer(current_ep_buffers[i], obs["record"][i])

        step_in_episode += 1

        if args_cli.save_data:
            success_per_env = infos["curr_success_per_env"]
            for i in range(num_envs):
                if success_per_env[i].item() and not success_saved[i].item():
                    np.savez_compressed(
                        os.path.join(data_dir, f"episode_{episodes_collected}.npz"),
                        **_episode_buffer_to_npz_kwargs(env.unwrapped, i, current_ep_buffers[i]),
                    )
                    episodes_collected += 1
                    success_saved[i] = True
                    print(
                        f"[INFO] Env {i} success at global step {step_in_episode}; "
                        f"saved episode {episodes_collected} / {args_cli.num_episodes}"
                    )
            if episodes_collected >= args_cli.num_episodes:
                break

        if max_steps_per_episode is not None and step_in_episode >= max_steps_per_episode:
            if args_cli.save_data:
                for i in range(num_envs):
                    if not success_saved[i].item():
                        print(f"[INFO] Env {i} discarded (max_steps={max_steps_per_episode} without success).")
            current_ep_buffers = [dict() for _ in range(num_envs)]
            success_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)
            step_in_episode = 0
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                with torch.inference_mode():
                    policy_nn.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
            with torch.inference_mode():
                obs, _ = wrapped.reset()

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    wrapped.close()


if __name__ == "__main__":
    if not args_cli.task:
        raise SystemExit("--task is required (e.g. Isaac-UR10eShadowHand-Pickup-Direct-v0).")
    try:
        main()  # pyright: ignore[reportCallIssue]
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
if __name__ == "__main__":
    raise SystemExit