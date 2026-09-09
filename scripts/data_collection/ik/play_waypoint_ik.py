# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect Forge/Franka trajectories with scripted Cartesian IK waypoints.

This entry reuses the registered RSL-RL configuration only to construct the
environment; it does not load or execute policy checkpoint weights.  The
environment receives zero normalized actions while ``set_franka_ik_target``
drives the arm.  Successful records are saved by default; ``--save-outcome
all`` can retain timeout attempts for data-chain diagnostics.
"""


import argparse
import os
import sys
import time
from datetime import datetime
import traceback
import gymnasium as gym
import numpy as np
import torch

# Shared importable helpers; executable collection logic remains in this file.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from isaaclab.app import AppLauncher

from rl import cli_args  # isort: skip

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect RSL-RL task data with scripted arm IK waypoints.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

    # data recording options
    parser.add_argument("--save_data", action="store_true", default=False, help="If set, save play trajectory data to disk.")
    parser.add_argument("--data_path", type=str, default=None, help="Directory to save recorded data.")
    
    # 新增：控制回合数的参数（多环境时为目标成功轨迹总数，各环境独立判断成功/超步/结束）
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of successful episodes to record when save_data is enabled (total across all envs).")
    
    parser.add_argument(
        "--max_steps", type=int, default=0,
        help=(
            "Maximum steps per attempt. A timeout is discarded by default or saved with --save-outcome all; "
            "it never increments the successful --num_episodes count. 0 disables the attempt limit."
        ),
    )
    parser.add_argument(
        "--save-outcome",
        choices=("success", "all"),
        default="success",
        help="Save successful trajectories only, or also save max-step timeout attempts for diagnostics.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Stop after this many success/timeout attempts across all environments; 0 keeps trying until the success target or app exit.",
    )
    parser.add_argument(
        "--waypoint_max_steps",
        type=int,
        default=0,
        help="Max steps allowed on one waypoint before skipping to the next waypoint (per-env). 0 disables skip-by-steps.",
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
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import ViTacLab.tasks  # noqa: F401

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


def _quat_angle_error(q_curr: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    """Return quaternion angular distance (rad) for batched wxyz quaternions."""
    q_curr_n = q_curr / torch.clamp(torch.linalg.norm(q_curr, dim=-1, keepdim=True), min=1e-8)
    q_tgt_n = q_tgt / torch.clamp(torch.linalg.norm(q_tgt, dim=-1, keepdim=True), min=1e-8)
    dot = torch.sum(q_curr_n * q_tgt_n, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _build_ik_waypoints(base_env, num_envs: int, device: torch.device):
    """
    Build waypoint list of (pos_env, quat_env) for IK control.
    Uses current EE pose as reference to keep sequence task-agnostic.
    """
    ee_idx = base_env.fingertip_body_idx
    ee_pos_env = base_env._robot.data.body_pos_w[:, ee_idx] - base_env.scene.env_origins
    cur_pos = ee_pos_env
    cur_quat = base_env._robot.data.body_quat_w[:, ee_idx]

    # fixed_pos in FactoryEnv/ForgeEnv is already env-local.
    
    # Compute XY compensation in env-local frame.
    # With delta = held - ee, to place held on fixed center:
    # ee_target = fixed - delta = fixed + (ee - held).
    held_pos_env = base_env._held_asset.data.root_pos_w - base_env.scene.env_origins
    bias_x = ee_pos_env[:, 0] - held_pos_env[:, 0]
    bias_y = ee_pos_env[:, 1] - held_pos_env[:, 1]
    
    print(f"bias_x: {bias_x}, bias_y: {bias_y}")
    
    fixed_pos_env = base_env.fixed_pos
    target_pos1 = fixed_pos_env.clone()
    target_pos1[:, 0] += bias_x
    target_pos1[:, 1] += bias_y
    target_pos1[:, 2] += 0.1
    
    # target_pos2 = fixed_pos_env.clone()
    # target_pos2[:, 0] += xy[0]
    # target_pos2[:, 1] += xy[1]
    # target_pos2[:, 2] += 0.069
    
    # target_pos3 = fixed_pos_env.clone()
    # target_pos3[:, 0] -= 0.00
    # target_pos3[:, 1] += 0.0
    # target_pos3[:, 2] += 0.1
    
    target_pos4 = fixed_pos_env.clone()
    target_pos4[:, 0] += bias_x
    target_pos4[:, 1] += bias_y
    target_pos4[:, 2] += 0.07
    
    target_pos5 = fixed_pos_env.clone()
    target_pos5[:, 0] += bias_x
    target_pos5[:, 1] += bias_y
    target_pos5[:, 2] += 0.04

    return [
        (cur_pos.clone(), cur_quat.clone()),
        (target_pos1, cur_quat.clone()),
        # (target_pos2, cur_quat.clone()),
        # (target_pos3, cur_quat.clone()),
        (target_pos4, cur_quat.clone()),
        (target_pos5, cur_quat.clone()),
    ]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with IK-only controller and record successful trajectories."""
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

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
    base_env = env.unwrapped
    base_env._use_rl_control = False
    if hasattr(base_env, "set_stiffness_damping"):
        base_env.set_stiffness_damping(True)
    dt = base_env.step_dt
    obs = env.get_observations()

    # 多环境：从 obs 或 env 获取 num_envs
    obs_policy = obs.get("policy", obs) if isinstance(obs, dict) else obs
    num_envs = obs_policy.shape[0] if hasattr(obs_policy, "shape") else getattr(base_env, "num_envs", 1)
    device = base_env.device

    # 每个环境独立的 episode 缓冲；所有 env 共享同一个 global episode 步数
    current_ep_buffers = [dict() for _ in range(num_envs)]
    step_in_episode = 0
    max_steps_per_episode = args_cli.max_steps if args_cli.max_steps > 0 else None
    pos_tol = 0.001  # meters
    rot_tol = np.deg2rad(5.0)  # radians

    print(f"[INFO] Starting IK simulation. num_envs={num_envs}, max_steps={max_steps_per_episode}.")

    if args_cli.save_data:
        seed_val = agent_cfg.seed if agent_cfg.seed is not None else -1
        if args_cli.data_path is not None:
            data_dir = os.path.abspath(args_cli.data_path)
        else:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("data", "rsl_rl", task_name + "_" + str(seed_val), current_time))
        os.makedirs(data_dir, exist_ok=True)

    # 定义 IK 目标位姿列表（位置、旋转）
    waypoints = _build_ik_waypoints(base_env, num_envs, device)
    n_waypoints = len(waypoints)
    target_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    waypoint_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    active_target_pos, active_target_quat = waypoints[0]
    base_env.set_franka_ik_target(active_target_pos, active_target_quat, ik_pos_tolerance=pos_tol)

    # 在主循环前，设置目标位姿并采集一帧数据
    if args_cli.save_data:
        for i in range(num_envs):
            _append_env_frame_to_buffer(
                current_ep_buffers[i],
                obs["record"][i],
            )

    action_dim = getattr(base_env.cfg, "action_space", 7)
    dummy_actions = torch.zeros(num_envs, action_dim, device=device, dtype=torch.float32)
    success_per_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
    log_every = 1  # print every N steps

    # Align save/reset logic with play.py
    success_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes_collected = 0
    attempts_completed = 0
    total_timestep = 0
    run_until_target = args_cli.save_data and args_cli.num_episodes > 0
    print(f"[INFO] IK Data recording target: {args_cli.num_episodes} successful trajectories.")
    
    while True:
        if not simulation_app.is_running():
            print("[INFO] App no longer running; exiting.")
            break
        if run_until_target and episodes_collected >= args_cli.num_episodes:
            break
        if args_cli.max_attempts > 0 and attempts_completed >= args_cli.max_attempts:
            print(f"[INFO] Reached max_attempts={args_cli.max_attempts}; stopping.")
            break
        if max_steps_per_episode is not None and step_in_episode >= max_steps_per_episode:
            print(f"[INFO] Attempt reached max_steps={max_steps_per_episode}.")
            for i in range(num_envs):
                if bool(success_saved[i]):
                    continue
                attempt_index = attempts_completed
                if args_cli.save_data and args_cli.save_outcome == "all":
                    ep_data_torch = current_ep_buffers[i]
                    timeout_path = os.path.join(data_dir, f"attempt_{attempt_index:04d}_timeout.npz")
                    np.savez_compressed(
                        timeout_path,
                        **_episode_buffer_to_npz_kwargs(base_env, i, ep_data_torch),
                    )
                    print(
                        f"[INFO] Env {i} timeout saved for diagnostics: {timeout_path} "
                        f"(max_steps={max_steps_per_episode}, success_count={episodes_collected})."
                    )
                else:
                    print(
                        f"[INFO] Env {i} timeout discarded "
                        f"(max_steps={max_steps_per_episode}, success_count={episodes_collected})."
                    )
                attempts_completed += 1
            if args_cli.max_attempts > 0 and attempts_completed >= args_cli.max_attempts:
                print(f"[INFO] Reached max_attempts={args_cli.max_attempts}; stopping.")
                break
            # Per play.py: reset env and clear buffers at global max_steps
            current_ep_buffers = [dict() for _ in range(num_envs)]
            success_saved = torch.zeros(num_envs, dtype=torch.bool, device=device)
            step_in_episode = 0
            obs, _ = env.reset()
            waypoints = _build_ik_waypoints(base_env, num_envs, device)
            n_waypoints = len(waypoints)
            target_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
            waypoint_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            active_target_pos, active_target_quat = waypoints[0]
            base_env.set_franka_ik_target(active_target_pos, active_target_quat, ik_pos_tolerance=pos_tol)
            continue

        start_time = time.time()

        # 先判定当前是否达到目标位姿，若达到则切换下一个目标
        ee_pos_env = obs["record"]["ee_pos_env"].to(device=device)
        ee_quat_env = obs["record"]["ee_quat_env"].to(device=device)
        pos_err = torch.linalg.norm(ee_pos_env - active_target_pos, dim=-1)
        quat_err = _quat_angle_error(ee_quat_env, active_target_quat)
        reached = torch.logical_and(pos_err < pos_tol, quat_err < rot_tol)
        if step_in_episode % log_every == 0:
            print(
                f"[IK] step={step_in_episode} target_idx={int(target_idx[0].item())} "
                f"pos_err={pos_err[0].item():.4f} "
                f"rot_err_deg={np.rad2deg(quat_err[0].item()):.2f}"
            )

        # Independent per-env waypoint progression.
        for i in range(num_envs):
            cur_idx = int(target_idx[i].item())
            has_next = cur_idx + 1 < n_waypoints
            timed_out = bool(args_cli.waypoint_max_steps > 0 and int(waypoint_step_count[i].item()) >= args_cli.waypoint_max_steps)
            should_advance = has_next and (bool(reached[i]) or timed_out)
            if should_advance:
                target_idx[i] += 1
                nxt = int(target_idx[i].item())
                active_target_pos[i] = waypoints[nxt][0][i]
                active_target_quat[i] = waypoints[nxt][1][i]
                waypoint_step_count[i] = 0
                if timed_out and not bool(reached[i]):
                    print(
                        f"[IK] env={i} skip waypoint {cur_idx} after "
                        f"{args_cli.waypoint_max_steps} steps."
                    )

        # 再下发（可能已更新的）当前 IK 目标。
        base_env.set_franka_ik_target(active_target_pos, active_target_quat, ik_pos_tolerance=pos_tol)

        obs, _, _, infos = env.step(dummy_actions)

        if args_cli.save_data:
            for i in range(num_envs):
                _append_env_frame_to_buffer(
                    current_ep_buffers[i],
                    obs["record"][i],
                )

        step_in_episode += 1
        total_timestep += 1
        waypoint_step_count += 1

        if "curr_success_per_env" in infos:
            success_per_env = infos["curr_success_per_env"].to(device=device)
            if args_cli.save_data:
                for i in range(num_envs):
                    if bool(success_per_env[i]) and not bool(success_saved[i]):
                        ep_data_torch = current_ep_buffers[i]
                        success_path = os.path.join(data_dir, f"episode_{episodes_collected:04d}_success.npz")
                        np.savez_compressed(
                            success_path,
                            **_episode_buffer_to_npz_kwargs(base_env, i, ep_data_torch),
                        )
                        episodes_collected += 1
                        attempts_completed += 1
                        success_saved[i] = True
                        print(
                            f"[INFO] Env {i} success at global step {step_in_episode}; "
                            f"saved {success_path} ({episodes_collected}/{args_cli.num_episodes})"
                        )
                if run_until_target and episodes_collected >= args_cli.num_episodes:
                    break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 保存任务成功的数据
    if args_cli.save_data:
        print(f"[INFO] Total saved successful trajectories: {episodes_collected}")
        print(f"[INFO] Total completed attempts: {attempts_completed}")

    # close the simulator
    env.close()

if __name__ == "__main__":
    try:
        main()  # type: ignore[misc]
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
