# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play / eval a **hand-only** RSL-RL policy with GPU differential IK arm (same stack as ``train_ik_rl_single.py``).

Examples::

    # Absolute path to checkpoint
    ./isaaclab.sh -p scripts/rsl_rl/ik_rl/play_ik_rl_single.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \\
        --num_envs 64 --checkpoint logs/rsl_rl/Isaac-UR10eShadowHand-Pickup-Direct-v0/2026-03-21_19-31-05/model_1000.pt

    # Resolve under logs/rsl_rl/<task>/ (same as training resume)
    ./isaaclab.sh -p scripts/rsl_rl/ik_rl/play_ik_rl_single.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \\
        --num_envs 4096 --headless --resume --load_run 2026-03-21_19-31-05 --checkpoint model_61250.pt

IK / trajectory flags should match training; defaults mirror ``train_ik_rl_single.py``.

Optional **data recording**: ``--record_data`` saves policy observations, **hand** actions, rewards, and dones for
``--record_env_index`` to ``--record_path`` (default: ``./play_records/<task>_<timestamp>/``), one compressed
``.npz`` per completed episode (plus ``*_partial.npz`` if play stops mid-episode). Use ``--record_max_episodes N``
to stop after ``N`` saved episodes.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
from datetime import datetime

# Local imports: ``utils/`` next to this file (``scripts/rsl_rl/ik_rl/utils``)
_IK_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_IK_UTILS = os.path.join(_IK_RL_DIR, "utils")
if _IK_UTILS not in sys.path:
    sys.path.insert(0, _IK_UTILS)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
import numpy as np

parser = argparse.ArgumentParser(description="Play hand-only RSL-RL policy with GPU IK arm (single-arm setup).")
parser.add_argument("--task", type=str, default=None, help="Registered Gym task (e.g. Isaac-UR10eShadowHand-Pickup-Direct-v0).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point (registry key)."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
# --checkpoint / --resume / --load_run come from cli_args.add_rsl_rl_args below.
parser.add_argument(
    "--max_play_steps",
    type=int,
    default=0,
    help="Stop after N env steps (0 = run until window closed or Ctrl+C).",
)
# Optional trajectory recording (one env index; .npz per episode)
parser.add_argument(
    "--record_data",
    action="store_true",
    help="Save policy obs / hand actions / rewards / dones for --record_env_index to --record_path (npz per episode).",
)
parser.add_argument(
    "--record_path",
    type=str,
    default=None,
    help="Directory for recorded .npz files. Default: play_records/<task>_<timestamp>/ under CWD.",
)
parser.add_argument(
    "--record_env_index",
    type=int,
    default=0,
    help="Which parallel env to record when --record_data (default: 0).",
)
parser.add_argument(
    "--record_max_episodes",
    type=int,
    default=0,
    help="Stop after saving this many completed episodes (0 = no episode limit; still respects --max_play_steps).",
)
parser.add_argument(
    "--play_success_interval",
    type=float,
    default=2.0,
    help="Print pickup success stats every N seconds (if env exposes get_episode_success_stats).",
)
parser.add_argument("--show_rgb", action="store_true", help="Show tactile RGB (implies --enable_cameras). Same idea as run_ur10e_shadowhand_single.py.")
parser.add_argument("--show_ff", action="store_true", help="Show tactile force-field RGB (implies --enable_cameras).")
parser.add_argument("--env_index", type=int, default=0, help="Which env index to visualize for tactile plots (default: 0).")
parser.add_argument("--fps", type=float, default=20.0, help="Target display FPS when --show_rgb / --show_ff (default: 20).")
# Palm + IK (keep in sync with train_ik_rl_single.py)
parser.add_argument(
    "--trajectory",
    type=str,
    default="object:150:0,goal:-1:0",
    help="name:env_steps:use_rot — name resolves to env.<name> (asset) or <name>_pos/_rot (tensors), or goal (legacy). "
    "Pour example: cup:150:0,goal_cup:-1:0",
)
parser.add_argument(
    "--object-to-palm-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.05),
    metavar=("OX", "OY", "OZ"),
)
parser.add_argument(
    "--palm-in-wrist-pos",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.35),
    metavar=("PX", "PY", "PZ"),
)
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
    help="YAML with task + palm/IK/trajectory (see configs/ik_rl_pickup.yaml). Omitted: auto-load if present. 'none' = off.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

from ik_rl_load_config import (
    apply_sys_argv_ik_yaml_defaults,
    default_pickup_ik_yaml_path,
    resolve_ik_config_path,
    warn_if_task_mismatch_with_ik_yaml,
)

apply_sys_argv_ik_yaml_defaults(parser)
args_cli, hydra_args = parser.parse_known_args()
_cfg_path = resolve_ik_config_path(sys.argv, default_pickup_ik_yaml_path())
if _cfg_path is not None:
    print(f"[INFO] IK palm/trajectory defaults merged from YAML: {_cfg_path}")
warn_if_task_mismatch_with_ik_yaml(_cfg_path, args_cli.task)

# Tactile viewer needs scene cameras / sensors (match scripts/debug/run_ur10e_shadowhand_single.py).
if getattr(args_cli, "show_rgb", False) or getattr(args_cli, "show_ff", False):
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import platform
import time

from packaging import version

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
    exit(1)

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401

import ViTacLab.tasks  # noqa: F401

from rsl_rl_log_utils import get_rsl_rl_log_root
from ik_rl_hand_vec_env import ArmIkHandActionExpander, IkHandRslRlVecEnvWrapper, IkRlHandArmCfg, parse_trajectory_phases

# Tactile visualization (aligned with scripts/debug/run_ur10e_shadowhand_single.py)
TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _render_tactile_ff_rgb(nf: np.ndarray, sf: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Render tactile normal/shear arrays into an RGB image (same as show_ur10e_shadowhand_records)."""
    nf = np.asarray(nf, dtype=np.float32)
    sf = np.asarray(sf, dtype=np.float32)
    if nf.ndim != 2 or sf.ndim != 3 or sf.shape[-1] != 2:
        raise ValueError(f"Invalid shapes for ff render: nf={nf.shape}, sf={sf.shape}")

    nf_scale = np.percentile(np.abs(nf), 99.0) + eps
    sf_scale = np.percentile(np.linalg.norm(sf, axis=-1), 99.0) + eps

    n = np.clip(nf / nf_scale, 0.0, 1.0)
    sx = np.clip(sf[..., 0] / sf_scale, -1.0, 1.0)
    sy = np.clip(sf[..., 1] / sf_scale, -1.0, 1.0)

    r = 0.5 + 0.5 * sx
    g = 0.5 + 0.5 * sy
    b = n
    img = np.stack([r, g, b], axis=-1)

    img = img * (0.3 + 0.7 * n[..., None])
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _resolve_checkpoint_path() -> str:
    log_root = get_rsl_rl_log_root(args_cli.task, getattr(args_cli, "experiment_name", None))
    cp = getattr(args_cli, "checkpoint", None)
    if cp and os.path.isfile(os.path.expanduser(cp)):
        return retrieve_file_path(cp)
    if getattr(args_cli, "resume", False) and getattr(args_cli, "load_run", None) and cp:
        return get_checkpoint_path(log_root, args_cli.load_run, cp)
    raise SystemExit(
        "Provide an existing --checkpoint path to model_*.pt, or "
        "--resume --load_run <run_dir_name> --checkpoint model_XXXX.pt under logs/rsl_rl/<task>/."
    )


def _save_play_episode_npz(out_dir: str, fname: str, bufs: dict):
    """Write one episode to ``fname`` under ``out_dir``; clear ``bufs``. Returns (path or None, T)."""
    if not bufs.get("policy_obs"):
        return None, 0
    T = len(bufs["dones"])
    path = os.path.join(out_dir, fname)
    np.savez_compressed(
        path,
        policy_obs=np.stack(bufs["policy_obs"], axis=0),
        actions=np.stack(bufs["actions"], axis=0),
        rewards=np.asarray(bufs["rewards"], dtype=np.float32),
        dones=np.asarray(bufs["dones"], dtype=np.bool_),
    )
    for k in bufs:
        bufs[k].clear()
    return path, T


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    _enable_cams = bool(getattr(args_cli, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    setattr(env_cfg, "enable_cameras", _enable_cams)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if getattr(args_cli, "enable_cameras", False) else None)

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
    print(
        f"[INFO] IK play: hand actions={expander.num_hand}, full actuated={expander.num_actuated}, "
        f"trajectory={args_cli.trajectory}"
    )

    wrapped = IkHandRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions, expander=expander)

    resume_path = _resolve_checkpoint_path()
    print(f"[INFO] Loading checkpoint: {resume_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=wrapped.device)
    policy_nn = None
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = getattr(runner.alg, "actor_critic", None)

    obs = wrapped.get_observations()
    print(f"[INFO] Policy obs batch shape: {tuple(obs['policy'].shape) if 'policy' in obs else obs}")

    record_dir = None
    rec_bufs = None
    rec_ep_idx = 0
    if args_cli.record_data:
        if args_cli.record_path:
            record_dir = os.path.abspath(os.path.expanduser(args_cli.record_path))
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            safe = (args_cli.task or "task").replace(":", "_").replace("/", "_")
            record_dir = os.path.join(os.getcwd(), "play_records", f"{safe}_{ts}")
        os.makedirs(record_dir, exist_ok=True)
        rec_bufs = {"policy_obs": [], "actions": [], "rewards": [], "dones": []}
        meta = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "record_env_index": int(args_cli.record_env_index),
            "trajectory": args_cli.trajectory,
            "num_envs": int(wrapped.num_envs),
        }
        with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] --record_data: saving trajectories under {record_dir}")

    show_tactile = bool(args_cli.show_rgb or args_cli.show_ff)
    fig = None
    ax_rgb = None
    ax_ff = None
    rgb_ims: list = []
    ff_ims: list = []
    nrows, ncols = 20, 25

    if show_tactile:
        import matplotlib

        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        if args_cli.show_rgb and args_cli.show_ff:
            fig, axes = plt.subplots(2, 5, figsize=(16, 6))
            ax_rgb = axes[0]
            ax_ff = axes[1]
        elif args_cli.show_rgb:
            fig, ax_rgb = plt.subplots(1, 5, figsize=(16, 3))
        elif args_cli.show_ff:
            fig, ax_ff = plt.subplots(1, 5, figsize=(16, 3))

    _scene_env = wrapped.unwrapped
    if show_tactile:
        for name in TACTILE_SENSOR_NAMES:
            if name in _scene_env.scene.sensors:
                try:
                    _scene_env.scene[name].get_initial_render()
                except Exception:
                    pass

        if args_cli.show_ff and fig is not None:
            for name in TACTILE_SENSOR_NAMES:
                if name in _scene_env.scene.sensors:
                    try:
                        nrows, ncols = _scene_env.scene[name].cfg.tactile_array_size
                    except Exception:
                        pass
                    break

        if fig is not None:
            import matplotlib.pyplot as plt

            if args_cli.show_rgb and ax_rgb is not None:
                zero_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
                axes_rgb = ax_rgb if isinstance(ax_rgb, np.ndarray) else [ax_rgb]
                for i, name in enumerate(TACTILE_SENSOR_NAMES):
                    if i >= len(axes_rgb):
                        break
                    title = name.replace("tactile_sensor_", "").upper()
                    im = axes_rgb[i].imshow(zero_rgb)
                    axes_rgb[i].set_title(f"{title} RGB")
                    axes_rgb[i].axis("off")
                    rgb_ims.append(im)

            if args_cli.show_ff and ax_ff is not None:
                zero_ff = np.zeros((nrows * 30, ncols * 30, 3), dtype=np.uint8)
                axes_ff = ax_ff if isinstance(ax_ff, np.ndarray) else [ax_ff]
                for i, name in enumerate(TACTILE_SENSOR_NAMES):
                    if i >= len(axes_ff):
                        break
                    title = name.replace("tactile_sensor_", "").upper()
                    im = axes_ff[i].imshow(zero_ff)
                    axes_ff[i].set_title(f"{title} FF")
                    axes_ff[i].axis("off")
                    ff_ims.append(im)

            plt.tight_layout()
            fig.canvas.draw()
            plt.pause(0.1)

    render_ff = _render_tactile_ff_rgb if args_cli.show_ff else None
    target_dt = 1.0 / max(1e-3, float(args_cli.fps))
    env_idx = max(0, min(int(args_cli.env_index), wrapped.num_envs - 1))
    record_ei = max(0, min(int(args_cli.record_env_index), wrapped.num_envs - 1))

    step = 0
    last_print = time.time()
    interval = max(0.1, float(args_cli.play_success_interval))

    while simulation_app.is_running():
        t0 = time.time() if show_tactile else None
        step += 1
        obs_k_np = None
        if args_cli.record_data and rec_bufs is not None and obs is not None:
            pol = obs.get("policy", None) if isinstance(obs, dict) else None
            if pol is not None:
                obs_k_np = pol[record_ei].detach().cpu().numpy()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _rew, dones, _extras = wrapped.step(actions)
            if policy_nn is not None:
                policy_nn.reset(dones)

        if args_cli.record_data and rec_bufs is not None and obs_k_np is not None:
            rec_bufs["policy_obs"].append(obs_k_np)
            rec_bufs["actions"].append(actions[record_ei].detach().cpu().numpy())
            rec_bufs["rewards"].append(float(_rew[record_ei].detach().cpu().item()))
            rec_bufs["dones"].append(bool(dones[record_ei].detach().cpu().item()))
            if bool(dones[record_ei].detach().cpu().item()):
                path, T = _save_play_episode_npz(record_dir, f"episode_{rec_ep_idx:04d}.npz", rec_bufs)
                if path is not None:
                    print(f"[INFO] recorded {path} (T={T})")
                    rec_ep_idx += 1
                    if args_cli.record_max_episodes > 0 and rec_ep_idx >= args_cli.record_max_episodes:
                        break

        now = time.time()
        if now - last_print >= interval:
            ue = wrapped.unwrapped
            if hasattr(ue, "get_episode_success_stats"):
                n_ok, n_ep, rate = ue.get_episode_success_stats()
                ema = getattr(ue, "get_episode_success_rate_ema", None)
                ema_s = f" ema={ema():.4f}" if callable(ema) else ""
                print(
                    f"[play] step={step}  success_rate(all_time)={rate:.4f} ({n_ok}/{n_ep}){ema_s}  "
                    f"n_envs={wrapped.num_envs}"
                )
            last_print = now

        if fig is not None and (rgb_ims or ff_ims):
            import matplotlib.pyplot as plt

            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if name not in _scene_env.scene.sensors:
                    continue
                data = _scene_env.scene[name].data

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

        if args_cli.max_play_steps > 0 and step >= int(args_cli.max_play_steps):
            break

        if show_tactile and t0 is not None:
            elapsed = time.time() - t0
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

    if args_cli.record_data and rec_bufs is not None and record_dir is not None and rec_bufs.get("policy_obs"):
        path, T = _save_play_episode_npz(record_dir, f"episode_{rec_ep_idx:04d}_partial.npz", rec_bufs)
        if path is not None:
            print(f"[INFO] recorded incomplete episode {path} (T={T})")

    wrapped.close()
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    print(f"[INFO] play finished after {step} steps.")


if __name__ == "__main__":
    if not args_cli.task:
        raise SystemExit("--task is required (e.g. Isaac-UR10eShadowHand-Pickup-Direct-v0).")
    main()
    simulation_app.close()
