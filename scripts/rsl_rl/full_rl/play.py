# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Shared helpers with ``ik_rl`` live under ``scripts/rsl_rl/ik_rl/utils``
_RSL_RL_ROOT = os.path.dirname(os.path.abspath(__file__))
_IK_UTILS = os.path.abspath(os.path.join(_RSL_RL_ROOT, "..", "ik_rl", "utils"))
if _IK_UTILS not in sys.path:
    sys.path.insert(0, _IK_UTILS)

from isaaclab.app import AppLauncher

# local imports
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

# clear out sys.argv for Hydra
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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ViTacLab.tasks  # noqa: F401
from ViTacLab.utils.vitaclab_marl_rsl import multi_agent_to_single_agent

from rsl_rl_log_utils import get_rsl_rl_log_root


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

        from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import compute_tactile_shear_image

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

    # specify directory for logging experiments (default: folder name = ``--task`` id; override: ``--experiment_name``)
    log_root_path = get_rsl_rl_log_root(args_cli.task, getattr(args_cli, "experiment_name", None))
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

    # ViTacLab direct tasks: match sensor spawning to AppLauncher / ENABLE_CAMERAS.
    _enable_cams = bool(getattr(args_cli, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    setattr(env_cfg, "enable_cameras", _enable_cams)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
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

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

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
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
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
    # run the main function
    main()
    # close sim app
    simulation_app.close()
