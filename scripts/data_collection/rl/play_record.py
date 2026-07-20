# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical RSL-RL rollout data collector.

Runs a trained RSL-RL checkpoint and, with ``--save_data``, writes successful
task trajectories from ``obs['record']`` as compressed NPZ episodes.  This file
contains the collector implementation directly; it does not dispatch to a
script under ``scripts/rsl_rl``.
"""


import argparse
import os
import sys
from datetime import datetime

# Importable shared helpers are deliberately separate from executable collectors.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from isaaclab.app import AppLauncher

# local imports
from rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Run an RSL-RL checkpoint and optionally collect successful NPZ episodes.")
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
parser.add_argument("--save_data", action="store_true", default=False, help="If set, save play trajectory data to disk.")
parser.add_argument("--data_path", type=str, default=None, help="Directory to save recorded data.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=50,
    help="Number of successful trajectories to save when --save_data is enabled (total across all envs).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="When --save_data: max steps per rollout before reset; 0 means no explicit max-step reset.",
)
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
if args_cli.video or args_cli.show_rgb or args_cli.show_ff or args_cli.save_data:
    args_cli.enable_cameras = True

_MATPLOTLIB_PYPLOT = None
if args_cli.show_rgb or args_cli.show_ff:
    import matplotlib

    backend_from_env = os.environ.get("MPLBACKEND", "").strip()
    if backend_from_env:
        backend_candidates = [backend_from_env]
    elif os.environ.get("DISPLAY"):
        # Prefer Qt in desktop sessions; Tk can fail on some font/DPI stacks.
        backend_candidates = ["QtAgg", "TkAgg", "Agg"]
    else:
        backend_candidates = ["Agg"]

    last_exc = None
    for backend_name in backend_candidates:
        try:
            matplotlib.use(backend_name, force=True)
            import matplotlib.pyplot as plt

            # Force backend load/initialization now (before AppLauncher mutates runtime state).
            _probe_fig = plt.figure(figsize=(1.0, 1.0))
            _probe_fig.canvas.draw()
            plt.close(_probe_fig)
            plt.rcParams["font.size"] = 10.0
            plt.rcParams["figure.dpi"] = 100.0
            plt.rcParams["savefig.dpi"] = 100.0
            _MATPLOTLIB_PYPLOT = plt
            print(f"[INFO] Matplotlib backend for tactile viewer: {backend_name}")
            break
        except Exception as exc:
            last_exc = exc
            _MATPLOTLIB_PYPLOT = None

    if _MATPLOTLIB_PYPLOT is None and last_exc is not None:
        raise RuntimeError(
            "Failed to initialize any matplotlib backend for --show_rgb/--show_ff. "
            f"Tried: {backend_candidates}. Last error: {last_exc}"
        )

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


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

from rl.rsl_rl_log_utils import get_rsl_rl_log_root


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


def _append_record_row(buf: dict[str, list[np.ndarray]], row: dict[str, np.ndarray] | None) -> None:
    if row is None:
        return
    for k, v in row.items():
        if k not in buf:
            buf[k] = []
        buf[k].append(np.asarray(v))


def _clear_episode_buffer(buf: dict[str, list[np.ndarray]]) -> None:
    for k in buf:
        buf[k].clear()


def _episode_buffer_to_npz_kwargs(buf: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack(v, axis=0) for k, v in buf.items() if len(v) > 0}


def _extract_record_obs(obs: Any, base_env: Any) -> Any:
    """Get obs['record'] from wrapper output and unwrapped env chain."""
    initial_record = obs["record"] if isinstance(obs, dict) and "record" in obs else None

    def _from_obs_container(container: Any) -> Any:
        if isinstance(container, dict) and "record" in container:
            return container["record"]
        return None

    # Probe base_env and then walk through common wrapper links.
    cur = base_env
    seen: set[int] = set()
    for _ in range(20):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))

        for method_name in ("get_observations", "_get_observations"):
            fn = getattr(cur, method_name, None)
            if callable(fn):
                try:
                    out = fn()
                    rec = _from_obs_container(out)
                    if rec is not None:
                        initial_record = rec
                except Exception:
                    pass

        build_record = getattr(cur, "_build_record_dict", None)
        if callable(build_record):
            try:
                record = build_record()
                if isinstance(record, dict):
                    if isinstance(initial_record, dict):
                        merged = dict(record)
                        merged.update(initial_record)
                        return merged
                    return record
            except Exception:
                pass

        for attr_name in ("unwrapped", "env"):
            nxt = getattr(cur, attr_name, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return initial_record


def _extract_record_row(record_obs: Any, env_index: int) -> dict[str, np.ndarray] | None:
    """Keep tensor keys only and slice the parallel-environment dimension at axis 0."""
    if not isinstance(record_obs, dict):
        return None
    row: dict[str, np.ndarray] = {}
    for k, v in record_obs.items():
        if torch.is_tensor(v):
            if v.ndim == 0:
                row[k] = np.asarray(v.detach().cpu().numpy())
            else:
                ei = max(0, min(int(env_index), int(v.shape[0]) - 1))
                row[k] = np.asarray(v[ei].detach().cpu().numpy())
    return row if row else None


def _extract_success_per_env(infos: Any, base_env: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    """Resolve per-environment success from common info layouts or task state."""

    candidates: list[Any] = []
    if isinstance(infos, dict):
        candidates.extend([infos.get("curr_success_per_env"), infos.get("successes"), infos.get("success")])
        for nested_key in ("extras", "record", "log"):
            nested = infos.get(nested_key)
            if isinstance(nested, dict):
                candidates.extend(
                    [nested.get("curr_success_per_env"), nested.get("successes"), nested.get("success")]
                )

    cur = base_env
    seen: set[int] = set()
    for _ in range(20):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        for attr in ("curr_success_per_env", "successes", "success"):
            candidates.append(getattr(cur, attr, None))
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            nxt = getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt

    for candidate in candidates:
        if candidate is None:
            continue
        value = torch.as_tensor(candidate, device=device, dtype=torch.bool).reshape(-1)
        if value.numel() == 1:
            return value.expand(num_envs)
        if value.numel() == num_envs:
            return value
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


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
    # Vision tasks use one canonical Gym registration. Switch the embedded
    # feature extractor to inference mode here instead of maintaining a second
    # ``*-Play-v0`` environment/config registration.
    feature_extractor_cfg = getattr(env_cfg, "feature_extractor", None)
    if feature_extractor_cfg is not None:
        if hasattr(feature_extractor_cfg, "train"):
            feature_extractor_cfg.train = False
        if hasattr(feature_extractor_cfg, "load_checkpoint"):
            feature_extractor_cfg.load_checkpoint = True

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

    base_env = env.unwrapped
    dt = base_env.step_dt

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
            plt = _MATPLOTLIB_PYPLOT
            if plt is None:
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
                if args_cli.show_rgb and i < len(ax_rgb):
                    im = ax_rgb[i].imshow(zero_rgb)
                    # Avoid text rendering on TkAgg (can trigger invalid ppem/fontsize crashes).
                    ax_rgb[i].set_title("")
                    ax_rgb[i].set_xticks([])
                    ax_rgb[i].set_yticks([])
                    ax_rgb[i].axis("off")
                    rgb_ims.append(im)
                if args_cli.show_ff and i < len(ax_ff):
                    zero_ff = np.zeros(
                        (max(1, nrows) * _TACTILE_SHEAR_VIZ_RESOLUTION, max(1, ncols) * _TACTILE_SHEAR_VIZ_RESOLUTION, 3),
                        dtype=np.uint8,
                    )
                    im = ax_ff[i].imshow(zero_ff)
                    ax_ff[i].set_title("")
                    ax_ff[i].set_xticks([])
                    ax_ff[i].set_yticks([])
                    ax_ff[i].axis("off")
                    ff_ims.append(im)
            # Keep fixed spacing to avoid layout text metrics.
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02)
            fig.canvas.draw()
            plt.pause(0.1)
            render_ff = _tactile_shear_image_rgb_uint8 if args_cli.show_ff else None
            print(
                f"[INFO] Tactile viewer: {len(tactile_keys)} sensor(s), env_index={env_idx}, fps={float(args_cli.fps):.1f}"
            )

    # reset environment
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    timestep = 0
    max_steps_per_rollout = int(args_cli.max_steps) if int(args_cli.max_steps) > 0 else None
    data_dir: str | None = None
    run_until_target = False
    current_ep_buffers: list[dict[str, list[np.ndarray]]] = []
    success_saved: torch.Tensor | None = None
    episodes_collected = 0
    step_in_rollout = 0
    num_envs = int(getattr(env, "num_envs", 1))
    if args_cli.save_data:
        record_obs = _extract_record_obs(obs, base_env)
        if record_obs is None:
            obs_keys = list(obs.keys()) if isinstance(obs, dict) else []
            raise RuntimeError(
                "--save_data requires observations to contain obs['record'].\n"
                f"Current reset obs type={type(obs).__name__}, keys={obs_keys}.\n"
                "No 'record' found along wrapper chain via get_observations/_get_observations."
            )
        seed_val = agent_cfg.seed if agent_cfg.seed is not None else -1
        if args_cli.data_path is not None:
            data_dir = os.path.abspath(args_cli.data_path)
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data_dir = os.path.abspath(os.path.join("data", "rsl_rl", f"{task_name}_{seed_val}", stamp))
        os.makedirs(data_dir, exist_ok=True)
        run_until_target = int(args_cli.num_episodes) > 0
        current_ep_buffers = [dict() for _ in range(num_envs)]
        success_saved = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
        for i in range(num_envs):
            _append_record_row(current_ep_buffers[i], _extract_record_row(record_obs, i))
        print(f"[INFO] Data collection enabled. save_dir={data_dir}")
        print(
            f"[INFO] Target successful trajectories={int(args_cli.num_episodes)}, "
            f"max_steps_per_rollout={max_steps_per_rollout if max_steps_per_rollout is not None else 'None'}"
        )

    # simulate environment
    while simulation_app.is_running():
        if args_cli.save_data and run_until_target and episodes_collected >= int(args_cli.num_episodes):
            print(f"[INFO] Reached target successful trajectories: {episodes_collected}")
            break
        start_time = time.time()
        # Keep inference_mode only for policy forward.
        # Running env.step/reset inside inference_mode can turn internal state tensors into inference tensors.
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, infos = env.step(actions)
        # reset recurrent states for episodes that have terminated
        policy_nn.reset(dones)
        if args_cli.save_data:
            record_obs = _extract_record_obs(obs, base_env)
            if record_obs is None:
                obs_keys = list(obs.keys()) if isinstance(obs, dict) else []
                raise RuntimeError(
                    "obs['record'] is required when --save_data is enabled.\n"
                    f"Current step obs type={type(obs).__name__}, keys={obs_keys}.\n"
                    "No 'record' found along wrapper chain via get_observations/_get_observations."
                )
            for i in range(num_envs):
                if success_saved is not None and bool(success_saved[i]):
                    continue
                _append_record_row(current_ep_buffers[i], _extract_record_row(record_obs, i))
            step_in_rollout += 1

            if success_saved is not None and data_dir is not None:
                success_per_env = _extract_success_per_env(
                    infos, base_env, num_envs, env.unwrapped.device
                )
                for i in range(num_envs):
                    if bool(success_per_env[i]) and not bool(success_saved[i]):
                        np.savez_compressed(
                            os.path.join(data_dir, f"episode_{episodes_collected}.npz"),
                            **_episode_buffer_to_npz_kwargs(current_ep_buffers[i]),
                        )
                        episodes_collected += 1
                        success_saved[i] = True
                        _clear_episode_buffer(current_ep_buffers[i])
                        print(
                            f"[INFO] Env {i} success at rollout_step={step_in_rollout}; "
                            f"saved episode {episodes_collected}/{int(args_cli.num_episodes)}"
                        )

            if max_steps_per_rollout is not None and step_in_rollout >= max_steps_per_rollout:
                if success_saved is not None and hasattr(base_env, "object_pos") and hasattr(base_env, "goal_pos"):
                    try:
                        pos_err = torch.norm(base_env.object_pos - base_env.goal_pos, p=2, dim=-1)
                        tol = getattr(getattr(base_env, "cfg", None), "success_pos_tol", None)
                        for i in range(num_envs):
                            if not bool(success_saved[i]):
                                tol_str = f", tol={float(tol):.6f}" if tol is not None else ""
                                print(
                                    f"[TIMEOUT_ERR] Env {i} timeout without success: "
                                    f"pos_err={float(pos_err[i]):.6f}{tol_str}"
                                )
                    except Exception as e:
                        print(f"[WARN] Failed to compute timeout position error: {e}")
                print(
                    f"[INFO] Rollout reached max_steps={max_steps_per_rollout}; "
                    "resetting env and clearing unsaved buffers."
                )
                current_ep_buffers = [dict() for _ in range(num_envs)]
                if success_saved is not None:
                    success_saved = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
                step_in_rollout = 0
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                record_obs = _extract_record_obs(obs, base_env)
                if record_obs is not None:
                    for i in range(num_envs):
                        _append_record_row(current_ep_buffers[i], _extract_record_row(record_obs, i))
                continue

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
    if args_cli.save_data:
        print(f"[INFO] Total saved successful trajectories: {episodes_collected}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
