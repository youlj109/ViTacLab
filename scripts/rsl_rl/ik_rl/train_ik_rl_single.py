# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train Shadow **hand** with RSL-RL; UR10e **arm** follows GPU differential IK + EE waypoints.

Mirrors ``train.py`` but wraps with :class:`IkHandRslRlVecEnvWrapper` (policy = hand only).

**Trajectory**: YAML list of ``{pos: [x,y,z], quat: [w,x,y,z], steps: int}`` — world-frame pose of ``ee_body``
(default ``wrist_3_link``). ``steps`` = env steps to hold; ``-1`` = until episode end.

Example::

    ./isaaclab.sh -p scripts/rsl_rl/ik_rl/train_ik_rl_single.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \\
        --num_envs 16

IK defaults load from ``configs/ik_rl_pickup.yaml`` when present (``--ik-config PATH`` or ``none``).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Local imports: ``utils/`` next to this file (``scripts/rsl_rl/ik_rl/utils``)
_IK_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_IK_UTILS = os.path.join(_IK_RL_DIR, "utils")
if _IK_UTILS not in sys.path:
    sys.path.insert(0, _IK_UTILS)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train hand-only RL with GPU differential IK arm (single-arm setup).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--max_episode_length",
    type=int,
    default=None,
    help="Override env horizon in RL **env steps** (DirectRLEnv: sets episode_length_s = steps * sim.dt * decimation).",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# IK: EE waypoints from YAML (list of {pos, quat, steps})
parser.add_argument("--trajectory", default=None, help=argparse.SUPPRESS)
parser.add_argument("--ee-body", type=str, default=None, dest="ee_body", help="EE link for IK (default: wrist_3_link).")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=("pinv", "svd", "trans", "dls"),
    default="dls",
    help="Differential IK Jacobian method.",
)
parser.add_argument(
    "--ik-lambda",
    type=float,
    default=None,
    help="dls damping lambda override; default = 0.005 in ik_rl (Isaac default is 0.01).",
)
parser.add_argument(
    "--ik-k-val",
    type=float,
    default=None,
    dest="ik_k_val",
    help="pinv/svd/trans: scale step size (Isaac k_val); ignored for dls.",
)
parser.add_argument(
    "--ik-delta-scale",
    type=float,
    default=1.0,
    dest="ik_delta_scale",
    help="Multiply joint-space IK delta each step (>1 = faster EE motion, may overshoot).",
)
parser.add_argument(
    "--ik-waypoints-world-frame",
    action="store_true",
    dest="ik_waypoints_world_frame",
    help="YAML pos is global sim world (do not add env_origins). Default: env-local for multi-env cloning.",
)
parser.add_argument(
    "--hand-freeze-phase-target",
    type=str,
    default=None,
    help="When set along with --hand-freeze-yaml: freeze hand joints during trajectory phases whose target matches this string (e.g. pickup: 'goal').",
)
parser.add_argument(
    "--hand-freeze-yaml",
    type=str,
    default=None,
    help="YAML with hand_joint_pos_shadow_order (24 floats) to freeze hand joints to during grasp phase.",
)
parser.add_argument(
    "--ik-config",
    type=str,
    default=None,
    help="YAML with task + trajectory (see configs/ik_rl_pickup.yaml). "
    "If omitted, that file is loaded when present. Pass 'none' to disable.",
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
RESOLVED_IK_CONFIG_YAML = _cfg_path
if _cfg_path is not None:
    print(f"[INFO] IK trajectory defaults merged from YAML: {_cfg_path}")
warn_if_task_mismatch_with_ik_yaml(RESOLVED_IK_CONFIG_YAML, args_cli.task)

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging
import shutil
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl_log_utils import check_rsl_rl_lib_version, get_rsl_rl_log_root
from ik_rl_hand_vec_env import ArmIkHandActionExpander, IkHandRslRlVecEnvWrapper, build_ik_cfg_from_trajectory_args

logger = logging.getLogger(__name__)

check_rsl_rl_lib_version()

import ViTacLab.tasks  # noqa: F401
from ViTacLab.utils.vitaclab_marl_rsl import multi_agent_to_single_agent

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train hand-only policy with GPU differential IK arm."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    if args_cli.max_episode_length is not None:
        n = int(args_cli.max_episode_length)
        if n < 1:
            raise ValueError("--max_episode_length must be >= 1")
        if hasattr(env_cfg, "max_episode_length"):
            env_cfg.max_episode_length = n
        elif hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg, "decimation"):
            step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
            env_cfg.episode_length_s = float(n) * step_dt
            logger.info(
                "[train_ik_rl_single] --max_episode_length=%d → episode_length_s=%.6f (step_dt=%.6f)",
                n,
                env_cfg.episode_length_s,
                step_dt,
            )
        else:
            logger.warning(
                "[train_ik_rl_single] --max_episode_length ignored: env_cfg has no usable max_episode_length / "
                "episode_length_s + sim + decimation."
            )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
    elif args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root_path = get_rsl_rl_log_root(args_cli.task, getattr(args_cli, "experiment_name", None))
    if getattr(args_cli, "experiment_name", None) is None:
        agent_cfg.experiment_name = os.path.basename(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir

    _enable_cams = bool(getattr(args_cli, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    setattr(env_cfg, "enable_cameras", _enable_cams)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base: DirectRLEnv = env.unwrapped

    env.reset()

    ik_cfg = build_ik_cfg_from_trajectory_args(args_cli, arm="single")
    expander = ArmIkHandActionExpander(base, ik_cfg)
    print(
        f"[INFO] Hand-only RL: policy actions={expander.num_hand}, full actuated={expander.num_actuated}, "
        f"EE waypoints={len(ik_cfg.waypoints)}"
    )

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = IkHandRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions, expander=expander)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if RESOLVED_IK_CONFIG_YAML is not None:
        try:
            shutil.copy2(
                RESOLVED_IK_CONFIG_YAML,
                os.path.join(log_dir, "params", "ik_config_source.yaml"),
            )
        except OSError:
            pass
    dump_yaml(
        os.path.join(log_dir, "params", "ik_rl_hand.yaml"),
        {
            "task": args_cli.task,
            "ik_config_source_yaml": str(RESOLVED_IK_CONFIG_YAML) if RESOLVED_IK_CONFIG_YAML else None,
            "trajectory": args_cli.trajectory,
            "ee_body_name": ik_cfg.ee_body_name,
            "ik_method": ik_cfg.ik_method,
            "ik_lambda": ik_cfg.ik_lambda,
            "ik_k_val": ik_cfg.ik_k_val,
            "ik_delta_scale": ik_cfg.ik_delta_scale,
            "ik_waypoints_world_frame": getattr(args_cli, "ik_waypoints_world_frame", False),
            "num_hand_actions": expander.num_hand,
        },
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
