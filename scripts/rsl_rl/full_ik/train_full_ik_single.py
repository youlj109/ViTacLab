# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""**full_ik**: scripted pregrasp + grasp, then UR10e **arm** via GPU IK + pour trajectory.

Default YAML (``--full-ik-config``) sets ``freeze_hand_after_script: true`` so the **hand stays at the grasp YAML**
after scripted phases; PPO gets a **1-d dummy action** (arm motion is IK-only). Set ``freeze_hand_after_script: false``
to learn hand joints with PPO instead.

Example::

    ./python.sh scripts/rsl_rl/full_ik/train_full_ik_single.py --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \\
        --num_envs 16 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

# ``ik_rl/utils`` (IK expander) + ``full_ik/utils`` (phased wrapper)
_FULL_IK_DIR = os.path.dirname(os.path.abspath(__file__))
_IK_RL_DIR = os.path.join(_FULL_IK_DIR, "..", "ik_rl")
_IK_UTILS = os.path.join(_IK_RL_DIR, "utils")
_FULL_IK_UTILS = os.path.join(_FULL_IK_DIR, "utils")
for _p in (_FULL_IK_UTILS, _IK_UTILS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_FULL_IK_YAML = Path(_FULL_IK_DIR) / "configs" / "full_ik_pour.yaml"
FULL_IK_PHASE_SCHEDULE: list = []
RESOLVED_FULL_IK_YAML: Path | None = None
FULL_IK_FREEZE_HAND: bool = False
FULL_IK_FREEZE_HAND_YAML: str | None = None
FULL_IK_CUP_REL_STABLE_CUP_ROT: bool = True

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
import numpy as np
parser = argparse.ArgumentParser(
    description="full_ik: scripted phases + GPU IK arm; optional frozen hand (default YAML) or PPO hand control."
)
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
# Palm + IK (minimal)
parser.add_argument(
    "--trajectory",
    type=str,
    default="object:150:0,goal:-1:0",
    help="Comma-separated phases: name:env_steps:use_rotation (0/1). "
    "name = env asset (e.g. object, cup) or tensor prefix (e.g. goal_cup → goal_cup_pos/rot), or goal (legacy). "
    "steps=-1 = until episode end.",
)
parser.add_argument(
    "--object-to-palm-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.05),
    metavar=("OX", "OY", "OZ"),
    help="Offset from trajectory anchor to palm origin (world if use_rotation=0, anchor frame if use_rotation=1).",
)
parser.add_argument(
    "--palm-in-wrist-pos",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.35),
    metavar=("PX", "PY", "PZ"),
    help="Palm origin in wrist_3 frame (m).",
)
parser.add_argument(
    "--palm-in-wrist-euler",
    type=float,
    nargs=3,
    default=(np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0),
    metavar=("RX", "RY", "RZ"),
    help="Palm in wrist_3 euler xyz (rad).",
)
parser.add_argument(
    "--palm-orient",
    type=str,
    choices=("fixed", "pickup_down"),
    default="pickup_down",
    help="When trajectory phase has use_rotation=0: fixed euler or pickup_down.",
)
parser.add_argument(
    "--palm-normal-local",
    type=float,
    nargs=3,
    default=(0.0, 1.0, 0.0),
    metavar=("NX", "NY", "NZ"),
    help="Palm-frame axis to align with --world-down (pickup_down).",
)
parser.add_argument(
    "--palm-yaw-offset",
    type=float,
    default=0.0,
    help="Extra yaw (rad) about world Z after pickup_down alignment.",
)
parser.add_argument(
    "--world-down",
    type=float,
    nargs=3,
    default=(0.0, 0.0, -1.0),
    metavar=("DX", "DY", "DZ"),
    help="World down direction for pickup_down.",
)
parser.add_argument(
    "--palm-euler",
    type=float,
    nargs=3,
    default=(0.0, 2.2, 0.0),
    metavar=("RX", "RY", "RZ"),
    help="Palm world euler xyz (rad) when --palm-orient fixed.",
)
parser.add_argument(
    "--palm-euler-in-anchor",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("RX", "RY", "RZ"),
    help="When use_rotation=1: euler xyz (rad) of palm relative to anchor frame (applied after anchor quat).",
)
parser.add_argument("--ee-body", type=str, default="wrist_3_link", help="End-effector link for Jacobian.")
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
    help="dls damping lambda override; default = Isaac controller.",
)
parser.add_argument(
    "--ik-config",
    type=str,
    default=None,
    help="Optional extra IK YAML merge (pickup-style). For full_ik, prefer --full-ik-config; pass 'none' to skip.",
)
parser.add_argument(
    "--full-ik-config",
    type=str,
    default=str(_DEFAULT_FULL_IK_YAML),
    help="YAML with phase_schedule + palm/IK/trajectory (see full_ik/configs/full_ik_pour.yaml).",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

import yaml

from ik_rl_load_config import (
    IK_YAML_KEYS,
    _coerce,
    load_ik_yaml_into_parser,
    resolve_ik_config_path,
    warn_if_task_mismatch_with_ik_yaml,
)


def _merge_optional_ik_config_cli_only(parser: argparse.ArgumentParser) -> None:
    """Apply ``--ik-config`` when present; do **not** load ``ik_rl_pickup.yaml`` by default.

    ``apply_sys_argv_ik_yaml_defaults(..., default_file=None)`` still falls back to pickup YAML, which would
    overwrite ``full_ik_pour.yaml`` trajectory (e.g. with ``object:...``).
    """
    load_ik_yaml_into_parser(parser, resolve_ik_config_path(sys.argv, None))


def _apply_full_ik_yaml_defaults(parser: argparse.ArgumentParser) -> None:
    global FULL_IK_PHASE_SCHEDULE, RESOLVED_FULL_IK_YAML, FULL_IK_FREEZE_HAND, FULL_IK_FREEZE_HAND_YAML, FULL_IK_CUP_REL_STABLE_CUP_ROT
    p = _DEFAULT_FULL_IK_YAML
    if "--full-ik-config" in sys.argv:
        i = sys.argv.index("--full-ik-config")
        if i + 1 < len(sys.argv):
            raw = sys.argv[i + 1].strip()
            low = raw.lower()
            if low in ("none", "false", ""):
                FULL_IK_PHASE_SCHEDULE = []
                RESOLVED_FULL_IK_YAML = None
                FULL_IK_FREEZE_HAND = False
                FULL_IK_FREEZE_HAND_YAML = None
                FULL_IK_CUP_REL_STABLE_CUP_ROT = True
                _merge_optional_ik_config_cli_only(parser)
                return
            p = Path(raw).expanduser()
    if not p.is_file():
        FULL_IK_PHASE_SCHEDULE = []
        RESOLVED_FULL_IK_YAML = None
        FULL_IK_FREEZE_HAND = False
        FULL_IK_FREEZE_HAND_YAML = None
        FULL_IK_CUP_REL_STABLE_CUP_ROT = True
        _merge_optional_ik_config_cli_only(parser)
        return
    RESOLVED_FULL_IK_YAML = p.resolve()
    data = yaml.safe_load(p.read_text()) or {}
    FULL_IK_PHASE_SCHEDULE = list(data.get("phase_schedule") or [])
    FULL_IK_FREEZE_HAND = bool(data.get("freeze_hand_after_script", False))
    _fhy = data.get("freeze_hand_yaml")
    FULL_IK_FREEZE_HAND_YAML = str(_fhy).strip() if _fhy else None
    FULL_IK_CUP_REL_STABLE_CUP_ROT = bool(data.get("cup_relative_stable_cup_rotation", True))
    kwargs: dict = {}
    for k in IK_YAML_KEYS:
        if k in data and data[k] is not None:
            kwargs[k] = _coerce(k, data[k])
    if kwargs:
        parser.set_defaults(**kwargs)
    _merge_optional_ik_config_cli_only(parser)


_apply_full_ik_yaml_defaults(parser)
args_cli, hydra_args = parser.parse_known_args()
_extra_ik = resolve_ik_config_path(sys.argv, None)
if _extra_ik is not None and _extra_ik.is_file():
    _exd = yaml.safe_load(_extra_ik.read_text()) or {}
    for _k in IK_YAML_KEYS:
        if _k not in _exd or _exd[_k] is None:
            continue
        if hasattr(args_cli, _k):
            setattr(args_cli, _k, _coerce(_k, _exd[_k]))

if RESOLVED_FULL_IK_YAML is not None:
    print(f"[INFO] full_ik defaults merged from YAML: {RESOLVED_FULL_IK_YAML}")
warn_if_task_mismatch_with_ik_yaml(RESOLVED_FULL_IK_YAML, args_cli.task)
RESOLVED_IK_CONFIG_YAML = _extra_ik if _extra_ik is not None else RESOLVED_FULL_IK_YAML

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""RSL-RL version check (same as train.py)."""

import importlib.metadata as metadata
import platform

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

import logging
import os
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
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl_log_utils import get_rsl_rl_log_root
from full_ik_hand_vec_env import PhasedArmIkHandExpander, PhasedIkHandRslRlVecEnvWrapper
from ik_rl_hand_vec_env import IkRlHandArmCfg, parse_trajectory_phases

logger = logging.getLogger(__name__)

import ViTacLab.tasks  # noqa: F401


def _apply_train_env_overrides_from_full_ik_yaml(env_cfg: object) -> None:
    """Merge ``train_env_overrides`` from resolved ``--full-ik-config`` into ``env_cfg`` (if present)."""
    if RESOLVED_FULL_IK_YAML is None or not RESOLVED_FULL_IK_YAML.is_file():
        return
    try:
        data = yaml.safe_load(RESOLVED_FULL_IK_YAML.read_text()) or {}
    except OSError:
        return
    ovr = data.get("train_env_overrides")
    if not isinstance(ovr, dict) or not ovr:
        return
    for k, v in ovr.items():
        if hasattr(env_cfg, k):
            setattr(env_cfg, k, v)
            logger.info("[full_ik] train_env_overrides: %s = %r", k, v)
        else:
            logger.warning("[full_ik] train_env_overrides: env_cfg has no attribute %r — skipped", k)

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

    _apply_train_env_overrides_from_full_ik_yaml(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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
    if not FULL_IK_PHASE_SCHEDULE:
        raise RuntimeError(
            "full_ik: phase_schedule is empty. Use --full-ik-config PATH to a YAML that defines "
            "phase_schedule (see scripts/rsl_rl/full_ik/configs/full_ik_pour.yaml), or fix the file path."
        )
    _proj_root = Path(__file__).resolve().parents[3]
    expander = PhasedArmIkHandExpander(
        base,
        ik_cfg,
        FULL_IK_PHASE_SCHEDULE,
        project_root=_proj_root,
        freeze_hand_after_script=FULL_IK_FREEZE_HAND,
        freeze_hand_yaml=FULL_IK_FREEZE_HAND_YAML,
        cup_relative_stable_cup_rotation=FULL_IK_CUP_REL_STABLE_CUP_ROT,
    )
    _scripted_h = int(sum(int(p.get("env_steps", 0)) for p in FULL_IK_PHASE_SCHEDULE))
    _pol_dim = 1 if FULL_IK_FREEZE_HAND else expander.num_hand
    print(
        f"[INFO] full_ik: scripted_horizon={_scripted_h} steps, then IK trajectory={args_cli.trajectory!r}; "
        f"policy_action_dim={_pol_dim} freeze_hand={FULL_IK_FREEZE_HAND}, full actuated={expander.num_actuated}"
    )
    if FULL_IK_FREEZE_HAND:
        print(
            "[WARN] full_ik: freeze_hand_after_script=True → hand fixed from YAML during IK; "
            "PPO action does not move fingers (1-d dummy). Useful for critic-only / smoke tests; "
            "disable in full_ik YAML to learn hand control."
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

    env = PhasedIkHandRslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        expander=expander,
        freeze_hand_after_script=FULL_IK_FREEZE_HAND,
    )

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
            "object_to_palm_offset": list(ik_cfg.object_to_palm_offset),
            "palm_in_wrist_pos": list(ik_cfg.palm_in_wrist_pos),
            "palm_in_wrist_euler_xyz": list(ik_cfg.palm_in_wrist_euler_xyz),
            "palm_orientation_mode": ik_cfg.palm_orientation_mode,
            "palm_euler_xyz": list(ik_cfg.palm_euler_xyz),
            "palm_normal_in_palm_frame": list(ik_cfg.palm_normal_in_palm_frame),
            "world_down": list(ik_cfg.world_down),
            "palm_yaw_offset_rad": ik_cfg.palm_yaw_offset_rad,
            "palm_euler_in_anchor_frame": list(ik_cfg.palm_euler_in_anchor_frame),
            "ee_body_name": ik_cfg.ee_body_name,
            "ik_method": ik_cfg.ik_method,
            "ik_lambda": ik_cfg.ik_lambda,
            "num_hand_actions": expander.num_hand,
            "freeze_hand_after_script": FULL_IK_FREEZE_HAND,
            "freeze_hand_yaml": FULL_IK_FREEZE_HAND_YAML,
        },
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
