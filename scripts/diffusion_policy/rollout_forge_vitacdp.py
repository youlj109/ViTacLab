#!/usr/bin/env python3
"""Roll out a ViTacDP checkpoint on forge dexhand tasks."""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Load NumPy submodules before Isaac prepends its pip_prebundle paths. Otherwise
# late imports from gymnasium/trimesh can mix conda NumPy with Isaac's bundled NumPy.
import numpy.lib.recfunctions  # noqa: F401
import numpy.lib.stride_tricks  # noqa: F401
import numpy.ma  # noqa: F401
import numpy.random  # noqa: F401
import torch
from isaaclab.app import AppLauncher


_TASK_PRESETS: dict[str, dict[str, str]] = {
    "forge_insert": {
        "gym_id": "Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0",
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgePegInsertEnvCfg",
    },
    "forge_gear": {
        "gym_id": "Isaac-UR10eShadowHand-ForgeGearMesh-Direct-v0",
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeGearMeshEnvCfg",
    },
    "forge_nut": {
        "gym_id": "Isaac-UR10eShadowHand-ForgeNutThread-Direct-v0",
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeNutThreadEnvCfg",
    },
}
_GYM_ID_TO_PRESET = {v["gym_id"]: k for k, v in _TASK_PRESETS.items()}


def _repo_root() -> Path:
    path = Path(__file__).resolve().parent
    for _ in range(10):
        if (path / "source").is_dir() and (path / "policy").is_dir():
            return path
        if path.parent == path:
            break
        path = path.parent
    return Path(__file__).resolve().parents[2]


def _load_symbol(entry: str) -> Any:
    module_name, symbol_name = entry.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _tensor_to_policy_value(value: Any, env_index: int) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim > 0:
        env_index = max(0, min(int(env_index), int(value.shape[0]) - 1))
        value = value[env_index]
    return value.detach().cpu()


def _policy_observation_from_record(obs: object, num_envs: int) -> list[dict[str, torch.Tensor]]:
    if not isinstance(obs, dict) or not isinstance(obs.get("record"), dict):
        raise RuntimeError("Expected env observation dict with obs['record']; run with cameras enabled.")

    record = obs["record"]
    required = (
        "third_person_camera",
        "third_person_camera_pos",
        "twist_camera",
        "twist_camera_pos",
        "tactile_normal_force",
        "tactile_shear_force",
        "tactile_rgb_image",
        "tactile_pos",
        "joint_pos",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise RuntimeError(f"obs['record'] missing ViTacDP fields: {missing}. Pass --enable_cameras.")

    out: list[dict[str, torch.Tensor]] = []
    for env_i in range(num_envs):
        out.append(
            {
                "third_person_camera": _tensor_to_policy_value(record["third_person_camera"], env_i),
                "third_person_camera_pos": _tensor_to_policy_value(record["third_person_camera_pos"], env_i),
                "twist_camera": _tensor_to_policy_value(record["twist_camera"], env_i),
                "twist_camera_pos": _tensor_to_policy_value(record["twist_camera_pos"], env_i),
                "tactile_normal_force": _tensor_to_policy_value(record["tactile_normal_force"], env_i),
                "tactile_shear_force": _tensor_to_policy_value(record["tactile_shear_force"], env_i),
                "tactile_rgb_image": _tensor_to_policy_value(record["tactile_rgb_image"], env_i),
                "tactile_pos": _tensor_to_policy_value(record["tactile_pos"], env_i),
                "joint_pos": _tensor_to_policy_value(record["joint_pos"], env_i),
            }
        )
    return out


def _reset_env(env: Any) -> object:
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _reset_policy_state(policy: Any) -> None:
    policy.action_idx = 0
    policy.actions = None
    try:
        policy.policy.policy.reset_obs()
    except AttributeError:
        pass


def _joint_pos_to_normalized_action(env: Any, joint_pos: np.ndarray) -> np.ndarray:
    """Convert actuated joint positions in radians to IsaacLab normalized action."""
    action = np.asarray(joint_pos, dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if action.shape[-1] != int(env.num_actions):
        raise ValueError(
            f"Expected ViTacDP joint_pos output dim {env.num_actions}, got {action.shape[-1]}."
        )

    joint_ids = list(env.actuated_dof_indices)
    lower = env.robot_dof_lower_limits[:, joint_ids].detach().cpu().numpy()
    upper = env.robot_dof_upper_limits[:, joint_ids].detach().cpu().numpy()
    if lower.shape[0] == 1 and action.shape[0] > 1:
        lower = np.repeat(lower, action.shape[0], axis=0)
        upper = np.repeat(upper, action.shape[0], axis=0)
    lower = lower[: action.shape[0]]
    upper = upper[: action.shape[0]]

    denom = upper - lower
    normalized = np.zeros_like(action, dtype=np.float32)
    valid = denom > 1.0e-6
    normalized[valid] = (2.0 * (action[valid] - lower[valid]) / denom[valid]) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def _add_joint_pos_delta(env: Any, joint_pos: np.ndarray, joint_tokens: tuple[str, ...], delta_rad: float) -> np.ndarray:
    action = np.asarray(joint_pos, dtype=np.float32).copy()
    if abs(float(delta_rad)) <= 0.0:
        return action
    if action.ndim == 1:
        action = action.reshape(1, -1)

    for local_i, joint_id in enumerate(env.actuated_dof_indices):
        joint_name = str(env.robot.joint_names[int(joint_id)])
        if any(token in joint_name or joint_name.endswith(token) for token in joint_tokens):
            action[:, local_i] += float(delta_rad)
    return action


def _is_success_now(env: Any) -> torch.Tensor:
    try:
        if hasattr(env, "_get_curr_successes") and hasattr(env, "cfg_task"):
            check_rot = bool(getattr(env.cfg_task, "name", "") == "nut_thread")
            threshold = float(getattr(env.cfg_task, "success_threshold", 1.0))
            successes = env._get_curr_successes(success_threshold=threshold, check_rot=check_rot)  # noqa: SLF001
            if torch.is_tensor(successes):
                return successes.detach().bool().cpu()
    except Exception as exc:
        print(f"[WARN] Could not query forge success state: {exc}")
    return torch.zeros((int(getattr(env, "num_envs", 1)),), dtype=torch.bool)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=str,
        default="forge_insert",
        choices=sorted([*_TASK_PRESETS.keys(), *_GYM_ID_TO_PRESET.keys()]),
        help="Forge task alias or Gym id.",
    )
    parser.add_argument("--env", type=str, default="", help="Override env entry point module:Class.")
    parser.add_argument("--cfg", type=str, default="", help="Override cfg entry point module:Class.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=float, default=0.0, help="If >0, sleep to approximate this rollout FPS.")
    parser.add_argument("--max_steps", type=int, default=2000, help="0 = run until app closes.")
    parser.add_argument("--max_episode_steps", type=int, default=250, help="Max env.step calls per episode batch.")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=20,
        help="Total sub-environment episodes to evaluate across all parallel envs; 0 = unlimited.",
    )
    parser.add_argument(
        "--env_spacing",
        type=float,
        default=3.0,
        help="Distance between cloned environments. Increase this for camera/tactile-heavy parallel eval.",
    )
    parser.add_argument("--task_name", type=str, default="", help="ViTacDP checkpoint task name; default uses the Gym id.")
    parser.add_argument("--data_num", type=int, required=True, help="ViTacDP data count used in checkpoint directory name.")
    parser.add_argument("--checkpoint_num", type=int, required=True, help="ViTacDP checkpoint number without .ckpt.")
    parser.add_argument(
        "--index_middle_j3_delta_rad",
        type=float,
        default=0.0,
        help="Add this radian offset to ShadowHand FFJ3 and MFJ3 during rollout. Only applies in joint_pos mode.",
    )
    parser.add_argument(
        "--policy_action_mode",
        choices=("joint_pos", "normalized_action"),
        default="joint_pos",
        help=(
            "How to interpret ViTacDP's 30D output. Use 'joint_pos' for checkpoints trained "
            "from data['joint_pos'][1:], or 'normalized_action' for checkpoints trained "
            "from recorded data['action']."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    repo_root = _repo_root()
    vitacdp_dir = repo_root / "policy" / "ViTacDP"
    # ViTacDP uses top-level `diffusion_policy` imports internally; keep its copy first.
    for path in (vitacdp_dir, repo_root, repo_root / "source"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401
    from policy.ViTacDP.deploy_policy import Encapsulation

    preset_key = _GYM_ID_TO_PRESET.get(str(args.task), str(args.task))
    preset = _TASK_PRESETS[preset_key]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]
    dp_task_name = str(args.task_name).strip() or preset["gym_id"]

    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    if hasattr(cfg.scene, "env_spacing"):
        cfg.scene.env_spacing = float(args.env_spacing)
    cfg.device = getattr(args, "device", None) or "cuda:0"
    setattr(cfg, "enable_cameras", True)
    if hasattr(cfg, "enable_twist_camera"):
        setattr(cfg, "enable_twist_camera", True)
    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(cfg, "seed"):
            setattr(cfg, "seed", seed)
        print(f"[INFO] Using seed={seed}")

    env = EnvCls(cfg)
    if not hasattr(env, "_use_rl_control"):
        env._use_rl_control = True

    init_dict = {
        "task_name": dp_task_name,
        "data_num": int(args.data_num),
        "checkpoint_num": int(args.checkpoint_num),
    }
    policy = Encapsulation(init_dict, env.num_envs)

    obs = _reset_env(env)
    _reset_policy_state(policy)
    total_episode_count = 0
    total_success_count = 0
    per_env_episode_count = np.zeros((env.num_envs,), dtype=np.int64)
    per_env_success_count = np.zeros((env.num_envs,), dtype=np.int64)
    total_steps = 0
    batch_episode_steps = 0
    env_completed = torch.zeros((env.num_envs,), dtype=torch.bool)
    target_dt = 1.0 / float(args.fps) if float(args.fps) > 0 else 0.0

    print(
        "[INFO] Starting ViTacDP rollout: "
        f"task={preset_key}, dp_task_name={dp_task_name}, num_envs={env.num_envs}, "
        f"checkpoint={args.checkpoint_num}, data_num={args.data_num}, "
        f"policy_action_mode={args.policy_action_mode}, env_spacing={cfg.scene.env_spacing}, "
        f"index_middle_j3_delta_rad={args.index_middle_j3_delta_rad}"
    )

    while simulation_app.is_running():
        t0 = time.time()
        total_steps += 1
        batch_episode_steps += 1

        policy_obs = _policy_observation_from_record(obs, env.num_envs)
        action_np = np.asarray(policy.get_action(policy_obs), dtype=np.float32)
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)
        if args.policy_action_mode == "joint_pos":
            action_np = _add_joint_pos_delta(
                env,
                action_np,
                joint_tokens=("FFJ3", "MFJ3"),
                delta_rad=float(args.index_middle_j3_delta_rad),
            )
            action_np = _joint_pos_to_normalized_action(env, action_np)
        actions = torch.as_tensor(action_np, dtype=torch.float32, device=env.device)

        obs, _rew, terminated, truncated, _extras = env.step(actions)
        success = _is_success_now(env)
        done = success.clone()
        if torch.is_tensor(terminated):
            done = done | terminated.detach().bool().cpu()
        if torch.is_tensor(truncated):
            done = done | truncated.detach().bool().cpu()
        if args.max_episode_steps > 0 and batch_episode_steps >= int(args.max_episode_steps):
            done = torch.ones_like(done, dtype=torch.bool)

        newly_done = done & ~env_completed
        if bool(torch.any(newly_done).item()):
            done_indices = torch.nonzero(newly_done, as_tuple=False).squeeze(-1).tolist()
            for env_i in done_indices:
                env_i = int(env_i)
                is_success = bool(success[env_i].item())
                per_env_episode_count[env_i] += 1
                total_episode_count += 1
                if is_success:
                    per_env_success_count[env_i] += 1
                    total_success_count += 1
                print(
                    f"[EPISODE {total_episode_count}] env={env_i} success={is_success} "
                    f"batch_steps={batch_episode_steps} total_steps={total_steps}"
                )
            env_completed |= newly_done
            if args.max_episodes > 0 and total_episode_count >= int(args.max_episodes):
                break
            if bool(torch.all(env_completed).item()):
                obs = _reset_env(env)
                _reset_policy_state(policy)
                env_completed.zero_()
                batch_episode_steps = 0
                continue

        if args.max_steps > 0 and total_steps >= int(args.max_steps):
            break

        if target_dt > 0:
            elapsed = time.time() - t0
            if target_dt > elapsed:
                time.sleep(target_dt - elapsed)

    rate = total_success_count / max(1, total_episode_count)
    print(
        f"[RESULT] episodes={total_episode_count} successes={total_success_count} "
        f"success_rate={rate:.3f} total_steps={total_steps}"
    )
    for env_i in range(env.num_envs):
        env_rate = per_env_success_count[env_i] / max(1, per_env_episode_count[env_i])
        print(
            f"[RESULT][env={env_i}] episodes={per_env_episode_count[env_i]} "
            f"successes={per_env_success_count[env_i]} success_rate={env_rate:.3f}"
        )
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
