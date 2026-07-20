"""Run and validate a trained Diffusion Policy or ViTacDP checkpoint.

The script creates one canonical ViTacLab Gym task, obtains the task's
``record`` observation, executes policy action chunks, and writes one MP4 plus
one success flag per parallel environment under ``data/validation``.  Use
``--policy-output`` when the checkpoint output dimension is ambiguous.  Run
``python scripts/policy/play_policy.py --help`` for every supported option.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import traceback
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import imageio
import cv2

from isaaclab.app import AppLauncher
# Shared non-executable RL CLI helpers.
_POLICY_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.abspath(os.path.join(_POLICY_SCRIPT_DIR, "..", "common"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from rl import cli_args  # isort: skip
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Diffusion Policy or ViTacDP in a ViTacLab task and save validation videos/results."
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (default: 1).")
    # Keep this optional during parser construction. Isaac Lab 0.54.x's
    # AppLauncher temporarily removes -h/--help and calls parse_known_args();
    # argparse-required options would therefore make ``--help`` exit with 2.
    parser.add_argument("--task", type=str, default=None, help="Registered ViTacLab Gym task ID (required for execution).")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Gym metadata key used only to obtain wrapper/clip settings (default: rsl_rl_cfg_entry_point).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed; omit to use the agent config default.")
    parser.add_argument(
        "--data_num",
        type=int,
        default=200,
        help="Training-dataset count embedded in the default checkpoint directory name (default: 200).",
    )
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=1000,
        help="Checkpoint filename without .ckpt when --policy-checkpoint is omitted (default: 1000).",
    )
    parser.add_argument(
        "--policy_name",
        "--policy-name",
        dest="policy_name",
        choices=("Diffusion_Policy", "ViTacDP"),
        default="ViTacDP",
        help="Policy family to load (default: ViTacDP).",
    )
    parser.add_argument(
        "--observation-profile",
        "--version",
        dest="observation_profile",
        default=None,
        help="Checkpoint observation profile such as rgb or force. --version is a deprecated compatibility alias.",
    )
    parser.add_argument(
        "--checkpoint_task_name",
        type=str,
        default=None,
        help="Checkpoint folder prefix. Defaults to the task ID, optionally suffixed by --observation-profile.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Exact .ckpt file. Relative paths are resolved from the project working directory; overrides folder/number lookup.",
    )
    parser.add_argument(
        "--policy-output",
        type=str,
        choices=("auto", "joint_pos", "action"),
        default="auto",
        help="Policy output semantics: joint targets or normalized env actions. 'auto' infers by output dim.",
    )
    
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum policy steps per rollout; 0 waits until all envs terminate.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of rollout batches. Each batch produces num_envs videos and success flags (default: 1).",
    )
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=None,
        help="Override the task's internal timeout in steps. Omit to preserve the environment definition.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Zero-action physics steps after each reset before policy inference (default: 5).",
    )
    parser.add_argument(
        "--debug-observations",
        action="store_true",
        help="Print finite-value statistics for the first policy observation of every rollout.",
    )
    parser.add_argument(
        "--joint_err_print_interval",
        type=int,
        default=0,
        help="Print joint tracking/jitter diagnostics every N steps; <=0 disables them (default: disabled).",
    )
    parser.add_argument(
        "--jitter_topk",
        type=int,
        default=5,
        help="Number of largest finger-joint jitter values shown when diagnostics are enabled (default: 5).",
    )

    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_arg_parser()
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.task:
    parser.error("--task is required when executing policy inference.")

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

def _apply_farthest_point_sample(buf):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in buf.items():
        if "pointcloud" in k:
            buf[k] = _farthest_point_sample(v.to(device=device), 2048).cpu()
    return buf

def _append_env_frame_to_buffer(buf, obs_record):
    """Append one environment's record tensors as CPU NumPy arrays."""

    for k, v in obs_record.items():
        if k not in buf:
            buf[k] = []
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)
        buf[k].append(v)


def _resolve_primary_robot(base_env):
    """Return a representative robot Articulation for the env.

    单臂任务有 ``base_env.robot``，直接返回（行为与之前完全一致）。双臂任务没有
    ``robot``，而是 ``right_hand``/``left_hand``，或在 scene 中注册为
    ``right_robot``/``left_robot``。此处仅用于读取 dtype/joint 维度与 joint_names，
    随便取一个有效的 articulation 即可，不影响动作下发逻辑。
    """
    robot = getattr(base_env, "robot", None)
    if robot is not None:
        return robot
    for attr in ("right_hand", "left_hand", "robot_right", "robot_left"):
        cand = getattr(base_env, attr, None)
        if cand is not None:
            return cand
    scene = getattr(base_env, "scene", None)
    arts = getattr(scene, "articulations", None) if scene is not None else None
    if arts:
        for key in ("robot", "right_robot", "left_robot"):
            if key in arts:
                return arts[key]
        for cand in arts.values():
            if cand is not None:
                return cand
    return None


def _obs_has_key(obs, key: str) -> bool:
    try:
        if hasattr(obs, "keys"):
            return key in obs.keys()
        return key in obs
    except Exception:
        return False


def _resolve_task_env(env):
    """Unwrap to the task env that implements ``_get_observations`` / ``record`` (not the MARL shell)."""
    cur = env
    for _ in range(16):
        inner = getattr(cur, "env", None)
        if inner is not None and inner is not cur:
            cur = inner
            continue
        nxt = getattr(cur, "unwrapped", None)
        if nxt is not None and nxt is not cur:
            cur = nxt
            continue
        break
    return cur


def _merge_record_into_obs(obs, rec: dict, num_envs: int):
    from tensordict import TensorDict

    rec_td = TensorDict(dict(rec), batch_size=[num_envs])
    if isinstance(obs, TensorDict):
        merged = {k: obs[k] for k in obs.keys()}
        merged["record"] = rec_td
        return TensorDict(merged, batch_size=[num_envs])
    if isinstance(obs, dict):
        out = dict(obs)
        out["record"] = rec_td
        return out
    return TensorDict({"policy": obs, "record": rec_td}, batch_size=[num_envs])


def _record_is_policy_complete(record) -> bool:
    """Return whether a record already contains the selected policy's required sensor schema."""

    try:
        keys = set(record.keys())
    except Exception:
        return False
    camera_keys = [key for key in keys if key.endswith("camera") and not key.endswith("_pos")]
    if "joint_pos" not in keys or not camera_keys:
        return False
    if args_cli.policy_name == "Diffusion_Policy":
        return True
    has_tactile = "tactile_rgb_image" in keys or {
        "tactile_normal_force",
        "tactile_shear_force",
    }.issubset(keys)
    if "tactile_pos" not in keys or not has_tactile:
        return False
    return all(f"{key}_pos" in keys for key in camera_keys)


def _ensure_record_in_obs(base_env, obs):
    """Make sure ``obs['record']`` exists, recomputing it for MARL tasks if needed.

    单臂 DirectRLEnv 的 obs 自带 ``record``（``RslRlVecEnvWrapper`` 已包成 TensorDict），
    此函数直接返回（行为不变）。

    双臂经 ``multi_agent_to_single_agent`` 后 wrapper 的 obs 只有 ``policy``；需从真实
    task env（``base_env.env`` 或 ``unwrapped``）重算 ``record`` 并合并回 TensorDict。
    """
    task_env = _resolve_task_env(base_env)
    if _obs_has_key(obs, "record"):
        if _record_is_policy_complete(obs["record"]):
            return obs
        build_record = getattr(task_env, "_build_record_dict", None)
        if not callable(build_record):
            return obs
        supplemental = build_record()
        current = obs["record"]
        missing = {key: value for key, value in supplemental.items() if key not in current.keys()}
        if not missing:
            return obs
        rec = {key: current[key] for key in current.keys()}
        rec.update(missing)
        n = int(getattr(task_env, "num_envs", getattr(base_env, "num_envs", 1)))
        return _merge_record_into_obs(obs, rec, n)

    if hasattr(task_env, "_compute_intermediate_values"):
        task_env._compute_intermediate_values()

    full = task_env._get_observations()
    if isinstance(full, dict) and "record" in full:
        rec = full["record"]
    else:
        build_record = getattr(task_env, "_build_record_dict", None)
        rec = build_record() if callable(build_record) else None
    if not isinstance(rec, dict):
        raise RuntimeError(
            "Could not build obs['record'] from task env "
            f"({type(task_env).__name__}); keys={list(full.keys()) if isinstance(full, dict) else type(full)}"
        )
    n = int(getattr(task_env, "num_envs", getattr(base_env, "num_envs", 1)))
    return _merge_record_into_obs(obs, rec, n)


def _infer_env_action_dim(base_env, env) -> int:
    """Total env action dim, summing per-agent spaces for MARL (single-arm unchanged)."""
    n = int(getattr(base_env, "num_actions", 0) or 0)
    if n > 0:
        return n
    n = int(getattr(env, "num_actions", 0) or 0)
    if n > 0:
        return n
    import gymnasium as _gym

    action_spaces = getattr(base_env.cfg, "action_spaces", None)
    possible_agents = getattr(base_env.cfg, "possible_agents", None) or (
        list(action_spaces.keys()) if isinstance(action_spaces, dict) else None
    )
    if isinstance(action_spaces, dict) and possible_agents:
        try:
            return int(sum(_gym.spaces.flatdim(action_spaces[a]) for a in possible_agents))
        except Exception:
            pass
    action_space = getattr(base_env.cfg, "action_space", None)
    if isinstance(action_space, int):
        return action_space
    return 7


def _find_non_finite_paths(x, prefix: str = "") -> list[str]:
    """Recursively collect paths containing NaN/Inf in numpy/torch payloads."""
    bad: list[str] = []
    if torch.is_tensor(x):
        if not torch.isfinite(x).all():
            bad.append(prefix or "<root>")
        return bad
    if isinstance(x, np.ndarray):
        if not np.isfinite(x).all():
            bad.append(prefix or "<root>")
        return bad
    if isinstance(x, dict):
        for k, v in x.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            bad.extend(_find_non_finite_paths(v, key))
        return bad
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            bad.extend(_find_non_finite_paths(v, key))
        return bad
    return bad


def _extract_success_per_env(infos, obs, base_env, num_envs: int, device: torch.device) -> torch.Tensor:
    """Read task success from the common ViTacLab/Isaac Lab info layouts."""

    candidates = []
    if isinstance(infos, dict):
        candidates.extend(
            [
                infos.get("curr_success_per_env"),
                infos.get("successes"),
                infos.get("success"),
            ]
        )
        for nested_key in ("extras", "log", "record"):
            nested = infos.get(nested_key)
            if isinstance(nested, dict):
                candidates.extend(
                    [nested.get("curr_success_per_env"), nested.get("successes"), nested.get("success")]
                )
    try:
        record = obs["record"]
        for key in ("curr_success_per_env", "successes", "success"):
            if key in record.keys():
                candidates.append(record[key])
    except Exception:
        pass

    cur = base_env
    seen: set[int] = set()
    for _ in range(20):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        for key in ("curr_success_per_env", "successes", "success"):
            candidates.append(getattr(cur, key, None))
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


def _debug_print_obs_stats(obs_dict: dict, *, max_items: int = 8) -> None:
    """Print per-key finite stats for policy input observation."""
    printed = 0
    for key in sorted(obs_dict.keys()):
        if printed >= max_items:
            remain = len(obs_dict.keys()) - printed
            if remain > 0:
                print(f"[ObsDebug] ... {remain} more keys omitted")
            break
        val = obs_dict[key]
        if torch.is_tensor(val):
            arr = val.detach()
            finite_mask = torch.isfinite(arr)
            num_total = int(arr.numel())
            num_bad = int((~finite_mask).sum().item())
            if num_total > 0:
                arr_f32 = arr.to(torch.float32)
                vmin = float(torch.nan_to_num(arr_f32, nan=0.0, posinf=0.0, neginf=0.0).min().item())
                vmax = float(torch.nan_to_num(arr_f32, nan=0.0, posinf=0.0, neginf=0.0).max().item())
                vmean = float(torch.nan_to_num(arr_f32, nan=0.0, posinf=0.0, neginf=0.0).mean().item())
                flat = arr_f32.flatten()
                preview = ", ".join([f"{float(x):.4f}" for x in flat[:5].detach().cpu().tolist()])
            else:
                vmin = vmax = vmean = 0.0
                preview = ""
            print(
                f"[ObsDebug] key={key} shape={tuple(arr.shape)} bad={num_bad}/{num_total} "
                f"min={vmin:.4f} max={vmax:.4f} mean={vmean:.4f} first5=[{preview}]"
            )
            printed += 1
        elif isinstance(val, np.ndarray):
            finite_mask = np.isfinite(val)
            num_total = int(val.size)
            num_bad = int(np.size(val) - np.count_nonzero(finite_mask))
            if num_total > 0:
                safe = np.nan_to_num(val.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                vmin = float(safe.min())
                vmax = float(safe.max())
                vmean = float(safe.mean())
                preview = ", ".join([f"{float(x):.4f}" for x in safe.reshape(-1)[:5].tolist()])
            else:
                vmin = vmax = vmean = 0.0
                preview = ""
            print(
                f"[ObsDebug] key={key} shape={val.shape} bad={num_bad}/{num_total} "
                f"min={vmin:.4f} max={vmax:.4f} mean={vmean:.4f} first5=[{preview}]"
            )
            printed += 1


def compute_tactile_shear_image(
    tactile_normal_force: np.ndarray,
    tactile_shear_force: np.ndarray,
    normal_force_threshold: float = 0.00008,
    shear_force_threshold: float = 0.0005,
    resolution: int = 30,
) -> np.ndarray:
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]
    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)
    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            sx = float(tactile_shear_force[row, col, 0])
            sy = float(tactile_shear_force[row, col, 1])
            nf_v = float(tactile_normal_force[row, col])
            loc1_x = float(loc0_x) + sx / shear_force_threshold * resolution
            loc1_y = float(loc0_y) + sy / shear_force_threshold * resolution
            color = (
                0.0,
                float(max(0.0, 1.0 - nf_v / normal_force_threshold)),
                float(min(1.0, nf_v / normal_force_threshold)),
            )
            cv2.arrowedLine(
                imgs_tactile,
                (int(loc0_y), int(loc0_x)),
                (int(loc1_y), int(loc1_x)),
                color,
                6,
                tipLength=0.4,
            )
    return imgs_tactile


def _squeeze_tactile_normal(nf: np.ndarray) -> np.ndarray:
    nf = np.asarray(nf, dtype=np.float32)
    if nf.ndim == 4 and nf.shape[-1] == 1:
        nf = nf[..., 0]
    return nf


def concat_tactile_rgb_image(tactile_rgb_image) -> np.ndarray:
    x = np.asarray(tactile_rgb_image)
    if x.ndim == 4 and x.shape[0] > 1:
        return np.concatenate([x[i] for i in range(x.shape[0])], axis=1)
    return np.asarray(x)


def frame_tactile_force_field(
    tactile_normal_force_t: np.ndarray,
    tactile_shear_force_t: np.ndarray,
    normal_thr: float = 0.00008,
    shear_thr: float = 0.0005,
    resolution: int = 30,
) -> np.ndarray:
    nf = _squeeze_tactile_normal(tactile_normal_force_t)
    sf = np.asarray(tactile_shear_force_t, dtype=np.float32)
    if nf.shape[0] != sf.shape[0]:
        raise ValueError(f"Finger count mismatch: normal {nf.shape}, shear {sf.shape}")
    pieces = []
    for f in range(nf.shape[0]):
        img = compute_tactile_shear_image(
            nf[f],
            sf[f],
            normal_force_threshold=normal_thr,
            shear_force_threshold=shear_thr,
            resolution=resolution,
        )
        pieces.append((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.concatenate(pieces, axis=1)


def _to_uint8_image(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Video frame must be HxW, HxWx3, or HxWx4; got {arr.shape}.")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.nanmax(arr)) if arr.size > 0 else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _stack_rows_with_padding(rows: list[np.ndarray]) -> np.ndarray:
    max_width = max(r.shape[1] for r in rows)
    padded_rows = []
    for row in rows:
        row_u8 = _to_uint8_image(row)
        if row_u8.shape[1] < max_width:
            pad = np.zeros((row_u8.shape[0], max_width - row_u8.shape[1], 3), dtype=np.uint8)
            row_u8 = np.concatenate([row_u8, pad], axis=1)
        padded_rows.append(row_u8)
    return np.concatenate(padded_rows, axis=0)


def save_result(save_dir, episode_count, current_ep_buffers, success_episode):
    """Save one validation video and one success flag per parallel environment."""

    success_values = torch.as_tensor(success_episode, dtype=torch.bool).detach().cpu().tolist()
    for env_index, ep_buffer in enumerate(current_ep_buffers):
        idx = episode_count + env_index
        camera_sequences = []
        for key, values in sorted(ep_buffer.items()):
            if (key.endswith("camera") or key.endswith("camera_rgb")) and not key.endswith("_pos") and values:
                sequence = np.asarray(values)
                if sequence.ndim == 4 and sequence.shape[-1] == 4:
                    sequence = sequence[..., :3]
                camera_sequences.append(sequence)
        camera_views = np.concatenate(camera_sequences, axis=2) if camera_sequences else None

        tactile_rgb_values = ep_buffer.get("tactile_rgb_image", [])
        tactile_rgb_seq = np.asarray(tactile_rgb_values) if tactile_rgb_values else None
        normal_values = ep_buffer.get("tactile_normal_force", [])
        shear_values = ep_buffer.get("tactile_shear_force", [])
        tactile_force_seq = None
        if normal_values and shear_values:
            tactile_force_seq = (np.asarray(normal_values), np.asarray(shear_values))

        seq_lens = []
        if camera_views is not None:
            seq_lens.append(camera_views.shape[0])
        if tactile_rgb_seq is not None:
            seq_lens.append(tactile_rgb_seq.shape[0])
        if tactile_force_seq is not None:
            seq_lens.append(tactile_force_seq[0].shape[0])
        if not seq_lens:
            print(f"[WARNING] Episode {idx} has no camera or tactile frames; MP4 was not written.")
            continue

        video_path = os.path.join(save_dir, f"episode_{idx}.mp4")
        video_writer = imageio.get_writer(video_path, fps=20)
        num_frames = min(seq_lens)
        try:
            for frame_index in range(num_frames):
                rows = []
                if camera_views is not None:
                    rows.append(camera_views[frame_index])
                if tactile_rgb_seq is not None:
                    rows.append(concat_tactile_rgb_image(tactile_rgb_seq[frame_index]))
                if tactile_force_seq is not None:
                    force_img = frame_tactile_force_field(
                        tactile_force_seq[0][frame_index], tactile_force_seq[1][frame_index]
                    )
                    rows.append(
                        cv2.resize(force_img, dsize=None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
                    )
                video_writer.append_data(_stack_rows_with_padding(rows))
        finally:
            video_writer.close()
        print(f"[INFO] Saved validation video: {video_path}")

    with open(os.path.join(save_dir, "all_success.txt"), "a", encoding="utf-8") as stream:
        for success in success_values:
            stream.write(f"{bool(success)}\n")
        

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]

    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be at least 1.")
    if args_cli.num_episodes < 1:
        raise ValueError("--num-episodes must be at least 1.")
    if args_cli.max_steps < 0:
        raise ValueError("--max_steps must be >= 0.")
    if args_cli.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0.")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    seed_suffix = f"_seed{int(agent_cfg.seed)}" if agent_cfg.seed is not None else ""
    save_dir = os.path.join(
        "data",
        "validation",
        task_name,
        args_cli.policy_name + "_" + args_cli.observation_profile
        if args_cli.observation_profile
        else args_cli.policy_name,
        f"{args_cli.data_num}_{args_cli.checkpoint_num}{seed_suffix}",
    )
    start_episode_index = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    else:
        saved_episodes = [int(file.split("_")[-1].split(".")[0]) for file in os.listdir(save_dir) if file.endswith(".mp4")]
        if len(saved_episodes) == 0:
            start_episode_index = 0
        else:
            start_episode_index = max(saved_episodes) + 1
    print(f"[INFO] Start episode index: {start_episode_index}")
    
    # ForgeEnv (and similar) gate tactile + third_person_camera on cfg.enable_cameras, not only AppLauncher.
    # save_data reads obs["record"] which requires those sensors — mirror train.py injection.
    _enable_cams = bool(getattr(args_cli, "enable_cameras", False)) or bool(
        int(os.environ.get("ENABLE_CAMERAS", "0"))
    )
    if getattr(args_cli, "save_data", False):
        _enable_cams = True
    setattr(env_cfg, "enable_cameras", _enable_cams)

    # Preserve each canonical task timeout unless the operator explicitly overrides it.
    if args_cli.env_max_steps is not None:
        if args_cli.env_max_steps < 1:
            raise ValueError("--env-max-steps must be at least 1 when supplied.")
        if hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg, "decimation"):
            step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
            env_cfg.episode_length_s = float(args_cli.env_max_steps) * step_dt
        elif hasattr(env_cfg, "max_episode_length"):
            env_cfg.max_episode_length = int(args_cli.env_max_steps)
        else:
            raise AttributeError("This environment config exposes no supported episode-length field.")

    # Keep one canonical Vision task registration. Evaluation loads the CNN
    # checkpoint instead of relying on a duplicate ``*-Play-v0`` task.
    feature_extractor_cfg = getattr(env_cfg, "feature_extractor", None)
    if feature_extractor_cfg is not None:
        if hasattr(feature_extractor_cfg, "train"):
            feature_extractor_cfg.train = False
        if hasattr(feature_extractor_cfg, "load_checkpoint"):
            feature_extractor_cfg.load_checkpoint = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    base_env = env.unwrapped
    task_env = _resolve_task_env(base_env)
    task_env._use_rl_control = False
    robot_art = _resolve_primary_robot(base_env)
    if robot_art is None:
        raise AttributeError(
            "Could not resolve a robot Articulation from the env "
            "(tried base_env.robot / right_hand / left_hand / scene.articulations)."
        )
    joint_names = list(getattr(robot_art, "joint_names", []))
    arm_joint_keywords = ("shoulder", "elbow", "wrist", "ur10")
    finger_joint_indices = [
        i for i, name in enumerate(joint_names) if not any(k in name.lower() for k in arm_joint_keywords)
    ]
    if len(finger_joint_indices) == 0:
        finger_joint_indices = list(range(len(joint_names)))
    print(f"[INFO] Finger joint indices: {finger_joint_indices}")
    if len(joint_names) > 0:
        print(f"[INFO] Joint names: {joint_names}")
    # base_env.set_stiffness_damping(True)
    dt = base_env.step_dt
    # Explicitly reset once before collecting any frame.
    # For some env/wrapper stacks, get_observations() can return a pre-step transient state.
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    obs = _ensure_record_in_obs(base_env, obs)
    # Align joint position target with current state to avoid first-frame jerk.
    if (
        hasattr(task_env, "apply_joint_targets")
        and _obs_has_key(obs, "record")
        and "joint_pos" in obs["record"]
    ):
        task_env.apply_joint_targets(obs["record"]["joint_pos"].to(device=task_env.device))

    # 多环境：从 obs 或 env 获取 num_envs
    obs_policy = obs.get("policy", obs) if isinstance(obs, dict) else obs
    num_envs = obs_policy.shape[0] if hasattr(obs_policy, "shape") else getattr(base_env, "num_envs", 1)
    device = base_env.device

    # 模仿学习：DP 输出关节角，直接下发给 robot；用哑 action 推 env.step
    env_action_dim = _infer_env_action_dim(base_env, env)
    dummy_actions = torch.zeros(num_envs, env_action_dim, device=device, dtype=torch.float32)
    # Warm up a few physics steps so the first recorded frame is after reset initialization settles.
    for _ in range(args_cli.warmup_steps):
        obs, _, _, _ = env.step(dummy_actions)
        obs = _ensure_record_in_obs(base_env, obs)

    import importlib

    if args_cli.checkpoint_task_name:
        policy_task_name = str(args_cli.checkpoint_task_name).strip()
    elif args_cli.observation_profile:
        policy_task_name = task_name + "_" + args_cli.observation_profile
    else:
        policy_task_name = task_name

    Policy_Encapsulation = getattr(importlib.import_module(f"policy.{args_cli.policy_name}.deploy_policy"), "Encapsulation")
    init_dict = {
        "task_name": policy_task_name,
        "data_num": args_cli.data_num,
        "checkpoint_num": args_cli.checkpoint_num,
        "checkpoint_path": str(Path(args_cli.policy_checkpoint).expanduser().resolve())
        if args_cli.policy_checkpoint
        else None,
        "device": args_cli.device or str(device),
    }
    policy = Policy_Encapsulation(init_dict, num_envs)
    
    current_ep_buffers = [dict() for _ in range(num_envs)]

    # 每个环境独立的 episode 缓冲；所有 env 共享同一个 global episode 步数
    step_in_episode = 0  # global episode step counter (since last env.reset())
    
    success_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
    max_steps_per_episode = args_cli.max_steps

    success_count = 0
    episode_count = start_episode_index
    completed_rollouts = 0
    policy_control_mode: str | None = None  # "joint_pos" | "action"
    joint_err_hist_all: list[torch.Tensor] = []
    finger_err_hist_all: list[torch.Tensor] = []
    target_jitter_hist_all: list[torch.Tensor] = []
    finger_target_jitter_hist_all: list[torch.Tensor] = []
    tracking_jitter_hist_all: list[torch.Tensor] = []
    finger_tracking_jitter_hist_all: list[torch.Tensor] = []
    prev_joint_pos_cmd: torch.Tensor | None = None
    prev_actual_joint: torch.Tensor | None = None
    obs_debug_printed = False
    
    for i in range(num_envs):
        _append_env_frame_to_buffer(
            current_ep_buffers[i],
            obs["record"][i],
        )

    while True:
        if not simulation_app.is_running():
            break

        start_time = time.time()
        
        with torch.inference_mode():
            policy_input = _apply_farthest_point_sample(obs["record"])
            if args_cli.debug_observations and not obs_debug_printed:
                print("[ObsDebug] policy input summary (first iteration)")
                _debug_print_obs_stats(policy_input, max_items=12)
                obs_debug_printed = True
            bad_paths = _find_non_finite_paths(policy_input)
            if bad_paths:
                preview = ", ".join(bad_paths[:12])
                if len(bad_paths) > 12:
                    preview += f", ... (+{len(bad_paths) - 12} more)"
                raise RuntimeError(
                    "[NaNCheck] Non-finite values found in policy input obs['record'] before policy.get_action: "
                    f"{preview}"
                )
            policy_np = np.asarray(policy.get_action(policy_input), dtype=np.float32)
            if not np.isfinite(policy_np).all():
                nan_idx = np.argwhere(~np.isfinite(policy_np))
                first = tuple(int(v) for v in nan_idx[0].tolist()) if nan_idx.size > 0 else None
                extra = f" first_bad_index={first}" if first is not None else ""
                raise RuntimeError(
                    "[NaNCheck] Non-finite values found in policy output right after policy.get_action."
                    f"{extra}, shape={policy_np.shape}"
                )
            policy_out = torch.as_tensor(
                policy_np,
                device=device,
                dtype=robot_art.data.joint_pos.dtype,
            )
            if policy_out.ndim == 1:
                policy_out = policy_out.unsqueeze(0)
            out_dim = int(policy_out.shape[-1])

            if policy_control_mode is None:
                requested_mode = str(args_cli.policy_output).strip().lower()
                joint_dim = int(obs["record"]["joint_pos"].shape[-1])
                if requested_mode in ("joint_pos", "action"):
                    policy_control_mode = requested_mode
                else:
                    if out_dim == joint_dim:
                        policy_control_mode = "joint_pos"
                    elif out_dim == env_action_dim:
                        policy_control_mode = "action"
                    else:
                        raise RuntimeError(
                            f"Cannot infer policy output semantics: output_dim={out_dim}, "
                            f"joint_dim={joint_dim}, action_dim={env_action_dim}. "
                            f"Please set --policy-output joint_pos|action explicitly."
                        )
                task_env._use_rl_control = policy_control_mode == "action"
                print(
                    f"[INFO] Policy output mode locked: {policy_control_mode} "
                    f"(output_dim={out_dim}, action_dim={env_action_dim})"
                )

            if policy_control_mode == "joint_pos":
                if not hasattr(task_env, "apply_joint_targets"):
                    raise AttributeError(
                        f"Task {type(task_env).__name__} does not implement apply_joint_targets(), "
                        "so it cannot execute a joint-position policy. Use --policy-output action or adapt the task."
                    )
                joint_pos_cmd = policy_out
                if int(joint_pos_cmd.shape[-1]) != int(obs["record"]["joint_pos"].shape[-1]):
                    raise RuntimeError(
                        f"Joint-position output dim {joint_pos_cmd.shape[-1]} does not match record joint_pos dim "
                        f"{obs['record']['joint_pos'].shape[-1]}."
                    )
                task_env.apply_joint_targets(joint_pos_cmd)
                obs, rewards, dones, infos = env.step(dummy_actions)
                obs = _ensure_record_in_obs(base_env, obs)
                current_joint = obs["record"]["joint_pos"].to(device=device, dtype=joint_pos_cmd.dtype)
                joint_abs_error = torch.abs(current_joint - joint_pos_cmd)
                finger_abs_error = joint_abs_error[:, finger_joint_indices]
                joint_err_hist_all.append(joint_abs_error.detach().cpu())
                finger_err_hist_all.append(finger_abs_error.detach().cpu())
                if prev_joint_pos_cmd is None:
                    target_jitter = torch.zeros_like(joint_pos_cmd)
                else:
                    target_jitter = torch.abs(joint_pos_cmd - prev_joint_pos_cmd)
                if prev_actual_joint is None:
                    tracking_jitter = torch.zeros_like(current_joint)
                else:
                    tracking_jitter = torch.abs(current_joint - prev_actual_joint)
                finger_target_jitter = target_jitter[:, finger_joint_indices]
                finger_tracking_jitter = tracking_jitter[:, finger_joint_indices]
                target_jitter_hist_all.append(target_jitter.detach().cpu())
                finger_target_jitter_hist_all.append(finger_target_jitter.detach().cpu())
                tracking_jitter_hist_all.append(tracking_jitter.detach().cpu())
                finger_tracking_jitter_hist_all.append(finger_tracking_jitter.detach().cpu())
                prev_joint_pos_cmd = joint_pos_cmd.detach().clone()
                prev_actual_joint = current_joint.detach().clone()
            else:
                # state->action policy: action is normalized env action in [-1, 1].
                action_cmd = policy_out.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)
                if action_cmd.shape[0] == 1 and num_envs > 1:
                    action_cmd = action_cmd.expand(num_envs, -1)
                if int(action_cmd.shape[-1]) != env_action_dim:
                    raise RuntimeError(
                        f"Action dim mismatch: got {int(action_cmd.shape[-1])}, expected {env_action_dim}"
                    )
                obs, rewards, dones, infos = env.step(action_cmd)
                obs = _ensure_record_in_obs(base_env, obs)
            next_step_in_episode = step_in_episode + 1
            if args_cli.joint_err_print_interval > 0 and (next_step_in_episode % int(args_cli.joint_err_print_interval) == 0):
                if policy_control_mode == "joint_pos":
                    print(
                        "[JOINT_ERR] "
                        f"step={next_step_in_episode:04d} "
                        f"all_mean={joint_abs_error.mean().item():.6f} "
                        f"all_max={joint_abs_error.max().item():.6f} "
                        f"finger_mean={finger_abs_error.mean().item():.6f} "
                        f"finger_max={finger_abs_error.max().item():.6f}"
                    )
                    print(
                        "[TARGET_JITTER] "
                        f"step={next_step_in_episode:04d} "
                        f"all_mean={target_jitter.mean().item():.6f} "
                        f"all_max={target_jitter.max().item():.6f} "
                        f"finger_mean={finger_target_jitter.mean().item():.6f} "
                        f"finger_max={finger_target_jitter.max().item():.6f}"
                    )
                    print(
                        "[TRACKING_JITTER] "
                        f"step={next_step_in_episode:04d} "
                        f"all_mean={tracking_jitter.mean().item():.6f} "
                        f"all_max={tracking_jitter.max().item():.6f} "
                        f"finger_mean={finger_tracking_jitter.mean().item():.6f} "
                        f"finger_max={finger_tracking_jitter.max().item():.6f}"
                    )
                    topk_n = max(1, min(int(args_cli.jitter_topk), len(finger_joint_indices)))
                    finger_target_env0 = finger_target_jitter[0]
                    finger_tracking_env0 = finger_tracking_jitter[0]
                    target_vals, target_idx_local = torch.topk(finger_target_env0, k=topk_n)
                    tracking_vals, tracking_idx_local = torch.topk(finger_tracking_env0, k=topk_n)
                    target_topk_msg = ", ".join(
                        [
                            f"{joint_names[finger_joint_indices[int(local_idx)]]}:{float(val):.5f}"
                            for val, local_idx in zip(target_vals.detach().cpu().tolist(), target_idx_local.detach().cpu().tolist())
                        ]
                    )
                    tracking_topk_msg = ", ".join(
                        [
                            f"{joint_names[finger_joint_indices[int(local_idx)]]}:{float(val):.5f}"
                            for val, local_idx in zip(
                                tracking_vals.detach().cpu().tolist(), tracking_idx_local.detach().cpu().tolist()
                            )
                        ]
                    )
                    print(f"[TARGET_JITTER_TOPK] step={next_step_in_episode:04d} env0_fingers={target_topk_msg}")
                    print(f"[TRACKING_JITTER_TOPK] step={next_step_in_episode:04d} env0_fingers={tracking_topk_msg}")
                else:
                    print(
                        "[ACTION_STATS] "
                        f"step={next_step_in_episode:04d} "
                        f"mean={action_cmd.mean().item():.6f} "
                        f"max={action_cmd.max().item():.6f} "
                        f"min={action_cmd.min().item():.6f}"
                    )
            # 多环境：dones 展平为 (num_envs,)
            dones_flat = dones.flatten() if isinstance(dones, torch.Tensor) else torch.tensor(dones, device=device).flatten()
            if dones_flat.numel() != num_envs:
                dones_flat = dones_flat.expand(num_envs)

            step_in_episode += 1
            
            for i in range(num_envs):
                _append_env_frame_to_buffer(
                    current_ep_buffers[i],
                    obs["record"][i],
                )

            # 1) 成功：立即保存当前 buffer，但不 reset 环境；一个 global episode 中每个 env 只保存一次
            success_per_env = _extract_success_per_env(infos, obs, base_env, num_envs, device)
            for i in range(num_envs):
                if success_per_env[i].item() and not success_episode[i]:
                    # 任务成功
                    print(f"[INFO] Env {i} success at global step {step_in_episode}")
                    success_count += 1
                    success_episode[i] = True
                
            # 2) global max_steps：此时才调用 env.reset()，并整体清空缓冲
            reached_step_limit = bool(max_steps_per_episode) and step_in_episode >= max_steps_per_episode
            all_done = bool(torch.all(dones_flat).item())
            if reached_step_limit or all_done:
                print(
                    f"[INFO] Rollout {completed_rollouts + 1}/{args_cli.num_episodes} ended "
                    f"at step {step_in_episode}; success={success_episode.detach().cpu().tolist()}"
                )
                save_result(save_dir, episode_count, current_ep_buffers, success_episode)
                completed_rollouts += 1
                episode_count += num_envs
                if completed_rollouts >= args_cli.num_episodes:
                    break

                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                obs = _ensure_record_in_obs(base_env, obs)
                for _ in range(args_cli.warmup_steps):
                    obs, _, _, _ = env.step(dummy_actions)
                    obs = _ensure_record_in_obs(base_env, obs)
                policy.reset()
                current_ep_buffers = [dict() for _ in range(num_envs)]
                for env_index in range(num_envs):
                    _append_env_frame_to_buffer(current_ep_buffers[env_index], obs["record"][env_index])
                step_in_episode = 0
                success_episode.zero_()
                policy_control_mode = None
                prev_joint_pos_cmd = None
                prev_actual_joint = None
                obs_debug_printed = False

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    if len(joint_err_hist_all) > 0:
        joint_err_all = torch.cat(joint_err_hist_all, dim=0)
        finger_err_all = torch.cat(finger_err_hist_all, dim=0)
        print(
            "[JOINT_ERR_SUMMARY] "
            f"all_mean={joint_err_all.mean().item():.6f} "
            f"all_max={joint_err_all.max().item():.6f} "
            f"all_p95={torch.quantile(joint_err_all.flatten(), 0.95).item():.6f} "
            f"finger_mean={finger_err_all.mean().item():.6f} "
            f"finger_max={finger_err_all.max().item():.6f} "
            f"finger_p95={torch.quantile(finger_err_all.flatten(), 0.95).item():.6f}"
        )
    if len(target_jitter_hist_all) > 0:
        target_jitter_all = torch.cat(target_jitter_hist_all, dim=0)
        finger_target_jitter_all = torch.cat(finger_target_jitter_hist_all, dim=0)
        tracking_jitter_all = torch.cat(tracking_jitter_hist_all, dim=0)
        finger_tracking_jitter_all = torch.cat(finger_tracking_jitter_hist_all, dim=0)
        print(
            "[TARGET_JITTER_SUMMARY] "
            f"all_mean={target_jitter_all.mean().item():.6f} "
            f"all_max={target_jitter_all.max().item():.6f} "
            f"all_p95={torch.quantile(target_jitter_all.flatten(), 0.95).item():.6f} "
            f"finger_mean={finger_target_jitter_all.mean().item():.6f} "
            f"finger_max={finger_target_jitter_all.max().item():.6f} "
            f"finger_p95={torch.quantile(finger_target_jitter_all.flatten(), 0.95).item():.6f}"
        )
        print(
            "[TRACKING_JITTER_SUMMARY] "
            f"all_mean={tracking_jitter_all.mean().item():.6f} "
            f"all_max={tracking_jitter_all.max().item():.6f} "
            f"all_p95={torch.quantile(tracking_jitter_all.flatten(), 0.95).item():.6f} "
            f"finger_mean={finger_tracking_jitter_all.mean().item():.6f} "
            f"finger_max={finger_tracking_jitter_all.max().item():.6f} "
            f"finger_p95={torch.quantile(finger_tracking_jitter_all.flatten(), 0.95).item():.6f}"
        )
        topk_n = max(1, min(int(args_cli.jitter_topk), len(finger_joint_indices)))
        target_finger_p95 = torch.quantile(finger_target_jitter_all, 0.95, dim=0)
        tracking_finger_p95 = torch.quantile(finger_tracking_jitter_all, 0.95, dim=0)
        target_vals, target_idx_local = torch.topk(target_finger_p95, k=topk_n)
        tracking_vals, tracking_idx_local = torch.topk(tracking_finger_p95, k=topk_n)
        target_summary_topk = ", ".join(
            [
                f"{joint_names[finger_joint_indices[int(local_idx)]]}:{float(val):.5f}"
                for val, local_idx in zip(target_vals.detach().cpu().tolist(), target_idx_local.detach().cpu().tolist())
            ]
        )
        tracking_summary_topk = ", ".join(
            [
                f"{joint_names[finger_joint_indices[int(local_idx)]]}:{float(val):.5f}"
                for val, local_idx in zip(
                    tracking_vals.detach().cpu().tolist(), tracking_idx_local.detach().cpu().tolist()
                )
            ]
        )
        print(f"[TARGET_JITTER_TOPK_SUMMARY] finger_p95={target_summary_topk}")
        print(f"[TRACKING_JITTER_TOPK_SUMMARY] finger_p95={tracking_summary_topk}")
    
    print(f"[INFO] Success count: {success_count}")
    print(f"[INFO] Episode count: {episode_count}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
