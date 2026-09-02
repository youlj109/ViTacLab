#!/usr/bin/env python3
"""Record and inspect raw observations from canonical single-arm UR10e+ShadowHand tasks.

The entry supports zero/random actions, an RSL-RL checkpoint, tactile viewers,
and compressed per-step NPZ output while switching tasks through presets or
explicit environment/config entry strings.

Examples (inside Isaac Sim python):

    # Pour task
    ./python.sh scripts/data_collection/manual/record_observations.py --task pour --num_envs 1 --show_rgb --enable_cameras

    # Pickup task
    ./python.sh scripts/data_collection/manual/record_observations.py --task pickup --num_envs 1 --show_rgb --show_ff --random_actions --enable_cameras

    # In-hand cube reorientation (UR10e + ShadowHand, tactile)
    ./python.sh scripts/data_collection/manual/record_observations.py --task inhand --num_envs 1  --show_rgb --show_ff --random_actions --enable_cameras

    # Play trained policy (RSL-RL .pt), same enable_cameras as train/play
    ./python.sh scripts/data_collection/manual/record_observations.py --task inhand --num_envs 4096 --play \\
        --resume_path logs/rsl_rl/shadow_hand_tactile/2026-03-21_00-01-44/model_1000.pt --enable_cameras

    # Fully custom (module:Class)
    ./python.sh scripts/data_collection/manual/record_observations.py \\
        --env ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv \\
        --cfg ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg \\
        --num_envs 1 --show_rgb --enable_cameras
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher


TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)


_TASK_PRESETS: dict[str, dict[str, str]] = {
    "pour": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
    },
    "pickup": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg",
    },
    "inhand": {
        "env": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandTactileEnvCfg",
    },
}

# Gym-registered ids for loading `rsl_rl_cfg_entry_point` (must match training task / network layout).
_TASK_PRESET_GYM_TASK: dict[str, str] = {
    "pour": "Isaac-UR10eShadowHand-PourDeformable-Direct-v0",
    "pickup": "Isaac-UR10eShadowHand-Pickup-Direct-v0",
    "inhand": "Isaac-UR10eShadowHand-Repose-Cube-Tactile-Direct-v0",
}


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
    """Render tactile normal/shear arrays into an RGB image — same as show_ur10e_shadowhand_records.py.

    Args:
        nf: (H, W) normal force
        sf: (H, W, 2) shear force (x, y)
    Returns:
        (H, W, 3) uint8 image in [0, 255]
    """
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


def _load_symbol(entry: str) -> Any:
    """Load `module.path:SymbolName`."""
    if ":" not in entry:
        raise ValueError(f"Invalid entry '{entry}'. Expected 'module.path:SymbolName'.")
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _flatten_for_npz(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested record into a 1-level dict for npz saving."""
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            out.update(_flatten_for_npz(v, key))
        return out
    if torch.is_tensor(obj):
        out[prefix] = obj.detach().cpu().numpy()
        return out
    if isinstance(obj, np.ndarray):
        out[prefix] = obj
        return out
    if isinstance(obj, (int, float, bool, str)):
        out[prefix] = np.asarray(obj)
        return out
    # skip unsupported types (e.g., None)
    return out


def _select_record_environment(record: Any, env_index: int, num_envs: int) -> dict[str, Any] | None:
    """Select one environment from a canonical batched record dictionary.

    Canonical ViTacLab environments expose record tensors with the environment
    dimension first.  Keep that dimension (shape ``1, ...``) so files written
    by this per-step recorder remain compatible with the former
    ``_get_record(env_ids=[index])`` result.
    """

    if not isinstance(record, dict):
        return None

    selected: dict[str, Any] = {}
    index = max(0, min(int(env_index), max(0, int(num_envs) - 1)))

    for key, value in record.items():
        if isinstance(value, dict):
            nested = _select_record_environment(value, index, num_envs)
            if nested is not None:
                selected[key] = nested
            continue

        if torch.is_tensor(value):
            if value.ndim > 0 and int(value.shape[0]) == int(num_envs):
                selected[key] = value[index : index + 1].detach().cpu()
            else:
                selected[key] = value.detach().cpu()
            continue

        if isinstance(value, np.ndarray):
            if value.ndim > 0 and int(value.shape[0]) == int(num_envs):
                selected[key] = value[index : index + 1]
            else:
                selected[key] = value
            continue

        if isinstance(value, (int, float, bool, str)):
            selected[key] = value

    return selected or None


def _extract_record_snapshot(obs: Any, env: Any, env_index: int, num_envs: int) -> tuple[dict[str, Any], str]:
    """Return one canonical record snapshot and the interface used.

    Interface priority:

    1. the maintained observation contract, ``obs['record']``;
    2. the legacy task method, ``_get_record(env_ids=...)``;
    3. the maintained robot/task fallback builder, ``_build_record_dict()``.

    The final fallback is required by tasks such as Pour whose policy
    observation intentionally stays compact while the robot base still owns
    the complete tactile/camera record implementation.
    """

    if isinstance(obs, dict):
        selected = _select_record_environment(obs.get("record"), env_index, num_envs)
        if selected is not None:
            return selected, "obs.record"

    legacy_get_record = getattr(env, "_get_record", None)
    if callable(legacy_get_record):
        try:
            record = legacy_get_record(env_ids=[int(env_index)])
        except TypeError:
            record = legacy_get_record()
        selected = _select_record_environment(record, env_index, num_envs)
        if selected is not None:
            return selected, "env._get_record"

    build_record = getattr(env, "_build_record_dict", None)
    if callable(build_record):
        selected = _select_record_environment(build_record(), env_index, num_envs)
        if selected is not None:
            return selected, "env._build_record_dict"

    raise AttributeError(
        f"{type(env).__name__} exposes neither obs['record'], _get_record(), nor _build_record_dict()."
    )


def _resolve_record_paths(record_path: str, fmt: str) -> tuple[str, str]:
    """Return (output_dir, file_prefix)."""
    rp = os.path.expanduser(record_path)
    # If a directory is provided, keep it; else treat as prefix under its parent.
    if rp.endswith(os.sep) or os.path.isdir(rp):
        out_dir = rp.rstrip(os.sep)
        prefix = "record"
    else:
        out_dir = os.path.dirname(rp) or "."
        base = os.path.basename(rp)
        if base.endswith(f".{fmt}"):
            base = base[: -(len(fmt) + 1)]
        prefix = base or "record"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, prefix


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UR10e+ShadowHand single env and show tactile images.")
    parser.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="pour", help="Preset task.")
    parser.add_argument("--env", type=str, default="", help="Env entry: module:Class (overrides --task).")
    parser.add_argument("--cfg", type=str, default="", help="Cfg entry: module:Class (overrides --task).")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Use trained RSL-RL policy from --resume_path instead of random/zero actions.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="",
        help="Path to model_*.pt checkpoint (requires --play). Same format as scripts/rsl_rl/full_rl/play.py --checkpoint.",
    )
    parser.add_argument(
        "--gym_task",
        type=str,
        default="",
        help="Registered Gym task id for RSL-RL agent config (default: preset mapping). Required if --env/--cfg override breaks the preset.",
    )
    parser.add_argument(
        "--play_success_interval",
        type=float,
        default=2.0,
        help="When --play: print mean success rate over all envs every N seconds (in-hand task only; default: 2).",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    parser.add_argument("--env_index", type=int, default=0, help="Env index to visualize (default: 0).")
    parser.add_argument("--fps", type=float, default=20.0, help="Target display FPS (default: 20).")
    parser.add_argument("--max_steps", type=int, default=0, help="If >0, stop after N steps.")
    parser.add_argument("--random_actions", action="store_true", help="Apply random actions instead of zeros.")
    parser.add_argument("--show_rgb", action="store_true", help="Show tactile RGB images.")
    parser.add_argument("--show_ff", action="store_true", help="Show tactile force-field images (if enabled).")
    parser.add_argument(
        "--record_path",
        type=str,
        default="",
        help=(
            "If set, save the canonical record snapshot every step to this directory or file prefix. "
            "The source is obs['record'] with legacy/base-environment fallbacks."
        ),
    )
    parser.add_argument("--record_format", choices=["pt", "npz"], default="pt", help="Record file format.")
    parser.add_argument("--record_every", type=int, default=1, help="Record every N steps (default: 1).")
    parser.add_argument("--record_env_index", type=int, default=-1, help="Env index to record (default: env_index).")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.play and not str(args.resume_path).strip():
        raise SystemExit("--play requires --resume_path to a trained RSL-RL checkpoint (.pt).")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import tasks after app launch (gym registrations, etc.)
    import ViTacLab.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401  # registry side effects

    preset = _TASK_PRESETS[str(args.task)]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]

    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    # Matplotlib setup
    fig = None
    ax_rgb = None
    ax_ff = None
    rgb_ims: list = []
    ff_ims: list = []
    nrows, ncols = 20, 25

    if args.show_rgb or args.show_ff:
        import matplotlib

        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        if args.show_rgb and args.show_ff:
            fig, axes = plt.subplots(2, 5, figsize=(16, 6))
            ax_rgb = axes[0]
            ax_ff = axes[1]
        elif args.show_rgb:
            fig, ax_rgb = plt.subplots(1, 5, figsize=(16, 3))
        elif args.show_ff:
            fig, ax_ff = plt.subplots(1, 5, figsize=(16, 3))

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"

    # Match scripts/rsl_rl/full_rl/train.py / play.py: gate cameras / tactile sensors.
    _enable_cams = bool(getattr(args, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    setattr(cfg, "enable_cameras", _enable_cams)
    print(f"[INFO] cfg.enable_cameras={getattr(cfg, 'enable_cameras', None)} (CLI or ENABLE_CAMERAS)")

    print(f"Creating {EnvCls.__name__} (device={cfg.device}, num_envs={cfg.scene.num_envs}) ...")
    env = EnvCls(cfg)
    action_dim = env.num_actions
    print(f"Action dim: {action_dim}, Obs dim: {cfg.observation_space}")

    # RSL-RL policy + wrapper (play mode)
    policy = None
    policy_nn = None
    wrapped_env = env
    if args.play:
        from isaaclab.utils.assets import retrieve_file_path
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        gym_task = str(args.gym_task).strip() or _TASK_PRESET_GYM_TASK.get(str(args.task), "")
        if not gym_task:
            raise SystemExit(
                "Could not resolve --gym_task for RSL-RL agent config. "
                "Pass --gym_task explicitly (registered Gym id for this env/cfg)."
            )
        agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(gym_task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = str(cfg.device)

        resume_path = retrieve_file_path(str(args.resume_path).strip())
        print(f"[INFO] Play mode: gym_task={gym_task}, checkpoint={resume_path}")

        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        obs = wrapped_env.get_observations()
        policy_obs = obs.get("policy") if hasattr(obs, "get") else None
        if policy_obs is not None:
            print(f"Play ready. policy obs shape: {tuple(policy_obs.shape)}")
    else:
        obs, _ = env.reset()
        policy_obs = obs.get("policy")
        if policy_obs is not None:
            print(f"Reset ok. policy obs shape: {tuple(policy_obs.shape)}")

    # Initialize tactile sensors' nominal render (for camera tactile)
    _scene_env = wrapped_env.unwrapped
    for name in TACTILE_SENSOR_NAMES:
        if name in _scene_env.scene.sensors:
            try:
                _scene_env.scene[name].get_initial_render()
            except Exception:
                pass

    # Determine tactile array size for FF from sensor cfg
    if args.show_ff and fig is not None:
        for name in TACTILE_SENSOR_NAMES:
            if name in _scene_env.scene.sensors:
                try:
                    nrows, ncols = _scene_env.scene[name].cfg.tactile_array_size
                except Exception:
                    pass
                break

    # Create plot artists
    if fig is not None:
        import matplotlib.pyplot as plt

        if args.show_rgb and ax_rgb is not None:
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

        if args.show_ff and ax_ff is not None:
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

    # FF: same renderer as scripts/debug/show_ur10e_shadowhand_records.py (no isaaclab_contrib dependency)
    render_ff = _render_tactile_ff_rgb if args.show_ff else None

    target_dt = 1.0 / max(1e-3, float(args.fps))
    env_idx = max(0, min(int(args.env_index), wrapped_env.num_envs - 1))
    rec_env_idx = env_idx if int(args.record_env_index) < 0 else max(0, min(int(args.record_env_index), wrapped_env.num_envs - 1))
    step = 0
    do_record = bool(str(args.record_path).strip())
    record_dir = ""
    record_prefix = ""
    if do_record:
        record_dir, record_prefix = _resolve_record_paths(str(args.record_path).strip(), str(args.record_format))
        # Validate the current canonical record contract before entering the
        # loop.  Do not require the removed legacy ``_get_record`` method:
        # maintained environments normally expose ``obs['record']`` or the
        # shared ``_build_record_dict`` fallback.
        _, initial_record_source = _extract_record_snapshot(
            obs,
            wrapped_env.unwrapped,
            rec_env_idx,
            wrapped_env.num_envs,
        )
        print(f"Recording enabled: dir='{record_dir}', prefix='{record_prefix}', fmt={args.record_format}, every={args.record_every}, env={rec_env_idx}")
        print(f"Recording source: {initial_record_source}")

    print("Environment created. Starting viewer (Ctrl+C to stop).")
    last_play_success_print_ts = time.time() if args.play else 0.0
    play_sr_interval = max(0.1, float(args.play_success_interval))

    while simulation_app.is_running():
        t0 = time.time()
        step += 1

        if args.play:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = wrapped_env.step(actions)
                if policy_nn is not None:
                    policy_nn.reset(dones)
            now = time.time()
            if now - last_play_success_print_ts >= play_sr_interval:
                ue = wrapped_env.unwrapped
                if hasattr(ue, "get_episode_success_stats"):
                    n_ok, n_ep, rate = ue.get_episode_success_stats()
                    print(
                        f"[play] episode success rate: {rate:.4f} ({rate * 100:.2f}%)  "
                        f"({n_ok}/{n_ep} episodes)  n_envs={wrapped_env.num_envs}  step={step}"
                    )
                last_play_success_print_ts = now
        elif args.random_actions:
            actions = 0.3 * (2.0 * torch.rand(wrapped_env.num_envs, action_dim, device=wrapped_env.device) - 1.0)
            step_result = wrapped_env.step(actions)
            obs = step_result[0]
        else:
            actions = torch.zeros(wrapped_env.num_envs, action_dim, device=wrapped_env.device)
            step_result = wrapped_env.step(actions)
            obs = step_result[0]

        # Record
        if do_record and int(args.record_every) > 0 and (step % int(args.record_every) == 0):
            rec, record_source = _extract_record_snapshot(
                obs,
                wrapped_env.unwrapped,
                rec_env_idx,
                wrapped_env.num_envs,
            )
            fname = os.path.join(record_dir, f"{record_prefix}_step_{step:06d}.{args.record_format}")
            if str(args.record_format) == "pt":
                torch.save(rec, fname)
            else:
                flat = _flatten_for_npz(rec)
                if not flat:
                    raise RuntimeError(
                        f"Record source {record_source!r} produced no NPZ-compatible fields at step {step}."
                    )
                np.savez_compressed(fname, **flat)

        # Update plots
        if fig is not None and (rgb_ims or ff_ims):
            import matplotlib.pyplot as plt

            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if name not in _scene_env.scene.sensors:
                    continue
                data = _scene_env.scene[name].data

                # RGB
                if args.show_rgb and rgb_ims and i < len(rgb_ims):
                    img = getattr(data, "tactile_rgb_image", None)
                    if img is not None and img.ndim == 4:
                        e = min(env_idx, img.shape[0] - 1)
                        rgb_ims[i].set_data(_img_to_uint8(img[e].detach().cpu().numpy()))

                # FF (point-array → RGB, aligned with show_ur10e_shadowhand_records.py)
                if args.show_ff and ff_ims and i < len(ff_ims) and render_ff is not None:
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

        if args.max_steps > 0 and step >= int(args.max_steps):
            break

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    wrapped_env.close()
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    simulation_app.close()


if __name__ == "__main__":
    main()

