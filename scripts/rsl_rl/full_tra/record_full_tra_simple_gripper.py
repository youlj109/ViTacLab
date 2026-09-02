#!/usr/bin/env python3
"""Record open-loop Factory FORGE action trajectories for ``simple_gripper`` (Franka).

Drives ``ForgeEnv`` with 7-D actions in [-1, 1] via GUI sliders and saves trajectories for
``play_full_tra_simple_gripper.py``:

    [pos_x, pos_y, pos_z, roll, pitch, yaw, success_pred]

Examples (Isaac Sim python)::

    python scripts/rsl_rl/full_tra/record_full_tra_simple_gripper.py \\
        --task simple_forge_peg --num_envs 1 --enable_cameras

    python scripts/rsl_rl/full_tra/record_full_tra_simple_gripper.py \\
        --task Isaac-Forge-NutThread-Direct-v0 --num_envs 1 --enable_cameras --show_rgb
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from isaaclab.app import AppLauncher

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from full_tra_high_fidelity import add_high_fidelity_cli_args, apply_high_fidelity_cfg
from full_tra_task_entries import resolve_env_cfg_entries

ACTION_DIM = 7
ACTION_LABELS = ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw", "success_pred")
TACTILE_SENSOR_NAMES = ("tactile_sensor_left", "tactile_sensor_right")


def _repo_root() -> Path:
    p = _HERE
    for _ in range(10):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _load_symbol(entry: str) -> Any:
    mod_name, sym_name = entry.split(":", 1)
    return getattr(importlib.import_module(mod_name), sym_name)


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        return np.clip(img, 0.0, 1.0) * 255.0
    else:
        return np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _warmup_tactile_nominal(env: Any) -> list[str]:
    warmed: list[str] = []
    for name in TACTILE_SENSOR_NAMES:
        if name not in env.scene.sensors:
            continue
        for _ in range(3):
            try:
                env.scene[name].get_initial_render()
                warmed.append(name)
                break
            except RuntimeError:
                continue
            except Exception:
                break
    return warmed


def _float_model_set(model: Any, value: float) -> None:
    if hasattr(model, "set_value"):
        model.set_value(float(value))
    elif hasattr(model, "set_float"):
        model.set_float(float(value))


def _float_model_get(model: Any) -> float:
    if hasattr(model, "get_value_as_float"):
        return float(model.get_value_as_float())
    if hasattr(model, "as_float"):
        return float(model.as_float)
    return 0.0


def _int_model_get(model: Any, default: int = 1) -> int:
    if model is None:
        return int(default)
    if hasattr(model, "get_value_as_int"):
        return int(model.get_value_as_int())
    if hasattr(model, "as_int"):
        return int(model.as_int)
    return int(_float_model_get(model))


def _string_model_get(model: Any, default: str = "") -> str:
    if model is None:
        return str(default)
    if hasattr(model, "get_value_as_string"):
        return str(model.get_value_as_string())
    if hasattr(model, "as_string"):
        return str(model.as_string)
    return str(default)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record 7-D Factory FORGE action trajectories for simple_gripper tasks."
    )
    p.add_argument(
        "--task",
        type=str,
        default="simple_forge_peg",
        help=(
            "Preset (simple_forge_peg, simple_forge_gear, simple_forge_nut), "
            "Gym id alias (Isaac-Forge-*-Direct-v0), or any registered task with env_cfg_entry_point."
        ),
    )
    p.add_argument("--env", type=str, default="", help="Env entry module:Class (overrides --task).")
    p.add_argument("--cfg", type=str, default="", help="Cfg entry module:Class (overrides --task).")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--obs-mode", choices=("reduce", "full"), default="full")
    add_high_fidelity_cli_args(p)
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N sim steps (0 = until close).")
    p.add_argument(
        "--print-every",
        type=int,
        default=60,
        help="Print current action every N steps (0 = disable).",
    )
    p.add_argument(
        "--action-smoothing",
        type=float,
        default=0.0,
        help="EMA on commanded actions in [0,1); 0=off.",
    )
    _ag = p.add_mutually_exclusive_group()
    _ag.add_argument("--action-gui", dest="action_gui", action="store_true", help="Show 7 action sliders.")
    _ag.add_argument("--no-action-gui", dest="action_gui", action="store_false", help="Use env.actions after reset.")
    p.set_defaults(action_gui=True)
    reset_group = p.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--manual-reset-only",
        dest="manual_reset_only",
        action="store_true",
        help="Only reset from GUI Reset button.",
    )
    reset_group.add_argument(
        "--auto-reset",
        dest="manual_reset_only",
        action="store_false",
        help="Allow env auto-reset when episode ends.",
    )
    p.set_defaults(manual_reset_only=True)
    p.add_argument(
        "--show_rgb",
        action="store_true",
        help="Live matplotlib tactile RGB (implies --enable_cameras).",
    )
    p.add_argument("--env-index", type=int, default=0, help="Env index for tactile viewer.")
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.show_rgb:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401

    repo_root = _repo_root()
    env_entry, cfg_entry, preset_key = resolve_env_cfg_entries(
        task=str(args.task),
        env=str(args.env),
        cfg=str(args.cfg),
    )
    print(f"[INFO] env={env_entry}\n[INFO] cfg={cfg_entry}")

    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)
    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    cfg.obs_mode = str(args.obs_mode)
    if bool(getattr(args, "enable_cameras", False)):
        cfg.enable_cameras = True
    apply_high_fidelity_cfg(
        cfg,
        args,
        preset_key=preset_key,
        env_entry=env_entry,
        cfg_entry=cfg_entry,
    )

    env = EnvCls(cfg)
    allow_reset_gate = {"allow": not bool(args.manual_reset_only)}
    orig_reset_idx = getattr(env, "_reset_idx", None)
    if callable(orig_reset_idx):
        def _gated_reset_idx(env_ids):  # noqa: ANN001
            if allow_reset_gate["allow"]:
                return orig_reset_idx(env_ids)
            if env_ids is not None:
                ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
                env.episode_length_buf[ids] = 0
            return None

        env._reset_idx = _gated_reset_idx  # type: ignore[method-assign]
        if bool(args.manual_reset_only):
            print("[INFO] manual-reset-only: use GUI Reset; auto reset is blocked.")
    try:
        allow_reset_gate["allow"] = True
        env.reset()
    finally:
        allow_reset_gate["allow"] = not bool(args.manual_reset_only)
    warmed = _warmup_tactile_nominal(env)
    if warmed:
        print(f"[INFO] TacSL nominal render warmed: {warmed}")

    pending_manual_reset = {"flag": False}
    smoothing = float(np.clip(getattr(args, "action_smoothing", 0.0), 0.0, 0.999))
    prev_action = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)

    trajectory_path = (
        repo_root / "scripts/rsl_rl/full_tra/pose_keyframes" / f"{EnvCls.__name__}__{CfgCls.__name__}.json"
    )

    record_state: dict[str, Any] = {"active": False, "frames": []}
    record_path_model: Any = None
    record_name_model: Any = None
    record_status_model: Any = None
    hold_steps_model: Any = None
    action_gui_models: list[Any] = []
    action_gui_window: Any = None

    def _actions_from_sliders() -> np.ndarray:
        if action_gui_models and len(action_gui_models) == ACTION_DIM:
            return np.array(
                [float(np.clip(_float_model_get(m), -1.0, 1.0)) for m in action_gui_models],
                dtype=np.float32,
            )
        return _to_numpy(env.actions[0]).astype(np.float32).ravel()[:ACTION_DIM]

    def _sync_sliders_from_env() -> None:
        if not action_gui_models:
            return
        cur = _to_numpy(env.actions[0]).ravel()[:ACTION_DIM]
        for i, m in enumerate(action_gui_models):
            if i < len(cur):
                _float_model_set(m, float(cur[i]))

    def _record_status_set(msg: str) -> None:
        if record_status_model is not None and hasattr(record_status_model, "set_value"):
            record_status_model.set_value(str(msg))

    def _capture_frame() -> dict[str, Any]:
        action = _actions_from_sliders()
        hold = max(1, _int_model_get(hold_steps_model, 1))
        joint_pos = _to_numpy(env.robot.data.joint_pos[0]).ravel().tolist()
        return {
            "t_wall": float(time.time()),
            "action": action.tolist(),
            "hold_steps": int(hold),
            "joint_pos": joint_pos,
        }

    def _record_start_cb() -> None:
        record_state["active"] = True
        record_state["frames"] = []
        _record_status_set("Recording...")
        print("[INFO] Trajectory recording started.")

    def _record_snapshot_cb() -> None:
        if not record_state.get("active", False):
            _record_status_set("Not recording")
            return
        snap = _capture_frame()
        record_state["frames"].append(snap)
        _record_status_set(f"Snapshots: {len(record_state['frames'])}")
        print(f"[INFO] Snapshot #{len(record_state['frames'])} captured: action={snap['action']}")

    def _record_stop_cb() -> None:
        if not record_state.get("active", False):
            _record_status_set("Not recording")
            return
        rec_dir = "./scripts/rsl_rl/full_tra/records"
        rec_name = "traj"
        if record_path_model is not None:
            rec_dir = str(_string_model_get(record_path_model, rec_dir)).strip() or rec_dir
        if record_name_model is not None:
            rec_name = str(_string_model_get(record_name_model, rec_name)).strip() or rec_name
        out_dir = Path(rec_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (repo_root / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{rec_name}_{ts}.json"
        doc = {
            "format": "vitatlab_forge_action_trajectory_v1",
            "task": str(args.task),
            "preset_key": preset_key,
            "env_entry": env_entry,
            "cfg_entry": cfg_entry,
            "env_class": EnvCls.__name__,
            "cfg_class": CfgCls.__name__,
            "created_at": datetime.now().isoformat(),
            "num_frames": len(record_state["frames"]),
            "frames": list(record_state["frames"]),
        }
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        record_state["active"] = False
        _record_status_set(f"Saved: {out_path}")
        print(f"[INFO] Trajectory saved -> {out_path}")

        # Also refresh task keyframe file (last recording) for convenience.
        try:
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[INFO] Updated keyframe path -> {trajectory_path}")
        except Exception as exc:
            print(f"[WARN] Could not update {trajectory_path}: {exc}")

    def _manual_reset_cb() -> None:
        pending_manual_reset["flag"] = True
        _record_status_set("Reset requested")

    if args.action_gui:
        import omni.ui  # type: ignore

        init_a = _to_numpy(env.actions[0]).ravel()[:ACTION_DIM]
        try:
            action_gui_window = omni.ui.Window(
                "Forge FORGE Actions",
                width=420,
                height=520,
                visible=True,
                dock_preference=omni.ui.DockPreference.RIGHT_TOP,
            )
            with action_gui_window.frame:
                with omni.ui.ScrollingFrame(horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                    with omni.ui.VStack(spacing=4, height=0):
                        omni.ui.Label("simple_gripper trajectory recorder", height=18)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Dir", width=30)
                            record_path_model = omni.ui.SimpleStringModel()
                            record_path_model.set_value("./scripts/rsl_rl/full_tra/records")
                            omni.ui.StringField(model=record_path_model, height=24)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Name", width=30)
                            record_name_model = omni.ui.SimpleStringModel()
                            record_name_model.set_value("traj")
                            omni.ui.StringField(model=record_name_model, height=24)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("hold", width=30)
                            hold_steps_model = omni.ui.SimpleIntModel()
                            hold_steps_model.set_value(1)
                            omni.ui.IntField(model=hold_steps_model, height=24)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Button("Start Recording", clicked_fn=_record_start_cb)
                            omni.ui.Button("Snapshot", clicked_fn=_record_snapshot_cb)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Button("Stop Recording", clicked_fn=_record_stop_cb)
                            omni.ui.Button("Reset", clicked_fn=_manual_reset_cb)
                        record_status_model = omni.ui.SimpleStringModel()
                        record_status_model.set_value("Idle")
                        omni.ui.StringField(model=record_status_model, read_only=True, height=24)
                        omni.ui.Label(
                            f"Keyframe file: {trajectory_path.relative_to(repo_root)}",
                            word_wrap=True,
                        )
                        omni.ui.Label("Actions in [-1, 1] (Factory FORGE space)", height=18)
                        for i, label in enumerate(ACTION_LABELS):
                            m = omni.ui.SimpleFloatModel()
                            _float_model_set(m, float(init_a[i]) if i < len(init_a) else 0.0)
                            action_gui_models.append(m)
                            with omni.ui.HStack(spacing=6):
                                omni.ui.Label(label, width=72, alignment=omni.ui.Alignment.LEFT_CENTER)
                                omni.ui.FloatSlider(model=m, min=-1.0, max=1.0, step=0.01)
            print("[INFO] Opened Forge action slider panel.")
        except Exception as exc:
            action_gui_models = []
            action_gui_window = None
            print(f"[WARN] Could not build action GUI ({exc}); using env.actions only.")

    fig = None
    rgb_ims: list = []
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))
    if args.show_rgb:
        import matplotlib

        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        zero = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb_ims = [axes[0].imshow(zero), axes[1].imshow(zero)]
        axes[0].set_title("Tactile Left")
        axes[1].set_title("Tactile Right")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)

    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0

    while simulation_app.is_running():
        t0 = time.time()
        step += 1

        if pending_manual_reset["flag"]:
            try:
                allow_reset_gate["allow"] = True
                env.reset()
                _warmup_tactile_nominal(env)
                _sync_sliders_from_env()
                prev_action = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
                print("[INFO] Manual reset applied.")
            finally:
                allow_reset_gate["allow"] = not bool(args.manual_reset_only)
                pending_manual_reset["flag"] = False

        action_np = _actions_from_sliders()
        cmd = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)
        if smoothing > 0.0:
            cmd = smoothing * prev_action + (1.0 - smoothing) * cmd
        prev_action = cmd.clone()
        if env.num_envs > 1:
            actions = cmd.expand(env.num_envs, -1).clone()
        else:
            actions = cmd

        if args.print_every > 0 and step % int(args.print_every) == 0:
            print(f"[INFO] step={step} action={action_np.tolist()}")

        env.step(actions)

        if fig is not None and rgb_ims:
            import matplotlib.pyplot as plt

            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if name not in env.scene.sensors or i >= len(rgb_ims):
                    continue
                try:
                    data = env.scene[name].data
                except RuntimeError:
                    try:
                        env.scene[name].get_initial_render()
                        data = env.scene[name].data
                    except Exception:
                        continue
                img = getattr(data, "tactile_rgb_image", None)
                if img is not None and img.ndim == 4:
                    e = min(env_idx, img.shape[0] - 1)
                    rgb_ims[i].set_data(_img_to_uint8(img[e].detach().cpu().numpy()))
            fig.canvas.draw_idle()
            plt.pause(0.001)

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

        if args.max_steps > 0 and step >= int(args.max_steps):
            break

    if action_gui_window is not None:
        try:
            action_gui_window.visible = False
            action_gui_window.destroy()
        except Exception:
            pass

    env.close()
    simulation_app.close()
    print(f"[INFO] record session finished after {step} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
