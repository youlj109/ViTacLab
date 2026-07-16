#!/usr/bin/env python3
"""Tune UR10e arm joint targets from a visual marker pose + IK (no random arm actions).

Loads the same task presets as ``scripts/debug/run_ur10e_shadowhand_single.py``. Spawns a small
``VisualCuboid`` under ``/World/Debug/ArmIkTarget``. **Move/orient this prim in the
viewport** (e.g. with the move/rotate gizmo); each step reads its **world** pose,
runs the same pipeline as video teleop via ``VideoTeleopControl.compute``, and steps
the env.

Use this to find a good arm pose, then copy the printed joint dict into your cfg.

A docked **Shadow Hand Joints** window (``omni.ui`` sliders) drives the 24 ShadowHand
DoFs in radians; limits match the articulation. Use ``--no-hand-gui`` to fall back to
``--hand-joints`` (sim/zeros) only.

**Printing:** arm block is still **joint_pos (rad)** for ``ArticulationCfg.init_state``.
Hand block defaults to **normalized actions in ``[-1, 1]``** (same as ``env.step``), plus
a length-24 list in ``shadowhand_joint_names()`` order for direct hand control. Use
``--print-hand-rad`` to also print hand ``joint_pos`` in radians.

Examples (Isaac Sim python):

    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_shadowhand_arm_pose_from_marker.py \\
        --task pickup --num_envs 1 --enable_cameras

    # Custom initial marker pose (world frame, meters / rad euler xyz)
    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_shadowhand_arm_pose_from_marker.py \\
        --task pour --marker-pos 0.75 0.0 0.35 --marker-euler 0.0 0.78 0.0

    # Five-finger TacSL (requires ``--enable_cameras``; for ``--task inhand`` uses tactile cfg if not ``--cfg``):
    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_shadowhand_arm_pose_from_marker.py \\
        --task inhand --num_envs 1 --enable_cameras --show_rgb --show_ff

    # Forge peg (or ``--task Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0``):
    ./python.sh scripts/rsl_rl/full_tra/record_full_tra_single.py \\
        --task forge_peg --num_envs 1 --enable_cameras

    # Any registered single-arm task with ``env_cfg_entry_point``:
    ./python.sh scripts/rsl_rl/full_tra/record_full_tra_single.py \
        --task Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 --num_envs 1 --enable_cameras  --show_rgb --show_ff
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher

_FULL_TRA_DIR = Path(__file__).resolve().parent
if str(_FULL_TRA_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_TRA_DIR))
from full_tra_high_fidelity import add_high_fidelity_cli_args, apply_high_fidelity_cfg
from full_tra_task_entries import resolve_env_cfg_entries

MARKER_PRIM_PATH = "/World/Debug/ArmIkTarget"

# Same as ``scripts/debug/run_ur10e_shadowhand_single.py`` (five GelSight TacSL sensors).
TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)

_INHAND_TACTILE_CFG = (
    "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:"
    "UR10eShadowHandInHandTactileEnvCfg"
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


def _load_symbol(entry: str) -> Any:
    if ":" not in entry:
        raise ValueError(f"Invalid entry '{entry}'. Expected 'module.path:SymbolName'.")
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _make_T(pos_xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T


def _quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Isaac / XFormPrim orientation is usually w,x,y,z."""
    q = np.asarray(quat, dtype=np.float64).ravel()
    if q.size != 4:
        return np.zeros(3, dtype=np.float64)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_euler("xyz")


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _repo_root() -> Path:
    """ViTacLab repo root (contains ``source/``)."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _configure_viewer_window(fig: Any, *, topmost: bool) -> None:
    """Best-effort: keep tactile viewer from stealing focus."""
    if fig is None:
        return
    try:
        manager = fig.canvas.manager
        win = getattr(manager, "window", None)
        if win is None:
            return
        # Tk backend
        if hasattr(win, "attributes"):
            try:
                win.attributes("-topmost", bool(topmost))
            except Exception:
                pass
            if not topmost and hasattr(win, "wm_attributes"):
                try:
                    win.wm_attributes("-topmost", 0)
                except Exception:
                    pass
        # Qt backend
        if hasattr(win, "setWindowFlag"):
            try:
                import os as _os
                from PyQt5 import QtCore  # type: ignore

                win.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, bool(topmost))
                win.show()
                if not topmost and _os.name != "nt" and hasattr(win, "setAttribute"):
                    # Hint to avoid activating on show for some WMs.
                    win.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            except Exception:
                pass
    except Exception:
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR10e arm IK from a visual marker (same task presets as run_ur10e_shadowhand_single.py).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="pickup",
        help=(
            "Preset (pickup, pour, inhand, forge_peg, forge_gear, forge_nut), a forge Gym id alias, "
            "or any registered Gymnasium id with env_cfg_entry_point."
        ),
    )
    p.add_argument("--env", type=str, default="", help="Env entry module:Class (overrides --task).")
    p.add_argument("--cfg", type=str, default="", help="Cfg entry module:Class (overrides --task).")
    p.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    p.add_argument("--fps", type=float, default=30.0, help="Simulation loop target FPS.")
    p.add_argument(
        "--marker-pos",
        type=float,
        nargs=3,
        default=(0.3, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Initial marker position in world frame (m).",
    )
    p.add_argument(
        "--marker-euler",
        type=float,
        nargs=3,
        default=(0.0, 2.2, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Initial marker orientation euler xyz in world frame (rad).",
    )
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="VideoTeleopControl T_world_arm_base translation.")
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="VideoTeleopControl T_world_arm_base rotation (rad).")
    p.add_argument(
        "--object-init-index",
        type=int,
        default=0,
        help="BlindGrasp object init index: 0, 1, or 2.",
    )
    p.add_argument(
        "--hold-initial-pose",
        action="store_true",
        help="Keep the arm at its reset pose until the marker is moved or recording starts.",
    )
    p.add_argument(
        "--hand-joints",
        choices=["zeros", "sim"],
        default="sim",
        help="Initial hand vector when GUI is off, or initial slider values when --hand-gui: zeros(24) or sim hand.",
    )
    _hg = p.add_mutually_exclusive_group()
    _hg.add_argument(
        "--hand-gui",
        dest="hand_gui",
        action="store_true",
        help="Show Shadow Hand joint sliders (default).",
    )
    _hg.add_argument(
        "--no-hand-gui",
        dest="hand_gui",
        action="store_false",
        help="Disable hand sliders; use --hand-joints (sim/zeros) each step as before.",
    )
    p.set_defaults(hand_gui=True)
    p.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print arm joint dict every N steps (0 = disable periodic print).",
    )
    p.add_argument(
        "--print-on-change",
        action="store_true",
        help="Also print when IK arm joints or hand normalized actions change (thresholded).",
    )
    p.add_argument(
        "--print-hand-rad",
        action="store_true",
        help="Also print hand joint_pos (rad) for cfg init_state (default print is normalized [-1,1] actions).",
    )
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = run until close).")
    p.add_argument(
        "--show_rgb",
        action="store_true",
        help="Live matplotlib: GelSight tactile RGB (needs --enable_cameras and TacSL env).",
    )
    p.add_argument(
        "--show_ff",
        action="store_true",
        help="Live matplotlib: TacSL shear/FF via compute_tactile_shear_image (needs FF on sensors).",
    )
    p.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Which env index to read for tactile display (default: 0).",
    )
    p.add_argument(
        "--viewer-topmost",
        action="store_true",
        help="Keep tactile viewer on top. Default off to avoid stealing focus.",
    )
    add_high_fidelity_cli_args(p)
    reset_group = p.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--manual-reset-only",
        dest="manual_reset_only",
        action="store_true",
        help="Disable auto-reset in env.step; only reset when pressing GUI Reset.",
    )
    reset_group.add_argument(
        "--auto-reset",
        dest="manual_reset_only",
        action="store_false",
        help="Allow the environment to auto-reset when an episode ends.",
    )
    p.set_defaults(hand_gui=True, manual_reset_only=True, hold_initial_pose=True)
    AppLauncher.add_app_launcher_args(p)
    return p


def _sample_forge_offsets_env(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (fixed_offset, held_offset) in env frame if forge env exposes them."""
    z3 = np.zeros(3, dtype=np.float64)
    try:
        fixed_t = getattr(env, "fixed_pos_env_random", None)
        if torch.is_tensor(fixed_t) and fixed_t.ndim >= 2 and fixed_t.shape[-1] >= 3:
            fixed = _to_numpy(fixed_t[0]).ravel()[:3]
        else:
            fixed = z3.copy()
    except Exception:
        fixed = z3.copy()
    try:
        held_t = getattr(env, "held_pos_env_random", None)
        if torch.is_tensor(held_t) and held_t.ndim >= 2 and held_t.shape[-1] >= 3:
            held = _to_numpy(held_t[0]).ravel()[:3]
        else:
            held = z3.copy()
    except Exception:
        held = z3.copy()
    return fixed, held


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.show_rgb or args.show_ff:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401

    repo_root = _repo_root()
    source_dir = repo_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from video_teleop.core.video_teleop_control import ArmHandTargets, VideoTeleopControl
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names

    env_entry, cfg_entry, preset_key = resolve_env_cfg_entries(
        task=str(args.task),
        env=str(args.env),
        cfg=str(args.cfg),
    )
    print(f"[INFO] env={env_entry}\n[INFO] cfg={cfg_entry}")
    if (args.show_rgb or args.show_ff) and preset_key == "inhand" and not str(args.cfg).strip():
        cfg_entry = _INHAND_TACTILE_CFG
        print(f"[INFO] Using tactile cfg for inhand: {cfg_entry}")
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    fig = None
    ax_rgb = None
    ax_ff = None
    rgb_ims: list = []
    ff_ims: list = []
    nrows, ncols = 20, 25

    if args.show_rgb or args.show_ff:
        import matplotlib

        # Prevent matplotlib from repeatedly raising the viewer window above Isaac UI.
        matplotlib.rcParams["figure.raise_window"] = False
        #matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
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
    apply_high_fidelity_cfg(
        cfg,
        args,
        preset_key=preset_key,
        env_entry=env_entry,
        cfg_entry=cfg_entry,
    )
    if hasattr(cfg, "object_init_choice"):
        cfg.object_init_choice = int(args.object_init_index)

    _enable_cams = bool(getattr(args, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    setattr(cfg, "enable_cameras", _enable_cams)
    print(f"[INFO] cfg.enable_cameras={getattr(cfg, 'enable_cameras', None)}")

    print(f"[INFO] Creating {EnvCls.__name__} (num_envs={cfg.scene.num_envs}) ...")
    env = EnvCls(cfg)
    robot = env.robot
    joint_names = list(robot.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    sh_names = shadowhand_joint_names()

    allow_reset_gate = {"allow": not bool(args.manual_reset_only)}
    reset_mode_state = {"manual_only": bool(args.manual_reset_only)}
    orig_reset_idx = getattr(env, "_reset_idx", None)
    if callable(orig_reset_idx):
        def _gated_reset_idx(env_ids):
            if allow_reset_gate["allow"]:
                return orig_reset_idx(env_ids)
            # Reset only the episode timer so the env doesn't fire "done" every step,
            # but leave physics state (robot + object) untouched.
            if env_ids is not None:
                import torch as _torch
                ids = _torch.as_tensor(env_ids, device=env.device, dtype=_torch.long)
                env.episode_length_buf[ids] = 0
            return None
        env._reset_idx = _gated_reset_idx  # type: ignore[attr-defined]
        if reset_mode_state["manual_only"]:
            print("[INFO] manual-reset-only enabled: auto reset is blocked; use GUI Reset button.")
        else:
            print("[INFO] auto-reset enabled: environment resets automatically when an episode ends.")

    try:
        allow_reset_gate["allow"] = True
        env.reset()
    finally:
        allow_reset_gate["allow"] = not bool(reset_mode_state["manual_only"])
    action_dim = env.num_actions
    print(f"[INFO] action_dim={action_dim}, actuated_dof_indices count={len(env.actuated_dof_indices)}")
    if args.show_rgb or args.show_ff:
        print(
            "[INFO] Tactile viewer: move marker / hand; requires scene.sensors tactile_* "
            "(use --enable_cameras; inhand defaults to tactile cfg unless --cfg is set)."
        )

    _scene_env = env
    if args.show_rgb or args.show_ff:
        warmed = []
        for name in TACTILE_SENSOR_NAMES:
            if name in _scene_env.scene.sensors:
                try:
                    _scene_env.scene[name].get_initial_render()
                    warmed.append(name)
                except Exception as e:
                    print(f"[WARN] tactile warmup failed for {name}: {e}")
        if warmed:
            print(f"[INFO] tactile nominal warmup done: {warmed}")
    if args.show_ff and fig is not None:
        for name in TACTILE_SENSOR_NAMES:
            if name in _scene_env.scene.sensors:
                try:
                    nrows, ncols = _scene_env.scene[name].cfg.tactile_array_size
                except Exception:
                    pass
                break

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
            zero_ff = np.zeros((nrows * _TACTILE_SHEAR_VIZ_RESOLUTION, ncols * _TACTILE_SHEAR_VIZ_RESOLUTION, 3), dtype=np.uint8)
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
        _configure_viewer_window(fig, topmost=bool(args.viewer_topmost))
        plt.pause(0.1)

    render_ff = _tactile_shear_image_rgb_uint8 if args.show_ff else None
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))
    print(f"[INFO] Move prim '{MARKER_PRIM_PATH}' in the viewport; arm IK follows its world pose.")
    if len(env.actuated_dof_indices) < len(arm_indices) + len(hand_indices):
        print(
            "[WARN] This env does not actuate all arm+hand DOFs (e.g. inhand is hand-only). "
            "Printed arm joints are still valid for copying into robot_cfg.init_state; "
            "env.step only applies the actuated subset.",
        )

    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.prims import XFormPrim

    VisualCuboid(
        prim_path=MARKER_PRIM_PATH,
        size=0.04,
        position=np.array(args.marker_pos, dtype=np.float64),
        visible=True,
        color=np.array([1.0, 0.2, 0.8]),
    )
    marker_xf = XFormPrim(prim_paths_expr=MARKER_PRIM_PATH, name="ArmIkTarget", usd=True)
    T0 = _make_T(np.array(args.marker_pos, dtype=np.float64), np.array(args.marker_euler, dtype=np.float64))
    pos0 = T0[:3, 3]
    quat_wxyz = R.from_matrix(T0[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_wxyz[3], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2]], dtype=np.float32)
    marker_xf.set_world_poses(
        positions=torch.tensor([pos0], dtype=torch.float32, device="cpu"),
        orientations=torch.tensor([quat_wxyz], dtype=torch.float32, device="cpu"),
    )
    initial_marker_pos = pos0.copy()
    initial_marker_quat_wxyz = quat_wxyz.copy()

    T_world_arm_base = _make_T(np.array(args.arm_base_pos, dtype=np.float64), np.array(args.arm_base_euler, dtype=np.float64))
    control = VideoTeleopControl(T_world_arm_base=T_world_arm_base)

    def _hand_joints_shadow_from_sim() -> np.ndarray:
        out = np.zeros(24, dtype=np.float64)
        jpos = robot.data.joint_pos[0].detach().cpu().numpy()
        for sh_i, sh_name in enumerate(sh_names):
            for idx in hand_indices:
                n = joint_names[idx]
                if sh_name in n or n.endswith(sh_name):
                    out[sh_i] = float(jpos[idx])
                    break
        return out

    def _hand_joint_for_robot_name(name: str, hand_joints: np.ndarray) -> float:
        for sh_idx, sh_name in enumerate(sh_names):
            if sh_name in name or name.endswith(sh_name):
                return float(hand_joints[sh_idx])
        return 0.0

    def _arm_joints_from_sim() -> np.ndarray:
        jpos = robot.data.joint_pos[0].detach().cpu().numpy()
        return np.array([float(jpos[idx]) for idx in arm_indices], dtype=np.float64)

    def _build_action_numpy(arm_joints: np.ndarray, hand_joints: np.ndarray) -> np.ndarray:
        """Map desired joint positions (rad) to env actions in [-1, 1] (numpy, one env row).

        Matches :meth:`UR10eShadowHandDirectBaseEnv._pre_physics_step` (clamp) and
        ``_scale``: joint = 0.5 * (action + 1) * (upper - lower) + lower.
        """
        arm_joints = np.asarray(arm_joints, dtype=np.float64).ravel()
        hand_joints = np.asarray(hand_joints, dtype=np.float64).ravel()
        full_dof = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_indices):
            if i < len(arm_joints):
                full_dof[idx] = arm_joints[i]
        for idx in hand_indices:
            full_dof[idx] = _hand_joint_for_robot_name(joint_names[idx], hand_joints)
        actuated = full_dof[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lower = env.robot_dof_lower_limits[0, env.actuated_dof_indices].cpu().numpy()
        upper = env.robot_dof_upper_limits[0, env.actuated_dof_indices].cpu().numpy()
        eps = 1e-6
        scale = np.where(upper - lower > eps, 2.0 * (actuated - lower) / (upper - lower) - 1.0, 0.0)
        scale = np.clip(scale, -1.0, 1.0)
        return scale

    def _build_action(arm_joints: np.ndarray, hand_joints: np.ndarray) -> torch.Tensor:
        scale = _build_action_numpy(arm_joints, hand_joints)
        return torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)

    def _print_arm_cfg_block(arm_j: np.ndarray) -> None:
        print("[INFO] Arm joint_pos snippet for ArticulationCfg.init_state (rad):")
        print('    joint_pos={')
        for i, idx in enumerate(arm_indices):
            if i < len(arm_j):
                print(f'        "{joint_names[idx]}": {float(arm_j[i]):.16f},')
        print("    },")

    def _robot_joint_name_for_shadow(sh_name: str) -> str:
        for idx in hand_indices:
            n = joint_names[idx]
            if sh_name in n or n.endswith(sh_name):
                return str(n)
        return sh_name

    def _shadow_limits_rad(sh_i: int) -> tuple[float, float]:
        sh_name = sh_names[sh_i]
        for idx in hand_indices:
            n = joint_names[idx]
            if sh_name in n or n.endswith(sh_name):
                lo = float(env.robot_dof_lower_limits[0, idx].item())
                hi = float(env.robot_dof_upper_limits[0, idx].item())
                if hi - lo < 1e-6:
                    return (-1.5707963267948966, 1.5707963267948966)
                return lo, hi
        return (-1.5707963267948966, 1.5707963267948966)

    def _print_hand_cfg_block(hand_j: np.ndarray) -> None:
        print("[INFO] Hand joint_pos snippet for ArticulationCfg.init_state (rad):")
        print('    joint_pos={')
        for i, sh_name in enumerate(sh_names):
            if i < len(hand_j):
                print(f'        "{_robot_joint_name_for_shadow(sh_name)}": {float(hand_j[i]):.16f},')
        print("    },")

    def _hand_norm_shadow_order(arm_j: np.ndarray, hand_j: np.ndarray) -> np.ndarray:
        """Length-24 normalized actions [-1,1] in shadowhand_joint_names order (for change detection)."""
        act = _build_action_numpy(arm_j, hand_j)
        ad = list(env.actuated_dof_indices)
        out = np.zeros(24, dtype=np.float64)
        for sh_i, sh_name in enumerate(sh_names):
            jid = None
            for jidx in hand_indices:
                n = joint_names[jidx]
                if sh_name in n or n.endswith(sh_name):
                    jid = jidx
                    break
            if jid is None:
                continue
            try:
                ai = ad.index(jid)
            except ValueError:
                continue
            out[sh_i] = float(act[ai])
        return out

    def _print_hand_action_block(hand_j: np.ndarray, arm_j: Optional[np.ndarray]) -> None:
        """Print hand as normalized env actions (and shadow-order list) for direct control."""
        aj = _arm_joints_from_sim() if arm_j is None else np.asarray(arm_j, dtype=np.float64).ravel()
        hj = np.asarray(hand_j, dtype=np.float64).ravel()
        act = _build_action_numpy(aj, hj)
        ad = list(env.actuated_dof_indices)
        hand_set = set(hand_indices)
        print("[INFO] Hand normalized actions in [-1, 1] (same as env.step), dict by joint name:")
        print("    {")
        for ai, jid in enumerate(ad):
            if jid in hand_set:
                print(f'        "{joint_names[jid]}": {float(act[ai]):.16f},')
        print("    },")
        row = _hand_norm_shadow_order(aj, hj)
        print("[INFO] Same hand actions in shadowhand_joint_names() order (length 24), for direct control:")
        print(f"    {row.tolist()}")

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

    def _string_model_get(model: Any, default: str = "") -> str:
        if model is None:
            return str(default)
        if hasattr(model, "get_value_as_string"):
            return str(model.get_value_as_string())
        if hasattr(model, "as_string"):
            return str(model.as_string)
        return str(default)

    def _string_model_set(model: Any, value: str) -> None:
        if model is None:
            return
        if hasattr(model, "set_value"):
            model.set_value(str(value))

    def _bool_model_get(model: Any, default: bool = False) -> bool:
        if model is None:
            return bool(default)
        if hasattr(model, "get_value_as_bool"):
            return bool(model.get_value_as_bool())
        if hasattr(model, "as_bool"):
            return bool(model.as_bool)
        return bool(default)

    import omni.ui  # type: ignore

    hand_pose_preset_path = (
        (_repo_root() / "scripts/rsl_rl/full_tra/hand_pose_presets")
        / f"{EnvCls.__name__}__{CfgCls.__name__}.json"
    )
    pose_keyframe_path = (
        (_repo_root() / "scripts/rsl_rl/full_tra/pose_keyframes")
        / f"{EnvCls.__name__}__{CfgCls.__name__}.json"
    )
    hand_gui_models: list[Any] = []
    hand_gui_window: Any = None
    record_state: dict[str, Any] = {
        "active": False,
        "frames": [],
        "task": str(args.task),
        "preset_key": preset_key,
        "env_entry": env_entry,
        "cfg_entry": cfg_entry,
    }
    record_path_model: Any = None
    record_name_model: Any = None
    record_status_model: Any = None
    hand_pose_status_model: Any = None
    keyframe_name_model: Any = None
    keyframe_selected_model: Any = None
    keyframe_status_model: Any = None
    reset_mode_model: Any = None
    camera_name_model: Any = None
    camera_capture_dir_model: Any = None
    camera_capture_prefix_model: Any = None
    camera_capture_status_model: Any = None
    camera_save_tactile_image_model: Any = None
    camera_save_tactile_grid_model: Any = None
    pending_manual_reset = {"flag": False}
    pending_keyframe_apply: dict[str, Any] = {"name": None}
    hold_initial_pose_state = {"active": bool(args.hold_initial_pose), "marker_moved": False}
    hand_pose_state: dict[str, Optional[np.ndarray]] = {
        "open": None,
        "close": None,
    }
    pose_keyframes: dict[str, dict[str, Any]] = {}

    def _record_status_set(msg: str) -> None:
        if record_status_model is None:
            return
        if hasattr(record_status_model, "set_value"):
            record_status_model.set_value(str(msg))

    def _set_manual_reset_only(enabled: bool) -> None:
        reset_mode_state["manual_only"] = bool(enabled)
        allow_reset_gate["allow"] = not reset_mode_state["manual_only"]
        mode_text = "Manual reset only" if reset_mode_state["manual_only"] else "Auto reset"
        _record_status_set(f"Reset mode: {mode_text}")
        print(f"[INFO] Reset mode -> {mode_text}")

    def _reset_mode_label(enabled: bool) -> str:
        return "Manual Only" if bool(enabled) else "Auto"

    def _hand_pose_status_set(msg: str) -> None:
        if hand_pose_status_model is None:
            return
        if hasattr(hand_pose_status_model, "set_value"):
            hand_pose_status_model.set_value(str(msg))

    def _keyframe_status_set(msg: str) -> None:
        if keyframe_status_model is None:
            return
        if hasattr(keyframe_status_model, "set_value"):
            keyframe_status_model.set_value(str(msg))

    def _camera_capture_status_set(msg: str) -> None:
        if camera_capture_status_model is None:
            return
        if hasattr(camera_capture_status_model, "set_value"):
            camera_capture_status_model.set_value(str(msg))

    def _selected_keyframe_name() -> str:
        return _string_model_get(keyframe_selected_model, "").strip()

    def _set_selected_keyframe_name(name: str) -> None:
        _string_model_set(keyframe_selected_model, str(name))

    def _refresh_keyframe_selection(default_name: str = "") -> None:
        names = sorted(pose_keyframes.keys())
        target = str(default_name).strip()
        if target and target in pose_keyframes:
            _set_selected_keyframe_name(target)
            return
        current = _selected_keyframe_name()
        if current in pose_keyframes:
            return
        _set_selected_keyframe_name(names[0] if names else "")

    def _hand_pose_status_refresh(prefix: str = "") -> None:
        open_ok = hand_pose_state["open"] is not None
        close_ok = hand_pose_state["close"] is not None
        summary = f"open={'Y' if open_ok else 'N'} | close={'Y' if close_ok else 'N'}"
        msg = f"{prefix} ({summary})" if prefix else summary
        _hand_pose_status_set(msg)

    def _save_hand_pose_presets() -> None:
        hand_pose_preset_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "format": "vitatlab_hand_pose_presets_v1",
            "task": str(args.task),
            "preset_key": preset_key,
            "env_entry": env_entry,
            "cfg_entry": cfg_entry,
            "env_class": EnvCls.__name__,
            "cfg_class": CfgCls.__name__,
            "updated_at": datetime.now().isoformat(),
            "poses": {
                name: (pose.tolist() if pose is not None else None)
                for name, pose in hand_pose_state.items()
            },
        }
        hand_pose_preset_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_hand_pose_presets() -> None:
        if not hand_pose_preset_path.is_file():
            _hand_pose_status_refresh("No saved hand presets")
            return
        try:
            doc = json.loads(hand_pose_preset_path.read_text(encoding="utf-8"))
            poses = doc.get("poses") or {}
            for name in ("open", "close"):
                raw = poses.get(name)
                if raw is None:
                    hand_pose_state[name] = None
                    continue
                arr = np.asarray(raw, dtype=np.float64).ravel()
                if arr.size != 24:
                    raise ValueError(f"pose {name!r} must have 24 values, got {arr.size}")
                hand_pose_state[name] = arr
            _hand_pose_status_refresh("Loaded hand presets")
            print(f"[INFO] Hand pose presets loaded <- {hand_pose_preset_path}")
        except Exception as e:
            hand_pose_state["open"] = None
            hand_pose_state["close"] = None
            _hand_pose_status_refresh("Preset load failed")
            print(f"[WARN] Failed to load hand pose presets from {hand_pose_preset_path}: {e}")

    def _capture_pose_keyframe() -> dict[str, Any]:
        pos_t, ori_t = marker_xf.get_world_poses()
        marker_pos = _to_numpy(pos_t[0]).ravel()[:3]
        quat = _to_numpy(ori_t[0]).ravel()[:4]
        marker_euler = _quat_wxyz_to_euler_xyz(quat)
        arm_j = _arm_joints_from_sim()
        hand_j = _hand_vector_for_step()
        obj_pos = _sample_object_pos_env()
        obj_init = _object_init_pos_env()
        obj_rand = obj_pos - obj_init
        obj_rand[2] = 0.0
        fixed_rand, held_rand = _sample_forge_offsets_env(env)
        return {
            "name": "",
            "t_wall": float(time.time()),
            "marker_pos_w": marker_pos.tolist(),
            "marker_quat_wxyz": quat.tolist(),
            "marker_euler_xyz": marker_euler.tolist(),
            "arm_joint_pos": arm_j.tolist(),
            "hand_joint_pos_shadow_order": hand_j.tolist(),
            "object_pos_env": obj_pos.tolist(),
            "object_pos_env_random": obj_rand.tolist(),
            "fixed_pos_env_random": fixed_rand.tolist(),
            "held_pos_env_random": held_rand.tolist(),
            "object_init_index": int(getattr(cfg, "object_init_choice", 0)),
            "object_init_pos": obj_init.tolist(),
        }

    def _save_pose_keyframes() -> None:
        pose_keyframe_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "format": "vitatlab_pose_keyframes_v1",
            "task": str(args.task),
            "preset_key": preset_key,
            "env_entry": env_entry,
            "cfg_entry": cfg_entry,
            "env_class": EnvCls.__name__,
            "cfg_class": CfgCls.__name__,
            "updated_at": datetime.now().isoformat(),
            "keyframes": [
                {**pose_keyframes[name], "name": name}
                for name in sorted(pose_keyframes.keys())
            ],
        }
        pose_keyframe_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_pose_keyframes() -> None:
        pose_keyframes.clear()
        if not pose_keyframe_path.is_file():
            _refresh_keyframe_selection()
            _keyframe_status_set("No saved keyframes")
            return
        try:
            doc = json.loads(pose_keyframe_path.read_text(encoding="utf-8"))
            raw_keyframes = doc.get("keyframes") or []
            for item in raw_keyframes:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                arm_j = np.asarray(item.get("arm_joint_pos", []), dtype=np.float64).ravel()
                hand_j = np.asarray(item.get("hand_joint_pos_shadow_order", []), dtype=np.float64).ravel()
                marker_pos = np.asarray(item.get("marker_pos_w", []), dtype=np.float64).ravel()
                marker_euler = np.asarray(item.get("marker_euler_xyz", []), dtype=np.float64).ravel()
                if arm_j.size != len(arm_indices):
                    raise ValueError(f"keyframe {name!r} arm_joint_pos size {arm_j.size} != {len(arm_indices)}")
                if hand_j.size != 24:
                    raise ValueError(f"keyframe {name!r} hand_joint_pos size {hand_j.size} != 24")
                if marker_pos.size != 3 or marker_euler.size != 3:
                    raise ValueError(f"keyframe {name!r} marker pose is invalid")
                clean = dict(item)
                clean["arm_joint_pos"] = arm_j.tolist()
                clean["hand_joint_pos_shadow_order"] = hand_j.tolist()
                clean["marker_pos_w"] = marker_pos.tolist()
                clean["marker_euler_xyz"] = marker_euler.tolist()
                pose_keyframes[name] = clean
            _refresh_keyframe_selection()
            _keyframe_status_set(f"Loaded {len(pose_keyframes)} keyframes")
            print(f"[INFO] Pose keyframes loaded <- {pose_keyframe_path}")
        except Exception as e:
            pose_keyframes.clear()
            _refresh_keyframe_selection()
            _keyframe_status_set("Keyframe load failed")
            print(f"[WARN] Failed to load pose keyframes from {pose_keyframe_path}: {e}")

    def _requested_keyframe_name() -> str:
        typed = _string_model_get(keyframe_name_model, "").strip()
        if typed:
            return typed
        return _selected_keyframe_name()

    def _record_pose_keyframe_cb() -> None:
        name = _requested_keyframe_name()
        if not name:
            _keyframe_status_set("Enter keyframe name")
            return
        snap = _capture_pose_keyframe()
        snap["name"] = name
        pose_keyframes[name] = snap
        _save_pose_keyframes()
        _refresh_keyframe_selection(name)
        _set_selected_keyframe_name(name)
        _keyframe_status_set(f"Saved keyframe: {name}")
        print(f"[INFO] Pose keyframe '{name}' saved -> {pose_keyframe_path}")

    def _apply_pose_keyframe_cb() -> None:
        name = _selected_keyframe_name()
        if not name:
            _keyframe_status_set("No keyframe selected")
            return
        if name not in pose_keyframes:
            _keyframe_status_set(f"Keyframe not found: {name}")
            return
        pending_keyframe_apply["name"] = name
        hold_initial_pose_state["active"] = False
        _keyframe_status_set(f"Apply requested: {name}")
        print(f"[INFO] Pose keyframe '{name}' apply requested.")

    def _rename_pose_keyframe_cb() -> None:
        old_name = _selected_keyframe_name()
        new_name = _string_model_get(keyframe_name_model, "").strip()
        if not old_name:
            _keyframe_status_set("No keyframe selected")
            return
        if not new_name:
            _keyframe_status_set("Enter new keyframe name")
            return
        if old_name not in pose_keyframes:
            _keyframe_status_set(f"Keyframe not found: {old_name}")
            return
        if new_name != old_name and new_name in pose_keyframes:
            _keyframe_status_set(f"Keyframe exists: {new_name}")
            return
        data = dict(pose_keyframes.pop(old_name))
        data["name"] = new_name
        pose_keyframes[new_name] = data
        _save_pose_keyframes()
        _refresh_keyframe_selection(new_name)
        _set_selected_keyframe_name(new_name)
        _string_model_set(keyframe_name_model, new_name)
        _keyframe_status_set(f"Renamed: {old_name} -> {new_name}")
        print(f"[INFO] Pose keyframe renamed: '{old_name}' -> '{new_name}'.")

    def _delete_pose_keyframe_cb() -> None:
        name = _selected_keyframe_name()
        if not name:
            _keyframe_status_set("No keyframe selected")
            return
        if name not in pose_keyframes:
            _keyframe_status_set(f"Keyframe not found: {name}")
            return
        pose_keyframes.pop(name, None)
        _save_pose_keyframes()
        _refresh_keyframe_selection()
        _keyframe_status_set(f"Deleted keyframe: {name}")
        print(f"[INFO] Pose keyframe '{name}' deleted.")

    def _fill_keyframe_name_from_selected_cb() -> None:
        name = _selected_keyframe_name()
        if not name:
            _keyframe_status_set("No keyframe selected")
            return
        _string_model_set(keyframe_name_model, name)
        _keyframe_status_set(f"Name loaded: {name}")

    def _save_camera_frame_cb() -> None:
        cam_name = _string_model_get(camera_name_model, "third_person_camera").strip() or "third_person_camera"
        save_dir = _string_model_get(camera_capture_dir_model, "./scripts/rsl_rl/full_tra/camera_snaps").strip()
        save_prefix = _string_model_get(camera_capture_prefix_model, "camera").strip() or "camera"
        save_tactile_image = _bool_model_get(camera_save_tactile_image_model, default=False)
        save_tactile_grid = _bool_model_get(camera_save_tactile_grid_model, default=False)

        if cam_name not in env.scene.sensors:
            msg = f"Camera not found: {cam_name}"
            _camera_capture_status_set(msg)
            print(f"[WARN] {msg}")
            return

        cam = env.scene[cam_name]
        cam_out = getattr(cam.data, "output", {})
        rgb = cam_out.get("rgb", None) if isinstance(cam_out, dict) else None
        if rgb is None:
            msg = f"Camera has no rgb output: {cam_name}"
            _camera_capture_status_set(msg)
            print(f"[WARN] {msg}")
            return

        e = max(0, min(int(args.env_index), env.num_envs - 1))
        try:
            img = rgb[e].detach().cpu().numpy()
        except Exception as ex:
            msg = f"Read camera frame failed: {ex}"
            _camera_capture_status_set(msg)
            print(f"[WARN] {msg}")
            return

        if img.dtype != np.uint8:
            img_f = img.astype(np.float32)
            if img_f.max() <= 1.0:
                img_f = np.clip(img_f, 0.0, 1.0) * 255.0
            else:
                img_f = np.clip(img_f, 0.0, 255.0)
            img = img_f.astype(np.uint8)

        out_dir = Path(save_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (_repo_root() / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"{save_prefix}_{cam_name}_env{e}_{ts}.png"
        try:
            imageio.imwrite(out_path, img)
            saved_paths: list[str] = [str(out_path)]

            if save_tactile_image:
                for sensor_name in TACTILE_SENSOR_NAMES:
                    if sensor_name not in env.scene.sensors:
                        continue
                    sensor_obj = env.scene[sensor_name]
                    tactile_rgb = getattr(sensor_obj.data, "tactile_rgb_image", None)
                    if tactile_rgb is None:
                        continue
                    try:
                        t_img = tactile_rgb[e].detach().cpu().numpy()
                        t_img = _img_to_uint8(t_img)
                        t_path = out_dir / f"{save_prefix}_{sensor_name}_rgb_env{e}_{ts}.png"
                        imageio.imwrite(t_path, t_img)
                        saved_paths.append(str(t_path))
                    except Exception as ex:
                        print(f"[WARN] Save tactile image failed ({sensor_name}): {ex}")

            if save_tactile_grid:
                grid_payload: dict[str, np.ndarray] = {}
                for sensor_name in TACTILE_SENSOR_NAMES:
                    if sensor_name not in env.scene.sensors:
                        continue
                    sensor_obj = env.scene[sensor_name]
                    normal_force = getattr(sensor_obj.data, "tactile_normal_force", None)
                    shear_force = getattr(sensor_obj.data, "tactile_shear_force", None)
                    if normal_force is not None:
                        try:
                            grid_payload[f"{sensor_name}_normal_force"] = normal_force[e].detach().cpu().numpy()
                        except Exception as ex:
                            print(f"[WARN] Read tactile normal grid failed ({sensor_name}): {ex}")
                    if shear_force is not None:
                        try:
                            grid_payload[f"{sensor_name}_shear_force"] = shear_force[e].detach().cpu().numpy()
                        except Exception as ex:
                            print(f"[WARN] Read tactile shear grid failed ({sensor_name}): {ex}")
                if grid_payload:
                    g_path = out_dir / f"{save_prefix}_tactile_grid_env{e}_{ts}.npz"
                    np.savez_compressed(g_path, **grid_payload)
                    saved_paths.append(str(g_path))

            msg = f"Saved {len(saved_paths)} file(s)"
            _camera_capture_status_set(msg)
            print(f"[INFO] Camera/tactile snapshot saved -> {saved_paths}")
        except Exception as ex:
            msg = f"Save failed: {ex}"
            _camera_capture_status_set(msg)
            print(f"[WARN] {msg}")

    def _set_hand_gui_pose(hand_joints: np.ndarray) -> None:
        arr = np.asarray(hand_joints, dtype=np.float64).ravel()
        if arr.size != 24:
            raise ValueError(f"Expected 24 hand joints, got {arr.size}")
        if len(hand_gui_models) != 24:
            raise RuntimeError("Hand GUI sliders are not available.")
        for model, value in zip(hand_gui_models, arr, strict=False):
            _float_model_set(model, float(value))

    def _record_hand_pose(name: str) -> None:
        hand_j = np.asarray(_hand_vector_for_step(), dtype=np.float64).ravel()
        if hand_j.size != 24:
            _hand_pose_status_refresh(f"{name}_record failed")
            return
        hand_pose_state[name] = hand_j.copy()
        _save_hand_pose_presets()
        _hand_pose_status_refresh(f"{name}_record saved")
        print(f"[INFO] Hand pose '{name}' saved -> {hand_pose_preset_path}")

    def _apply_hand_pose(name: str) -> None:
        pose = hand_pose_state.get(name)
        if pose is None:
            _hand_pose_status_refresh(f"{name} not recorded")
            return
        try:
            _set_hand_gui_pose(pose)
        except Exception as e:
            _hand_pose_status_refresh(f"{name} apply failed")
            print(f"[WARN] Failed to apply hand pose '{name}': {e}")
            return
        _hand_pose_status_refresh(f"{name} applied")
        print(f"[INFO] Hand pose '{name}' applied.")

    def _open_record_cb() -> None:
        _record_hand_pose("open")

    def _open_cb() -> None:
        _apply_hand_pose("open")

    def _close_record_cb() -> None:
        _record_hand_pose("close")

    def _close_cb() -> None:
        _apply_hand_pose("close")

    # Preset / keyframe dicts are filled here; UI status & Selected are synced after omni.ui
    # models exist (see hand_gui block), otherwise _string_model_set is a no-op.
    _load_hand_pose_presets()
    _load_pose_keyframes()

    def _record_start_cb() -> None:
        record_state["active"] = True
        record_state["frames"] = []
        hold_initial_pose_state["active"] = False
        _record_status_set("Recording...")
        print("[INFO] Trajectory recording started.")

    def _record_snapshot_cb() -> None:
        if not record_state.get("active", False):
            _record_status_set("Not recording")
            return
        snap = _capture_pose_keyframe()
        snap.pop("name", None)
        record_state["frames"].append(snap)
        record_state["object_init_index"] = int(getattr(cfg, "object_init_choice", 0))
        record_state["object_init_pos"] = _object_init_pos_env().tolist()
        _record_status_set(f"Snapshots: {len(record_state['frames'])}")
        print(f"[INFO] Snapshot #{len(record_state['frames'])} captured.")

    def _record_stop_cb() -> None:
        if not record_state.get("active", False):
            _record_status_set("Not recording")
            return
        rec_dir = "."
        rec_name = "trajectory"
        if record_path_model is not None and hasattr(record_path_model, "get_value_as_string"):
            rec_dir = str(record_path_model.get_value_as_string()).strip() or "."
        if record_name_model is not None and hasattr(record_name_model, "get_value_as_string"):
            rec_name = str(record_name_model.get_value_as_string()).strip() or "trajectory"
        out_dir = Path(rec_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (_repo_root() / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{rec_name}_{ts}.json"
        doc = {
            "format": "vitatlab_full_tra_v1",
            "task": str(args.task),
            "preset_key": preset_key,
            "env_entry": env_entry,
            "cfg_entry": cfg_entry,
            "object_init_index": int(getattr(cfg, "object_init_choice", 0)),
            "object_init_pos": _object_init_pos_env().tolist(),
            "created_at": datetime.now().isoformat(),
            "num_frames": len(record_state["frames"]),
            "frames": list(record_state["frames"]),
        }
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        record_state["active"] = False
        _record_status_set(f"Saved: {out_path}")
        print(f"[INFO] Trajectory saved -> {out_path}")

    def _manual_reset_cb() -> None:
        pending_manual_reset["flag"] = True
        _record_status_set("Reset requested")

    def _toggle_reset_mode_cb() -> None:
        _set_manual_reset_only(not reset_mode_state["manual_only"])
        if reset_mode_model is not None and hasattr(reset_mode_model, "set_value"):
            reset_mode_model.set_value(_reset_mode_label(reset_mode_state["manual_only"]))

    if args.hand_gui:
        init_h = np.zeros(24, dtype=np.float64)
        if args.hand_joints == "sim":
            init_h = _hand_joints_shadow_from_sim()
        try:
            hand_gui_window = omni.ui.Window(
                "Shadow Hand Joints (rad)",
                width=460,
                height=720,
                visible=True,
                dock_preference=omni.ui.DockPreference.RIGHT_TOP,
            )
            with hand_gui_window.frame:
                with omni.ui.ScrollingFrame(horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                    with omni.ui.VStack(spacing=3, height=0):
                        omni.ui.Label("Trajectory Recorder", height=18)
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
                        omni.ui.Label("Hand Pose Presets", height=18)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Button("open_record", clicked_fn=_open_record_cb)
                            omni.ui.Button("open", clicked_fn=_open_cb)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Button("close_record", clicked_fn=_close_record_cb)
                            omni.ui.Button("close", clicked_fn=_close_cb)
                        hand_pose_status_model = omni.ui.SimpleStringModel()
                        hand_pose_status_model.set_value("")
                        omni.ui.StringField(model=hand_pose_status_model, read_only=True, height=24)
                        omni.ui.Label(
                            f"Preset file: {hand_pose_preset_path.relative_to(_repo_root())}",
                            word_wrap=True,
                        )
                        omni.ui.Label("Pose Keyframes (UR + Hand + Marker)", height=18)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Name", width=40)
                            keyframe_name_model = omni.ui.SimpleStringModel()
                            keyframe_name_model.set_value("pose_001")
                            omni.ui.StringField(model=keyframe_name_model, height=24)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Selected", width=56)
                            keyframe_selected_model = omni.ui.SimpleStringModel()
                            keyframe_selected_model.set_value("")
                            omni.ui.StringField(model=keyframe_selected_model, height=24)
                            omni.ui.Button("Use Selected", clicked_fn=_fill_keyframe_name_from_selected_cb, width=110)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Button("Save Current Pose", clicked_fn=_record_pose_keyframe_cb)
                            omni.ui.Button("Replay Selected", clicked_fn=_apply_pose_keyframe_cb)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Button("Rename Selected", clicked_fn=_rename_pose_keyframe_cb)
                            omni.ui.Button("Delete Selected", clicked_fn=_delete_pose_keyframe_cb)
                        keyframe_status_model = omni.ui.SimpleStringModel()
                        keyframe_status_model.set_value("")
                        omni.ui.StringField(model=keyframe_status_model, read_only=True, height=24)
                        omni.ui.Label(
                            f"Keyframe file: {pose_keyframe_path.relative_to(_repo_root())}",
                            word_wrap=True,
                        )
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Button("Start Recording", clicked_fn=_record_start_cb)
                            omni.ui.Button("Snapshot", clicked_fn=_record_snapshot_cb)
                            omni.ui.Button("Stop Recording", clicked_fn=_record_stop_cb)
                            omni.ui.Button("Reset", clicked_fn=_manual_reset_cb)
                        omni.ui.Label("Camera Snapshot", height=18)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Cam", width=30)
                            camera_name_model = omni.ui.SimpleStringModel()
                            camera_name_model.set_value("third_person_camera")
                            omni.ui.StringField(model=camera_name_model, height=24)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Dir", width=30)
                            camera_capture_dir_model = omni.ui.SimpleStringModel()
                            camera_capture_dir_model.set_value("./scripts/rsl_rl/full_tra/camera_snaps")
                            omni.ui.StringField(model=camera_capture_dir_model, height=24)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Name", width=30)
                            camera_capture_prefix_model = omni.ui.SimpleStringModel()
                            camera_capture_prefix_model.set_value("camera")
                            omni.ui.StringField(model=camera_capture_prefix_model, height=24)
                            omni.ui.Button("Save Camera Frame", clicked_fn=_save_camera_frame_cb, width=150)
                        with omni.ui.HStack(spacing=8):
                            camera_save_tactile_image_model = omni.ui.SimpleBoolModel()
                            camera_save_tactile_image_model.set_value(False)
                            omni.ui.CheckBox(model=camera_save_tactile_image_model, width=18)
                            omni.ui.Label("保存触觉图像", width=92)
                            camera_save_tactile_grid_model = omni.ui.SimpleBoolModel()
                            camera_save_tactile_grid_model.set_value(False)
                            omni.ui.CheckBox(model=camera_save_tactile_grid_model, width=18)
                            omni.ui.Label("保存点阵", width=70)
                        camera_capture_status_model = omni.ui.SimpleStringModel()
                        camera_capture_status_model.set_value("Idle")
                        omni.ui.StringField(model=camera_capture_status_model, read_only=True, height=24)
                        with omni.ui.HStack(spacing=4):
                            omni.ui.Label("Reset Mode", width=70)
                            reset_mode_model = omni.ui.SimpleStringModel()
                            reset_mode_model.set_value(_reset_mode_label(reset_mode_state["manual_only"]))
                            omni.ui.StringField(model=reset_mode_model, read_only=True, height=24)
                            omni.ui.Button("Toggle", clicked_fn=_toggle_reset_mode_cb, width=80)
                        record_status_model = omni.ui.SimpleStringModel()
                        record_status_model.set_value("Idle")
                        omni.ui.StringField(model=record_status_model, read_only=True, height=24)
                        _record_status_set(f"Reset mode: {'Manual reset only' if reset_mode_state['manual_only'] else 'Auto reset'}")
                        omni.ui.Label(
                            "Drag sliders; values feed IK + env (same order as shadowhand_joint_names).",
                            word_wrap=True,
                        )
                        for sh_i in range(24):
                            lo, hi = _shadow_limits_rad(sh_i)
                            mid = float(np.clip(init_h[sh_i], lo, hi))
                            m = omni.ui.SimpleFloatModel()
                            _float_model_set(m, mid)
                            hand_gui_models.append(m)
                            with omni.ui.HStack(spacing=6):
                                omni.ui.Label(f"{sh_names[sh_i]}", width=56, alignment=omni.ui.Alignment.LEFT_CENTER)
                                omni.ui.FloatSlider(
                                    model=m,
                                    min=lo,
                                    max=hi,
                                    step=max(1e-4, (hi - lo) / 500.0),
                                )
            _hand_pose_status_refresh()
            print("[INFO] Opened 'Shadow Hand Joints (rad)' panel; sliders control the 24-DoF hand.")
        except Exception as e:
            hand_gui_models = []
            hand_gui_window = None
            print(f"[WARN] Could not build hand GUI ({e}); use --hand-joints sim/zeros without sliders.")

        # Load ran before these models existed; UI was initialized with empty Selected. Sync
        # from pose_keyframes so Replay matches memory, and surface load failure in-panel.
        if keyframe_selected_model is not None:
            _refresh_keyframe_selection()
            if pose_keyframes:
                _keyframe_status_set(f"Loaded {len(pose_keyframes)} keyframes")
            elif not pose_keyframe_path.is_file():
                _keyframe_status_set("No saved keyframes")
            else:
                _keyframe_status_set("Keyframe load failed (see terminal WARN)")

    def _hand_vector_for_step() -> np.ndarray:
        if hand_gui_models and len(hand_gui_models) == 24:
            return np.array([_float_model_get(m) for m in hand_gui_models], dtype=np.float64)
        if args.hand_joints == "zeros":
            return np.zeros(24, dtype=np.float64)
        return _hand_joints_shadow_from_sim()

    def _sample_object_pos_env() -> np.ndarray:
        if hasattr(env, "object") and hasattr(env.object, "data"):
            return _to_numpy(env.object.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3]
        if hasattr(env, "cup") and hasattr(env.cup, "data"):
            return _to_numpy(env.cup.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3]
        return np.zeros(3, dtype=np.float64)

    def _object_init_pos_env() -> np.ndarray:
        if hasattr(cfg, "object_cfg"):
            pos = getattr(getattr(cfg, "object_cfg"), "init_state", None)
            if pos is not None and hasattr(pos, "pos"):
                return np.asarray(pos.pos, dtype=np.float64).ravel()[:3]
        if hasattr(cfg, "cup_cfg"):
            pos = getattr(getattr(cfg, "cup_cfg"), "init_state", None)
            if pos is not None and hasattr(pos, "pos"):
                return np.asarray(pos.pos, dtype=np.float64).ravel()[:3]
        return np.zeros(3, dtype=np.float64)

    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0
    last_arm_print: Optional[np.ndarray] = None
    last_hand_norm_print: Optional[np.ndarray] = None

    while simulation_app.is_running():
        t0 = time.time()
        step += 1
        if pending_manual_reset["flag"]:
            try:
                allow_reset_gate["allow"] = True
                env.reset()
                print("[INFO] Manual reset applied from GUI button.")
            finally:
                allow_reset_gate["allow"] = not bool(reset_mode_state["manual_only"])
                pending_manual_reset["flag"] = False

        pending_name = pending_keyframe_apply.get("name")
        if pending_name:
            keyframe = pose_keyframes.get(str(pending_name))
            if keyframe is None:
                _keyframe_status_set(f"Keyframe not found: {pending_name}")
            else:
                try:
                    marker_pos = np.asarray(keyframe["marker_pos_w"], dtype=np.float64).ravel()[:3]
                    marker_euler = np.asarray(keyframe["marker_euler_xyz"], dtype=np.float64).ravel()[:3]
                    T_apply = _make_T(marker_pos, marker_euler)
                    apply_pos = T_apply[:3, 3]
                    apply_quat_xyzw = R.from_matrix(T_apply[:3, :3]).as_quat()
                    apply_quat_wxyz = np.array(
                        [apply_quat_xyzw[3], apply_quat_xyzw[0], apply_quat_xyzw[1], apply_quat_xyzw[2]],
                        dtype=np.float32,
                    )
                    marker_xf.set_world_poses(
                        positions=torch.tensor([apply_pos], dtype=torch.float32, device="cpu"),
                        orientations=torch.tensor([apply_quat_wxyz], dtype=torch.float32, device="cpu"),
                    )
                    _set_hand_gui_pose(np.asarray(keyframe["hand_joint_pos_shadow_order"], dtype=np.float64).ravel())
                    initial_marker_pos[:] = apply_pos
                    initial_marker_quat_wxyz[:] = apply_quat_wxyz
                    hold_initial_pose_state["marker_moved"] = True
                    _keyframe_status_set(f"Applied keyframe: {pending_name}")
                    print(f"[INFO] Pose keyframe '{pending_name}' applied.")
                except Exception as e:
                    _keyframe_status_set(f"Apply failed: {pending_name}")
                    print(f"[WARN] Failed to apply pose keyframe '{pending_name}': {e}")
            pending_keyframe_apply["name"] = None

        pos_t, ori_t = marker_xf.get_world_poses()
        pos = _to_numpy(pos_t[0]).ravel()[:3]
        quat_wxyz = _to_numpy(ori_t[0]).ravel()[:4]
        if not hold_initial_pose_state["marker_moved"]:
            pos_delta = float(np.max(np.abs(pos - initial_marker_pos)))
            quat_delta = float(np.max(np.abs(quat_wxyz - initial_marker_quat_wxyz)))
            hold_initial_pose_state["marker_moved"] = (pos_delta > 1e-5) or (quat_delta > 1e-5)
        wrist_euler = _quat_wxyz_to_euler_xyz(quat_wxyz)

        hvec = _hand_vector_for_step()

        if hold_initial_pose_state["active"] and not hold_initial_pose_state["marker_moved"]:
            hold_arm = _arm_joints_from_sim()
            hold_hand = _hand_joints_shadow_from_sim() if args.hand_joints == "sim" else hvec
            actions = _build_action(hold_arm, hold_hand)
            targets = None
        else:
            targets: Optional[ArmHandTargets] = control.compute(pos, wrist_euler, hvec)
            if targets is None:
                if step % 60 == 0:
                    print("[WARN] IK failed for current marker pose.")
                actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            else:
                actions = _build_action(targets.arm_joints, targets.hand_joints)

        arm_for_norm = targets.arm_joints if targets is not None else None

        if args.print_every > 0 and step % int(args.print_every) == 0:
            if targets is not None:
                _print_arm_cfg_block(targets.arm_joints)
            _print_hand_action_block(hvec, arm_for_norm)
            if args.print_hand_rad:
                _print_hand_cfg_block(hvec)
        if args.print_on_change:
            if targets is not None and (
                last_arm_print is None or np.max(np.abs(targets.arm_joints - last_arm_print)) > 0.02
            ):
                _print_arm_cfg_block(targets.arm_joints)
                last_arm_print = targets.arm_joints.copy()
            cur_hand_norm = _hand_norm_shadow_order(
                _arm_joints_from_sim() if arm_for_norm is None else arm_for_norm,
                hvec,
            )
            if last_hand_norm_print is None or np.max(np.abs(cur_hand_norm - last_hand_norm_print)) > 0.02:
                _print_hand_action_block(hvec, arm_for_norm)
                if args.print_hand_rad:
                    _print_hand_cfg_block(hvec)
                last_hand_norm_print = cur_hand_norm.copy()

        if env.num_envs > 1:
            actions = actions.expand(env.num_envs, -1).clone()
        env.step(actions)

        if fig is not None and (rgb_ims or ff_ims):
            import matplotlib.pyplot as plt

            for i, name in enumerate(TACTILE_SENSOR_NAMES):
                if name not in _scene_env.scene.sensors:
                    continue
                try:
                    data = _scene_env.scene[name].data
                except RuntimeError as e:
                    if "Nominal tactile is not set" in str(e):
                        try:
                            _scene_env.scene[name].get_initial_render()
                            data = _scene_env.scene[name].data
                        except Exception:
                            continue
                    else:
                        raise

                if args.show_rgb and rgb_ims and i < len(rgb_ims):
                    img = getattr(data, "tactile_rgb_image", None)
                    if img is not None and img.ndim == 4:
                        e = min(env_idx, img.shape[0] - 1)
                        rgb_ims[i].set_data(_img_to_uint8(img[e].detach().cpu().numpy()))

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

        elapsed = time.time() - t0
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

        if args.max_steps > 0 and step >= int(args.max_steps):
            break

    if hand_gui_window is not None:
        try:
            hand_gui_window.visible = False
            hand_gui_window.destroy()
        except Exception:
            pass
    env.close()
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())