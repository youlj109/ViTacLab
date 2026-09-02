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

    # Factory forge peg (or ``--task Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0``):
    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_shadowhand_arm_pose_from_marker.py \\
        --task forge_peg --num_envs 1 --enable_cameras

    # Any registered single-arm task (reads ``entry_point`` + ``env_cfg_entry_point`` from Gym):
    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_shadowhand_arm_pose_from_marker.py \\
        --task Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 --num_envs 1 --enable_cameras
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher


# Same presets as scripts/debug/run_ur10e_shadowhand_single.py
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
        "cfg": "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandEnvCfg",
    },
    "forge_peg": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgePegInsertEnvCfg",
    },
    "forge_gear": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeGearMeshEnvCfg",
    },
    "forge_nut": {
        "env": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv",
        "cfg": "ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeNutThreadEnvCfg",
    },
}

# Registered Gym ids from ``ViTacLab.tasks.direct.medium_dexhand.forge_dexhand`` → preset key.
_TASK_GYM_ID_ALIASES: dict[str, str] = {
    "Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0": "forge_peg",
    "Isaac-UR10eShadowHand-ForgeGearMesh-Direct-v0": "forge_gear",
    "Isaac-UR10eShadowHand-ForgeNutThread-Direct-v0": "forge_nut",
}

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


def _render_tactile_ff_rgb(nf: np.ndarray, sf: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Render tactile normal/shear arrays into RGB (aligned with ``run_ur10e_shadowhand_single.py``)."""
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


def _entries_from_gym_registry(task_id: str) -> tuple[str, str]:
    """Resolve ``module:EnvClass`` and ``env_cfg_entry_point`` from a registered Gymnasium id."""

    import gymnasium as gym

    tid = task_id.split(":")[-1].strip()
    spec = gym.spec(tid)
    ep = spec.entry_point
    if callable(ep):
        env_entry = f"{ep.__module__}:{ep.__name__}"
    else:
        env_entry = str(ep)
    kwargs = spec.kwargs or {}
    cfg_ep = kwargs.get("env_cfg_entry_point")
    if not cfg_ep:
        raise ValueError(f"Registry task {tid!r} has no env_cfg_entry_point in spec.kwargs.")
    return env_entry, cfg_ep


def _resolve_env_cfg_entries(args: argparse.Namespace) -> tuple[str, str, str | None]:
    """Resolve env/cfg entry strings.

    Returns ``(env_entry, cfg_entry, preset_key)`` where ``preset_key`` is set when using a
    built-in preset name (after forge aliases), else ``None`` (e.g. arbitrary registered Gym id).
    """

    env_s = str(getattr(args, "env", "") or "").strip()
    cfg_s = str(getattr(args, "cfg", "") or "").strip()
    if env_s and cfg_s:
        return env_s, cfg_s, None
    if env_s or cfg_s:
        raise SystemExit("Provide both --env and --cfg, or neither and use --task.")

    task = str(getattr(args, "task", "") or "").strip()
    if task in _TASK_GYM_ID_ALIASES:
        task = _TASK_GYM_ID_ALIASES[task]
    if task in _TASK_PRESETS:
        p = _TASK_PRESETS[task]
        return p["env"], p["cfg"], task
    try:
        e, c = _entries_from_gym_registry(task)
        return e, c, None
    except Exception as exc:
        keys = ", ".join(sorted(_TASK_PRESETS.keys()))
        aliases = ", ".join(sorted(_TASK_GYM_ID_ALIASES.keys()))
        raise SystemExit(
            f"Unknown --task {task!r}. Use one of: {keys}; a forge Gym id alias ({aliases}); "
            f"or any registered Gymnasium id whose spec provides env_cfg_entry_point "
            f"(e.g. Isaac-UR10eShadowHand-BlindGrasp-Direct-v0). ({exc})"
        ) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR10e arm IK from a visual marker (same task presets as run_ur10e_shadowhand_single.py).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="pickup",
        help=(
            "Preset key (pickup, pour, inhand, forge_peg, forge_gear, forge_nut), a forge Gym id alias, "
            "or any registered ViTacLab Gymnasium id (resolved via env_cfg_entry_point), "
            "e.g. Isaac-UR10eShadowHand-BlindGrasp-Direct-v0."
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
        default=(0.65, 0.12, 0.42),
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
        help="Live matplotlib: tactile force-field RGB (needs FF enabled on sensors).",
    )
    p.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Which env index to read for tactile display (default: 0).",
    )
    AppLauncher.add_app_launcher_args(p)
    return p


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

    env_entry, cfg_entry, preset_key = _resolve_env_cfg_entries(args)
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

    print(
        "[INFO] env.reset(): starting (Forge uses joint-space reset only; "
        "first reset can still take a few seconds with cameras/TacSL — not frozen)."
    )
    env.reset()
    print("[INFO] env.reset(): done.")
    action_dim = env.num_actions
    print(f"[INFO] action_dim={action_dim}, actuated_dof_indices count={len(env.actuated_dof_indices)}")
    if args.show_rgb or args.show_ff:
        print(
            "[INFO] Tactile viewer: move marker / hand; requires scene.sensors tactile_* "
            "(use --enable_cameras; inhand defaults to tactile cfg unless --cfg is set)."
        )

    _scene_env = env
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

    render_ff = _render_tactile_ff_rgb if args.show_ff else None
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

    import omni.ui  # type: ignore

    hand_gui_models: list[Any] = []
    hand_gui_window: Any = None
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
            print("[INFO] Opened 'Shadow Hand Joints (rad)' panel; sliders control the 24-DoF hand.")
        except Exception as e:
            hand_gui_models = []
            hand_gui_window = None
            print(f"[WARN] Could not build hand GUI ({e}); use --hand-joints sim/zeros without sliders.")

    def _hand_vector_for_step() -> np.ndarray:
        if hand_gui_models and len(hand_gui_models) == 24:
            return np.array([_float_model_get(m) for m in hand_gui_models], dtype=np.float64)
        if args.hand_joints == "zeros":
            return np.zeros(24, dtype=np.float64)
        return _hand_joints_shadow_from_sim()

    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0
    last_arm_print: Optional[np.ndarray] = None
    last_hand_norm_print: Optional[np.ndarray] = None

    print(
        "[INFO] Main loop: first env.step may take 30–60s with TacSL + RTX (shader warmup). "
        "ManagerLiveVisualizer warnings for action/observation_manager are normal for DirectRLEnv."
    )

    while simulation_app.is_running():
        t0 = time.time()
        step += 1

        pos_t, ori_t = marker_xf.get_world_poses()
        pos = _to_numpy(pos_t[0]).ravel()[:3]
        quat_wxyz = _to_numpy(ori_t[0]).ravel()[:4]
        wrist_euler = _quat_wxyz_to_euler_xyz(quat_wxyz)

        hvec = _hand_vector_for_step()

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
                data = _scene_env.scene[name].data

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
