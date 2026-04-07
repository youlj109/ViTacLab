#!/usr/bin/env python3
"""Dual UR10e + Shadow Hand: tune each arm from a visual marker + IK (same idea as single-arm script).

Spawns two ``VisualCuboid`` markers under ``/World/Debug/ArmIkTargetRight`` and
``/World/Debug/ArmIkTargetLeft``. Move/orient each in the viewport; each step runs
:class:`VideoTeleopControl` with **T_world_arm_base** taken from the corresponding
articulation root pose in simulation, applies IK arm targets via
``set_joint_position_target`` (arm joints only), then :meth:`UR10eDualShadowHandOverEnv.step`
with hand-only actions (24 + 24 DoF), matching training.

Use this to copy printed ``joint_pos`` blocks into ``hand_over_env_cfg`` (or dual base cfg).

Examples (Isaac Sim python from ViTacLab repo root):

    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_dual_shadowhand_arm_pose_from_marker.py \\
        --num_envs 1 --enable_cameras

    ./python.sh scripts/teleoperation/gui_teleop/run_ur10e_dual_shadowhand_arm_pose_from_marker.py \\
        --marker-right-pos 0.55 1.0 0.65 --marker-left-pos 0.55 1.3 0.65
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

# Default env: dual-arm hand-over (hand-only policy in RL; this script drives arm via IK).
_DEFAULT_ENV = "ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env:UR10eDualShadowHandOverEnv"
_DEFAULT_CFG = "ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env_cfg:UR10eDualShadowHandOverEnvCfg"

MARKER_RIGHT_PRIM_PATH = "/World/Debug/ArmIkTargetRight"
MARKER_LEFT_PRIM_PATH = "/World/Debug/ArmIkTargetLeft"

# TacSL keys in dual base env: ``{left_,right_}tactile_sensor_*``
TACTILE_SENSOR_SUFFIXES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
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
    p = Path(__file__).resolve().parent
    for _ in range(12):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[4]


def _root_T_world_np(robot, env_idx: int) -> np.ndarray:
    """World pose of articulation root (4x4), base link frame in world."""
    pos = robot.data.root_pos_w[int(env_idx)].detach().cpu().numpy().ravel()[:3]
    q = robot.data.root_quat_w[int(env_idx)].detach().cpu().numpy().ravel()[:4]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_matrix()
    T[:3, 3] = pos
    return T


def _urdf_path(repo_root: Path, side: str) -> str:
    side = side.lower()
    if side not in ("left", "right"):
        raise ValueError(side)
    return str(
        repo_root
        / "source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e"
        / f"ur10e_shadow_{side}_hand_glb.urdf"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dual UR10e + Shadow Hand: two markers -> IK -> arm targets; hand via MARL actions.",
    )
    p.add_argument("--env", type=str, default=_DEFAULT_ENV, help="Env entry module:Class.")
    p.add_argument("--cfg", type=str, default=_DEFAULT_CFG, help="Cfg entry module:Class.")
    p.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    p.add_argument("--fps", type=float, default=30.0, help="Simulation loop target FPS.")
    p.add_argument(
        "--marker-right-pos",
        type=float,
        nargs=3,
        default=(0.55, 1.0, 0.62),
        metavar=("X", "Y", "Z"),
        help="Initial right marker position (world, m).",
    )
    p.add_argument(
        "--marker-right-euler",
        type=float,
        nargs=3,
        default=(0.0, 2.2, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Initial right marker euler xyz (rad).",
    )
    p.add_argument(
        "--marker-left-pos",
        type=float,
        nargs=3,
        default=(0.55, 1.3, 0.62),
        metavar=("X", "Y", "Z"),
        help="Initial left marker position (world, m).",
    )
    p.add_argument(
        "--marker-left-euler",
        type=float,
        nargs=3,
        default=(0.0, 2.2, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Initial left marker euler xyz (rad).",
    )
    _hg = p.add_mutually_exclusive_group()
    _hg.add_argument("--hand-gui", dest="hand_gui", action="store_true", help="Show 48 hand sliders (default).")
    _hg.add_argument("--no-hand-gui", dest="hand_gui", action="store_false", help="No sliders; use --hand-joints.")
    p.set_defaults(hand_gui=True)
    p.add_argument(
        "--hand-joints",
        choices=["zeros", "sim"],
        default="sim",
        help="When GUI off: initial/fallback hand vector per arm.",
    )
    p.add_argument("--print-every", type=int, default=30, help="Print arm joint dicts every N steps (0=off).")
    p.add_argument(
        "--print-on-change",
        action="store_true",
        help="Print when IK arm joints change (thresholded).",
    )
    p.add_argument("--print-hand-rad", action="store_true", help="Also print hand joint_pos (rad) blocks.")
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = run until close).")
    p.add_argument("--show_rgb", action="store_true", help="Matplotlib GelSight RGB (needs --enable_cameras).")
    p.add_argument("--show_ff", action="store_true", help="Matplotlib tactile FF RGB (needs --enable_cameras).")
    p.add_argument(
        "--tactile-arm",
        choices=("right", "left"),
        default="right",
        help="Which arm's TacSL sensors to display (scene keys are right_/left_ prefixed).",
    )
    p.add_argument("--env-index", type=int, default=0, help="Env index for tactile display.")
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

    from video_teleop.core.video_teleop_control import VideoTeleopControl
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names

    EnvCls = _load_symbol(str(args.env).strip())
    CfgCls = _load_symbol(str(args.cfg).strip())

    tactile_prefix = f"{args.tactile_arm}_"

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
    right = env.right_hand
    left = env.left_hand
    joint_names = list(right.joint_names)
    if list(left.joint_names) != joint_names:
        print("[WARN] Left/right joint_names differ; using right_hand names for both.")

    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    arm_indices.sort()
    hand_indices.sort()
    sh_names = shadowhand_joint_names()

    env.reset()
    print(
        f"[INFO] arm_indices={len(arm_indices)} hand_indices={len(hand_indices)} "
        f"actuated (env)={len(env.actuated_dof_indices)}",
    )

    _scene_env = env
    if args.show_ff and fig is not None:
        k0 = f"{tactile_prefix}{TACTILE_SENSOR_SUFFIXES[0]}"
        if k0 in _scene_env.scene.sensors:
            try:
                nrows, ncols = _scene_env.scene[k0].cfg.tactile_array_size
            except Exception:
                pass

    if fig is not None:
        import matplotlib.pyplot as plt

        if args.show_rgb and ax_rgb is not None:
            zero_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
            axes_rgb = ax_rgb if isinstance(ax_rgb, np.ndarray) else [ax_rgb]
            for i, suf in enumerate(TACTILE_SENSOR_SUFFIXES):
                if i >= len(axes_rgb):
                    break
                title = suf.replace("tactile_sensor_", "").upper()
                im = axes_rgb[i].imshow(zero_rgb)
                axes_rgb[i].set_title(f"{args.tactile_arm.upper()} {title} RGB")
                axes_rgb[i].axis("off")
                rgb_ims.append(im)

        if args.show_ff and ax_ff is not None:
            zero_ff = np.zeros((nrows * 30, ncols * 30, 3), dtype=np.uint8)
            axes_ff = ax_ff if isinstance(ax_ff, np.ndarray) else [ax_ff]
            for i, suf in enumerate(TACTILE_SENSOR_SUFFIXES):
                if i >= len(axes_ff):
                    break
                title = suf.replace("tactile_sensor_", "").upper()
                im = axes_ff[i].imshow(zero_ff)
                axes_ff[i].set_title(f"{args.tactile_arm.upper()} {title} FF")
                axes_ff[i].axis("off")
                ff_ims.append(im)

        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)

    render_ff = _render_tactile_ff_rgb if args.show_ff else None
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))

    # IK: one controller per arm (matching L/R URDF).
    T0_r = _root_T_world_np(right, env_idx)
    T0_l = _root_T_world_np(left, env_idx)
    control_right = VideoTeleopControl(urdf_path=_urdf_path(repo_root, "right"), T_world_arm_base=T0_r)
    control_left = VideoTeleopControl(urdf_path=_urdf_path(repo_root, "left"), T_world_arm_base=T0_l)

    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.prims import XFormPrim

    def _spawn_marker(path: str, pos: tuple[float, float, float], euler: tuple[float, float, float], color: np.ndarray):
        VisualCuboid(
            prim_path=path,
            size=0.04,
            position=np.array(pos, dtype=np.float64),
            visible=True,
            color=color,
        )
        xf = XFormPrim(prim_paths_expr=path, name=path.split("/")[-1], usd=True)
        T0 = _make_T(np.array(pos, dtype=np.float64), np.array(euler, dtype=np.float64))
        pos0 = T0[:3, 3]
        quat_xyzw = R.from_matrix(T0[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        xf.set_world_poses(
            positions=torch.tensor([pos0], dtype=torch.float32, device="cpu"),
            orientations=torch.tensor([quat_wxyz], dtype=torch.float32, device="cpu"),
        )
        return xf

    marker_right_xf = _spawn_marker(
        MARKER_RIGHT_PRIM_PATH,
        tuple(args.marker_right_pos),
        tuple(args.marker_right_euler),
        np.array([1.0, 0.2, 0.8]),
    )
    marker_left_xf = _spawn_marker(
        MARKER_LEFT_PRIM_PATH,
        tuple(args.marker_left_pos),
        tuple(args.marker_left_euler),
        np.array([0.2, 1.0, 0.8]),
    )

    print(f"[INFO] Move '{MARKER_RIGHT_PRIM_PATH}' (magenta-ish) and '{MARKER_LEFT_PRIM_PATH}' (cyan-ish) in the viewport.")

    def _hand_joints_from_robot(robot: Any, env_i: int) -> np.ndarray:
        out = np.zeros(24, dtype=np.float64)
        jpos = robot.data.joint_pos[int(env_i)].detach().cpu().numpy()
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

    def _hand_actions_from_rad(hand_rad: np.ndarray) -> np.ndarray:
        """Single arm: (24,) joint radians -> (24,) normalized [-1,1] using env hand limits."""
        hand_rad = np.asarray(hand_rad, dtype=np.float64).ravel()
        lower = env.hand_dof_lower_limits[int(env_idx)].detach().cpu().numpy()
        upper = env.hand_dof_upper_limits[int(env_idx)].detach().cpu().numpy()
        eps = 1e-6
        out = np.where(upper - lower > eps, 2.0 * (hand_rad - lower) / (upper - lower) - 1.0, 0.0)
        return np.clip(out, -1.0, 1.0)

    def _print_arm_block(label: str, arm_j: np.ndarray) -> None:
        print(f"[INFO] {label} arm joint_pos (rad) for cfg:")
        print("    joint_pos={")
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

    def _print_hand_rad_block(which: str, hand_j: np.ndarray) -> None:
        print(f"[INFO] {which} hand joint_pos (rad):")
        print("    joint_pos={")
        for i, sh_name in enumerate(sh_names):
            if i < len(hand_j):
                print(f'        "{_robot_joint_name_for_shadow(sh_name)}": {float(hand_j[i]):.16f},')
        print("    },")

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

    hand_gui_models_r: list[Any] = []
    hand_gui_models_l: list[Any] = []
    hand_gui_window: Any = None
    if args.hand_gui:

        def _init_vec(which: str) -> np.ndarray:
            if args.hand_joints == "zeros":
                return np.zeros(24, dtype=np.float64)
            rob = right if which == "right" else left
            return _hand_joints_from_robot(rob, env_idx)

        try:
            hand_gui_window = omni.ui.Window(
                "Dual Shadow Hand Joints (rad)",
                width=480,
                height=900,
                visible=True,
                dock_preference=omni.ui.DockPreference.RIGHT_TOP,
            )
            with hand_gui_window.frame:
                with omni.ui.ScrollingFrame(horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                    with omni.ui.VStack(spacing=3, height=0):
                        omni.ui.Label(
                            "Right arm: first 24 sliders. Left arm: next 24. Values feed env.step hand actions.",
                            word_wrap=True,
                        )
                        omni.ui.Label("— Right hand —", style={"font_size": 14})
                        init_r = _init_vec("right")
                        for sh_i in range(24):
                            lo, hi = _shadow_limits_rad(sh_i)
                            mid = float(np.clip(init_r[sh_i], lo, hi))
                            m = omni.ui.SimpleFloatModel()
                            _float_model_set(m, mid)
                            hand_gui_models_r.append(m)
                            with omni.ui.HStack(spacing=6):
                                omni.ui.Label(f"R {sh_names[sh_i]}", width=72, alignment=omni.ui.Alignment.LEFT_CENTER)
                                omni.ui.FloatSlider(
                                    model=m,
                                    min=lo,
                                    max=hi,
                                    step=max(1e-4, (hi - lo) / 500.0),
                                )
                        omni.ui.Label("— Left hand —", style={"font_size": 14})
                        init_l = _init_vec("left")
                        for sh_i in range(24):
                            lo, hi = _shadow_limits_rad(sh_i)
                            mid = float(np.clip(init_l[sh_i], lo, hi))
                            m = omni.ui.SimpleFloatModel()
                            _float_model_set(m, mid)
                            hand_gui_models_l.append(m)
                            with omni.ui.HStack(spacing=6):
                                omni.ui.Label(f"L {sh_names[sh_i]}", width=72, alignment=omni.ui.Alignment.LEFT_CENTER)
                                omni.ui.FloatSlider(
                                    model=m,
                                    min=lo,
                                    max=hi,
                                    step=max(1e-4, (hi - lo) / 500.0),
                                )
            print("[INFO] Opened dual hand joint panel (48 sliders).")
        except Exception as e:
            hand_gui_models_r = []
            hand_gui_models_l = []
            hand_gui_window = None
            print(f"[WARN] Could not build hand GUI ({e}); use --no-hand-gui --hand-joints sim/zeros.")

    def _hand_vector_pair() -> tuple[np.ndarray, np.ndarray]:
        if hand_gui_models_r and hand_gui_models_l and len(hand_gui_models_r) == 24 and len(hand_gui_models_l) == 24:
            hr = np.array([_float_model_get(m) for m in hand_gui_models_r], dtype=np.float64)
            hl = np.array([_float_model_get(m) for m in hand_gui_models_l], dtype=np.float64)
            return hr, hl
        if args.hand_joints == "zeros":
            z = np.zeros(24, dtype=np.float64)
            return z, z.copy()
        return _hand_joints_from_robot(right, env_idx), _hand_joints_from_robot(left, env_idx)

    def _set_arm_ik(robot: Any, arm_joints: np.ndarray) -> None:
        """Apply 6 UR10e joint targets (hand joints unchanged; env.step sets hand afterward)."""
        aj = np.asarray(arm_joints, dtype=np.float64).ravel()
        if len(arm_indices) != 6 or len(aj) < 6:
            return
        vals = torch.tensor(aj[:6], dtype=torch.float32, device=robot.device).unsqueeze(0).expand(env.num_envs, -1)
        robot.set_joint_position_target(vals, joint_ids=arm_indices)

    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0
    last_arm_r: Optional[np.ndarray] = None
    last_arm_l: Optional[np.ndarray] = None

    while simulation_app.is_running():
        t0 = time.time()
        step += 1

        # Refresh base frames from sim (roots are fixed but keeps numerics consistent).
        control_right.T_world_arm_base = _root_T_world_np(right, env_idx)
        control_left.T_world_arm_base = _root_T_world_np(left, env_idx)

        pos_r, ori_r = marker_right_xf.get_world_poses()
        pos_rw = _to_numpy(pos_r[0]).ravel()[:3]
        euler_r = _quat_wxyz_to_euler_xyz(_to_numpy(ori_r[0]).ravel()[:4])

        pos_l, ori_l = marker_left_xf.get_world_poses()
        pos_lw = _to_numpy(pos_l[0]).ravel()[:3]
        euler_l = _quat_wxyz_to_euler_xyz(_to_numpy(ori_l[0]).ravel()[:4])

        h_r, h_l = _hand_vector_pair()

        tgt_r = control_right.compute(pos_rw, euler_r, h_r)
        tgt_l = control_left.compute(pos_lw, euler_l, h_l)

        if tgt_r is not None:
            _set_arm_ik(right, tgt_r.arm_joints)
        if tgt_l is not None:
            _set_arm_ik(left, tgt_l.arm_joints)

        ar = _hand_actions_from_rad(h_r)
        al = _hand_actions_from_rad(h_l)
        actions = {
            "right_hand": torch.tensor(ar, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1),
            "left_hand": torch.tensor(al, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1),
        }

        if args.print_every > 0 and step % int(args.print_every) == 0:
            if tgt_r is not None:
                _print_arm_block("Right", tgt_r.arm_joints)
            else:
                print("[WARN] Right IK failed; not printing right arm dict.")
            if tgt_l is not None:
                _print_arm_block("Left", tgt_l.arm_joints)
            else:
                print("[WARN] Left IK failed; not printing left arm dict.")
            if args.print_hand_rad:
                _print_hand_rad_block("Right", h_r)
                _print_hand_rad_block("Left", h_l)

        if args.print_on_change:
            if tgt_r is not None and (last_arm_r is None or np.max(np.abs(tgt_r.arm_joints - last_arm_r)) > 0.02):
                _print_arm_block("Right", tgt_r.arm_joints)
                last_arm_r = tgt_r.arm_joints.copy()
            if tgt_l is not None and (last_arm_l is None or np.max(np.abs(tgt_l.arm_joints - last_arm_l)) > 0.02):
                _print_arm_block("Left", tgt_l.arm_joints)
                last_arm_l = tgt_l.arm_joints.copy()

        env.step(actions)

        if fig is not None and (rgb_ims or ff_ims):
            import matplotlib.pyplot as plt

            for i, suf in enumerate(TACTILE_SENSOR_SUFFIXES):
                key = f"{tactile_prefix}{suf}"
                if key not in _scene_env.scene.sensors:
                    continue
                data = _scene_env.scene[key].data

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

        if step % 120 == 0 and (tgt_r is None or tgt_l is None):
            miss = []
            if tgt_r is None:
                miss.append("right")
            if tgt_l is None:
                miss.append("left")
            print(f"[WARN] IK failed for: {', '.join(miss)}")

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
