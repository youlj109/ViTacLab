#!/usr/bin/env python3
"""Tune UR10e arm joint targets from a visual marker pose + IK (no random arm actions).

The canonical implementation includes:
- save hand/arm yaml (`--save-yaml`, hotkey `s`)
- optional fixed hand (`--fixed-hand-yaml`, hotkey `f` lock/unlock current)
- optional stop-on-done (`--no-auto-reset`) to avoid continuing after env auto-reset
- optional export full_ik pickup yaml (`--save-full-ik-yaml`) for play_full_ik_single
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher


# Same presets as scripts/data_collection/manual/record_observations.py
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
}

MARKER_PRIM_PATH = "/World/Debug/ArmIkTarget"
_DEFAULT_FULL_IK_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "ik" / "configs" / "full_ik" / "full_ik_pickup_fixed_hand.yaml"
)
_DEFAULT_FULL_IK_OUT = (
    Path(__file__).resolve().parents[1] / "ik" / "configs" / "full_ik" / "full_ik_pickup_fixed_hand.yaml"
)

# Same as ``scripts/data_collection/manual/record_observations.py`` (five GelSight TacSL sensors).
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
    """Render tactile normal/shear arrays into RGB (aligned with ``record_observations.py``)."""
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
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _configure_viewer_window(fig: Any, *, topmost: bool) -> None:
    if fig is None:
        return
    try:
        manager = fig.canvas.manager
        win = getattr(manager, "window", None)
        if win is None:
            return
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
        if hasattr(win, "setWindowFlag"):
            try:
                import os as _os
                from PyQt5 import QtCore  # type: ignore

                win.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, bool(topmost))
                win.show()
                if not topmost and _os.name != "nt" and hasattr(win, "setAttribute"):
                    win.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            except Exception:
                pass
    except Exception:
        pass


def _install_terminal_cbreak() -> tuple[int, Any] | None:
    if os.name != "posix" or not sys.stdin.isatty():
        return None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old


def _restore_terminal(state: tuple[int, Any] | None) -> None:
    if state is None:
        return
    fd, old = state
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _poll_key_nonblocking() -> str | None:
    if os.name != "posix" or not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not r:
        return None
    try:
        return sys.stdin.read(1)
    except Exception:
        return None


def _load_hand_shadow_24(yaml_path: Path) -> np.ndarray:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    seq = data.get("hand_joint_pos_shadow_order")
    if not isinstance(seq, list) or len(seq) != 24:
        raise ValueError(f"{yaml_path}: need hand_joint_pos_shadow_order: [24 floats]")
    return np.array([float(x) for x in seq], dtype=np.float64)


def _save_yaml(path: Path, task: str, arm_dict: dict[str, float], hand_vec: np.ndarray, sh_names: list[str]) -> None:
    doc = {
        "task": task,
        "arm_joint_pos": dict(arm_dict),
        "hand_joint_pos_shadow_order": [float(hand_vec[i]) for i in range(24)],
        "hand_joint_pos_named": {sh_names[i]: float(hand_vec[i]) for i in range(24)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


def _save_full_ik_yaml_from_state(
    out_path: Path,
    template_path: Path,
    hand_yaml_rel: str,
    marker_pos_w: np.ndarray,
    marker_euler_xyz: np.ndarray,
    object_pos_w: np.ndarray,
) -> None:
    data = yaml.safe_load(template_path.read_text()) or {}
    phases = list(data.get("phase_schedule") or [])
    for i, ph in enumerate(phases):
        if str(ph.get("name", "")).strip() == "apply_grasp_hand":
            phases[i]["hand_yaml"] = hand_yaml_rel
    data["phase_schedule"] = phases
    data["palm_orient"] = "fixed"
    data["palm_euler"] = [float(v) for v in np.asarray(marker_euler_xyz, dtype=np.float64).ravel()[:3]]
    data["palm_euler_in_anchor"] = [0.0, 0.0, 0.0]

    pwp = np.array(data.get("palm_in_wrist_pos", [0.0, 0.0, 0.35]), dtype=np.float64).ravel()[:3]
    pwe = np.array(data.get("palm_in_wrist_euler", [1.5707963267948966, -1.5707963267948966, 1.5707963267948966]), dtype=np.float64).ravel()[:3]
    T_world_wrist = _make_T(np.asarray(marker_pos_w, dtype=np.float64).ravel()[:3], np.asarray(marker_euler_xyz, dtype=np.float64).ravel()[:3])
    T_wrist_palm = _make_T(pwp, pwe)
    T_world_palm = T_world_wrist @ T_wrist_palm
    off = T_world_palm[:3, 3] - np.asarray(object_pos_w, dtype=np.float64).ravel()[:3]
    data["object_to_palm_offset"] = [float(off[0]), float(off[1]), float(off[2])]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UR10e arm IK from a visual marker.")
    p.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="pickup", help="Preset task.")
    p.add_argument("--env", type=str, default="", help="Env entry module:Class (overrides --task).")
    p.add_argument("--cfg", type=str, default="", help="Cfg entry module:Class (overrides --task).")
    p.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    p.add_argument("--fps", type=float, default=30.0, help="Simulation loop target FPS.")
    p.add_argument("--marker-pos", type=float, nargs=3, default=(0.65, 0.12, 0.42), metavar=("X", "Y", "Z"), help='Initial single-arm IK marker world position X Y Z in meters.')
    p.add_argument("--marker-euler", type=float, nargs=3, default=(0.0, 2.2, 0.0), metavar=("RX", "RY", "RZ"), help='Initial single-arm IK marker XYZ Euler orientation in radians.')
    p.add_argument("--arm-base-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0), help='Robot arm-base world position as X Y Z in meters.')
    p.add_argument("--arm-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0), help='Robot arm-base XYZ Euler orientation in radians.')
    p.add_argument("--hand-joints", choices=["zeros", "sim"], default="sim", help='Initial hand source: zeros or current simulator joint values.')
    _hg = p.add_mutually_exclusive_group()
    _hg.add_argument("--hand-gui", dest="hand_gui", action="store_true", help='Enable the interactive Shadow Hand joint-control window.')
    _hg.add_argument("--no-hand-gui", dest="hand_gui", action="store_false", help='Disable the interactive Shadow Hand joint-control window.')
    p.set_defaults(hand_gui=True)
    p.add_argument("--print-every", type=int, default=30, help='Print current arm/hand state every N steps; 0 disables periodic output.')
    p.add_argument("--print-on-change", action="store_true", help='Print state whenever a commanded joint target changes.')
    p.add_argument("--print-hand-rad", action="store_true", help='Include the 24 Shadow Hand joint values in radians in console output.')
    p.add_argument("--max-steps", type=int, default=0, help='Maximum control-loop steps; 0 runs until quit or environment termination.')
    p.add_argument("--show_rgb", action="store_true", help='Open the live tactile RGB viewer and enable cameras.')
    p.add_argument("--show_ff", action="store_true", help='Open the live tactile force-field viewer and enable cameras.')
    p.add_argument("--env-index", type=int, default=0, help='Parallel environment index displayed in the tactile viewer.')
    p.add_argument("--viewer-topmost", action="store_true", help='Keep the tactile visualization window above other windows.')
    # Canonical extra controls
    p.add_argument("--save-yaml", type=str, default="", help="Save hand/arm YAML on hotkey [s].")
    p.add_argument("--fixed-hand-yaml", type=str, default="", help="Lock hand vector from YAML.")
    p.add_argument("--disable-hotkeys", action="store_true", help='Disable keyboard shortcuts and require normal process termination.')
    p.add_argument("--no-auto-reset", action="store_true", help="Stop loop when done/timeout appears.")
    p.add_argument("--save-full-ik-yaml", type=str, default=str(_DEFAULT_FULL_IK_OUT), help="Export full_ik config on [s] for pickup.")
    p.add_argument("--full-ik-template", type=str, default=str(_DEFAULT_FULL_IK_TEMPLATE), help="Template full_ik yaml.")
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

    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import ArmHandTargets, VideoTeleopControl

    preset = _TASK_PRESETS[str(args.task)]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]
    if (args.show_rgb or args.show_ff) and str(args.task) == "inhand" and not str(args.cfg).strip():
        cfg_entry = _INHAND_TACTILE_CFG
        print(f"[INFO] Using tactile cfg for inhand: {cfg_entry}")
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    fig = None
    ax_rgb = None
    ax_ff = None
    rgb_ims: list[Any] = []
    ff_ims: list[Any] = []
    nrows, ncols = 20, 25

    if args.show_rgb or args.show_ff:
        import matplotlib

        matplotlib.rcParams["figure.raise_window"] = False
        matplotlib.rcParams["toolbar"] = "None"
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

    env.reset()
    action_dim = env.num_actions
    print(f"[INFO] action_dim={action_dim}, actuated_dof_indices count={len(env.actuated_dof_indices)}")

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
        _configure_viewer_window(fig, topmost=bool(args.viewer_topmost))
        plt.pause(0.1)

    render_ff = _render_tactile_ff_rgb if args.show_ff else None
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))

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
        return np.clip(scale, -1.0, 1.0)

    def _build_action(arm_joints: np.ndarray, hand_joints: np.ndarray) -> torch.Tensor:
        scale = _build_action_numpy(arm_joints, hand_joints)
        return torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)

    def _print_arm_cfg_block(arm_j: np.ndarray) -> None:
        print("[INFO] Arm joint_pos snippet for ArticulationCfg.init_state (rad):")
        print("    joint_pos={")
        for i, idx in enumerate(arm_indices):
            if i < len(arm_j):
                print(f'        "{joint_names[idx]}": {float(arm_j[i]):.16f},')
        print("    },")

    def _print_hand24_rad(hand_j: np.ndarray) -> None:
        print("[INFO] hand_joint_pos_shadow_order (24 rad):")
        print(f"    {np.asarray(hand_j, dtype=np.float64).tolist()}")

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
                        for sh_i in range(24):
                            m = omni.ui.SimpleFloatModel()
                            _float_model_set(m, float(init_h[sh_i]))
                            hand_gui_models.append(m)
                            with omni.ui.HStack(spacing=6):
                                omni.ui.Label(f"{sh_names[sh_i]}", width=56, alignment=omni.ui.Alignment.LEFT_CENTER)
                                omni.ui.FloatSlider(model=m, min=-1.8, max=1.8, step=0.002)
                                omni.ui.FloatField(m, width=72)
        except Exception as e:
            hand_gui_models = []
            hand_gui_window = None
            print(f"[WARN] Could not build hand GUI ({e}); use --hand-joints sim/zeros without sliders.")

    fixed_hand: Optional[np.ndarray] = None
    if str(args.fixed_hand_yaml).strip():
        p = Path(args.fixed_hand_yaml).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        fixed_hand = _load_hand_shadow_24(p)
        print(f"[INFO] fixed hand loaded from {p}")

    def _hand_vector_for_step() -> np.ndarray:
        if fixed_hand is not None:
            return fixed_hand.copy()
        if hand_gui_models and len(hand_gui_models) == 24:
            return np.array([_float_model_get(m) for m in hand_gui_models], dtype=np.float64)
        if args.hand_joints == "zeros":
            return np.zeros(24, dtype=np.float64)
        return _hand_joints_shadow_from_sim()

    def _save_now(arm_j: np.ndarray, hand_j: np.ndarray) -> None:
        save_arg = str(args.save_yaml).strip()
        if not save_arg:
            print("[WARN] --save-yaml not set; skip save.")
            return
        save_path = Path(save_arg).expanduser()
        if not save_path.is_absolute():
            save_path = (repo_root / save_path).resolve()
        arm_dict = {joint_names[idx]: float(arm_j[i]) for i, idx in enumerate(arm_indices) if i < len(arm_j)}
        _save_yaml(save_path, str(args.task), arm_dict, hand_j, list(sh_names))
        print(f"[INFO] Saved hand+arm YAML -> {save_path}")

        if str(args.task) == "pickup" and hasattr(env, "object"):
            out_cfg = Path(str(args.save_full_ik_yaml).strip() or str(_DEFAULT_FULL_IK_OUT)).expanduser()
            if not out_cfg.is_absolute():
                out_cfg = (repo_root / out_cfg).resolve()
            tpl = Path(str(args.full_ik_template).strip() or str(_DEFAULT_FULL_IK_TEMPLATE)).expanduser()
            if not tpl.is_absolute():
                tpl = (repo_root / tpl).resolve()
            pos_t, ori_t = marker_xf.get_world_poses()
            marker_pos = _to_numpy(pos_t[0]).ravel()[:3]
            marker_euler = _quat_wxyz_to_euler_xyz(_to_numpy(ori_t[0]).ravel()[:4])
            obj_pos = _to_numpy(env.object.data.root_pos_w[0]).ravel()[:3]
            hand_rel = str(save_path)
            try:
                hand_rel = str(save_path.relative_to(repo_root))
            except Exception:
                pass
            _save_full_ik_yaml_from_state(out_cfg, tpl, hand_rel, marker_pos, marker_euler, obj_pos)
            print(f"[INFO] Saved full_ik pickup YAML -> {out_cfg}")

    print("[INFO] Hotkeys: [p]=print hand24, [s]=save yaml(+full_ik), [f]=lock/unlock hand, [q]=quit")
    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0
    last_arm_print: Optional[np.ndarray] = None
    last_hand_print: Optional[np.ndarray] = None
    term_state = None

    try:
        if not bool(args.disable_hotkeys):
            term_state = _install_terminal_cbreak()
        while simulation_app.is_running():
            t0 = time.time()
            step += 1
            key = _poll_key_nonblocking() if not bool(args.disable_hotkeys) else None

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

            arm_for_print = _arm_joints_from_sim() if targets is None else np.asarray(targets.arm_joints, dtype=np.float64)
            if args.print_every > 0 and step % int(args.print_every) == 0:
                if targets is not None:
                    _print_arm_cfg_block(targets.arm_joints)
                _print_hand24_rad(hvec)
            if args.print_on_change:
                if targets is not None and (last_arm_print is None or np.max(np.abs(targets.arm_joints - last_arm_print)) > 0.02):
                    _print_arm_cfg_block(targets.arm_joints)
                    last_arm_print = targets.arm_joints.copy()
                if last_hand_print is None or np.max(np.abs(hvec - last_hand_print)) > 0.02:
                    _print_hand24_rad(hvec)
                    last_hand_print = hvec.copy()

            if env.num_envs > 1:
                actions = actions.expand(env.num_envs, -1).clone()
            _, _, terminated, truncated, _ = env.step(actions)
            done_any = bool(torch.any(terminated | truncated).item()) if torch.is_tensor(terminated) else False

            if key:
                k = key.lower()
                if k == "q":
                    break
                if k == "p":
                    if targets is not None:
                        _print_arm_cfg_block(targets.arm_joints)
                    _print_hand24_rad(hvec)
                elif k == "s":
                    _save_now(arm_for_print, hvec)
                elif k == "f":
                    if fixed_hand is None:
                        fixed_hand = hvec.copy()
                        print("[INFO] fixed hand locked.")
                    else:
                        fixed_hand = None
                        print("[INFO] fixed hand unlocked.")

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
                            nr, nc = nrows, ncols
                            if p != nr * nc:
                                nr = int(np.sqrt(p))
                                nc = max(1, p // max(1, nr))
                            nf_img = nf_flat.reshape(nr, nc)
                            sf_img = sf_flat.reshape(nr, nc, 2)
                            ff_ims[i].set_data(render_ff(nf_img, sf_img))
                fig.canvas.draw_idle()
                plt.pause(0.001)

            if bool(args.no_auto_reset) and done_any and step % 60 == 0:
                print("[INFO] done/timeout detected; keep running (no auto-exit).")
            elapsed = time.time() - t0
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)
            if args.max_steps > 0 and step >= int(args.max_steps):
                break
    finally:
        _restore_terminal(term_state)
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

