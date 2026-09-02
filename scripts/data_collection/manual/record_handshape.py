#!/usr/bin/env python3
"""Canonical interactive wrist, handshape, and tactile-visualization recorder.

Provides real-time tactile visualization while controlling handshape (the
matplotlib path is aligned with ``record_arm_pose.py``; default ``--viewer matplotlib``):
- --show_rgb : GelSight tactile RGB image per finger
- --show_ff  : tactile normal/shear rendered force-field RGB per finger
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import select
import traceback
import sys
import termios
import time
import tty
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import yaml

from isaaclab.app import AppLauncher

TASK_PRESETS: dict[str, dict[str, str]] = {
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

INHAND_TACTILE_CFG = (
    "ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:"
    "UR10eShadowHandInHandTactileEnvCfg"
)
TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)

GRASP_MARKER_PATH = "/World/Debug/GraspClosureTarget"
ARM_MARKER_PATH = "/World/Debug/GraspArmTarget"
THUMB_SLICE = slice(19, 24)
_DEFAULT_PICKUP_IK_YAML = (
    Path(__file__).resolve().parents[1] / "ik" / "configs" / "full_ik" / "full_ik_pickup_fixed_hand.yaml"
)
_DEFAULT_CLOSED_24: list[float] = [
    0.0,
    0.0,
    0.0,
    0.7909020471572877,
    0.7155780426661175,
    0.4896060291926067,
    0.0,
    0.7909020471572877,
    0.7155780426661175,
    0.4896060291926067,
    0.0,
    0.7909020471572877,
    0.7155780426661175,
    0.4896060291926067,
    0.07532400449117027,
    0.0,
    0.6628512395222983,
    0.587527235031128,
    0.4142820247014365,
    0.037662002245585136,
    0.286231217066447,
    0.15064800898234054,
    0.3615552215576172,
    0.7155780426661175,
]


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
    """Same as ``record_arm_pose.py`` / debug single."""
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
    mod_name, sym_name = entry.split(":", 1)
    return getattr(importlib.import_module(mod_name), sym_name)


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
    """Same as ``record_arm_pose.py``: avoid stealing Isaac UI focus."""
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


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _make_T(pos_xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T


def _quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).ravel()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_euler("xyz")


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _root_pose_w_to_T_44(pq: torch.Tensor) -> np.ndarray:
    pq = pq.detach().cpu().numpy().ravel()
    pos = pq[:3].astype(np.float64)
    w, x, y, z = [float(v) for v in pq[3:7]]
    rm = R.from_quat([x, y, z, w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rm
    T[:3, 3] = pos
    return T


def _pickup_down_euler(palm_normal_local: np.ndarray, world_down: np.ndarray, yaw_about_world_z: float) -> np.ndarray:
    n = np.asarray(palm_normal_local, dtype=np.float64).ravel()
    n = n / (np.linalg.norm(n) + 1e-12)
    d = np.asarray(world_down, dtype=np.float64).ravel()
    d = d / (np.linalg.norm(d) + 1e-12)
    r_align = R.align_vectors(d.reshape(1, 3), n.reshape(1, 3))[0]
    r_yaw = R.from_euler("z", float(yaw_about_world_z), degrees=False)
    return (r_yaw * r_align).as_euler("xyz")


def _T_from_pos_euler(pos: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).ravel()[:3]
    return T


def _T_inv(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ p
    return out


def _wrist_world_pickup_over_object(
    object_pos_w: np.ndarray,
    *,
    object_to_palm_offset: np.ndarray,
    palm_in_wrist_pos: np.ndarray,
    palm_in_wrist_euler_xyz: np.ndarray,
    palm_normal_local: np.ndarray,
    world_down: np.ndarray,
    palm_yaw_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    palm_pos = np.asarray(object_pos_w, dtype=np.float64).ravel()[:3] + np.asarray(object_to_palm_offset, dtype=np.float64).ravel()[:3]
    palm_euler = _pickup_down_euler(palm_normal_local, world_down, palm_yaw_offset)
    T_world_palm = _T_from_pos_euler(palm_pos, palm_euler)
    T_wrist_palm = _T_from_pos_euler(
        np.asarray(palm_in_wrist_pos, dtype=np.float64).ravel()[:3],
        np.asarray(palm_in_wrist_euler_xyz, dtype=np.float64).ravel()[:3],
    )
    T_world_wrist = T_world_palm @ _T_inv(T_wrist_palm)
    return T_world_wrist[:3, 3].copy(), R.from_matrix(T_world_wrist[:3, :3]).as_euler("xyz")


def _load_pickup_palm_cfg(yaml_path: Path | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "object_to_palm_offset": np.array([0.0, 0.0, 0.05], dtype=np.float64),
        "palm_in_wrist_pos": np.array([0.0, 0.0, 0.35], dtype=np.float64),
        "palm_in_wrist_euler": np.array([1.5707963267948966, -1.5707963267948966, 1.5707963267948966], dtype=np.float64),
        "palm_normal_local": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "world_down": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "palm_yaw_offset": 0.0,
    }
    if yaml_path is None or not yaml_path.is_file():
        return cfg
    data = yaml.safe_load(yaml_path.read_text()) or {}
    for k in ("object_to_palm_offset", "palm_in_wrist_pos", "palm_in_wrist_euler", "palm_normal_local", "world_down"):
        if k in data and isinstance(data[k], (list, tuple)) and len(data[k]) >= 3:
            cfg[k] = np.array([float(data[k][i]) for i in range(3)], dtype=np.float64)
    if "palm_yaw_offset" in data and data["palm_yaw_offset"] is not None:
        cfg["palm_yaw_offset"] = float(data["palm_yaw_offset"])
    return cfg


def _load_hand_shadow_24(yaml_path: Path) -> np.ndarray:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    seq = data.get("hand_joint_pos_shadow_order")
    if not isinstance(seq, list) or len(seq) != 24:
        raise ValueError(f"{yaml_path}: need hand_joint_pos_shadow_order: [24 floats]")
    return np.array([float(x) for x in seq], dtype=np.float64)


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
    except (OSError, IOError):
        return None


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


def _save_yaml(path: Path, task: str, alpha: float, finger_mode: str, arm_dict: dict[str, float], hand_vec: np.ndarray, sh_names: list[str]) -> None:
    doc = {
        "task": task,
        "grasp_closure_alpha": float(alpha),
        "finger_mode": finger_mode,
        "arm_joint_pos": dict(arm_dict),
        "hand_joint_pos_shadow_order": [float(hand_vec[i]) for i in range(24)],
        "hand_joint_pos_named": {sh_names[i]: float(hand_vec[i]) for i in range(24)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Handshape lab with tactile RGB/force visualization.")
    parser.add_argument("--task", choices=sorted(TASK_PRESETS.keys()), default="pickup", help='Supported task alias used to select the canonical environment/config pair.')
    parser.add_argument("--env", type=str, default="", help='Optional environment entry point in module:Class form; must be supplied together with --cfg.')
    parser.add_argument("--cfg", type=str, default="", help='Optional environment-config entry point in module:Class form; must be supplied together with --env.')
    parser.add_argument("--num_envs", type=int, default=1, help='Number of parallel simulation environments.')
    parser.add_argument("--fps", type=float, default=30.0, help='Target control and recording loop frequency in frames per second.')
    parser.add_argument("--max-episode-length", type=int, default=200000, help='Environment episode-length override in simulation steps.')
    parser.add_argument("--arm-control", choices=("marker", "fixed"), default="marker", help='Arm target source: interactive marker or fixed initial joints.')
    parser.add_argument("--arm-marker-pos", type=float, nargs=3, default=(0.65, 0.12, 0.42), help='Initial arm IK marker world position X Y Z in meters.')
    parser.add_argument("--arm-marker-euler", type=float, nargs=3, default=(0.0, 2.2, 0.0), help='Initial arm IK marker XYZ Euler orientation in radians.')
    parser.add_argument("--pickup-ik-yaml", type=str, default="", help='Optional pickup Full-IK YAML used to initialize palm/approach settings.')
    parser.add_argument("--skip-auto-approach", action="store_true", help='Skip the scripted initial approach and begin in manual tuning mode.')
    parser.add_argument("--grasp-cube-pos", type=float, nargs=3, default=(0.55, 0.12, 0.42), help='Initial grasp-control cube world position X Y Z in meters.')
    parser.add_argument("--closure-axis", choices=("x", "y", "z"), default="x", help='World axis used to map grasp-cube motion to hand closure.')
    parser.add_argument("--closure-min", type=float, default=0.42, help='Cube coordinate mapped to fully open hand closure.')
    parser.add_argument("--closure-max", type=float, default=0.72, help='Cube coordinate mapped to fully closed hand closure.')
    parser.add_argument("--closed-hand-yaml", type=str, default="", help='Optional YAML containing the fully closed 24-joint hand template.')
    parser.add_argument(
        "--slider-offset-limit",
        type=float,
        default=1.0,
        help="Per-joint offset slider half-range in radians; actual range is [-limit, +limit].",
    )
    parser.add_argument("--finger-mode", choices=("four", "five"), default="five", help='Use four-finger or five-finger closure behavior.')
    parser.add_argument("--save-yaml", type=str, default="", help='Output YAML path written by the save hotkey or on exit.')
    parser.add_argument("--disable-hotkeys", action="store_true", help='Disable keyboard shortcuts and require normal process termination.')
    parser.add_argument(
        "--ignore-first-key",
        action="store_true",
        default=True,
        help="Ignore one initial stdin key event to avoid accidental immediate hotkey-triggered exit.",
    )
    parser.add_argument("--print-hand-every", type=int, default=0, help='Print the current 24-joint hand vector every N steps; 0 disables it.')
    parser.add_argument("--max-steps", type=int, default=0, help='Maximum control-loop steps; 0 runs until quit or environment termination.')
    parser.add_argument(
        "--no-fallback-when-app-not-running",
        dest="fallback_when_app_not_running",
        action="store_false",
        help="Disable fallback stepping when simulation_app.is_running() is false.",
    )
    parser.set_defaults(fallback_when_app_not_running=True)
    parser.add_argument("--show_rgb", action="store_true", help="Live tactile RGB images.")
    parser.add_argument("--show_ff", action="store_true", help="Live tactile force-field rendering.")
    parser.add_argument("--env-index", type=int, default=0, help="Env index to display tactile for.")
    parser.add_argument(
        "--viewer-topmost",
        action="store_true",
        help="Keep tactile viewer on top. Default off to avoid stealing focus (matplotlib).",
    )
    parser.add_argument(
        "--viewer",
        choices=("cv2", "matplotlib"),
        default="matplotlib",
        help="Tactile viewer backend. matplotlib matches record_arm_pose.py; cv2 optional.",
    )
    parser.add_argument(
        "--save-tactile-video",
        type=str,
        default="",
        help="Optional mp4 path to save tactile canvas when no GUI window backend is available.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
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
    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.prims import XFormPrim
    import omni.ui  # type: ignore

    preset = TASK_PRESETS[str(args.task)]
    env_entry = str(args.env).strip() or preset["env"]
    cfg_entry = str(args.cfg).strip() or preset["cfg"]
    if (args.show_rgb or args.show_ff) and str(args.task) == "inhand" and not str(args.cfg).strip():
        cfg_entry = INHAND_TACTILE_CFG
        print(f"[INFO] Using tactile cfg for inhand: {cfg_entry}")
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    if int(args.max_episode_length) > 0 and hasattr(cfg, "episode_length_s") and hasattr(cfg, "sim") and hasattr(cfg, "decimation"):
        cfg.episode_length_s = float(args.max_episode_length) * float(cfg.sim.dt) * float(cfg.decimation)
    cfg.enable_cameras = bool(getattr(args, "enable_cameras", False)) or bool(int(os.environ.get("ENABLE_CAMERAS", "0")))
    print(f"[INFO] cfg.enable_cameras={cfg.enable_cameras}")
    env = EnvCls(cfg)

    def _never_done(_self):
        z = torch.zeros(_self.num_envs, dtype=torch.bool, device=_self.device)
        return z, z

    env._get_dones = types.MethodType(_never_done, env)  # type: ignore[method-assign]
    env.reset()

    # TacSL nominal render (same order as record_arm_pose.py / debug single).
    if args.show_rgb or args.show_ff:
        warmed0: list[str] = []
        for name in TACTILE_SENSOR_NAMES:
            if name in env.scene.sensors:
                try:
                    env.scene[name].get_initial_render()
                    warmed0.append(name)
                except Exception as e:
                    print(f"[WARN] tactile warmup failed for {name}: {e}")
        if warmed0:
            print(f"[INFO] tactile nominal warmup done: {warmed0}")

    robot = env.robot
    joint_names = list(robot.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    sh_names = shadowhand_joint_names()
    arm_j_live = np.array([float(robot.data.joint_pos[0, i].item()) for i in arm_indices], dtype=np.float64)
    body_ids, _ = robot.find_bodies("wrist_3_link")
    wrist_body_idx = int(body_ids[0]) if len(body_ids) > 0 else -1

    pickup_yaml_arg = str(args.pickup_ik_yaml).strip()
    pickup_ik_path = Path(pickup_yaml_arg).expanduser() if pickup_yaml_arg else _DEFAULT_PICKUP_IK_YAML
    if not pickup_ik_path.is_absolute():
        pickup_ik_path = (repo_root / pickup_ik_path).resolve()
    palm_cfg = _load_pickup_palm_cfg(pickup_ik_path if pickup_ik_path.is_file() else None)

    if str(args.closed_hand_yaml).strip():
        ch_path = Path(args.closed_hand_yaml).expanduser()
        if not ch_path.is_absolute():
            ch_path = (repo_root / ch_path).resolve()
        closed_ref = _load_hand_shadow_24(ch_path)
    else:
        closed_ref = np.array(_DEFAULT_CLOSED_24, dtype=np.float64)

    VisualCuboid(prim_path=GRASP_MARKER_PATH, size=0.04, position=np.array(args.grasp_cube_pos, dtype=np.float64), visible=True, color=np.array([0.2, 0.95, 0.95]))
    grasp_xf = XFormPrim(prim_paths_expr=GRASP_MARKER_PATH, name="GraspClosureTarget", usd=True)
    arm_xf: Optional[XFormPrim] = None
    arm_ik: Optional[VideoTeleopControl] = None
    if str(args.arm_control) == "marker":
        apos = np.array(args.arm_marker_pos, dtype=np.float64)
        VisualCuboid(prim_path=ARM_MARKER_PATH, size=0.04, position=apos, visible=True, color=np.array([0.95, 0.2, 0.95]))
        arm_xf = XFormPrim(prim_paths_expr=ARM_MARKER_PATH, name="GraspArmTarget", usd=True)
        T0 = _make_T(apos, np.array(args.arm_marker_euler, dtype=np.float64))
        q = R.from_matrix(T0[:3, :3]).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        arm_xf.set_world_poses(positions=torch.tensor([T0[:3, 3]], dtype=torch.float32, device="cpu"), orientations=torch.tensor([q_wxyz], dtype=torch.float32, device="cpu"))
        arm_ik = VideoTeleopControl()

    use_pickup_approach = str(args.task) == "pickup" and str(args.arm_control) == "marker" and not bool(args.skip_auto_approach)

    def _hand_joint_for_robot_name(name: str, hand_joints: np.ndarray) -> float:
        for sh_idx, sh_name in enumerate(sh_names):
            if sh_name in name or name.endswith(sh_name):
                return float(hand_joints[sh_idx])
        return 0.0

    def _build_action(hand_joints: np.ndarray) -> torch.Tensor:
        full_dof = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_indices):
            if i < len(arm_j_live):
                full_dof[idx] = arm_j_live[i]
        for idx in hand_indices:
            full_dof[idx] = _hand_joint_for_robot_name(joint_names[idx], hand_joints)
        actuated = full_dof[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lower = env.robot_dof_lower_limits[0, env.actuated_dof_indices].cpu().numpy()
        upper = env.robot_dof_upper_limits[0, env.actuated_dof_indices].cpu().numpy()
        scale = np.where(upper - lower > 1e-6, 2.0 * (actuated - lower) / (upper - lower) - 1.0, 0.0)
        return torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)

    def _set_arm_marker_pose(pos_w: np.ndarray, euler_xyz: np.ndarray) -> None:
        if arm_xf is None:
            return
        Tw = _T_from_pos_euler(pos_w, euler_xyz)
        q = R.from_matrix(Tw[:3, :3]).as_quat()
        qw = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        arm_xf.set_world_poses(
            positions=torch.tensor([Tw[:3, 3]], dtype=torch.float32, device="cpu"),
            orientations=torch.tensor([qw], dtype=torch.float32, device="cpu"),
        )

    def _sync_marker_to_sim_wrist() -> None:
        if arm_xf is None or wrist_body_idx < 0:
            return
        pw = _to_numpy(robot.data.body_pos_w[0, wrist_body_idx]).ravel()[:3]
        qw = _to_numpy(robot.data.body_quat_w[0, wrist_body_idx]).ravel()[:4]
        _set_arm_marker_pose(pw, _quat_wxyz_to_euler_xyz(qw))

    init_h = np.zeros(24, dtype=np.float64)
    hand_gui_models: list[Any] = []
    hand_gui_window = omni.ui.Window("Shadow Hand Offsets (rad)", width=520, height=720, visible=True, dock_preference=omni.ui.DockPreference.RIGHT_TOP)
    with hand_gui_window.frame:
        with omni.ui.ScrollingFrame(horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with omni.ui.VStack(spacing=3, height=0):
                omni.ui.Label("final_hand = alpha*closed_template + per-joint offset. Cyan cube controls alpha.", word_wrap=True)
                for sh_i in range(24):
                    lo = -float(args.slider_offset_limit)
                    hi = float(args.slider_offset_limit)
                    m = omni.ui.SimpleFloatModel()
                    _float_model_set(m, float(np.clip(init_h[sh_i], lo, hi)))
                    hand_gui_models.append(m)
                    with omni.ui.HStack(spacing=6):
                        omni.ui.Label(f"{sh_names[sh_i]}", width=56, alignment=omni.ui.Alignment.LEFT_CENTER)
                        omni.ui.FloatSlider(model=m, min=lo, max=hi, step=max(1e-4, (hi - lo) / 500.0))
                        omni.ui.FloatField(m, width=72)

    fig = None
    rgb_ims: list[Any] = []
    ff_ims: list[Any] = []
    nrows, ncols = 20, 25
    use_cv2 = False
    cv2_mod: Any = None
    cv2_window_enabled = False
    tactile_video_writer: Any = None
    tactile_video_path: Optional[Path] = None
    tactile_video_fps = max(1.0, float(args.fps))
    if (args.show_rgb or args.show_ff) and str(args.viewer) == "cv2":
        try:
            import cv2 as _cv2

            cv2_mod = _cv2
            use_cv2 = True
            print("[INFO] tactile viewer backend: cv2")
        except Exception as e:
            print(f"[WARN] cv2 viewer unavailable ({e}), fallback to matplotlib.")
            use_cv2 = False
    if str(args.save_tactile_video).strip():
        tactile_video_path = Path(args.save_tactile_video).expanduser()
        if not tactile_video_path.is_absolute():
            tactile_video_path = (repo_root / tactile_video_path).resolve()
        tactile_video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] tactile video recording path: {tactile_video_path}")

    if (args.show_rgb or args.show_ff) and not use_cv2:
        import matplotlib

        # Match record_arm_pose.py: avoid focus steal + toolbar crashes.
        matplotlib.rcParams["figure.raise_window"] = False
        # In some Isaac Sim + Tk builds, toolbar icon sizing can crash window creation.
        matplotlib.rcParams["toolbar"] = "None"

        backend_candidates: list[str] = []
        env_backend = str(os.environ.get("MPLBACKEND", "")).strip()
        if env_backend:
            backend_candidates.append(env_backend)
        if os.environ.get("DISPLAY"):
            backend_candidates.extend(["Qt5Agg", "TkAgg"])
        backend_candidates.append("Agg")

        backend_used: Optional[str] = None
        backend_err: Optional[Exception] = None
        for backend in backend_candidates:
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt

                if args.show_rgb and args.show_ff:
                    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
                    ax_rgb, ax_ff = axes[0], axes[1]
                elif args.show_rgb:
                    fig, ax_rgb = plt.subplots(1, 5, figsize=(16, 3))
                    ax_ff = None
                else:
                    fig, ax_ff = plt.subplots(1, 5, figsize=(16, 3))
                    ax_rgb = None
                backend_used = backend
                break
            except Exception as e:
                backend_err = e
                fig = None
                ax_rgb = None
                ax_ff = None
                continue

        if backend_used is None or fig is None:
            print(
                "[WARN] Failed to create matplotlib tactile window; "
                f"tried backends={backend_candidates}, last_error={backend_err}"
            )
            print("[WARN] Disabling tactile viewer this run.")
            args.show_rgb = False
            args.show_ff = False
        else:
            print(f"[INFO] tactile viewer backend: matplotlib/{backend_used}")

    if fig is not None:
        import matplotlib.pyplot as plt
        if args.show_ff:
            for name in TACTILE_SENSOR_NAMES:
                if name in env.scene.sensors:
                    try:
                        nrows, ncols = env.scene[name].cfg.tactile_array_size
                    except Exception:
                        pass
                    break
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
        available = [name for name in TACTILE_SENSOR_NAMES if name in env.scene.sensors]
        print(f"[INFO] Tactile viewer enabled. available_sensors={available}")
        print(f"[INFO] Tactile panel order: {[n.replace('tactile_sensor_', '').upper() for n in TACTILE_SENSOR_NAMES]}")
        if not available:
            print(
                "[WARN] No tactile_sensor_* found in scene.sensors. "
                "Check task cfg / --cfg and --enable_cameras."
            )

    render_ff = _render_tactile_ff_rgb if args.show_ff else None

    ax_i = _axis_index(str(args.closure_axis))
    lo = float(min(args.closure_min, args.closure_max))
    hi = float(max(args.closure_min, args.closure_max))
    span = hi - lo if abs(hi - lo) > 1e-9 else 1.0
    last_alpha = 0.0
    hand_locked = False
    last_hand = np.zeros(24, dtype=np.float64)
    save_path = Path(args.save_yaml).expanduser() if str(args.save_yaml).strip() else None
    if save_path and not save_path.is_absolute():
        save_path = (repo_root / save_path).resolve()

    def _slider_offset() -> np.ndarray:
        h = np.array([_float_model_get(m) for m in hand_gui_models], dtype=np.float64)
        if str(args.finger_mode) == "four":
            h[THUMB_SLICE] = 0.0
        return h

    def _closure_hand(alpha: float) -> np.ndarray:
        h = alpha * closed_ref.copy()
        if str(args.finger_mode) == "four":
            h[THUMB_SLICE] = 0.0
        return h

    def _compose_hand(alpha: float) -> np.ndarray:
        h = _closure_hand(alpha) + _slider_offset()
        for i, sh_name in enumerate(sh_names):
            for idx in hand_indices:
                n = joint_names[idx]
                if sh_name in n or n.endswith(sh_name):
                    lo_i = float(env.robot_dof_lower_limits[0, idx].item())
                    hi_i = float(env.robot_dof_upper_limits[0, idx].item())
                    h[i] = float(np.clip(h[i], lo_i, hi_i))
                    break
        if str(args.finger_mode) == "four":
            h[THUMB_SLICE] = 0.0
        return h

    def _print_hand24(vec: np.ndarray) -> None:
        print("[INFO] hand_joint_pos_shadow_order (24):")
        print(f"    {np.asarray(vec, dtype=np.float64).tolist()}")

    def _save_now() -> None:
        if save_path is None:
            print("[WARN] --save-yaml not set; skip save.")
            return
        arm_dict = {joint_names[idx]: float(arm_j_live[i]) for i, idx in enumerate(arm_indices) if i < len(arm_j_live)}
        _save_yaml(save_path, str(args.task), last_alpha, str(args.finger_mode), arm_dict, last_hand, sh_names)
        print(f"[INFO] Saved YAML now -> {save_path}")

    if use_pickup_approach and arm_xf is not None and hasattr(env, "object"):
        obj_pos = _to_numpy(env.object.data.root_pos_w[0]).ravel()[:3]
        wrist_pos, wrist_euler = _wrist_world_pickup_over_object(
            obj_pos,
            object_to_palm_offset=palm_cfg["object_to_palm_offset"],
            palm_in_wrist_pos=palm_cfg["palm_in_wrist_pos"],
            palm_in_wrist_euler_xyz=palm_cfg["palm_in_wrist_euler"],
            palm_normal_local=palm_cfg["palm_normal_local"],
            world_down=palm_cfg["world_down"],
            palm_yaw_offset=float(palm_cfg["palm_yaw_offset"]),
        )
        _set_arm_marker_pose(wrist_pos, wrist_euler)

    print("[INFO] Hotkeys: [f]=lock hand, [p]=print hand24, [s]=save, [g]=skip approach helper, [q]=quit")
    target_dt = 1.0 / max(1e-3, float(args.fps))
    term_state = None
    step = 0
    env_idx = max(0, min(int(args.env_index), env.num_envs - 1))
    exit_reason = "unknown"

    try:
        if not bool(args.disable_hotkeys):
            term_state = _install_terminal_cbreak()
        app_running_false_count = 0
        warned_app_not_running = False
        while True:
            app_running = bool(simulation_app.is_running())
            if not app_running:
                app_running_false_count += 1
                if not warned_app_not_running:
                    print(
                        "[WARN] simulation_app.is_running() is False; entering fallback stepping loop. "
                        "Use [q] to quit."
                    )
                    warned_app_not_running = True
                if not bool(args.fallback_when_app_not_running):
                    exit_reason = "app_not_running_no_fallback"
                    break
            else:
                app_running_false_count = 0

            t0 = time.time()
            step += 1
            key = _poll_key_nonblocking() if not bool(args.disable_hotkeys) else None
            if key:
                if bool(args.ignore_first_key) and step <= 2:
                    key = None
            if key:
                k = key.lower()
                if k == "q":
                    exit_reason = "hotkey_q"
                    break
                if k == "f":
                    hand_locked = not hand_locked
                if k == "g" and use_pickup_approach:
                    _sync_marker_to_sim_wrist()
                if k == "p":
                    _print_hand24(last_hand)
                if k == "s":
                    _save_now()

            pos_t, _ = grasp_xf.get_world_poses()
            pos = pos_t[0].detach().cpu().numpy().ravel()[:3]
            last_alpha = float(np.clip((float(pos[ax_i]) - lo) / span, 0.0, 1.0))
            if not hand_locked:
                last_hand = _compose_hand(last_alpha)

            if arm_ik is not None and arm_xf is not None:
                arm_ik.T_world_arm_base = _root_pose_w_to_T_44(env.robot.data.root_pose_w[0])
                pos_t2, ori_t2 = arm_xf.get_world_poses()
                wrist_pos = _to_numpy(pos_t2[0]).ravel()[:3]
                wrist_euler = _quat_wxyz_to_euler_xyz(_to_numpy(ori_t2[0]).ravel()[:4])
                targets: Optional[ArmHandTargets] = arm_ik.compute(wrist_pos, wrist_euler, last_hand)
                if targets is not None:
                    arm_j_live = np.array(targets.arm_joints, dtype=np.float64).ravel()[: len(arm_indices)]

            actions = _build_action(last_hand)
            if env.num_envs > 1:
                actions = actions.expand(env.num_envs, -1).clone()
            env.step(actions)

            if use_cv2 and cv2_mod is not None and (args.show_rgb or args.show_ff):
                rgb_tiles: list[np.ndarray] = []
                ff_tiles: list[np.ndarray] = []
                for name in TACTILE_SENSOR_NAMES:
                    if name not in env.scene.sensors:
                        continue
                    try:
                        data = env.scene[name].data
                    except RuntimeError as e:
                        if "Nominal tactile is not set" in str(e):
                            try:
                                env.scene[name].get_initial_render()
                                data = env.scene[name].data
                            except Exception:
                                continue
                        else:
                            raise
                    if args.show_rgb:
                        img = getattr(data, "tactile_rgb_image", None)
                        if img is not None and img.ndim == 4:
                            e = min(env_idx, img.shape[0] - 1)
                            rgb_tiles.append(_img_to_uint8(img[e].detach().cpu().numpy()))
                    if args.show_ff:
                        nf = getattr(data, "tactile_normal_force", None)
                        sf = getattr(data, "tactile_shear_force", None)
                        if nf is not None and sf is not None:
                            e = min(env_idx, nf.shape[0] - 1)
                            nf_flat = nf[e].detach().cpu().numpy().reshape(-1)
                            sf_flat = sf[e].detach().cpu().numpy().reshape(-1, 2)
                            p = int(nf_flat.shape[0])
                            nr, nc = nrows, ncols
                            if p != nr * nc:
                                nr = int(np.sqrt(p)) if p > 0 else 1
                                nc = max(1, p // max(1, nr))
                                if nr * nc != p:
                                    nr, nc = nrows, ncols
                            if p == nr * nc and render_ff is not None:
                                try:
                                    ff_tiles.append(render_ff(nf_flat.reshape(nr, nc), sf_flat.reshape(nr, nc, 2)))
                                except Exception:
                                    pass

                def _row_panel(tiles: list[np.ndarray], tile_h: int, tile_w: int) -> Optional[np.ndarray]:
                    if not tiles:
                        return None
                    out: list[np.ndarray] = []
                    for t in tiles:
                        if t.ndim == 2:
                            t = np.repeat(t[..., None], 3, axis=-1)
                        out.append(cv2_mod.resize(t, (tile_w, tile_h), interpolation=cv2_mod.INTER_NEAREST))
                    return np.concatenate(out, axis=1)

                panel_rgb = _row_panel(rgb_tiles, 180, 240) if args.show_rgb else None
                panel_ff = _row_panel(ff_tiles, 180, 240) if args.show_ff else None
                if panel_rgb is not None or panel_ff is not None:
                    if panel_rgb is not None and panel_ff is not None:
                        w = max(panel_rgb.shape[1], panel_ff.shape[1])
                        if panel_rgb.shape[1] < w:
                            pad = np.zeros((panel_rgb.shape[0], w - panel_rgb.shape[1], 3), dtype=np.uint8)
                            panel_rgb = np.concatenate([panel_rgb, pad], axis=1)
                        if panel_ff.shape[1] < w:
                            pad = np.zeros((panel_ff.shape[0], w - panel_ff.shape[1], 3), dtype=np.uint8)
                            panel_ff = np.concatenate([panel_ff, pad], axis=1)
                        canvas = np.concatenate([panel_rgb, panel_ff], axis=0)
                    else:
                        canvas = panel_rgb if panel_rgb is not None else panel_ff
                    if use_cv2 and cv2_mod is not None:
                        if cv2_window_enabled:
                            try:
                                cv2_mod.imshow("Tactile (RGB top / FF bottom)", canvas)
                                cv2_mod.waitKey(1)
                            except Exception as e:
                                cv2_window_enabled = False
                                print(f"[WARN] cv2 window update failed, disable window mode: {e}")
                        else:
                            try:
                                cv2_mod.imshow("Tactile (RGB top / FF bottom)", canvas)
                                cv2_mod.waitKey(1)
                                cv2_window_enabled = True
                            except Exception as e:
                                print(f"[WARN] cv2 imshow unavailable, running headless viewer mode: {e}")

                    if tactile_video_path is not None and cv2_mod is not None:
                        if tactile_video_writer is None:
                            h, w = int(canvas.shape[0]), int(canvas.shape[1])
                            fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
                            tactile_video_writer = cv2_mod.VideoWriter(str(tactile_video_path), fourcc, tactile_video_fps, (w, h))
                            if not tactile_video_writer.isOpened():
                                tactile_video_writer = None
                                print(f"[WARN] could not open tactile video writer: {tactile_video_path}")
                            else:
                                print(f"[INFO] recording tactile video -> {tactile_video_path}")
                        if tactile_video_writer is not None:
                            tactile_video_writer.write(canvas[:, :, ::-1])

            if fig is not None and (rgb_ims or ff_ims):
                import matplotlib.pyplot as plt
                for i, name in enumerate(TACTILE_SENSOR_NAMES):
                    if name not in env.scene.sensors:
                        continue
                    try:
                        data = env.scene[name].data
                    except RuntimeError as e:
                        if "Nominal tactile is not set" in str(e):
                            try:
                                env.scene[name].get_initial_render()
                                data = env.scene[name].data
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

            if int(args.print_hand_every) > 0 and step % int(args.print_hand_every) == 0:
                _print_hand24(last_hand)
            dt = target_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                exit_reason = "max_steps_reached"
                break
            if not app_running and app_running_false_count > 100000:
                print("[WARN] fallback loop safety break triggered.")
                exit_reason = "fallback_safety_break"
                break
    except BaseException as e:
        if exit_reason == "unknown":
            exit_reason = f"exception_{type(e).__name__}"
        print(f"[ERROR] Loop aborted by exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    finally:
        _restore_terminal(term_state)
        try:
            hand_gui_window.visible = False
            hand_gui_window.destroy()
        except Exception:
            pass
        if save_path is not None:
            _save_now()
        env.close()
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close("all")
        if use_cv2 and cv2_mod is not None:
            try:
                cv2_mod.destroyAllWindows()
            except Exception:
                pass
        if tactile_video_writer is not None:
            try:
                tactile_video_writer.release()
                print(f"[INFO] tactile video saved: {tactile_video_path}")
            except Exception:
                pass
        if exit_reason == "unknown":
            if step == 0:
                exit_reason = "loop_not_entered"
            else:
                exit_reason = "loop_ended_without_explicit_break"
        print(f"[INFO] Exit reason: {exit_reason}, steps={step}")
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

