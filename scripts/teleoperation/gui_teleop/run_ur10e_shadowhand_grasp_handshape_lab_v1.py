#!/usr/bin/env python3
"""Standalone wrist+handshape lab for pickup/pour/inhand.

Features:
- Magenta cube controls wrist pose through IK.
- Cyan cube controls closure alpha in [0, 1].
- 24-DoF hand sliders for per-joint micro tuning.
- Five-finger mode and finger lock for lift verification.
- Print/save current 24-d hand shape quickly.
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
        "env": "ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
        "cfg": "ViTacLab.tasks.direct.difficult_dexhand.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
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

GRASP_MARKER_PATH = "/World/Debug/GraspClosureTarget"
ARM_MARKER_PATH = "/World/Debug/GraspArmTarget"
THUMB_SLICE = slice(19, 24)


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
    parser = argparse.ArgumentParser(description="Standalone handshape lab: wrist marker + hand sliders + closure cube.")
    parser.add_argument("--task", choices=sorted(TASK_PRESETS.keys()), default="pickup")
    parser.add_argument("--env", type=str, default="")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-episode-length", type=int, default=200000)
    parser.add_argument("--arm-control", choices=("marker", "fixed"), default="marker")
    parser.add_argument("--arm-marker-pos", type=float, nargs=3, default=(0.65, 0.12, 0.42))
    parser.add_argument("--arm-marker-euler", type=float, nargs=3, default=(0.0, 2.2, 0.0))
    parser.add_argument("--grasp-cube-pos", type=float, nargs=3, default=(0.55, 0.12, 0.42))
    parser.add_argument("--closure-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--closure-min", type=float, default=0.42)
    parser.add_argument("--closure-max", type=float, default=0.72)
    parser.add_argument("--finger-mode", choices=("four", "five"), default="five")
    parser.add_argument("--save-yaml", type=str, default="")
    parser.add_argument("--disable-hotkeys", action="store_true")
    parser.add_argument("--print-hand-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

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
    EnvCls = _load_symbol(str(args.env).strip() or preset["env"])
    CfgCls = _load_symbol(str(args.cfg).strip() or preset["cfg"])

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    if int(args.max_episode_length) > 0 and hasattr(cfg, "episode_length_s") and hasattr(cfg, "sim") and hasattr(cfg, "decimation"):
        cfg.episode_length_s = float(args.max_episode_length) * float(cfg.sim.dt) * float(cfg.decimation)
    env = EnvCls(cfg)
    # no runtime reset hotkey behavior; tune hand shape continuously.
    def _never_done(_self):
        z = torch.zeros(_self.num_envs, dtype=torch.bool, device=_self.device)
        return z, z
    env._get_dones = types.MethodType(_never_done, env)  # type: ignore[method-assign]
    env.reset()

    robot = env.robot
    joint_names = list(robot.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    sh_names = shadowhand_joint_names()
    arm_j_live = np.array([float(robot.data.joint_pos[0, i].item()) for i in arm_indices], dtype=np.float64)

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

    init_h = np.zeros(24, dtype=np.float64)
    hand_gui_models: list[Any] = []
    hand_gui_window = omni.ui.Window("Shadow Hand Joints (rad)", width=460, height=720, visible=True, dock_preference=omni.ui.DockPreference.RIGHT_TOP)
    with hand_gui_window.frame:
        with omni.ui.ScrollingFrame(horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with omni.ui.VStack(spacing=3, height=0):
                omni.ui.Label("24-DoF sliders. Cyan cube = closure alpha fallback.", word_wrap=True)
                for sh_i in range(24):
                    lo, hi = -1.57, 1.57
                    for idx in hand_indices:
                        n = joint_names[idx]
                        if sh_names[sh_i] in n or n.endswith(sh_names[sh_i]):
                            lo = float(env.robot_dof_lower_limits[0, idx].item())
                            hi = float(env.robot_dof_upper_limits[0, idx].item())
                            break
                    m = omni.ui.SimpleFloatModel()
                    _float_model_set(m, float(np.clip(init_h[sh_i], lo, hi)))
                    hand_gui_models.append(m)
                    with omni.ui.HStack(spacing=6):
                        omni.ui.Label(f"{sh_names[sh_i]}", width=56, alignment=omni.ui.Alignment.LEFT_CENTER)
                        omni.ui.FloatSlider(model=m, min=lo, max=hi, step=max(1e-4, (hi - lo) / 500.0))

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

    def _slider_hand() -> np.ndarray:
        h = np.array([_float_model_get(m) for m in hand_gui_models], dtype=np.float64)
        if str(args.finger_mode) == "four":
            h[THUMB_SLICE] = 0.0
        return h

    def _closure_hand(alpha: float) -> np.ndarray:
        h = alpha * _slider_hand()
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

    print("[INFO] Hotkeys: [f]=lock hand, [p]=print hand24, [s]=save now, [q]=quit")
    target_dt = 1.0 / max(1e-3, float(args.fps))
    term_state = None
    step = 0
    try:
        if not bool(args.disable_hotkeys):
            term_state = _install_terminal_cbreak()
        while simulation_app.is_running():
            t0 = time.time()
            step += 1
            key = _poll_key_nonblocking() if not bool(args.disable_hotkeys) else None
            if key:
                k = key.lower()
                if k == "q":
                    break
                if k == "f":
                    hand_locked = not hand_locked
                    print(f"[INFO] finger lock: {'ON' if hand_locked else 'OFF'}")
                if k == "p":
                    _print_hand24(last_hand)
                if k == "s":
                    _save_now()

            pos_t, _ = grasp_xf.get_world_poses()
            pos = pos_t[0].detach().cpu().numpy().ravel()[:3]
            alpha = float(np.clip((float(pos[ax_i]) - lo) / span, 0.0, 1.0))
            last_alpha = alpha
            if not hand_locked:
                last_hand = _slider_hand() if hand_gui_models else _closure_hand(alpha)

            if arm_ik is not None and arm_xf is not None:
                root = env.robot.data.root_pose_w[0].detach().cpu().numpy().ravel()
                T = np.eye(4, dtype=np.float64)
                T[:3, 3] = root[:3]
                arm_ik.T_world_arm_base = T
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
            if int(args.print_hand_every) > 0 and step % int(args.print_hand_every) == 0:
                _print_hand24(last_hand)
            dt = target_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                break
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
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

