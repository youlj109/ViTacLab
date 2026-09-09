#!/usr/bin/env python3
"""Wrist + 24-DoF hand-shape lab (pickup aligns with full_ik pre-grasp geometry).

**Pickup workflow** (matches ``full_ik`` train: object anchor + world palm offset + ``pickup_down``):

1. **Approach** — Arm moves with open hand so the wrist/palm frame matches differential-IK semantics
   (same ``object_to_palm_offset``, ``palm_in_wrist_*``, ``palm_normal_local`` / ``world_down`` as
   ``scripts/rsl_rl/full_ik/configs/full_ik_pickup_fixed_hand.yaml`` unless overridden by ``--pickup-ik-yaml``).
2. **Hand tune** — Arm is frozen; adjust **24 sliders** only. Magenta cube follows the wrist visually (no IK).
3. **Wrist verify** — Press ``w``: magenta cube drives wrist via IK again; hand stays at slider pose to test lift.

**Pour / inhand** — No object anchor: starts in wrist-verify mode (marker IK + sliders), like the old script
without the cyan closure cube.

Hotkeys: ``w`` wrist verify / ``b`` back to hand tune (pickup), ``f`` lock sliders, ``p`` print hand24,
``s`` save, ``g`` skip approach, ``q`` quit.
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
from enum import Enum
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

ARM_MARKER_PATH = "/World/Debug/GraspArmTarget"
_DEFAULT_PICKUP_IK_YAML = (
    Path(__file__).resolve().parents[2] / "rsl_rl" / "full_ik" / "configs" / "full_ik_pickup_fixed_hand.yaml"
)
# Effectively disable horizon-based reset; ``_never_done`` also clears timeout / fall / OOB dones for teleop lab.
_LAB_NO_TIMEOUT_ENV_STEPS = 1_000_000_000


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


def _quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).ravel()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_euler("xyz")


def _pickup_down_euler(
    palm_normal_local: np.ndarray,
    world_down: np.ndarray,
    yaw_about_world_z: float,
) -> np.ndarray:
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
    """Same geometry as ``ArmIkHandActionExpander`` object phase with ``use_rotation=False`` (pickup_down)."""
    off = np.asarray(object_to_palm_offset, dtype=np.float64).ravel()[:3]
    palm_pos = np.asarray(object_pos_w, dtype=np.float64).ravel()[:3] + off
    palm_euler = _pickup_down_euler(palm_normal_local, world_down, palm_yaw_offset)
    T_world_palm = _T_from_pos_euler(palm_pos, palm_euler)
    T_wrist_palm = _T_from_pos_euler(
        np.asarray(palm_in_wrist_pos, dtype=np.float64).ravel()[:3],
        np.asarray(palm_in_wrist_euler_xyz, dtype=np.float64).ravel()[:3],
    )
    T_world_wrist = T_world_palm @ _T_inv(T_wrist_palm)
    wpos = T_world_wrist[:3, 3].copy()
    weuler = R.from_matrix(T_world_wrist[:3, :3]).as_euler("xyz")
    return wpos, weuler


def _load_pickup_palm_cfg(yaml_path: Path | None) -> dict[str, Any]:
    """Defaults match ``full_ik_pickup_fixed_hand.yaml`` palm / offset keys."""
    cfg: dict[str, Any] = {
        "object_to_palm_offset": np.array([0.0, 0.0, 0.05], dtype=np.float64),
        "palm_in_wrist_pos": np.array([0.0, 0.0, 0.35], dtype=np.float64),
        "palm_in_wrist_euler": np.array(
            [1.5707963267948966, -1.5707963267948966, 1.5707963267948966], dtype=np.float64
        ),
        "palm_normal_local": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "world_down": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "palm_yaw_offset": 0.0,
    }
    path = yaml_path
    if path is None or not path.is_file():
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    vec3 = (
        "object_to_palm_offset",
        "palm_in_wrist_pos",
        "palm_in_wrist_euler",
        "palm_normal_local",
        "world_down",
    )
    for k in vec3:
        if k in data and isinstance(data[k], (list, tuple)) and len(data[k]) >= 3:
            cfg[k] = np.array([float(data[k][i]) for i in range(3)], dtype=np.float64)
    if "palm_yaw_offset" in data and data["palm_yaw_offset"] is not None:
        cfg["palm_yaw_offset"] = float(data["palm_yaw_offset"])
    return cfg


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


def _save_yaml(path: Path, task: str, phase_note: str, arm_dict: dict[str, float], hand_vec: np.ndarray, sh_names: list[str]) -> None:
    doc = {
        "task": task,
        "handshape_lab_note": phase_note,
        "arm_joint_pos": dict(arm_dict),
        "hand_joint_pos_shadow_order": [float(hand_vec[i]) for i in range(24)],
        "hand_joint_pos_named": {sh_names[i]: float(hand_vec[i]) for i in range(24)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


class LabPhase(str, Enum):
    APPROACH = "approach"
    HAND_TUNE = "hand_tune"
    WRIST_VERIFY = "wrist_verify"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hand-shape lab: pickup auto-approach (full_ik geometry) → 24 sliders → wrist cube IK."
    )
    parser.add_argument("--task", choices=sorted(TASK_PRESETS.keys()), default="pickup")
    parser.add_argument("--env", type=str, default="")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=0,
        help="Env horizon in RL steps (for cfg.episode_length_s). Default 0 = use built-in huge value so the sim does "
        "not time-out reset; override only for debugging.",
    )
    parser.add_argument("--arm-control", choices=("marker", "fixed"), default="marker")
    parser.add_argument(
        "--arm-marker-pos",
        type=float,
        nargs=3,
        default=(0.65, 0.12, 0.42),
        help="Initial magenta cube (pour/inhand, or pickup if --skip-auto-approach).",
    )
    parser.add_argument("--arm-marker-euler", type=float, nargs=3, default=(0.0, 2.2, 0.0))
    parser.add_argument(
        "--pickup-ik-yaml",
        type=str,
        default="",
        help="YAML with palm/offset keys like full_ik_pickup_fixed_hand.yaml (default: that file if it exists).",
    )
    parser.add_argument(
        "--skip-auto-approach",
        action="store_true",
        help="Pickup: skip object-overhead approach; start with wrist IK + sliders.",
    )
    parser.add_argument("--approach-max-steps", type=int, default=400, help="Pickup: max steps in approach phase.")
    parser.add_argument(
        "--approach-hold",
        type=int,
        default=12,
        help="Pickup: end approach after wrist target is stable for this many consecutive steps.",
    )
    parser.add_argument(
        "--approach-wrist-tol",
        type=float,
        default=0.012,
        help="Pickup: max change in target wrist position (m) per step to count as 'stable'.",
    )
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
    if hasattr(cfg, "episode_length_s") and hasattr(cfg, "sim") and hasattr(cfg, "decimation"):
        step_dt = float(cfg.sim.dt) * float(int(cfg.decimation))
        n_ep = int(args.max_episode_length)
        if n_ep <= 0:
            n_ep = _LAB_NO_TIMEOUT_ENV_STEPS
        cfg.episode_length_s = float(n_ep) * step_dt
    env = EnvCls(cfg)

    def _never_done(_self):
        """No automatic reset from timeout, object fall, or OOB — teleop lab runs until you quit."""
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

    body_ids, _ = robot.find_bodies("wrist_3_link")
    wrist_body_idx = int(body_ids[0]) if len(body_ids) > 0 else -1

    pickup_yaml_arg = str(args.pickup_ik_yaml).strip()
    pickup_ik_path = Path(pickup_yaml_arg).expanduser() if pickup_yaml_arg else _DEFAULT_PICKUP_IK_YAML
    if not pickup_ik_path.is_absolute():
        pickup_ik_path = (repo_root / pickup_ik_path).resolve()
    palm_cfg = _load_pickup_palm_cfg(pickup_ik_path if pickup_ik_path.is_file() else None)

    use_pickup_approach = str(args.task) == "pickup" and str(args.arm_control) == "marker" and not bool(args.skip_auto_approach)
    phase = LabPhase.APPROACH if use_pickup_approach else LabPhase.WRIST_VERIFY
    open_hand24 = np.zeros(24, dtype=np.float64)
    last_wrist_target_pos = np.zeros(3, dtype=np.float64)
    approach_stable_ctr = 0
    approach_step = 0

    apos = np.array(args.arm_marker_pos, dtype=np.float64)
    VisualCuboid(
        prim_path=ARM_MARKER_PATH,
        size=0.04,
        position=apos,
        visible=True,
        color=np.array([0.95, 0.2, 0.95]),
    )
    arm_xf = XFormPrim(prim_paths_expr=ARM_MARKER_PATH, name="GraspArmTarget", usd=True)
    T0 = _T_from_pos_euler(apos, np.array(args.arm_marker_euler, dtype=np.float64))
    q = R.from_matrix(T0[:3, :3]).as_quat()
    q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    arm_xf.set_world_poses(
        positions=torch.tensor([T0[:3, 3]], dtype=torch.float32, device="cpu"),
        orientations=torch.tensor([q_wxyz], dtype=torch.float32, device="cpu"),
    )
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

    def _set_arm_marker_pose(pos_w: np.ndarray, euler_xyz: np.ndarray) -> None:
        Tw = _T_from_pos_euler(pos_w, euler_xyz)
        q = R.from_matrix(Tw[:3, :3]).as_quat()
        qw = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        arm_xf.set_world_poses(
            positions=torch.tensor([Tw[:3, 3]], dtype=torch.float32, device="cpu"),
            orientations=torch.tensor([qw], dtype=torch.float32, device="cpu"),
        )

    def _sync_marker_to_sim_wrist() -> None:
        if wrist_body_idx < 0:
            return
        pw = _to_numpy(robot.data.body_pos_w[0, wrist_body_idx]).ravel()[:3]
        qw = _to_numpy(robot.data.body_quat_w[0, wrist_body_idx]).ravel()[:4]
        eu = _quat_wxyz_to_euler_xyz(qw)
        _set_arm_marker_pose(pw, eu)

    init_h = np.zeros(24, dtype=np.float64)
    hand_gui_models: list[Any] = []
    phase_hint = (
        "Pickup: [1] Approach (auto) → [2] sliders only (arm frozen) → press [w] → [3] wrist cube IK. "
        "[b] back to sliders. Pour/inhand: wrist IK + sliders."
    )
    hand_gui_window = omni.ui.Window("Shadow Hand Joints (rad)", width=480, height=720, visible=True, dock_preference=omni.ui.DockPreference.RIGHT_TOP)
    with hand_gui_window.frame:
        with omni.ui.ScrollingFrame(horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with omni.ui.VStack(spacing=3, height=0):
                omni.ui.Label(phase_hint, word_wrap=True)
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

    hand_locked = False
    last_hand = np.zeros(24, dtype=np.float64)
    save_path = Path(args.save_yaml).expanduser() if str(args.save_yaml).strip() else None
    if save_path and not save_path.is_absolute():
        save_path = (repo_root / save_path).resolve()

    def _slider_hand() -> np.ndarray:
        return np.array([_float_model_get(m) for m in hand_gui_models], dtype=np.float64)

    def _print_hand24(vec: np.ndarray) -> None:
        print("[INFO] hand_joint_pos_shadow_order (24):")
        print(f"    {np.asarray(vec, dtype=np.float64).tolist()}")

    def _save_now() -> None:
        if save_path is None:
            print("[WARN] --save-yaml not set; skip save.")
            return
        arm_dict = {joint_names[idx]: float(arm_j_live[i]) for i, idx in enumerate(arm_indices) if i < len(arm_j_live)}
        _save_yaml(save_path, str(args.task), f"phase={phase.value}", arm_dict, last_hand, sh_names)
        print(f"[INFO] Saved YAML now -> {save_path}")

    print(
        "[INFO] Hotkeys: [w]=wrist IK verify (pickup), [b]=back to hand sliders, [f]=lock sliders, "
        "[p]=print hand24, [s]=save, [g]=skip approach, [q]=quit"
    )
    if use_pickup_approach:
        print(f"[INFO] Pickup approach uses palm/offset from: {pickup_ik_path if pickup_ik_path.is_file() else 'built-in defaults'}")

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
                    print(f"[INFO] slider lock: {'ON' if hand_locked else 'OFF'}")
                if k == "p":
                    _print_hand24(last_hand)
                if k == "s":
                    _save_now()
                if k == "g" and phase == LabPhase.APPROACH:
                    phase = LabPhase.HAND_TUNE
                    print("[INFO] phase → HAND_TUNE (skipped approach)")
                if k == "w" and use_pickup_approach and phase == LabPhase.HAND_TUNE:
                    phase = LabPhase.WRIST_VERIFY
                    _sync_marker_to_sim_wrist()
                    print("[INFO] phase → WRIST_VERIFY (magenta cube drives wrist IK)")
                if k == "b" and use_pickup_approach and phase == LabPhase.WRIST_VERIFY:
                    phase = LabPhase.HAND_TUNE
                    print("[INFO] phase → HAND_TUNE (arm frozen, sliders only)")

            if not hand_locked:
                last_hand = _slider_hand() if hand_gui_models else last_hand

            arm_ik.T_world_arm_base = _root_pose_w_to_T_44(env.robot.data.root_pose_w[0])

            targets: Optional[ArmHandTargets] = None
            if str(args.arm_control) != "marker":
                targets = None
            elif phase == LabPhase.APPROACH and hasattr(env, "object"):
                obj = env.object
                op = _to_numpy(obj.data.root_pos_w[0]).ravel()[:3]
                wpos, weuler = _wrist_world_pickup_over_object(
                    op,
                    object_to_palm_offset=palm_cfg["object_to_palm_offset"],
                    palm_in_wrist_pos=palm_cfg["palm_in_wrist_pos"],
                    palm_in_wrist_euler_xyz=palm_cfg["palm_in_wrist_euler"],
                    palm_normal_local=palm_cfg["palm_normal_local"],
                    world_down=palm_cfg["world_down"],
                    palm_yaw_offset=float(palm_cfg["palm_yaw_offset"]),
                )
                shift = float(np.linalg.norm(wpos - last_wrist_target_pos))
                if shift < float(args.approach_wrist_tol):
                    approach_stable_ctr += 1
                else:
                    approach_stable_ctr = 0
                last_wrist_target_pos = wpos.copy()
                _set_arm_marker_pose(wpos, weuler)
                targets = arm_ik.compute(wpos, weuler, open_hand24)
                if targets is not None:
                    arm_j_live = np.array(targets.arm_joints, dtype=np.float64).ravel()[: len(arm_indices)]
                approach_step += 1
                if approach_stable_ctr >= int(args.approach_hold) or approach_step >= int(args.approach_max_steps):
                    phase = LabPhase.HAND_TUNE
                    print("[INFO] phase → HAND_TUNE (approach done; adjust sliders, then press [w] for wrist cube)")
            elif phase == LabPhase.WRIST_VERIFY:
                pos_t2, ori_t2 = arm_xf.get_world_poses()
                wrist_pos = _to_numpy(pos_t2[0]).ravel()[:3]
                wrist_euler = _quat_wxyz_to_euler_xyz(_to_numpy(ori_t2[0]).ravel()[:4])
                targets = arm_ik.compute(wrist_pos, wrist_euler, last_hand)
                if targets is not None:
                    arm_j_live = np.array(targets.arm_joints, dtype=np.float64).ravel()[: len(arm_indices)]
            elif phase == LabPhase.HAND_TUNE:
                pass
            else:
                # WRIST_VERIFY for pour/inhand initial, or marker mode non-pickup
                pos_t2, ori_t2 = arm_xf.get_world_poses()
                wrist_pos = _to_numpy(pos_t2[0]).ravel()[:3]
                wrist_euler = _quat_wxyz_to_euler_xyz(_to_numpy(ori_t2[0]).ravel()[:4])
                targets = arm_ik.compute(wrist_pos, wrist_euler, last_hand)
                if targets is not None:
                    arm_j_live = np.array(targets.arm_joints, dtype=np.float64).ravel()[: len(arm_indices)]

            actions = _build_action(last_hand)
            if env.num_envs > 1:
                actions = actions.expand(env.num_envs, -1).clone()
            env.step(actions)

            if phase == LabPhase.HAND_TUNE:
                _sync_marker_to_sim_wrist()

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
