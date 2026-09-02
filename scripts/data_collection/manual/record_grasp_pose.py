#!/usr/bin/env python3
"""Fix arm from YAML, drive hand closure from a **cyan** visual cube (pour / pickup presets).

Loads the same task presets as ``record_arm_pose.py``. Spawns
``/World/Debug/GraspClosureTarget`` (cyan). Move it along ``--closure-axis``; its world
coordinate in ``[closure_min, closure_max]`` maps to grasp alpha in ``[0, 1]``. Finger flexion
interpolates from open to a reference closed pose (tunable via ``--closed-hand-yaml``).

By default this script allows all five fingers to move and record thumb flexion. Set
``--finger-mode four`` to lock thumb joints (THJ*) to 0 (four-finger preset).

On clean exit (close Isaac window or stop the script), if ``--save-yaml`` is set, writes a
file compatible with ``full_ik`` / GUI configs (``arm_joint_pos``, ``hand_joint_pos_shadow_order``).

Examples (Isaac Sim python)::

    ./python.sh scripts/data_collection/manual/record_grasp_pose.py \\
        --task pour --enable_cameras \\
        --arm-yaml scripts/data_collection/manual/config/pour_grasp.yaml \\
        --save-yaml scripts/data_collection/manual/config/tuned_pour_grasp.yaml
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
import types
import tty
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import yaml

from isaaclab.app import AppLauncher

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

GRASP_MARKER_PATH = "/World/Debug/GraspClosureTarget"
ARM_MARKER_PATH = "/World/Debug/GraspArmTarget"
# Shadow order: thumb = THJ5..THJ1 at indices 19..23
_THUMB_SLICE = slice(19, 24)
# Shadow order flex joint slices (FFJ3..1, MFJ3..1, RFJ3..1, LFJ3..1)
_FF_FLEX_SLICE = slice(3, 6)
_MF_FLEX_SLICE = slice(7, 10)
_RF_FLEX_SLICE = slice(11, 14)
_LF_FLEX_SLICE = slice(16, 19)
# Pickup-cube thumb target (THJ5..THJ1) used in pickup_cube profile.
_PICKUP_CUBE_THUMB_CLOSED = np.array([0.10, 0.40, 0.22, 0.55, 0.85], dtype=np.float64)

# Default “closed” template (from recorded_grasp_3); override with --closed-hand-yaml
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


def _load_symbol(entry: str) -> Any:
    if ":" not in entry:
        raise ValueError(f"Invalid entry '{entry}'. Expected 'module.path:SymbolName'.")
    mod_name, sym_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _load_arm_joint_dict(yaml_path: Path) -> dict[str, float]:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    jp = data.get("joint_pos")
    if jp is None:
        jp = data.get("arm_joint_pos")
    if jp is None or not isinstance(jp, dict):
        raise ValueError(f"{yaml_path}: need top-level joint_pos or arm_joint_pos (UR10e rad)")
    return {str(k): float(v) for k, v in jp.items()}


def _load_hand_shadow_24(yaml_path: Path) -> np.ndarray:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    seq = data.get("hand_joint_pos_shadow_order")
    if not isinstance(seq, list) or len(seq) != 24:
        raise ValueError(f"{yaml_path}: need hand_joint_pos_shadow_order: [24 floats]")
    return np.array([float(x) for x in seq], dtype=np.float64)


def _load_yaml_doc(yaml_path: Path) -> dict:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR10e arm fixed from YAML; hand closure from cyan cube (Shadow Hand pour/pickup presets).",
    )
    p.add_argument("--task", choices=sorted(_TASK_PRESETS.keys()), default="pour", help="Preset task.")
    p.add_argument("--env", type=str, default="", help="Env entry module:Class (overrides --task).")
    p.add_argument("--cfg", type=str, default="", help="Cfg entry module:Class (overrides --task).")
    p.add_argument("--num_envs", type=int, default=1, help="Number of envs (default: 1).")
    p.add_argument(
        "--max-episode-length",
        type=int,
        default=200000,
        help="Override episode length in env steps (large default avoids automatic timeout reset while tuning).",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Simulation loop target FPS.")
    p.add_argument(
        "--arm-yaml",
        type=str,
        default="",
        help="Optional arm seed YAML (joint_pos or arm_joint_pos). If omitted, uses current env-reset arm joints.",
    )
    p.add_argument(
        "--closed-hand-yaml",
        type=str,
        default="",
        help="Optional YAML with hand_joint_pos_shadow_order (24) used as alpha=1 pose; default = built-in template.",
    )
    p.add_argument(
        "--save-yaml",
        type=str,
        default="",
        help="On exit, write grasp snapshot (arm + hand + closure metadata) to this path.",
    )
    p.add_argument(
        "--closure-axis",
        choices=("x", "y", "z"),
        default="x",
        help="World axis of the cyan cube position used for closure alpha.",
    )
    p.add_argument(
        "--closure-min",
        type=float,
        default=0.42,
        help="Cube world coordinate on closure-axis → alpha=0 (open). Tune to your layout.",
    )
    p.add_argument(
        "--closure-max",
        type=float,
        default=0.72,
        help="Cube world coordinate on closure-axis → alpha=1 (closed template).",
    )
    p.add_argument(
        "--invert-closure",
        action="store_true",
        help="Use alpha <- 1 - alpha after mapping.",
    )
    p.add_argument(
        "--grasp-cube-pos",
        type=float,
        nargs=3,
        default=(0.55, 0.12, 0.42),
        metavar=("X", "Y", "Z"),
        help="Initial world position of the cyan grasp cube (m).",
    )
    p.add_argument(
        "--arm-control",
        choices=("fixed", "marker", "ik_trajectory", "cup_relative"),
        default="fixed",
        help="Arm mode: fixed=freeze from --arm-yaml, marker=magenta marker IK, ik_trajectory=follow ik_rl-style cup anchor, cup_relative=follow full_ik cup frame offsets.",
    )
    p.add_argument(
        "--arm-marker-pos",
        type=float,
        nargs=3,
        default=(0.65, 0.12, 0.42),
        metavar=("X", "Y", "Z"),
        help="Initial world position of the magenta arm marker (marker mode).",
    )
    p.add_argument(
        "--arm-marker-euler",
        type=float,
        nargs=3,
        default=(0.0, 2.2, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Initial world orientation (euler xyz, rad) of the magenta arm marker (marker mode).",
    )
    p.add_argument(
        "--print-arm-every",
        type=int,
        default=60,
        help="In marker mode, print solved arm joints every N steps (0=off).",
    )
    p.add_argument("--object-to-palm-offset", type=float, nargs=3, default=(0.0, 0.05, -0.03), help='Desired object-to-palm XYZ offset in meters for trajectory IK.')
    p.add_argument("--palm-in-wrist-pos", type=float, nargs=3, default=(0.0, 0.0, 0.35), help='Palm XYZ position expressed in the wrist frame, in meters.')
    p.add_argument(
        "--palm-in-wrist-euler",
        type=float,
        nargs=3,
        default=(1.5707963267948966, -1.5707963267948966, 1.5707963267948966),
        help='Palm XYZ Euler orientation expressed in the wrist frame, in radians.',
    )
    p.add_argument("--palm-orient", choices=("fixed", "pickup_down"), default="pickup_down", help='Palm orientation strategy: fixed Euler angles or pickup-down alignment.')
    p.add_argument("--palm-euler", type=float, nargs=3, default=(0.0, 2.2, 0.0), help='Fixed palm world XYZ Euler orientation in radians.')
    p.add_argument("--palm-normal-local", type=float, nargs=3, default=(0.0, 0.0, 1.0), help='Local palm normal vector used by pickup-down alignment.')
    p.add_argument("--world-down", type=float, nargs=3, default=(0.0, 0.0, -1.0), help='World-frame down vector used by pickup-down alignment.')
    p.add_argument("--palm-yaw-offset", type=float, default=3.141592653589793, help='Additional palm yaw offset in radians after alignment.')
    p.add_argument(
        "--sync-full-ik-config",
        type=str,
        default="",
        help="Optional full_ik YAML path. If set, sync palm/trajectory-aligned arm params from this file before start.",
    )
    p.add_argument(
        "--wrist-pos-in-cup-frame",
        type=float,
        nargs=3,
        default=(0.03, 0.0, 0.11),
        metavar=("X", "Y", "Z"),
        help="cup_relative wrist position offset in cup frame (meters). Use same values as full_ik phase.",
    )
    p.add_argument(
        "--wrist-euler-in-cup-frame",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="cup_relative wrist euler offset in cup frame (rad). Use same values as full_ik phase.",
    )
    p.add_argument(
        "--disable-hotkeys",
        action="store_true",
        help="Disable terminal hotkeys. By default: [f]=toggle finger lock, [t]=toggle thumb (4/5-finger), [r]=manual reset, [q]=quit.",
    )
    p.add_argument(
        "--finger-mode",
        choices=("four", "five"),
        default="five",
        help="Four: thumb joints (THJ*) forced to 0 (cup-style preset). Five: thumb joints follow closure alpha.",
    )
    p.add_argument(
        "--grasp-profile",
        choices=("uniform", "pickup_cube"),
        default="uniform",
        help="Hand closure mapping. pickup_cube uses non-uniform finger gains + opposed thumb for cube grasping.",
    )
    p.add_argument(
        "--manual-reset-only",
        action="store_true",
        default=True,
        help="Disable env auto-reset on done; only reset when pressing [r].",
    )
    p.add_argument(
        "--allow-auto-reset",
        action="store_true",
        help="Allow environment auto-reset on done/timeout (overrides --manual-reset-only).",
    )
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = run until close).")
    AppLauncher.add_app_launcher_args(p)
    return p


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _hand_joint_for_robot_name(name: str, hand_joints: np.ndarray, sh_names: list[str]) -> float:
    for sh_idx, sh_name in enumerate(sh_names):
        if sh_name in name or name.endswith(sh_name):
            return float(hand_joints[sh_idx])
    return 0.0


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


def _root_pose_w_to_T_44(pq: torch.Tensor) -> np.ndarray:
    pq = pq.detach().cpu().numpy().ravel()
    pos = pq[:3].astype(np.float64)
    w, x, y, z = [float(v) for v in pq[3:7]]
    rm = R.from_quat([x, y, z, w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rm
    T[:3, 3] = pos
    return T


def _quat_wxyz_to_rot(q_wxyz: np.ndarray) -> R:
    q = np.asarray(q_wxyz, dtype=np.float64).ravel()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return R.from_quat([x, y, z, w])


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


def _save_grasp_yaml(
    path: Path,
    *,
    task: str,
    closure_axis: str,
    closure_min: float,
    closure_max: float,
    invert_closure: bool,
    alpha: float,
    finger_mode: str,
    arm_dict: dict[str, float],
    hand_vec: np.ndarray,
    sh_names: list[str],
) -> None:
    named = {sh_names[i]: float(hand_vec[i]) for i in range(24)}
    doc = {
        "task": task,
        "closure_axis": closure_axis,
        "closure_min": float(closure_min),
        "closure_max": float(closure_max),
        "invert_closure": bool(invert_closure),
        "grasp_closure_alpha": float(alpha),
        "finger_mode": str(finger_mode),
        "arm_joint_pos": dict(arm_dict),
        "hand_joint_pos_shadow_order": [float(hand_vec[i]) for i in range(24)],
        "hand_joint_pos_named": named,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


def _install_terminal_cbreak() -> tuple[int, Any] | None:
    if os.name != "posix":
        return None
    if not sys.stdin.isatty():
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


def main() -> int:
    args = _build_arg_parser().parse_args()
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
    EnvCls = _load_symbol(env_entry)
    CfgCls = _load_symbol(cfg_entry)

    arm_path: Path | None = None
    arm_dict: dict[str, float] | None = None
    if str(args.arm_yaml).strip():
        arm_path = Path(args.arm_yaml).expanduser()
        if not arm_path.is_absolute():
            arm_path = (repo_root / arm_path).resolve()
        arm_dict = _load_arm_joint_dict(arm_path)

    if str(args.closed_hand_yaml).strip():
        ch_path = Path(args.closed_hand_yaml).expanduser()
        if not ch_path.is_absolute():
            ch_path = (repo_root / ch_path).resolve()
        closed_ref = _load_hand_shadow_24(ch_path)
    else:
        closed_ref = np.array(_DEFAULT_CLOSED_24, dtype=np.float64)

    if str(args.sync_full_ik_config).strip():
        fk_path = Path(args.sync_full_ik_config).expanduser()
        if not fk_path.is_absolute():
            fk_path = (repo_root / fk_path).resolve()
        if not fk_path.is_file():
            raise FileNotFoundError(f"--sync-full-ik-config not found: {fk_path}")
        fk = _load_yaml_doc(fk_path)
        if "object_to_palm_offset" in fk:
            args.object_to_palm_offset = tuple(float(x) for x in fk["object_to_palm_offset"])
        if "palm_in_wrist_pos" in fk:
            args.palm_in_wrist_pos = tuple(float(x) for x in fk["palm_in_wrist_pos"])
        if "palm_in_wrist_euler" in fk:
            args.palm_in_wrist_euler = tuple(float(x) for x in fk["palm_in_wrist_euler"])
        if "palm_orient" in fk:
            args.palm_orient = str(fk["palm_orient"])
        if "palm_euler" in fk:
            args.palm_euler = tuple(float(x) for x in fk["palm_euler"])
        if "palm_normal_local" in fk:
            args.palm_normal_local = tuple(float(x) for x in fk["palm_normal_local"])
        if "world_down" in fk:
            args.world_down = tuple(float(x) for x in fk["world_down"])
        if "palm_yaw_offset" in fk:
            args.palm_yaw_offset = float(fk["palm_yaw_offset"])

        # If full_ik YAML has a grasp hand_yaml and user didn't pass --closed-hand-yaml, use it as closed template.
        if not str(args.closed_hand_yaml).strip():
            for ph in (fk.get("phase_schedule") or []):
                if not isinstance(ph, dict):
                    continue
                hy = ph.get("hand_yaml")
                if not hy:
                    continue
                try:
                    hy_p = Path(str(hy)).expanduser()
                    if not hy_p.is_absolute():
                        hy_p = (repo_root / hy_p).resolve()
                    if hy_p.is_file():
                        closed_ref = _load_hand_shadow_24(hy_p)
                except Exception:
                    pass
        print(f"[INFO] Synced arm params from full_ik YAML: {fk_path}")

    cfg = CfgCls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    if int(args.max_episode_length) > 0:
        n = int(args.max_episode_length)
        if hasattr(cfg, "max_episode_length"):
            cfg.max_episode_length = n
        elif hasattr(cfg, "episode_length_s") and hasattr(cfg, "sim") and hasattr(cfg, "decimation"):
            cfg.episode_length_s = float(n) * float(cfg.sim.dt) * float(cfg.decimation)

    print(f"[INFO] Creating {EnvCls.__name__} (num_envs={cfg.scene.num_envs}) ...")
    env = EnvCls(cfg)
    if bool(args.manual_reset_only) and not bool(args.allow_auto_reset):
        def _never_done(_self):
            z = torch.zeros(_self.num_envs, dtype=torch.bool, device=_self.device)
            return z, z

        env._get_dones = types.MethodType(_never_done, env)  # type: ignore[method-assign]
        print("[INFO] manual-reset-only: auto done/reset disabled; press [r] to reset.")
    robot = env.robot
    joint_names = list(robot.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = [i for i, n in enumerate(joint_names) if re.match(arm_expr, n)]
    hand_indices = [i for i, n in enumerate(joint_names) if re.match(hand_expr, n)]
    sh_names = shadowhand_joint_names()

    env.reset()

    if arm_dict is None:
        jpos = robot.data.joint_pos[0].detach().cpu().numpy()
        arm_j_fixed = np.array([float(jpos[j]) for j in arm_indices], dtype=np.float64)
        arm_source_desc = "env reset joint_pos (auto)"
    else:
        arm_j_fixed = np.array([float(arm_dict.get(joint_names[j], 0.0)) for j in arm_indices], dtype=np.float64)
        arm_source_desc = f"YAML: {arm_path}"
    arm_j_live = arm_j_fixed.copy()

    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.prims import XFormPrim

    gpos = np.array(args.grasp_cube_pos, dtype=np.float64)
    VisualCuboid(
        prim_path=GRASP_MARKER_PATH,
        size=0.04,
        position=gpos,
        visible=True,
        color=np.array([0.2, 0.95, 0.95]),
    )
    grasp_xf = XFormPrim(prim_paths_expr=GRASP_MARKER_PATH, name="GraspClosureTarget", usd=True)
    arm_xf: Optional[XFormPrim] = None
    arm_ik: Optional[VideoTeleopControl] = None
    ik_anchor_name: Optional[str] = None
    if str(args.arm_control) == "marker":
        apos = np.array(args.arm_marker_pos, dtype=np.float64)
        VisualCuboid(
            prim_path=ARM_MARKER_PATH,
            size=0.04,
            position=apos,
            visible=True,
            color=np.array([0.95, 0.2, 0.95]),
        )
        arm_xf = XFormPrim(prim_paths_expr=ARM_MARKER_PATH, name="GraspArmTarget", usd=True)
        T0 = _make_T(apos, np.array(args.arm_marker_euler, dtype=np.float64))
        pos0 = T0[:3, 3]
        quat_xyzw = R.from_matrix(T0[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        arm_xf.set_world_poses(
            positions=torch.tensor([pos0], dtype=torch.float32, device="cpu"),
            orientations=torch.tensor([quat_wxyz], dtype=torch.float32, device="cpu"),
        )
        arm_ik = VideoTeleopControl()
    elif str(args.arm_control) == "cup_relative":
        if not hasattr(env, "cup"):
            raise RuntimeError("arm-control=cup_relative requires env.cup (e.g. pour task).")
        arm_ik = VideoTeleopControl()
    elif str(args.arm_control) == "ik_trajectory":
        if hasattr(env, "cup"):
            ik_anchor_name = "cup"
        elif hasattr(env, "object"):
            ik_anchor_name = "object"
        else:
            raise RuntimeError("arm-control=ik_trajectory requires env.cup or env.object as trajectory anchor.")
        arm_ik = VideoTeleopControl()

    ax_i = _axis_index(str(args.closure_axis))
    lo = float(min(args.closure_min, args.closure_max))
    hi = float(max(args.closure_min, args.closure_max))
    span = hi - lo if abs(hi - lo) > 1e-9 else 1.0

    def _alpha_from_pos(pos: np.ndarray) -> float:
        t = (float(pos[ax_i]) - lo) / span
        t = float(np.clip(t, 0.0, 1.0))
        if args.invert_closure:
            t = 1.0 - t
        return t

    thumb_zeroed = str(args.finger_mode).lower() == "four"

    def _hand_from_alpha(alpha: float) -> np.ndarray:
        h = alpha * closed_ref.copy()
        if str(args.grasp_profile) == "pickup_cube":
            # For cube grasping: avoid 4-finger flat pressing by reducing RF/LF closure
            # and making thumb opposition more active.
            h[_FF_FLEX_SLICE] *= 1.05
            h[_MF_FLEX_SLICE] *= 1.00
            h[_RF_FLEX_SLICE] *= 0.88
            h[_LF_FLEX_SLICE] *= 0.80
            if not thumb_zeroed:
                h[_THUMB_SLICE] = alpha * _PICKUP_CUBE_THUMB_CLOSED
        if thumb_zeroed:
            h[_THUMB_SLICE] = 0.0
        return h

    def _build_action(hand_joints: np.ndarray) -> torch.Tensor:
        full_dof = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_indices):
            if i < len(arm_j_live):
                full_dof[idx] = arm_j_live[i]
        for idx in hand_indices:
            full_dof[idx] = _hand_joint_for_robot_name(joint_names[idx], hand_joints, sh_names)
        actuated = full_dof[np.array(env.actuated_dof_indices, dtype=np.int64)]
        lower = env.robot_dof_lower_limits[0, env.actuated_dof_indices].cpu().numpy()
        upper = env.robot_dof_upper_limits[0, env.actuated_dof_indices].cpu().numpy()
        eps = 1e-6
        scale = np.where(upper - lower > eps, 2.0 * (actuated - lower) / (upper - lower) - 1.0, 0.0)
        return torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)

    save_path = Path(args.save_yaml).expanduser() if str(args.save_yaml).strip() else None
    if save_path and not save_path.is_absolute():
        save_path = (repo_root / save_path).resolve()

    last_alpha = 0.0
    last_hand = _hand_from_alpha(0.0)
    hand_locked = False

    print(f"[INFO] Arm seed source: {arm_source_desc}")
    if str(args.arm_control) == "fixed":
        print("[INFO] Arm mode=fixed (frozen to --arm-yaml).")
    elif str(args.arm_control) == "marker":
        print(f"[INFO] Arm mode=marker. Move magenta cube '{ARM_MARKER_PATH}' to control wrist pose via IK.")
    elif str(args.arm_control) == "ik_trajectory":
        print(
            f"[INFO] Arm mode=ik_trajectory. Wrist follows ik_rl-style anchor='{ik_anchor_name}' "
            "+ object_to_palm_offset; this matches train-time arm approach semantics."
        )
    else:
        print(
            "[INFO] Arm mode=cup_relative. Wrist tracks cup frame offsets: "
            f"pos={tuple(float(v) for v in args.wrist_pos_in_cup_frame)}, "
            f"euler={tuple(float(v) for v in args.wrist_euler_in_cup_frame)}"
        )
    print(f"[INFO] Move cyan cube '{GRASP_MARKER_PATH}'; axis={args.closure_axis!r} maps [{lo:.3f},{hi:.3f}] → alpha.")
    print(f"[INFO] grasp_profile={args.grasp_profile!r}, finger_mode={args.finger_mode!r}")
    print("[INFO] Suggested flow: move cyan cube -> press [f] lock fingers -> verify cup lift/move -> press [r] reset.")
    if save_path:
        print(f"[INFO] On exit, writing grasp YAML → {save_path}")
    print(f"[INFO] max_episode_length steps = {int(args.max_episode_length)}")
    if not bool(args.disable_hotkeys):
        print("[INFO] Hotkeys: [f] toggle finger lock, [t] toggle thumb (4/5 fingers), [r] reset env, [q] quit.")

    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0
    term_state: tuple[int, list[int]] | None = None

    try:
        if not bool(args.disable_hotkeys):
            term_state = _install_terminal_cbreak()
        while simulation_app.is_running():
            t0 = time.time()
            step += 1

            key = _poll_key_nonblocking() if not bool(args.disable_hotkeys) else None
            if key is not None:
                k = key.lower()
                if k == "q":
                    break
                if k == "f":
                    hand_locked = not hand_locked
                    state = "ON" if hand_locked else "OFF"
                    print(f"[INFO] finger lock: {state}")
                if k == "t":
                    thumb_zeroed = not thumb_zeroed
                    last_hand = _hand_from_alpha(last_alpha)
                    print(f"[INFO] finger-mode: {'four' if thumb_zeroed else 'five'} (thumb_zeroed={thumb_zeroed})")
                if k == "r":
                    env.reset()
                    arm_j_live = arm_j_fixed.copy()
                    print("[INFO] Manual reset applied.")

            pos_t, _ori_t = grasp_xf.get_world_poses()
            pos = pos_t[0].detach().cpu().numpy().ravel()[:3]
            if not hand_locked:
                alpha = _alpha_from_pos(pos)
                last_alpha = alpha
                last_hand = _hand_from_alpha(alpha)
            if arm_ik is not None:
                arm_ik.T_world_arm_base = _root_pose_w_to_T_44(env.robot.data.root_pose_w[0])
                if str(args.arm_control) == "marker" and arm_xf is not None:
                    pos_t2, ori_t2 = arm_xf.get_world_poses()
                    wrist_pos = _to_numpy(pos_t2[0]).ravel()[:3]
                    wrist_euler = _quat_wxyz_to_euler_xyz(_to_numpy(ori_t2[0]).ravel()[:4])
                elif str(args.arm_control) == "cup_relative":
                    cup_pos = env.cup.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64).ravel()[:3]
                    cup_q = (
                        env.cup.data.sim_element_quat_w[0, 0, :].detach().cpu().numpy().astype(np.float64).ravel()[:4]
                    )
                    cup_q = cup_q / (np.linalg.norm(cup_q) + 1e-12)
                    r_cup = _quat_wxyz_to_rot(cup_q)
                    off_p = np.array(args.wrist_pos_in_cup_frame, dtype=np.float64).ravel()[:3]
                    off_e = np.array(args.wrist_euler_in_cup_frame, dtype=np.float64).ravel()[:3]
                    wrist_pos = cup_pos + r_cup.apply(off_p)
                    wrist_euler = (r_cup * R.from_euler("xyz", off_e, degrees=False)).as_euler("xyz")
                elif str(args.arm_control) == "ik_trajectory":
                    if ik_anchor_name == "cup":
                        anchor_pos = env.cup.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64).ravel()[:3]
                    elif ik_anchor_name == "object":
                        anchor_pos = env.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64).ravel()[:3]
                    else:
                        raise RuntimeError("ik_trajectory anchor is not initialized.")
                    off = np.array(args.object_to_palm_offset, dtype=np.float64).ravel()[:3]
                    palm_pos = anchor_pos + off
                    if str(args.palm_orient) == "fixed":
                        palm_e = np.array(args.palm_euler, dtype=np.float64).ravel()[:3]
                    else:
                        palm_e = _pickup_down_euler(
                            np.array(args.palm_normal_local, dtype=np.float64).ravel()[:3],
                            np.array(args.world_down, dtype=np.float64).ravel()[:3],
                            float(args.palm_yaw_offset),
                        )
                    T_world_palm = _make_T(palm_pos, palm_e)
                    T_wrist_palm = _make_T(
                        np.array(args.palm_in_wrist_pos, dtype=np.float64).ravel()[:3],
                        np.array(args.palm_in_wrist_euler, dtype=np.float64).ravel()[:3],
                    )
                    T_world_wrist = T_world_palm @ np.linalg.inv(T_wrist_palm)
                    wrist_pos = T_world_wrist[:3, 3]
                    wrist_euler = R.from_matrix(T_world_wrist[:3, :3]).as_euler("xyz")
                else:
                    wrist_pos = np.zeros(3, dtype=np.float64)
                    wrist_euler = np.zeros(3, dtype=np.float64)
                targets: Optional[ArmHandTargets] = arm_ik.compute(wrist_pos, wrist_euler, last_hand)
                if targets is not None:
                    arm_j_live = np.array(targets.arm_joints, dtype=np.float64).ravel()[: len(arm_indices)]
                elif step % 60 == 0:
                    print("[WARN] Arm IK failed at current target pose; keeping last valid arm joints.")

            actions = _build_action(last_hand)
            if env.num_envs > 1:
                actions = actions.expand(env.num_envs, -1).clone()
            env.step(actions)

            if step % 60 == 0:
                print(f"[INFO] grasp_closure_alpha ≈ {last_alpha:.4f} (cube {args.closure_axis}={pos[ax_i]:.4f})")
            if arm_ik is not None and int(args.print_arm_every) > 0 and step % int(args.print_arm_every) == 0:
                print("[INFO] marker arm_joint_pos:")
                for i, idx in enumerate(arm_indices):
                    if i < len(arm_j_live):
                        print(f'    "{joint_names[idx]}": {float(arm_j_live[i]):.6f}')

            elapsed = time.time() - t0
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

            if args.max_steps > 0 and step >= int(args.max_steps):
                break
    finally:
        _restore_terminal(term_state)
        if save_path is not None:
            finger_mode = "four" if thumb_zeroed else "five"
            arm_dict_save = {
                joint_names[idx]: float(arm_j_live[i]) for i, idx in enumerate(arm_indices) if i < len(arm_j_live)
            }
            _save_grasp_yaml(
                save_path,
                task=str(args.task),
                closure_axis=str(args.closure_axis),
                closure_min=float(args.closure_min),
                closure_max=float(args.closure_max),
                invert_closure=bool(args.invert_closure),
                alpha=last_alpha,
                finger_mode=finger_mode,
                arm_dict=arm_dict_save,
                hand_vec=last_hand,
                sh_names=sh_names,
            )
            print(f"[INFO] Wrote {save_path}")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
