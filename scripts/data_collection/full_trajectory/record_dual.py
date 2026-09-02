#!/usr/bin/env python3
"""Record dual-arm full trajectory with marker-driven IK + hand sliders."""

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
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R

_FULL_TRA_DIR = Path(__file__).resolve().parent
if str(_FULL_TRA_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_TRA_DIR))
from task_entries_dual import resolve_env_cfg_entries

MARKER_RIGHT_PRIM_PATH = "/World/Debug/ArmIkTargetRight"
MARKER_LEFT_PRIM_PATH = "/World/Debug/ArmIkTargetLeft"
POSE_KEYFRAME_DUAL_JSON = _FULL_TRA_DIR / "pose_keyframes_dual.json"


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


def _make_T(pos_xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64), degrees=False).as_matrix()
    t[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return t


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


def _root_T_world_np(robot, env_idx: int) -> np.ndarray:
    pos = robot.data.root_pos_w[int(env_idx)].detach().cpu().numpy().ravel()[:3]
    q = robot.data.root_quat_w[int(env_idx)].detach().cpu().numpy().ravel()[:4]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = R.from_quat(np.array([x, y, z, w], dtype=np.float64)).as_matrix()
    t[:3, 3] = pos
    return t


def _urdf_path(repo_root: Path, side: str) -> str:
    side = side.lower()
    return str(
        repo_root
        / "source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e"
        / f"ur10e_shadow_{side}_hand_glb.urdf"
    )


def _load_dual_init_keyframe(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    keyframes = data.get("keyframes", {})
    init = keyframes.get("Init")
    if not isinstance(init, dict):
        raise RuntimeError(f"Missing 'Init' keyframe in {path}")
    return init


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dual-arm full trajectory recorder.")
    p.add_argument(
        "--task",
        type=str,
        default="bi_blind_grasp",
        help="Dual-UR10e alias or any registered dual-arm Gym ID with env_cfg_entry_point.",
    )
    p.add_argument("--env", type=str, default="", help="module:EnvClass")
    p.add_argument("--cfg", type=str, default="", help="module:CfgClass")
    p.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1, help='Number of parallel simulation environments.')
    p.add_argument("--fps", type=float, default=30.0, help='Target control and recording loop frequency in frames per second.')
    p.add_argument("--object-init-index", type=int, default=0, help='Object initialization preset index; -1 uses the trajectory/keyframe default.')
    p.add_argument("--record-dir", type=str, default="./scripts/data_collection/full_trajectory/records_dual", help='Directory in which the recorded trajectory JSON is written.')
    p.add_argument("--record-name", type=str, default="traj_dual", help='Output trajectory filename; omit to generate a timestamped name.')
    p.add_argument("--hand-joints", choices=["zeros", "sim"], default="sim", help='Initial hand source: zeros or current simulator joint values.')
    hg = p.add_mutually_exclusive_group()
    hg.add_argument("--hand-gui", dest="hand_gui", action="store_true", help="Show Shadow Hand joint sliders (default).")
    hg.add_argument("--no-hand-gui", dest="hand_gui", action="store_false", help="Disable hand sliders.")
    p.add_argument(
        "--hold-initial-pose",
        action="store_true",
        help="Keep both arms at reset pose until either marker is moved or recording starts.",
    )
    p.add_argument("--print-every", type=int, default=0, help="Print arm/hand command info every N steps (0=disable).")
    p.add_argument("--print-on-change", action="store_true", help="Print when commands change.")
    p.add_argument("--print-hand-rad", action="store_true", help="Also print hand target joints in rad.")
    p.add_argument("--marker-right-pos", type=float, nargs=3, default=(0.3, 0.00, 0.58), help='Initial right-arm IK marker world position X Y Z in meters.')
    p.add_argument("--marker-right-euler", type=float, nargs=3, default=(0.0, 1.57, 0.0), help='Initial right-arm IK marker XYZ Euler orientation in radians.')
    p.add_argument("--marker-left-pos", type=float, nargs=3, default=(1, 0.00, 0.58), help='Initial left-arm IK marker world position X Y Z in meters.')
    p.add_argument("--marker-left-euler", type=float, nargs=3, default=(0.0, 1.57, 0.0), help='Initial left-arm IK marker XYZ Euler orientation in radians.')
    p.add_argument("--max-steps", type=int, default=0, help='Maximum control-loop steps; 0 runs until quit or environment termination.')
    p.add_argument("--show_rgb", action="store_true", help="Reserved for compatibility with single-arm CLI.")
    p.add_argument("--show_ff", action="store_true", help="Reserved for compatibility with single-arm CLI.")
    reset_group = p.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--manual-reset-only",
        dest="manual_reset_only",
        action="store_true",
        help="Disable auto-reset in env.step; only reset from UI Reset button.",
    )
    reset_group.add_argument(
        "--auto-reset",
        dest="manual_reset_only",
        action="store_false",
        help="Allow the environment to auto-reset when episode ends.",
    )
    p.set_defaults(manual_reset_only=False, hand_gui=True, hold_initial_pose=True)
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.show_rgb or args.show_ff:
        args.enable_cameras = True
    app = AppLauncher(args).app

    import ViTacLab.tasks  # noqa: F401

    repo_root = _repo_root()
    source_dir = repo_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    init_keyframe = _load_dual_init_keyframe(POSE_KEYFRAME_DUAL_JSON)

    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.prims import XFormPrim
    from video_teleop.core.shadowhand_joints import shadowhand_joint_names
    from video_teleop.core.video_teleop_control import VideoTeleopControl

    env_entry, cfg_entry, preset_key = resolve_env_cfg_entries(task=args.task, env=args.env, cfg=args.cfg)
    print(f"[INFO] env={env_entry}\n[INFO] cfg={cfg_entry}")
    env_cls = _load_symbol(env_entry)
    cfg_cls = _load_symbol(cfg_entry)

    cfg = cfg_cls()
    cfg.scene.num_envs = max(1, int(args.num_envs))
    cfg.device = getattr(args, "device", None) or "cuda:0"
    if hasattr(cfg, "object_init_choice"):
        cfg.object_init_choice = int(args.object_init_index)
    setattr(cfg, "enable_cameras", bool(getattr(args, "enable_cameras", False)))
    print(f"[INFO] cfg.enable_cameras={getattr(cfg, 'enable_cameras', None)}")

    print(f"[INFO] Creating {env_cls.__name__} (num_envs={cfg.scene.num_envs}) ...")
    env = env_cls(cfg)
    allow_reset_gate = {"allow": not bool(args.manual_reset_only)}
    reset_mode_state = {"manual_only": bool(args.manual_reset_only)}
    orig_reset_idx = getattr(env, "_reset_idx", None)
    if callable(orig_reset_idx):

        def _gated_reset_idx(env_ids):
            if allow_reset_gate["allow"]:
                return orig_reset_idx(env_ids)
            if env_ids is not None:
                ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
                env.episode_length_buf[ids] = 0
            return None

        env._reset_idx = _gated_reset_idx  # type: ignore[attr-defined]
        if reset_mode_state["manual_only"]:
            print("[INFO] manual-reset-only enabled: auto reset is blocked; use UI Reset button.")
        else:
            print("[INFO] auto-reset enabled.")
    try:
        allow_reset_gate["allow"] = True
        env.reset()
    finally:
        allow_reset_gate["allow"] = not bool(reset_mode_state["manual_only"])
    right = env.right_hand
    left = env.left_hand
    env_idx = 0

    joint_names = list(right.joint_names)
    arm_expr = getattr(cfg, "arm_joint_expr", ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*")
    hand_expr = getattr(cfg, "hand_joint_expr", ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*")
    arm_indices = sorted([i for i, n in enumerate(joint_names) if re.match(arm_expr, n)])
    hand_indices = sorted([i for i, n in enumerate(joint_names) if re.match(hand_expr, n)])
    sh_names = shadowhand_joint_names()
    print(f"[INFO] right/left joint count={len(joint_names)}, actuated={len(env.actuated_dof_indices)}")

    control_right = VideoTeleopControl(
        urdf_path=_urdf_path(repo_root, "right"), T_world_arm_base=_root_T_world_np(right, env_idx)
    )
    control_left = VideoTeleopControl(
        urdf_path=_urdf_path(repo_root, "left"), T_world_arm_base=_root_T_world_np(left, env_idx)
    )

    def _spawn_marker(path: str, pos: tuple[float, float, float], euler: tuple[float, float, float], color: np.ndarray):
        VisualCuboid(prim_path=path, size=0.04, position=np.array(pos, dtype=np.float64), visible=True, color=color)
        xf = XFormPrim(prim_paths_expr=path, name=path.split("/")[-1], usd=True)
        t0 = _make_T(np.array(pos, dtype=np.float64), np.array(euler, dtype=np.float64))
        p0 = t0[:3, 3]
        q_xyzw = R.from_matrix(t0[:3, :3]).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
        xf.set_world_poses(
            positions=torch.tensor([p0], dtype=torch.float32, device="cpu"),
            orientations=torch.tensor([q_wxyz], dtype=torch.float32, device="cpu"),
        )
        return xf

    marker_right_pos_default = tuple(np.asarray(init_keyframe.get("marker_right_pos_w", args.marker_right_pos), dtype=np.float64).ravel()[:3])
    marker_right_euler_default = tuple(np.asarray(init_keyframe.get("marker_right_euler_xyz", args.marker_right_euler), dtype=np.float64).ravel()[:3])
    marker_left_pos_default = tuple(np.asarray(init_keyframe.get("marker_left_pos_w", args.marker_left_pos), dtype=np.float64).ravel()[:3])
    marker_left_euler_default = tuple(np.asarray(init_keyframe.get("marker_left_euler_xyz", args.marker_left_euler), dtype=np.float64).ravel()[:3])

    marker_right_xf = _spawn_marker(
        MARKER_RIGHT_PRIM_PATH, marker_right_pos_default, marker_right_euler_default, np.array([1.0, 0.2, 0.8])
    )
    marker_left_xf = _spawn_marker(
        MARKER_LEFT_PRIM_PATH, marker_left_pos_default, marker_left_euler_default, np.array([0.2, 1.0, 0.8])
    )

    def _hand_joints_from_robot(robot: Any, env_i: int) -> np.ndarray:
        out = np.zeros(24, dtype=np.float64)
        q = robot.data.joint_pos[int(env_i)].detach().cpu().numpy()
        for sh_i, sh_name in enumerate(sh_names):
            for idx in hand_indices:
                n = joint_names[idx]
                if sh_name in n or n.endswith(sh_name):
                    out[sh_i] = float(q[idx])
                    break
        return out

    def _hand_joint_for_robot_name(name: str, hand_joints: np.ndarray) -> float:
        for sh_idx, sh_name in enumerate(sh_names):
            if sh_name in name or name.endswith(sh_name):
                return float(hand_joints[sh_idx])
        return 0.0

    def _arm_joints_from_robot(robot: Any, env_i: int) -> np.ndarray:
        q = robot.data.joint_pos[int(env_i)].detach().cpu().numpy()
        return np.array([float(q[idx]) for idx in arm_indices], dtype=np.float64)

    def _build_actions_for_robot(robot: Any, arm_joints: np.ndarray, hand_joints: np.ndarray) -> np.ndarray:
        full = np.zeros(robot.num_joints, dtype=np.float64)
        for i, idx in enumerate(arm_indices):
            if i < len(arm_joints):
                full[idx] = float(arm_joints[i])
        for idx in hand_indices:
            full[idx] = _hand_joint_for_robot_name(joint_names[idx], hand_joints)
        act_idx = np.array(env.actuated_dof_indices, dtype=np.int64)
        joints = full[act_idx]
        lo = env.robot_dof_lower_limits[0, env.actuated_dof_indices].detach().cpu().numpy()
        hi = env.robot_dof_upper_limits[0, env.actuated_dof_indices].detach().cpu().numpy()
        scale = np.where(hi - lo > 1e-6, 2.0 * (joints - lo) / (hi - lo) - 1.0, 0.0)
        return np.clip(scale, -1.0, 1.0)

    import omni.ui  # type: ignore

    hand_gui_models_r: list[Any] = []
    hand_gui_models_l: list[Any] = []
    hand_pose_status_model: Any = None
    keyframe_status_model: Any = None
    keyframe_name_model: Any = None
    keyframe_selected_model: Any = None
    record_state: dict[str, Any] = {"active": False, "frames": []}
    record_status_model: Any = None
    reset_mode_model: Any = None
    record_path_model: Any = None
    record_name_model: Any = None
    pending_reset = {"flag": False}
    hold_initial_pose_state = {"active": bool(args.hold_initial_pose), "marker_moved": False}
    initial_marker_state = {
        "right_pos": np.asarray(marker_right_pos_default, dtype=np.float64).ravel()[:3].copy(),
        "right_quat": None,
        "left_pos": np.asarray(marker_left_pos_default, dtype=np.float64).ravel()[:3].copy(),
        "left_quat": None,
    }
    hand_pose_preset_path = repo_root / "scripts/data_collection/full_trajectory/hand_pose_presets_dual.json"
    pose_keyframe_path = repo_root / "scripts/data_collection/full_trajectory/pose_keyframes_dual.json"
    hand_pose_state: dict[str, dict[str, np.ndarray]] = {}
    pose_keyframes: dict[str, dict[str, Any]] = {}
    pending_keyframe_apply: dict[str, str | None] = {"name": None}

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

    def _string_model_set(model: Any, value: str) -> None:
        if model is not None and hasattr(model, "set_value"):
            model.set_value(str(value))

    def _string_model_get(model: Any, default: str = "") -> str:
        if model is not None and hasattr(model, "get_value_as_string"):
            return str(model.get_value_as_string()).strip()
        return str(default)

    def _set_status(msg: str) -> None:
        if record_status_model is not None and hasattr(record_status_model, "set_value"):
            record_status_model.set_value(str(msg))

    def _hand_pose_status_set(msg: str) -> None:
        _string_model_set(hand_pose_status_model, msg)

    def _keyframe_status_set(msg: str) -> None:
        _string_model_set(keyframe_status_model, msg)

    def _reset_mode_label(manual_only: bool) -> str:
        return "Manual reset only" if manual_only else "Auto reset"

    def _set_manual_reset_only(enabled: bool) -> None:
        reset_mode_state["manual_only"] = bool(enabled)
        allow_reset_gate["allow"] = not bool(enabled)
        _set_status(f"Reset mode: {_reset_mode_label(reset_mode_state['manual_only'])}")
        print(f"[INFO] Reset mode -> {_reset_mode_label(reset_mode_state['manual_only'])}")

    def _set_hand_gui_pose_pair(hand_right: np.ndarray, hand_left: np.ndarray) -> None:
        if len(hand_gui_models_r) != 24 or len(hand_gui_models_l) != 24:
            raise RuntimeError("Hand GUI sliders are not available.")
        hr = np.asarray(hand_right, dtype=np.float64).ravel()
        hl = np.asarray(hand_left, dtype=np.float64).ravel()
        if hr.size != 24 or hl.size != 24:
            raise ValueError("Expected 24 hand joints per side.")
        for model, value in zip(hand_gui_models_r, hr, strict=False):
            _float_model_set(model, float(value))
        for model, value in zip(hand_gui_models_l, hl, strict=False):
            _float_model_set(model, float(value))

    def _load_hand_pose_presets() -> None:
        hand_pose_state.clear()
        if not hand_pose_preset_path.is_file():
            return
        try:
            data = json.loads(hand_pose_preset_path.read_text(encoding="utf-8"))
            for name, payload in dict(data.get("poses", {})).items():
                r = np.asarray(payload.get("right", []), dtype=np.float64).ravel()
                l = np.asarray(payload.get("left", []), dtype=np.float64).ravel()
                if r.size == 24 and l.size == 24:
                    hand_pose_state[str(name)] = {"right": r, "left": l}
        except Exception as e:
            print(f"[WARN] Failed to load hand pose presets: {e}")

    def _save_hand_pose_presets() -> None:
        hand_pose_preset_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "vitatlab_hand_pose_presets_dual_v1",
            "updated_at": datetime.now().isoformat(),
            "poses": {
                n: {"right": v["right"].tolist(), "left": v["left"].tolist()} for n, v in hand_pose_state.items()
            },
        }
        hand_pose_preset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_hand_pose(name: str) -> None:
        hr, hl = _hand_pair_for_step()
        hand_pose_state[str(name)] = {"right": hr.copy(), "left": hl.copy()}
        _save_hand_pose_presets()
        _hand_pose_status_set(f"{name}_record saved")
        print(f"[INFO] Hand pose '{name}' saved -> {hand_pose_preset_path}")

    def _apply_hand_pose(name: str) -> None:
        pose = hand_pose_state.get(str(name))
        if pose is None:
            _hand_pose_status_set(f"{name} not recorded")
            return
        try:
            _set_hand_gui_pose_pair(pose["right"], pose["left"])
            _hand_pose_status_set(f"{name} applied")
            print(f"[INFO] Hand pose '{name}' applied.")
        except Exception as e:
            _hand_pose_status_set(f"{name} apply failed")
            print(f"[WARN] Failed to apply hand pose '{name}': {e}")

    def _load_pose_keyframes() -> None:
        pose_keyframes.clear()
        if not pose_keyframe_path.is_file():
            return
        try:
            data = json.loads(pose_keyframe_path.read_text(encoding="utf-8"))
            for name, frame in dict(data.get("keyframes", {})).items():
                pose_keyframes[str(name)] = dict(frame)
        except Exception as e:
            print(f"[WARN] Failed to load pose keyframes: {e}")

    def _save_pose_keyframes() -> None:
        pose_keyframe_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "vitatlab_pose_keyframes_dual_v1",
            "updated_at": datetime.now().isoformat(),
            "keyframes": pose_keyframes,
        }
        pose_keyframe_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _selected_keyframe_name() -> str:
        return _string_model_get(keyframe_selected_model, "")

    def _refresh_keyframe_selection() -> None:
        names = sorted(pose_keyframes.keys())
        cur = _selected_keyframe_name()
        if cur and cur in pose_keyframes:
            _string_model_set(keyframe_selected_model, cur)
            return
        _string_model_set(keyframe_selected_model, names[0] if names else "")

    def _capture_pose_keyframe(name: str) -> dict[str, Any]:
        snap = _capture_snapshot()
        snap["name"] = str(name)
        return snap

    def _hand_pair_for_step() -> tuple[np.ndarray, np.ndarray]:
        if hand_gui_models_r and hand_gui_models_l and len(hand_gui_models_r) == 24 and len(hand_gui_models_l) == 24:
            hr = np.array([_float_model_get(m) for m in hand_gui_models_r], dtype=np.float64)
            hl = np.array([_float_model_get(m) for m in hand_gui_models_l], dtype=np.float64)
            return hr, hl
        if args.hand_joints == "zeros":
            z = np.zeros(24, dtype=np.float64)
            return z, z.copy()
        return _hand_joints_from_robot(right, env_idx), _hand_joints_from_robot(left, env_idx)

    def _capture_snapshot() -> dict[str, Any]:
        pos_r_t, ori_r_t = marker_right_xf.get_world_poses()
        pos_l_t, ori_l_t = marker_left_xf.get_world_poses()
        pos_r = _to_numpy(pos_r_t[0]).ravel()[:3]
        pos_l = _to_numpy(pos_l_t[0]).ravel()[:3]
        q_r = _to_numpy(ori_r_t[0]).ravel()[:4]
        q_l = _to_numpy(ori_l_t[0]).ravel()[:4]
        e_r = _quat_wxyz_to_euler_xyz(q_r)
        e_l = _quat_wxyz_to_euler_xyz(q_l)
        h_r, h_l = _hand_pair_for_step()
        snap = {
            "t_wall": float(time.time()),
            "marker_right_pos_w": pos_r.tolist(),
            "marker_right_quat_wxyz": q_r.tolist(),
            "marker_right_euler_xyz": e_r.tolist(),
            "marker_left_pos_w": pos_l.tolist(),
            "marker_left_quat_wxyz": q_l.tolist(),
            "marker_left_euler_xyz": e_l.tolist(),
            "arm_right_joint_pos": _arm_joints_from_robot(right, env_idx).tolist(),
            "arm_left_joint_pos": _arm_joints_from_robot(left, env_idx).tolist(),
            "hand_right_joint_pos_shadow_order": h_r.tolist(),
            "hand_left_joint_pos_shadow_order": h_l.tolist(),
            "object_init_index": int(getattr(cfg, "object_init_choice", 0)),
        }
        if hasattr(env, "object") and hasattr(env.object, "data"):
            snap["object_pos_env"] = _to_numpy(env.object.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3].tolist()
        if hasattr(env, "trash_can") and hasattr(env.trash_can, "data"):
            snap["trash_can_pos_env"] = _to_numpy(env.trash_can.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[
                :3
            ].tolist()
        if hasattr(env, "hole") and hasattr(env.hole, "data"):
            snap["hole_pos_env"] = _to_numpy(env.hole.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3].tolist()
        if hasattr(env, "peg") and hasattr(env.peg, "data"):
            snap["peg_pos_env"] = _to_numpy(env.peg.data.root_pos_w[0] - env.scene.env_origins[0]).ravel()[:3].tolist()
        return snap

    def _record_start_cb() -> None:
        record_state["active"] = True
        record_state["frames"] = []
        hold_initial_pose_state["active"] = False
        _set_status("Recording...")
        print("[INFO] Trajectory recording started.")

    def _record_snapshot_cb() -> None:
        if not record_state["active"]:
            _set_status("Not recording")
            return
        snap = _capture_pose_keyframe("")
        snap.pop("name", None)
        record_state["frames"].append(snap)
        _set_status(f"Snapshots: {len(record_state['frames'])}")
        print(f"[INFO] Snapshot #{len(record_state['frames'])} captured.")

    def _record_stop_cb() -> None:
        if not record_state["active"]:
            _set_status("Not recording")
            return
        rec_dir = str(args.record_dir).strip() or "."
        rec_name = str(args.record_name).strip() or "traj_dual"
        if record_path_model is not None and hasattr(record_path_model, "get_value_as_string"):
            rec_dir = str(record_path_model.get_value_as_string()).strip() or rec_dir
        if record_name_model is not None and hasattr(record_name_model, "get_value_as_string"):
            rec_name = str(record_name_model.get_value_as_string()).strip() or rec_name
        out_dir = Path(rec_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (repo_root / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{rec_name}_{ts}.json"
        doc = {
            "format": "vitatlab_full_tra_dual_v1",
            "task": str(args.task),
            "preset_key": preset_key,
            "env_entry": env_entry,
            "cfg_entry": cfg_entry,
            "created_at": datetime.now().isoformat(),
            "object_init_index": int(getattr(cfg, "object_init_choice", 0)),
            "num_frames": len(record_state["frames"]),
            "frames": list(record_state["frames"]),
        }
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        record_state["active"] = False
        _set_status(f"Saved: {out_path}")
        print(f"[INFO] Trajectory saved -> {out_path}")

    def _manual_reset_cb() -> None:
        pending_reset["flag"] = True

    def _toggle_reset_mode_cb() -> None:
        _set_manual_reset_only(not reset_mode_state["manual_only"])
        if reset_mode_model is not None and hasattr(reset_mode_model, "set_value"):
            reset_mode_model.set_value(_reset_mode_label(reset_mode_state["manual_only"]))

    def _open_record_cb() -> None:
        _record_hand_pose("open")

    def _open_cb() -> None:
        _apply_hand_pose("open")

    def _close_record_cb() -> None:
        _record_hand_pose("close")

    def _close_cb() -> None:
        _apply_hand_pose("close")

    def _record_pose_keyframe_cb() -> None:
        name = _string_model_get(keyframe_name_model, "").strip()
        if not name:
            _keyframe_status_set("Keyframe name is empty")
            return
        pose_keyframes[name] = _capture_pose_keyframe(name)
        _save_pose_keyframes()
        _refresh_keyframe_selection()
        _keyframe_status_set(f"Saved keyframe: {name}")
        print(f"[INFO] Pose keyframe '{name}' saved -> {pose_keyframe_path}")

    def _apply_pose_keyframe_cb() -> None:
        name = _selected_keyframe_name()
        if not name:
            _keyframe_status_set("No keyframe selected")
            return
        pending_keyframe_apply["name"] = str(name)

    def _rename_pose_keyframe_cb() -> None:
        src = _selected_keyframe_name()
        dst = _string_model_get(keyframe_name_model, "").strip()
        if not src:
            _keyframe_status_set("No keyframe selected")
            return
        if not dst:
            _keyframe_status_set("New name is empty")
            return
        if src not in pose_keyframes:
            _keyframe_status_set(f"Keyframe not found: {src}")
            return
        if dst != src and dst in pose_keyframes:
            _keyframe_status_set(f"Name exists: {dst}")
            return
        frame = dict(pose_keyframes.pop(src))
        frame["name"] = dst
        pose_keyframes[dst] = frame
        _save_pose_keyframes()
        _refresh_keyframe_selection()
        _string_model_set(keyframe_selected_model, dst)
        _keyframe_status_set(f"Renamed: {src} -> {dst}")
        print(f"[INFO] Pose keyframe renamed: {src} -> {dst}")

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

    _load_hand_pose_presets()
    _load_pose_keyframes()

    if args.hand_gui:
        window = omni.ui.Window(
            "Dual FullTra Recorder",
            width=520,
            height=920,
            visible=True,
            dock_preference=omni.ui.DockPreference.RIGHT_TOP,
        )
        with window.frame:
            with omni.ui.ScrollingFrame(horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                with omni.ui.VStack(spacing=3, height=0):
                    omni.ui.Label("Trajectory Recorder", height=18)
                    with omni.ui.HStack(spacing=4):
                        omni.ui.Label("Dir", width=30)
                        record_path_model = omni.ui.SimpleStringModel()
                        record_path_model.set_value(str(args.record_dir))
                        omni.ui.StringField(model=record_path_model, height=24)
                    with omni.ui.HStack(spacing=4):
                        omni.ui.Label("Name", width=40)
                        record_name_model = omni.ui.SimpleStringModel()
                        record_name_model.set_value(str(args.record_name))
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
                        f"Preset file: {hand_pose_preset_path.relative_to(repo_root)}",
                        word_wrap=True,
                    )
                    omni.ui.Label("Pose Keyframes (Dual Marker + Hands)", height=18)
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
                        f"Keyframe file: {pose_keyframe_path.relative_to(repo_root)}",
                        word_wrap=True,
                    )
                    with omni.ui.HStack(spacing=4):
                        omni.ui.Button("Start Recording", clicked_fn=_record_start_cb)
                        omni.ui.Button("Snapshot", clicked_fn=_record_snapshot_cb)
                        omni.ui.Button("Stop Recording", clicked_fn=_record_stop_cb)
                        omni.ui.Button("Reset", clicked_fn=_manual_reset_cb)
                    with omni.ui.HStack(spacing=4):
                        omni.ui.Label("Reset Mode", width=80)
                        reset_mode_model = omni.ui.SimpleStringModel()
                        reset_mode_model.set_value(_reset_mode_label(reset_mode_state["manual_only"]))
                        omni.ui.StringField(model=reset_mode_model, read_only=True, height=24)
                        omni.ui.Button("Toggle", clicked_fn=_toggle_reset_mode_cb, width=80)
                    record_status_model = omni.ui.SimpleStringModel()
                    record_status_model.set_value("Idle")
                    omni.ui.StringField(model=record_status_model, read_only=True, height=24)
                    _set_status(f"Reset mode: {_reset_mode_label(reset_mode_state['manual_only'])}")
                    omni.ui.Label("Right hand", style={"font_size": 14})
                    init_r = np.asarray(init_keyframe.get("hand_right_joint_pos_shadow_order", []), dtype=np.float64).ravel()[:24]
                    init_l = np.asarray(init_keyframe.get("hand_left_joint_pos_shadow_order", []), dtype=np.float64).ravel()[:24]
                    if init_r.size != 24:
                        init_r = _hand_joints_from_robot(right, env_idx) if args.hand_joints == "sim" else np.zeros(24, dtype=np.float64)
                    if init_l.size != 24:
                        init_l = _hand_joints_from_robot(left, env_idx) if args.hand_joints == "sim" else np.zeros(24, dtype=np.float64)
                    for sh_i, sh_name in enumerate(sh_names):
                        lo = float(env.robot_dof_lower_limits[0, hand_indices[min(sh_i, len(hand_indices) - 1)]].item())
                        hi = float(env.robot_dof_upper_limits[0, hand_indices[min(sh_i, len(hand_indices) - 1)]].item())
                        if hi - lo < 1e-6:
                            lo, hi = -1.5, 1.5
                        m = omni.ui.SimpleFloatModel()
                        _float_model_set(m, float(np.clip(init_r[sh_i], lo, hi)))
                        hand_gui_models_r.append(m)
                        with omni.ui.HStack(spacing=6):
                            omni.ui.Label(f"R {sh_name}", width=78)
                            omni.ui.FloatSlider(model=m, min=lo, max=hi, step=max(1e-4, (hi - lo) / 500.0))
                    omni.ui.Label("Left hand", style={"font_size": 14})
                    for sh_i, sh_name in enumerate(sh_names):
                        lo = float(env.robot_dof_lower_limits[0, hand_indices[min(sh_i, len(hand_indices) - 1)]].item())
                        hi = float(env.robot_dof_upper_limits[0, hand_indices[min(sh_i, len(hand_indices) - 1)]].item())
                        if hi - lo < 1e-6:
                            lo, hi = -1.5, 1.5
                        m = omni.ui.SimpleFloatModel()
                        _float_model_set(m, float(np.clip(init_l[sh_i], lo, hi)))
                        hand_gui_models_l.append(m)
                        with omni.ui.HStack(spacing=6):
                            omni.ui.Label(f"L {sh_name}", width=78)
                            omni.ui.FloatSlider(model=m, min=lo, max=hi, step=max(1e-4, (hi - lo) / 500.0))
        _refresh_keyframe_selection()
        if pose_keyframes:
            _keyframe_status_set(f"Loaded {len(pose_keyframes)} keyframes")
        elif not pose_keyframe_path.is_file():
            _keyframe_status_set("No saved keyframes")
        else:
            _keyframe_status_set("Keyframe load failed (see terminal WARN)")
    else:
        window = None

    target_dt = 1.0 / max(1e-3, float(args.fps))
    step = 0
    last_arm_print_r: np.ndarray | None = None
    last_arm_print_l: np.ndarray | None = None
    last_hand_print_r: np.ndarray | None = None
    last_hand_print_l: np.ndarray | None = None
    while app.is_running():
        t0 = time.time()
        step += 1
        if pending_reset["flag"]:
            try:
                allow_reset_gate["allow"] = True
                env.reset()
                hold_initial_pose_state["active"] = True
                hold_initial_pose_state["marker_moved"] = False
                initial_marker_state["right_quat"] = None
                initial_marker_state["left_quat"] = None
                _set_status("Reset done; holding Init pose")
            finally:
                allow_reset_gate["allow"] = not bool(reset_mode_state["manual_only"])
                pending_reset["flag"] = False

        pending_name = pending_keyframe_apply.get("name")
        if pending_name:
            keyframe = pose_keyframes.get(str(pending_name))
            if keyframe is None:
                _keyframe_status_set(f"Keyframe not found: {pending_name}")
            else:
                try:
                    mr_pos = np.asarray(keyframe["marker_right_pos_w"], dtype=np.float64).ravel()[:3]
                    mr_euler = np.asarray(keyframe["marker_right_euler_xyz"], dtype=np.float64).ravel()[:3]
                    ml_pos = np.asarray(keyframe["marker_left_pos_w"], dtype=np.float64).ravel()[:3]
                    ml_euler = np.asarray(keyframe["marker_left_euler_xyz"], dtype=np.float64).ravel()[:3]
                    tr = _make_T(mr_pos, mr_euler)
                    tl = _make_T(ml_pos, ml_euler)
                    qr_xyzw = R.from_matrix(tr[:3, :3]).as_quat()
                    ql_xyzw = R.from_matrix(tl[:3, :3]).as_quat()
                    qr_wxyz = np.array([qr_xyzw[3], qr_xyzw[0], qr_xyzw[1], qr_xyzw[2]], dtype=np.float32)
                    ql_wxyz = np.array([ql_xyzw[3], ql_xyzw[0], ql_xyzw[1], ql_xyzw[2]], dtype=np.float32)
                    marker_right_xf.set_world_poses(
                        positions=torch.tensor([tr[:3, 3]], dtype=torch.float32, device="cpu"),
                        orientations=torch.tensor([qr_wxyz], dtype=torch.float32, device="cpu"),
                    )
                    marker_left_xf.set_world_poses(
                        positions=torch.tensor([tl[:3, 3]], dtype=torch.float32, device="cpu"),
                        orientations=torch.tensor([ql_wxyz], dtype=torch.float32, device="cpu"),
                    )
                    _set_hand_gui_pose_pair(
                        np.asarray(keyframe["hand_right_joint_pos_shadow_order"], dtype=np.float64).ravel()[:24],
                        np.asarray(keyframe["hand_left_joint_pos_shadow_order"], dtype=np.float64).ravel()[:24],
                    )
                    initial_marker_state["right_pos"] = tr[:3, 3].copy()
                    initial_marker_state["left_pos"] = tl[:3, 3].copy()
                    initial_marker_state["right_quat"] = qr_wxyz.astype(np.float64).copy()
                    initial_marker_state["left_quat"] = ql_wxyz.astype(np.float64).copy()
                    hold_initial_pose_state["marker_moved"] = True
                    _keyframe_status_set(f"Applied keyframe: {pending_name}")
                    print(f"[INFO] Pose keyframe '{pending_name}' applied.")
                except Exception as e:
                    _keyframe_status_set(f"Apply failed: {pending_name}")
                    print(f"[WARN] Failed to apply pose keyframe '{pending_name}': {e}")
            pending_keyframe_apply["name"] = None

        control_right.T_world_arm_base = _root_T_world_np(right, env_idx)
        control_left.T_world_arm_base = _root_T_world_np(left, env_idx)

        pos_r_t, ori_r_t = marker_right_xf.get_world_poses()
        pos_l_t, ori_l_t = marker_left_xf.get_world_poses()
        pos_rw = _to_numpy(pos_r_t[0]).ravel()[:3]
        pos_lw = _to_numpy(pos_l_t[0]).ravel()[:3]
        euler_r = _quat_wxyz_to_euler_xyz(_to_numpy(ori_r_t[0]).ravel()[:4])
        euler_l = _quat_wxyz_to_euler_xyz(_to_numpy(ori_l_t[0]).ravel()[:4])
        h_r, h_l = _hand_pair_for_step()

        if hold_initial_pose_state["active"] and not hold_initial_pose_state["marker_moved"]:
            quat_rw = _to_numpy(ori_r_t[0]).ravel()[:4]
            quat_lw = _to_numpy(ori_l_t[0]).ravel()[:4]
            if initial_marker_state["right_quat"] is None:
                initial_marker_state["right_quat"] = quat_rw.copy()
            if initial_marker_state["left_quat"] is None:
                initial_marker_state["left_quat"] = quat_lw.copy()
            dr_pos = np.linalg.norm(pos_rw - initial_marker_state["right_pos"])
            dl_pos = np.linalg.norm(pos_lw - initial_marker_state["left_pos"])
            if dr_pos > 1e-3 or dl_pos > 1e-3:
                hold_initial_pose_state["marker_moved"] = True

        tgt_r = control_right.compute(pos_rw, euler_r, h_r)
        tgt_l = control_left.compute(pos_lw, euler_l, h_l)
        if hold_initial_pose_state["active"] and not hold_initial_pose_state["marker_moved"]:
            arm_r = _arm_joints_from_robot(right, env_idx)
            arm_l = _arm_joints_from_robot(left, env_idx)
        else:
            arm_r = _arm_joints_from_robot(right, env_idx) if tgt_r is None else np.asarray(tgt_r.arm_joints, dtype=np.float64)
            arm_l = _arm_joints_from_robot(left, env_idx) if tgt_l is None else np.asarray(tgt_l.arm_joints, dtype=np.float64)

        act_r = _build_actions_for_robot(right, arm_r, h_r)
        act_l = _build_actions_for_robot(left, arm_l, h_l)
        actions = {
            "right_hand": torch.tensor(act_r, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1),
            "left_hand": torch.tensor(act_l, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1),
        }
        env.step(actions)

        should_print = False
        if int(args.print_every) > 0 and (step % int(args.print_every) == 0):
            should_print = True
        if bool(args.print_on_change):
            if (
                last_arm_print_r is None
                or np.max(np.abs(arm_r - last_arm_print_r)) > 1e-4
                or np.max(np.abs(arm_l - last_arm_print_l)) > 1e-4
                or np.max(np.abs(h_r - last_hand_print_r)) > 1e-4
                or np.max(np.abs(h_l - last_hand_print_l)) > 1e-4
            ):
                should_print = True
        if should_print:
            msg = {
                "step": int(step),
                "arm_right_joint_pos": arm_r.tolist(),
                "arm_left_joint_pos": arm_l.tolist(),
                "hand_right_norm": act_r[-24:].tolist(),
                "hand_left_norm": act_l[-24:].tolist(),
            }
            if bool(args.print_hand_rad):
                msg["hand_right_rad"] = h_r.tolist()
                msg["hand_left_rad"] = h_l.tolist()
            print(json.dumps(msg, ensure_ascii=False))
        last_arm_print_r = arm_r.copy()
        last_arm_print_l = arm_l.copy()
        last_hand_print_r = h_r.copy()
        last_hand_print_l = h_l.copy()

        if args.max_steps > 0 and step >= int(args.max_steps):
            break
        dt = target_dt - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)

    if window is not None:
        try:
            window.visible = False
            window.destroy()
        except Exception:
            pass
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
