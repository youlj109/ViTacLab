# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: Wuji hand + grabbable object — 26-link grid tactile + hand-shaped plots.

Scene: ground plane, dome light, Wuji right hand at a fixed spawn pose, and **one** rigid body
(**cube** or **sphere**, see ``--drop_shape``) at ``--drop_pos`` under **default world gravity** (~9.81 m/s²
down), **no scripted object velocity** at init. You can also pose the object in the **viewport** (``DropObject``,
gizmo) before/during the run. Size: ``--drop_cube_size`` or
``--drop_sphere_radius``; optional ``--drop_mass``. Hand **root is not driven** by script or UI. **Finger joints** use a docked
**Wuji Hand Joints (rad)** ``omni.ui`` panel (same pattern as ``run_ur10e_shadowhand_arm_pose_from_marker.py``:
fixed-height window + scroll). Sliders follow **Wuji URDF / articulation joint names and order**, not
Shadow Hand's 24-DoF ``shadowhand_joint_names()`` layout.

Tactile filters include the drop object only (no ground). Visualization: :func:`render_hand_tactile_pair`.

**Schematic vs your eyes:** the composite uses **palmar** layout: for ``--wuji-hand-side right``, the thumb
(finger1) is on the **left** side of the plot — that *is* a right hand palm facing you, not a left hand.
If your Isaac camera shows the thumb on the screen's other side, pass ``--tactile-schematic-mirror-x``.
Use ``--wuji-hand-side left`` when loading ``usd/left/...`` so link names match the schematic.

**PhysX / GPU:** If Kit logs ``PxArticulationLink::addForce()`` illegal with ``eENABLE_DIRECT_GPU_API``,
GPU PhysX forbids that articulation force API on some paths. Re-launch with ``--device cpu``, or set
``--physics-device cpu`` (keep ``--device`` consistent, preferably both ``cpu``).

**Tactile pad frame:** ``f·n`` uses **outward** ``n`` in each **link body** frame (not world). Wuji URDF
convention: **+Z along the finger toward the fingertip**. By default the builder aligns grid **v** (2nd
extent / ``patch_center_offset[1]``) with that axis. Set ``pad_normal_axis`` to 0 or 1 so the skin normal
is ±X or ±Y — **not** 2 if Z is the bone direction. Use ``--wuji-no-fingertip-align`` and
``--wuji-pad-swap-tangent-axes`` for fully manual u/v.

**Viewport palm grid:** pass ``--draw-palm-tactile-grid`` to overlay the palm :class:`GridTactileSensor`
patch (cyan outline + cell lines) in Isaac Sim via ``debug_draw``, aligned with the link frame used for
binning. Use ``--no-palm-tactile-grid-cells`` for outline only.

**UI extras** (same idea as ``scripts/rsl_rl/full_tra/record_full_tra_single.py``): **Reset object** moves the
cube/sphere back to ``--drop_pos`` (velocities cleared). **open_record** / **close_record** save current slider joint vector;
**open** / **close** load and apply saved poses. Presets are JSON (Wuji joint names + length, not Shadow's 24).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from isaaclab.app import AppLauncher

_VI_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WUJI_USD = (
    _VI_ROOT
    / "source/ViTacLab/ViTacLab/assets/data/Robots/wuji-hand-description-main/usd/right/wujihand.usd"
)

parser = argparse.ArgumentParser(description="Wuji hand + rigid object, gravity, 26-link tactile composite.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--usd",
    type=str,
    default=str(_DEFAULT_WUJI_USD),
    help="Path to Wuji hand USD (right hand default).",
)
parser.add_argument("--scale", type=float, default=1.0, help="Uniform spawn scale for the hand USD.")
parser.add_argument(
    "--hand_pos",
    type=float,
    nargs=3,
    default=(0.52, 0.5, 0.13),
    metavar=("X", "Y", "Z"),
    help="Hand root position (m), world frame (fixed for the whole run).",
)
parser.add_argument(
    "--hand_quat",
    type=float,
    nargs=4,
    default=(0.70711, 0.0, -0.70711, 0.0),
    metavar=("W", "X", "Y", "Z"),
    help="Hand root orientation (w,x,y,z), world frame (fixed for the whole run).",
)
parser.add_argument(
    "--drop_shape",
    type=str,
    choices=("cube", "sphere"),
    default="sphere",
    help="Rigid shape (affected by gravity; place/drag in viewport or use --drop_pos / Reset).",
)
parser.add_argument(
    "--drop_pos",
    type=float,
    nargs=3,
    default=(0.45, 0.5, 0.22),
    metavar=("X", "Y", "Z"),
    help="Object center (m, world) at init and on reset; e.g. above the palm for manual placement.",
)
parser.add_argument(
    "--drop_cube_size",
    type=float,
    nargs=3,
    default=(0.03, 0.03, 0.03),
    metavar=("SX", "SY", "SZ"),
    help="Cuboid edge lengths (m, full size per axis) when --drop_shape cube.",
)
parser.add_argument(
    "--drop_sphere_radius",
    type=float,
    default=0.03,
    help="Sphere radius (m) when --drop_shape sphere.",
)
parser.add_argument(
    "--drop_mass",
    type=float,
    default=None,
    help="Mass (kg). If omitted: 0.18 for cube, 0.08 for sphere.",
)
parser.add_argument(
    "--plot_interval",
    type=int,
    default=4,
    help="Update matplotlib tactile composite every N physics steps (larger = faster).",
)
parser.add_argument(
    "--no_plot",
    action="store_true",
    help="Disable matplotlib windows.",
)
_hand_gui = parser.add_mutually_exclusive_group()
_hand_gui.add_argument(
    "--hand-gui",
    dest="hand_gui",
    action="store_true",
    help="Show Wuji hand joint sliders (default; same idea as Shadow script --hand-gui).",
)
_hand_gui.add_argument(
    "--no-hand-gui",
    dest="hand_gui",
    action="store_false",
    help="Disable hand sliders (no omni.ui panel).",
)
parser.set_defaults(hand_gui=True)
parser.add_argument(
    "--hand-joints-init",
    choices=("sim", "zeros"),
    default="sim",
    help="Initial slider values when opening GUI: current sim pose or zeros (clamped to URDF limits).",
)
parser.add_argument(
    "--wuji_articulation_root",
    type=str,
    default="/root_joint",
    help="Path under spawned {ENV_REGEX_NS}/WujiHand to the prim with ArticulationRootAPI.",
)
parser.add_argument(
    "--periodic_reset",
    action="store_true",
    help="Every ~8s, move the object back to --drop_pos and clear its velocity (hand unchanged).",
)
parser.add_argument(
    "--hand-preset-path",
    type=str,
    default="scripts/demo/wuji_hand_pose_presets.json",
    help="JSON file for open/close hand presets (repo-relative to ViTacLab root or absolute).",
)
parser.add_argument(
    "--wuji-hand-side",
    type=str,
    choices=("right", "left"),
    default="right",
    help="Must match Wuji USD link names (right_* vs left_*). Drives sensors + schematic layout.",
)
parser.add_argument(
    "--tactile-schematic-mirror-x",
    action="store_true",
    help="Flip tactile schematic horizontally if thumb/pinky look swapped vs your 3D view (see script doc).",
)
parser.add_argument(
    "--tactile-plot-no-swap-uv",
    action="store_true",
    help="Matplotlib composite: keep raw grid axes (iu→rows, iv→cols); default swaps so v is vertical like the hand.",
)
parser.add_argument(
    "--tactile-plot-no-flip",
    action="store_true",
    help="After UV swap, do not flipud/fliplr (default flips fix 180° vs schematic; use if patch looks mirrored).",
)
parser.add_argument(
    "--tactile-normal-colormap",
    type=str,
    default="heatmap",
    metavar="NAME",
    help="Normal |f·n| heatmap colormap. Default: heatmap (blue→red→yellow). Also: wuji_heat, bry, turbo, viridis.",
)
parser.add_argument(
    "--physics-device",
    type=str,
    default=None,
    metavar="DEVICE",
    help=(
        'Override SimulationCfg.device (e.g. "cpu" to silence PxArticulationLink::addForce + '
        "eENABLE_DIRECT_GPU_API warnings with GPU PhysX). Default: AppLauncher --device."
    ),
)
parser.add_argument(
    "--wuji-pad-normal-axis",
    type=int,
    choices=(0, 1, 2),
    default=0,
    help="GridTactile: link body axis (0=x,1=y,2=z) of **outward** pad normal; must match Wuji USD.",
)
parser.add_argument(
    "--wuji-pad-normal-sign",
    type=int,
    choices=(-1, 1),
    default=1,
    help="+1 if outward is +axis; -1 if skin faces -axis along --wuji-pad-normal-axis.",
)
parser.add_argument(
    "--wuji-pad-swap-tangent-axes",
    action="store_true",
    help="Manual only: swap u/v when --wuji-no-fingertip-align (ignored otherwise; builder picks swap for +Z).",
)
parser.add_argument(
    "--wuji-fingertip-body-axis",
    type=int,
    choices=(0, 1, 2),
    default=2,
    help="Link body axis toward fingertip; grid v aligns to it (default 2 = +Z).",
)
parser.add_argument(
    "--wuji-no-fingertip-align",
    action="store_true",
    help="Disable auto v↔fingertip alignment; use --wuji-pad-swap-tangent-axes only.",
)
parser.add_argument(
    "--draw-palm-tactile-grid",
    action="store_true",
    help="Draw palm :class:`GridTactileSensor` patch in the viewport (outline + optional cell lines).",
)
parser.add_argument(
    "--no-palm-tactile-grid-cells",
    action="store_true",
    help="With --draw-palm-tactile-grid, draw only the rectangle outline (skip interior grid).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_usd_path = Path(args_cli.usd)
if not _usd_path.is_file():
    print(
        "[ERROR]: Wuji USD not found:\n"
        f"         {_usd_path.resolve()}\n"
        "         Pass --usd or install the asset under the default repo path."
    )
    sys.exit(1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

from ViTacLab.assets.sensor import GridTactileSensorCfg, build_wuji_hand_grid_tactile_sensor_cfgs
from ViTacLab.assets.sensor.grid_tactile.grid_tactile_sensor import _tangent_axes_from_normal
from ViTacLab.assets.sensor.wuji_hand_grid_tactile import HandTactilePlotCfg, render_hand_tactile_pair
from ViTacLab.assets.sensor.wuji_hand_grid_tactile.wuji_hand_grid_tactile_cfg import wuji_link_names

WUJI_HAND_PRESET_FORMAT = "vitaclab_wuji_hand_pose_presets_v1"


def _float_ui_model_get(model) -> float:
    if hasattr(model, "get_value_as_float"):
        return float(model.get_value_as_float())
    if hasattr(model, "as_float"):
        return float(model.as_float)
    return 0.0


def _float_ui_model_set(model, value: float) -> None:
    if hasattr(model, "set_value"):
        model.set_value(float(value))
    elif hasattr(model, "set_float"):
        model.set_float(float(value))


def _wuji_urdf_for_hand_usd(usd_path: Path) -> Path:
    r = usd_path.resolve()
    return r.parent.parent.parent / "urdf" / f"{r.parent.name}.urdf"


def _joint_pos_mid_from_urdf(urdf_path: Path) -> dict[str, float]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: dict[str, float] = {}
    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        name = joint.get("name")
        if not name or jtype != "revolute":
            continue
        lim = joint.find("limit")
        if lim is None:
            continue
        lower_s, upper_s = lim.get("lower"), lim.get("upper")
        if lower_s is None or upper_s is None:
            continue
        lower = float(lower_s)
        upper = float(upper_s)
        out[name] = 0.5 * (lower + upper)
    return out


def _joint_limits_from_urdf(urdf_path: Path) -> dict[str, tuple[float, float]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        name = joint.get("name")
        if not name or jtype != "revolute":
            continue
        lim = joint.find("limit")
        if lim is None:
            continue
        lower_s, upper_s = lim.get("lower"), lim.get("upper")
        if lower_s is None or upper_s is None:
            continue
        out[name] = (float(lower_s), float(upper_s))
    return out


def _resolve_repo_path(p: str | Path) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_VI_ROOT / path).resolve()


def _save_wuji_hand_pose_presets(
    path: Path,
    joint_names: list[str],
    poses: dict[str, list[float] | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pose_doc: dict[str, list[float] | None] = {}
    for name in ("open", "close"):
        pose = poses.get(name)
        pose_doc[name] = None if pose is None else [float(x) for x in pose]
    doc = {
        "format": WUJI_HAND_PRESET_FORMAT,
        "joint_names": list(joint_names),
        "updated_at": datetime.now().isoformat(),
        "poses": pose_doc,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_wuji_hand_pose_presets(path: Path, joint_names: list[str]) -> dict[str, list[float] | None]:
    out: dict[str, list[float] | None] = {"open": None, "close": None}
    if not path.is_file():
        return out
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != WUJI_HAND_PRESET_FORMAT:
        raise ValueError(f"Expected format {WUJI_HAND_PRESET_FORMAT!r}, got {doc.get('format')!r}")
    file_names = list(doc.get("joint_names") or [])
    if file_names != list(joint_names):
        raise ValueError(
            f"preset joint_names mismatch (file has {len(file_names)} joints, sim has {len(joint_names)})."
        )
    poses = doc.get("poses") or {}
    nj = len(joint_names)
    for name in ("open", "close"):
        raw = poses.get(name)
        if raw is None:
            out[name] = None
            continue
        arr = [float(x) for x in raw]
        if len(arr) != nj:
            raise ValueError(f"pose {name!r} length {len(arr)} != {nj}")
        out[name] = arr
    return out


def _apply_hand_joints_from_ui(scene: InteractiveScene, ui: dict) -> None:
    hand = scene[ui["hand_key"]]
    env_id = int(ui["env_id"])
    models: list = ui["models"]
    joint_count = int(ui["joint_count"])
    jp = hand.data.joint_pos.clone()
    for i in range(joint_count):
        jp[env_id, i] = float(_float_ui_model_get(models[i]))
    zv = torch.zeros_like(jp)
    hand.set_joint_position_target(jp)
    hand.write_joint_state_to_sim(jp, zv)


def _sync_hand_joint_ui_from_sim(scene: InteractiveScene, ui: dict) -> None:
    hand = scene[ui["hand_key"]]
    env_id = int(ui["env_id"])
    jp = hand.data.joint_pos[env_id].detach().cpu().tolist()
    models: list = ui["models"]
    for i, m in enumerate(models):
        _float_ui_model_set(m, float(jp[i]))


def _try_create_wuji_joint_ui(
    scene: InteractiveScene,
    *,
    hand_key: str,
    env_id: int,
    joint_limits: dict[str, tuple[float, float]],
    preset_path: Path,
    shared_sim_state: dict,
) -> dict | None:
    """Docked hand GUI — layout like ``record_full_tra_single.py`` open/close presets (Wuji joint count)."""
    if getattr(args_cli, "headless", False) or not bool(getattr(args_cli, "hand_gui", True)):
        if getattr(args_cli, "headless", False):
            print("[INFO]: Hand GUI disabled (--headless).")
        else:
            print("[INFO]: Hand GUI disabled (--no-hand-gui).")
        return None
    try:
        import omni.ui  # type: ignore
    except Exception as exc:
        print(f"[WARN]: Hand GUI skipped (omni.ui unavailable: {exc}).")
        return None

    hand = scene[hand_key]
    jn = getattr(hand, "joint_names", None)
    if jn is None:
        jn = getattr(hand.data, "joint_names", ())
    names = [str(x) for x in jn]
    nj = len(names)
    if nj == 0:
        print("[WARN]: Hand GUI skipped (no joint names on articulation).")
        return None

    pose_sim = hand.data.joint_pos[int(env_id)].detach().cpu().tolist()
    use_zeros = str(getattr(args_cli, "hand_joints_init", "sim")) == "zeros"

    ui: dict = {
        "hand_key": hand_key,
        "env_id": int(env_id),
        "models": [],
        "live": None,
        "joint_count": nj,
        "window": None,
        "joint_names": names,
        "preset_path": preset_path,
        "pose_presets": {"open": None, "close": None},
    }

    hand_pose_status_model: object | None = None

    def _hand_pose_status_set(msg: str) -> None:
        m = hand_pose_status_model
        if m is None:
            return
        if hasattr(m, "set_value"):
            cast(Any, m).set_value(str(msg))

    def _hand_pose_status_refresh(prefix: str = "") -> None:
        st = ui["pose_presets"]
        o_ok = st.get("open") is not None
        c_ok = st.get("close") is not None
        summary = f"open={'Y' if o_ok else 'N'} | close={'Y' if c_ok else 'N'}"
        msg = f"{prefix} ({summary})" if prefix else summary
        _hand_pose_status_set(msg)

    def _record_hand_pose(name: str) -> None:
        models: list = ui["models"]
        if len(models) != nj:
            _hand_pose_status_refresh(f"{name}_record failed")
            return
        vec = [float(_float_ui_model_get(models[i])) for i in range(nj)]
        ui["pose_presets"][name] = vec
        try:
            _save_wuji_hand_pose_presets(preset_path, names, ui["pose_presets"])
            _hand_pose_status_refresh(f"{name}_record saved")
            print(f"[INFO]: Wuji hand pose {name!r} saved -> {preset_path}")
        except Exception as e:
            _hand_pose_status_refresh(f"{name}_save failed")
            print(f"[WARN]: Failed to save presets: {e}")

    def _apply_hand_pose(name: str) -> None:
        pose = ui["pose_presets"].get(name)
        if pose is None:
            _hand_pose_status_refresh(f"{name} not recorded")
            print(f"[WARN]: No saved pose {name!r} (use {name}_record first).")
            return
        models: list = ui["models"]
        if len(pose) != len(models):
            _hand_pose_status_refresh(f"{name} apply failed")
            return
        try:
            for i, jname in enumerate(names):
                lo, hi = joint_limits.get(jname, (-math.pi, math.pi))
                if hi <= lo:
                    hi = lo + 1e-3
                v = float(np.clip(float(pose[i]), lo, hi))
                _float_ui_model_set(models[i], v)
            _apply_hand_joints_from_ui(scene, ui)
            _hand_pose_status_refresh(f"{name} applied")
            print(f"[INFO]: Wuji hand pose {name!r} applied ({nj} joints).")
        except Exception as e:
            _hand_pose_status_refresh(f"{name} apply failed")
            print(f"[WARN]: Apply {name!r} failed: {e}")

    def _reset_drop_cb() -> None:
        shared_sim_state["pending_drop_reset"] = True
        _hand_pose_status_set("Reset object requested")
        print("[INFO]: Reset object — will return to --drop_pos (velocity cleared) next step.")

    try:
        scroll_kwargs = {}
        try:
            scroll_kwargs["horizontal_scrollbar_policy"] = omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED  # type: ignore[attr-defined]
        except AttributeError:
            pass

        win = omni.ui.Window(
            "Wuji Hand Joints (rad)",
            width=460,
            height=720,
            visible=True,
            dock_preference=omni.ui.DockPreference.RIGHT_TOP,
        )
        ui["window"] = win

        with win.frame:
            with omni.ui.ScrollingFrame(**scroll_kwargs):
                with omni.ui.VStack(spacing=3, height=0):
                    omni.ui.Label("Drop object (gravity on)", height=18)
                    omni.ui.Button("Reset object", clicked_fn=_reset_drop_cb)
                    omni.ui.Label("Hand pose presets (cf. record_full_tra_single.py)", height=18)
                    with omni.ui.HStack(spacing=4):
                        omni.ui.Button("open_record", clicked_fn=lambda: _record_hand_pose("open"))
                        omni.ui.Button("open", clicked_fn=lambda: _apply_hand_pose("open"))
                    with omni.ui.HStack(spacing=4):
                        omni.ui.Button("close_record", clicked_fn=lambda: _record_hand_pose("close"))
                        omni.ui.Button("close", clicked_fn=lambda: _apply_hand_pose("close"))
                    hand_pose_status_model = omni.ui.SimpleStringModel()
                    cast(Any, hand_pose_status_model).set_value("")
                    omni.ui.StringField(model=hand_pose_status_model, read_only=True, height=28)
                    try:
                        rel = preset_path.relative_to(_VI_ROOT)
                    except ValueError:
                        rel = preset_path
                    omni.ui.Label(f"Preset file: {rel}", word_wrap=True)
                    omni.ui.Label(
                        "Wuji URDF joint order (not Shadow 24-DoF). Drag sliders; "
                        "live apply drives each physics step.",
                        word_wrap=True,
                    )
                    live_m = omni.ui.SimpleBoolModel()
                    live_m.set_value(True)
                    ui["live"] = live_m
                    with omni.ui.HStack():
                        omni.ui.Label("Apply each physics step", width=160)
                        omni.ui.CheckBox(model=live_m)
                    with omni.ui.HStack(spacing=6):
                        omni.ui.Button(
                            "Apply once",
                            clicked_fn=lambda: _apply_hand_joints_from_ui(scene, ui),
                        )
                        omni.ui.Button(
                            "Read from sim",
                            clicked_fn=lambda: _sync_hand_joint_ui_from_sim(scene, ui),
                        )

                    for i, jname in enumerate(names):
                        lo, hi = joint_limits.get(jname, (-math.pi, math.pi))
                        if hi <= lo:
                            hi = lo + 1e-3
                        if use_zeros:
                            mid = float(np.clip(0.0, lo, hi))
                        else:
                            mid = float(np.clip(float(pose_sim[i]) if i < len(pose_sim) else 0.0, lo, hi))
                        m = omni.ui.SimpleFloatModel()
                        _float_ui_model_set(m, mid)
                        ui["models"].append(m)
                        step = max(1e-4, (float(hi) - float(lo)) / 500.0)
                        with omni.ui.HStack(spacing=6):
                            omni.ui.Label(jname, width=200, alignment=omni.ui.Alignment.LEFT_CENTER)
                            omni.ui.FloatSlider(model=m, min=float(lo), max=float(hi), step=step)
    except Exception as exc:
        print(f"[WARN]: Could not build Wuji hand GUI ({exc}); try --no-hand-gui.")
        ui["window"] = None
        return None

    try:
        if preset_path.is_file():
            loaded = _load_wuji_hand_pose_presets(preset_path, names)
            ui["pose_presets"].update(loaded)
            _hand_pose_status_refresh("Loaded presets")
            print(f"[INFO]: Wuji hand presets loaded <- {preset_path}")
        else:
            _hand_pose_status_refresh("No preset file yet")
    except Exception as e:
        _hand_pose_status_refresh("Preset load failed")
        print(f"[WARN]: Could not load presets from {preset_path}: {e}")

    print(
        "[INFO]: Opened 'Wuji Hand Joints (rad)' panel (dock right). "
        "Joint names/order match this articulation, not Shadow Hand."
    )
    return ui


def _joint_ui_live_on(ui: dict | None) -> bool:
    if ui is None:
        return False
    m = ui.get("live")
    if m is None:
        return False
    if hasattr(m, "get_value_as_bool"):
        return bool(m.get_value_as_bool())
    if hasattr(m, "as_bool"):
        return bool(m.as_bool)
    return False


def _make_hand_tactile_plot_figure(*, canvas_hw: tuple[int, int]):
    import matplotlib.pyplot as plt

    plt.ion()
    ch, cw = canvas_hw
    z = np.zeros((ch, cw, 3), dtype=np.uint8)
    fig, (ax_n, ax_f) = plt.subplots(1, 2, num="Wuji hand — tactile composite", figsize=(12.5, 5.8))
    im_n = ax_n.imshow(z, origin="upper", interpolation="nearest")
    im_f = ax_f.imshow(z, origin="upper", interpolation="nearest")
    ax_n.set_title("Normal |f·n| (heatmap, schematic hand)")
    ax_f.set_title("Tangential friction (arrows)")
    for ax in (ax_n, ax_f):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.show()
    return fig, im_n, im_f


def _update_hand_tactile_plot(im_n, im_f, img_n, img_f) -> None:
    import matplotlib.pyplot as plt

    im_n.set_data(img_n)
    im_f.set_data(img_f)
    fig = im_n.figure
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def _gather_wuji_tactile_dicts(scene: InteractiveScene, *, env_id: int) -> tuple[dict, dict]:
    """Build ``link_name -> ndarray`` for :func:`render_hand_tactile_pair`."""
    normal: dict = {}
    friction: dict = {}
    e = int(env_id)
    b = 0
    for link in wuji_link_names(str(args_cli.wuji_hand_side)):
        key = f"wuji_grid_tactile_{link}"
        sens = scene[key]
        fg = sens.data.force_grid
        if fg is not None and b < fg.shape[1]:
            normal[link] = fg[e, b].sum(dim=0).detach().float().cpu().numpy()
        fr = sens.data.friction_grid_uv
        if fr is not None and b < fr.shape[1]:
            friction[link] = fr[e, b].sum(dim=0).detach().float().cpu().numpy()
    return normal, friction


def _grid_tangent_axis_indices(cfg: GridTactileSensorCfg) -> tuple[int, int]:
    tu, tv = _tangent_axes_from_normal(int(cfg.pad_normal_axis))
    if cfg.swap_tangent_axes:
        tu, tv = tv, tu
    return tu, tv


def _palm_grid_debug_line_batches(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    cfg: GridTactileSensorCfg,
    *,
    draw_cells: bool,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[float]]:
    """World-frame segments for ``omni_debug_draw``; geometry matches :class:`GridTactileSensor` binning."""
    tu, tv = _grid_tangent_axis_indices(cfg)
    ext_u, ext_v = float(cfg.patch_extent[0]), float(cfg.patch_extent[1])
    cu, cv = float(cfg.patch_center_offset[0]), float(cfg.patch_center_offset[1])
    u0, u1 = cu - 0.5 * ext_u, cu + 0.5 * ext_u
    v0, v1 = cv - 0.5 * ext_v, cv + 0.5 * ext_v

    def pt_body(u: float, v: float) -> list[float]:
        p = [0.0, 0.0, 0.0]
        p[tu] = u
        p[tv] = v
        return p

    corners = [pt_body(u0, v0), pt_body(u1, v0), pt_body(u1, v1), pt_body(u0, v1)]
    starts_b: list[list[float]] = []
    ends_b: list[list[float]] = []
    for i in range(4):
        starts_b.append(corners[i])
        ends_b.append(corners[(i + 1) % 4])

    gh, gw = int(cfg.grid_resolution[0]), int(cfg.grid_resolution[1])
    if draw_cells:
        if gh > 1:
            for i in range(1, gh):
                u = u0 + (ext_u / gh) * i
                starts_b.append(pt_body(u, v0))
                ends_b.append(pt_body(u, v1))
        if gw > 1:
            for j in range(1, gw):
                v = v0 + (ext_v / gw) * j
                starts_b.append(pt_body(u0, v))
                ends_b.append(pt_body(u1, v))

    dev = body_pos_w.device
    dt = body_pos_w.dtype
    s_t = torch.tensor(starts_b, device=dev, dtype=dt)
    e_t = torch.tensor(ends_b, device=dev, dtype=dt)
    q = body_quat_w.unsqueeze(0)
    n_seg = s_t.shape[0]
    sw = quat_apply(q.expand(n_seg, -1), s_t) + body_pos_w.unsqueeze(0)
    ew = quat_apply(q.expand(n_seg, -1), e_t) + body_pos_w.unsqueeze(0)
    starts_w = sw.detach().cpu().tolist()
    ends_w = ew.detach().cpu().tolist()

    colors: list[list[float]] = []
    thicknesses: list[float] = []
    cyan = [0.1, 0.95, 0.95, 1.0]
    dim = [0.35, 0.75, 0.85, 0.85]
    thick_outline = 5.0
    thick_grid = 2.0
    for i in range(n_seg):
        if i < 4:
            colors.append(cyan)
            thicknesses.append(thick_outline)
        else:
            colors.append(dim)
            thicknesses.append(thick_grid)
    return starts_w, ends_w, colors, thicknesses


def _build_scene_cfg() -> InteractiveSceneCfg:
    hand_rot_wxyz = tuple(float(x) for x in args_cli.hand_quat)
    s = float(args_cli.scale)
    usd_str = str(_usd_path.resolve())

    urdf_primary = _wuji_urdf_for_hand_usd(_usd_path)
    urdf_fallback = (
        _VI_ROOT
        / "source/ViTacLab/ViTacLab/assets/data/Robots/wuji-hand-description-main/urdf"
        / f"{_usd_path.resolve().parent.name}.urdf"
    )
    urdf_path = urdf_primary if urdf_primary.is_file() else urdf_fallback
    if not urdf_path.is_file():
        print(
            "[ERROR]: Need Wuji URDF for valid ``joint_pos`` defaults.\n"
            f"         Tried: {urdf_primary}\n"
            f"         Tried: {urdf_fallback}"
        )
        sys.exit(1)
    wuji_joint_pos = _joint_pos_mid_from_urdf(urdf_path)
    if not wuji_joint_pos:
        print(f"[ERROR]: No revolute limits parsed from {urdf_path}")
        sys.exit(1)

    filter_exprs = ["{ENV_REGEX_NS}/DropObject"]

    _ft_axis = None if args_cli.wuji_no_fingertip_align else int(args_cli.wuji_fingertip_body_axis)
    tactile_cfgs = build_wuji_hand_grid_tactile_sensor_cfgs(
        hand_root_prim_path_expr="{ENV_REGEX_NS}/WujiHand",
        filter_prim_paths_expr=filter_exprs,
        side=str(args_cli.wuji_hand_side),  # type: ignore[arg-type]
        pad_normal_axis=int(args_cli.wuji_pad_normal_axis),
        pad_normal_sign=int(args_cli.wuji_pad_normal_sign),  # type: ignore[arg-type]
        swap_tangent_axes=bool(args_cli.wuji_pad_swap_tangent_axes),
        fingertip_body_axis=_ft_axis,
    )

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    root_suffix = str(args_cli.wuji_articulation_root).strip()
    if not root_suffix.startswith("/"):
        root_suffix = "/" + root_suffix

    wuji_hand = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/WujiHand",
        articulation_root_prim_path=root_suffix,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_str,
            scale=(s, s, s),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(args_cli.hand_pos),
            rot=hand_rot_wxyz,
            joint_pos=wuji_joint_pos,
            joint_vel={},
        ),
        actuators={},
    )

    drop_xyz = tuple(float(x) for x in args_cli.drop_pos)
    cube_sz = tuple(max(1e-4, float(x)) for x in args_cli.drop_cube_size)
    sph_r = max(1e-4, float(args_cli.drop_sphere_radius))
    if args_cli.drop_mass is not None:
        drop_mass = max(1e-6, float(args_cli.drop_mass))
    else:
        drop_mass = 0.18 if args_cli.drop_shape == "cube" else 0.08

    if args_cli.drop_shape == "cube":
        drop_spawn = sim_utils.CuboidCfg(
            size=cube_sz,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=drop_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.95, dynamic_friction=0.95),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.2), metallic=0.05),
        )
    else:
        drop_spawn = sim_utils.SphereCfg(
            radius=sph_r,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=drop_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.95), metallic=0.12),
        )

    drop_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DropObject",
        spawn=drop_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=drop_xyz),
    )

    annotations: dict[str, type] = {
        "ground": AssetBaseCfg,
        "dome_light": AssetBaseCfg,
        "wuji_hand": ArticulationCfg,
        "drop_object": RigidObjectCfg,
    }
    for _k, _cfg in tactile_cfgs.items():
        annotations[_k] = GridTactileSensorCfg

    class_body: dict = {
        "__annotations__": annotations,
        "ground": ground,
        "dome_light": dome_light,
        "wuji_hand": wuji_hand,
        "drop_object": drop_object,
    }
    class_body.update(tactile_cfgs)

    SceneCfg = configclass(type("WujiHandGraspTactileSceneCfg", (InteractiveSceneCfg,), class_body))
    return SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    *,
    plot_cfg: HandTactilePlotCfg,
    joint_ui: dict | None,
    shared_sim_state: dict,
):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    env_id = 0

    plot_handles = None
    if not args_cli.no_plot:
        try:
            plot_handles = _make_hand_tactile_plot_figure(canvas_hw=(plot_cfg.canvas_height, plot_cfg.canvas_width))
            print("[INFO]: Matplotlib composite tactile window opened (close with --no_plot to disable).")
        except Exception as exc:
            print(f"[WARN]: Matplotlib failed ({exc}). Continuing without plot.")
            plot_handles = None

    reset_cadence = int(round(8.0 / max(sim_dt, 1e-6))) if args_cli.periodic_reset else 0

    palm_draw_iface = None
    palm_sensor_key = f"wuji_grid_tactile_{args_cli.wuji_hand_side}_palm_link"
    if bool(args_cli.draw_palm_tactile_grid) and not bool(getattr(args_cli, "headless", False)):
        try:
            palm_draw_iface = omni_debug_draw.acquire_debug_draw_interface()
            print("[INFO]: Palm tactile grid debug draw enabled (viewport overlay).")
        except Exception as exc:
            print(f"[WARN]: Could not acquire debug draw for palm grid ({exc}).")

    while simulation_app.is_running():
        if shared_sim_state.get("pending_drop_reset"):
            shared_sim_state["pending_drop_reset"] = False
            obj = scene["drop_object"]
            root = obj.data.default_root_state.clone()
            root[:, :3] += scene.env_origins
            root[:, 7:] = 0.0
            obj.write_root_pose_to_sim(root[:, :7])
            obj.write_root_velocity_to_sim(root[:, 7:])
            print("[INFO]: Object reset to --drop_pos (velocity cleared).")

        if _joint_ui_live_on(joint_ui):
            try:
                _apply_hand_joints_from_ui(scene, joint_ui)  # type: ignore[arg-type]
            except Exception:
                pass

        if reset_cadence > 0 and count > 0 and count % reset_cadence == 0:
            obj = scene["drop_object"]
            root = obj.data.default_root_state.clone()
            root[:, :3] += scene.env_origins
            root[:, 7:] = 0.0
            obj.write_root_pose_to_sim(root[:, :7])
            obj.write_root_velocity_to_sim(root[:, 7:])
            print("[INFO]: Periodic reset: object back to --drop_pos (velocity cleared).")

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        if palm_draw_iface is not None:
            try:
                hand = scene["wuji_hand"]
                palm_link = f"{args_cli.wuji_hand_side}_palm_link"
                ids, _ = hand.find_bodies(palm_link, preserve_order=True)
                if ids:
                    bi = int(ids[0])
                    pos_w = hand.data.body_pos_w[int(env_id), bi]
                    quat_w = hand.data.body_quat_w[int(env_id), bi]
                    cfg = scene[palm_sensor_key].cfg
                    palm_draw_iface.clear_lines()
                    sw, ew, col, th = _palm_grid_debug_line_batches(
                        pos_w,
                        quat_w,
                        cfg,
                        draw_cells=not bool(args_cli.no_palm_tactile_grid_cells),
                    )
                    palm_draw_iface.draw_lines(sw, ew, col, th)
            except Exception:
                pass

        if plot_handles is not None and count % max(1, int(args_cli.plot_interval)) == 0:
            try:
                n_dict, f_dict = _gather_wuji_tactile_dicts(scene, env_id=env_id)
                img_n, img_f = render_hand_tactile_pair(
                    n_dict, f_dict, side=str(args_cli.wuji_hand_side), cfg=plot_cfg
                )
                _update_hand_tactile_plot(plot_handles[1], plot_handles[2], img_n, img_f)
            except Exception as exc:
                print(f"[WARN]: Tactile plot update failed ({exc}); disabling plot.")
                plot_handles = None

    win = joint_ui.get("window") if joint_ui else None
    if win is not None:
        try:
            win.visible = False
            win.destroy()
        except Exception:
            pass


def main():
    urdf_primary = _wuji_urdf_for_hand_usd(_usd_path)
    urdf_fallback = (
        _VI_ROOT
        / "source/ViTacLab/ViTacLab/assets/data/Robots/wuji-hand-description-main/urdf"
        / f"{_usd_path.resolve().parent.name}.urdf"
    )
    urdf_path = urdf_primary if urdf_primary.is_file() else urdf_fallback
    joint_limits = _joint_limits_from_urdf(urdf_path) if urdf_path.is_file() else {}

    sim_device = args_cli.physics_device if args_cli.physics_device is not None else args_cli.device
    if args_cli.physics_device is not None and args_cli.physics_device != args_cli.device:
        print(
            f"[WARN]: --physics-device {args_cli.physics_device!r} != --device {args_cli.device!r}; "
            "prefer a single device (e.g. both cpu) to avoid tensor/physics mismatches."
        )
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=sim_device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.35, 1.05, 0.55], target=[0.52, 0.52, 0.08])

    scene_cfg = _build_scene_cfg()
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    for _ in range(24):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    print(
        f"[INFO]: Wuji grasp-tactile demo — default world gravity; shape={args_cli.drop_shape!r} pos={tuple(args_cli.drop_pos)}; "
        f"size={'cube ' + str(tuple(args_cli.drop_cube_size)) if args_cli.drop_shape == 'cube' else 'sphere r=' + str(args_cli.drop_sphere_radius)}; "
        f"mass={args_cli.drop_mass if args_cli.drop_mass is not None else 'default'}; "
        "26 tactile sensors; --hand-gui for sliders."
    )
    print(
        "[INFO]: DropObject uses gravity; init has zero velocity. You can still move/rotate in the viewport; "
        "Reset object (or --periodic_reset) teleports back to --drop_pos and clears velocity."
    )
    print(f"[INFO]: Hand USD: {_usd_path}")

    shared_sim_state = {"pending_drop_reset": False}
    preset_path = _resolve_repo_path(args_cli.hand_preset_path)

    joint_ui = _try_create_wuji_joint_ui(
        scene,
        hand_key="wuji_hand",
        env_id=0,
        joint_limits=joint_limits,
        preset_path=preset_path,
        shared_sim_state=shared_sim_state,
    )

    _swap_uv = not bool(args_cli.tactile_plot_no_swap_uv)
    _do_flip = _swap_uv and not bool(args_cli.tactile_plot_no_flip)
    plot_cfg = HandTactilePlotCfg(
        canvas_width=720,
        canvas_height=720,
        friction_arrow_subsample=5,
        friction_scale=5.0,
        schematic_mirror_x=bool(args_cli.tactile_schematic_mirror_x),
        normal_colormap=str(args_cli.tactile_normal_colormap).strip() or "heatmap",
        normal_vmin=0.0,
        normal_vmax=2.0,
        normal_plot_swap_uv_axes=_swap_uv,
        normal_plot_flip_rows=_do_flip,
        normal_plot_flip_cols=_do_flip,
    )

    run_simulator(
        sim,
        scene,
        plot_cfg=plot_cfg,
        joint_ui=joint_ui,
        shared_sim_state=shared_sim_state,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
