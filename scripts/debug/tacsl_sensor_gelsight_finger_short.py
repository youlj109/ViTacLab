#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ViTacLab: TacSL tactile sensor smoke test using the local GelSight R15 *short* finger USD.
#
# Based on ``IsaacLab/scripts/demos/sensors/tacsl_sensor.py``. Sensor cfg fields follow
# ``ViTacLab/.../ur10e_shadowhand_direct_base_cfg.py`` (``build_ur10e_shadowhand_tactile_sensor_cfgs`` / ``_mk``).
#
# Inner prim paths match the single-finger TacSL layout (same as IsaacLab nucleus
# ``gelsight_r15_finger.usd``): ``.../elastomer/tactile_sensor`` and ``.../elastomer_tip/cam``.
# If your short USD uses different names, adjust ``--tactile-sensor-subpath`` /
# ``--tactile-cam-subpath`` or edit this file.
#
# Robot is loaded with :class:`ArticulationCfg` + ``UsdFileWithCompliantContactCfg`` (same as
# ``IsaacLab/scripts/demos/sensors/tacsl_sensor.py``). The USD must expose ``ArticulationRootAPI``.
#
# Examples (from ViTacLab repo root, Isaac Sim python):
#
#     ./python.sh scripts/debug/tacsl_sensor_gelsight_finger_short.py \\
#         --use_tactile_rgb --use_tactile_ff --enable_cameras --num_envs 2
#
#     ./python.sh scripts/debug/tacsl_sensor_gelsight_finger_short.py \\
#         --contact_object_type nut --save_viz --enable_cameras
#
# Real-time viewer (same idea as ``run_ur10e_shadowhand_single.py`` -- matplotlib, one sensor):
#
#     ./python.sh scripts/debug/tacsl_sensor_gelsight_finger_short.py \\
#         --use_tactile_rgb --use_tactile_ff --show_rgb --show_ff --enable_cameras --num_envs 1
#
# Viewport drag: by default each frame reads the contact object USD world pose into PhysX; use
# ``--no-sync-viewport-pose`` to disable. Optional ``--sync-robot-root-from-viewport`` for the Robot root.
#
# UR10e + Shadow Hand (five GelSight fingertips) — use ``--asset ur10e_shadowhand`` and the processed USD
# ``ur10e_shadow_left_hand_glb_withtac_v2_no_gelsight_articulation.usd`` (single articulation root). Example::
#
# Depth-based force field (no SDF), same CLI otherwise — add ``--use-visuo-tactile-v2`` and keep FF on
# (camera is enabled automatically when ``--use_tactile_ff`` is set with V2):
#
#     ./isaaclab.sh -p scripts/debug/tacsl_sensor_gelsight_finger_short.py --asset finger \\
#         --use_tactile_ff --use-visuo-tactile-v2
#
#     ./isaaclab.sh -p scripts/debug/tacsl_sensor_gelsight_finger_short.py --asset ur10e_shadowhand \\
#         --use_tactile_rgb --use_tactile_ff --enable_cameras --num_envs 1

"""TacSL VisuoTactileSensor demo: single short finger USD, or UR10e + Shadow Hand with five GelSight tips."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher


def _repo_root() -> Path:
    """ViTacLab repo root (directory that contains ``source/``)."""
    p = Path(__file__).resolve().parent
    for _ in range(12):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _default_usd_path() -> str:
    root = _repo_root()
    return str(
        (
            root
            / "source/ViTacLab/ViTacLab/assets/data/Sensors/Gelsight_finger/gelsight_r15_finger_short_v2.usd"
        ).resolve()
    )


def _default_usd_path_ur10e_shadowhand() -> str:
    root = _repo_root()
    return str(
        (
            root
            / "source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/"
            / "ur10e_shadow_left_hand_glb_withtac_v2_no_gelsight_articulation.usd"
        ).resolve()
    )


def _dex_cube_usd_path() -> str:
    """DexCube with SDF collision — same asset as :class:`UR10eShadowHandInHandEnvCfg`.object_cfg."""
    return str(
        (_repo_root() / "source/ViTacLab/ViTacLab/assets/data/Objects/DexCube/dex_cube_sdf.usd").resolve()
    )


# Must match ``build_ur10e_shadowhand_tactile_sensor_cfgs`` keys in ``ur10e_shadowhand_direct_base_cfg.py``.
_TACTILE_FIVE_NAMES: tuple[str, ...] = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    """Match ``scripts/debug/run_ur10e_shadowhand_single.py``."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


parser = argparse.ArgumentParser(description="TacSL tactile sensor (ViTacLab short finger USD).")
parser.add_argument(
    "--asset",
    type=str,
    choices=("finger", "ur10e_shadowhand"),
    default="finger",
    help="finger = single GelSight short USD; ur10e_shadowhand = UR10e + Shadow Hand with five GelSight tips.",
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--normal_contact_stiffness", type=float, default=1.0, help="Tactile normal stiffness.")
parser.add_argument("--tangential_stiffness", type=float, default=0.1, help="Tactile tangential stiffness.")
parser.add_argument("--friction_coefficient", type=float, default=2.0, help="Tactile friction coefficient.")
parser.add_argument(
    "--tactile_compliance_stiffness",
    type=float,
    default=None,
    help="Optional: override compliant contact stiffness (default: USD).",
)
parser.add_argument(
    "--tactile_compliant_damping",
    type=float,
    default=None,
    help="Optional: override compliant contact damping (default: USD).",
)
parser.add_argument("--save_viz", action="store_true", help="Save tactile RGB / force-field visualizations.")
parser.add_argument("--save_viz_dir", type=str, default="data/tactile_record", help="Output directory for saves.")
parser.add_argument("--use_tactile_rgb", action="store_true", help="Enable camera tactile (RGB) pipeline.")
parser.add_argument("--use_tactile_ff", action="store_true", help="Enable force-field pipeline.")
parser.add_argument(
    "--use-visuo-tactile-v2",
    action="store_true",
    help="Use VisuoTactileSensorV2 (depth-based force field, no object SDF). Requires camera when FF is on.",
)
parser.add_argument("--debug_sdf_closest_pts", action="store_true", help="Visualize closest SDF points.")
parser.add_argument("--debug_tactile_sensor_pts", action="store_true", help="Visualize tactile sensor points.")
parser.add_argument("--trimesh_vis_tactile_points", action="store_true", help="Trimesh visualization for tactile points.")
parser.add_argument(
    "--contact_object_type",
    type=str,
    default="nut",
    choices=["none", "cube", "nut"],
    help="Contact object: none, cube (DexCube dex_cube_sdf.usd, same as in-hand task), or nut.",
)
parser.add_argument(
    "--contact-object-scale",
    type=float,
    default=1.0,
    help="Uniform USD root scale for nut / DexCube contact object (default: 1).",
)
parser.add_argument(
    "--usd-path",
    type=str,
    default="",
    help="Override USD: short finger asset, or UR10e+hand (see --asset; default path depends on asset).",
)
parser.add_argument(
    "--tactile-sensor-subpath",
    type=str,
    default="elastomer/tactile_sensor",
    help="Prim path under {{ENV_REGEX_NS}}/Robot/ for VisuoTactileSensorCfg.",
)
parser.add_argument(
    "--tactile-cam-subpath",
    type=str,
    default="elastomer_tip/cam",
    help="Prim path under {{ENV_REGEX_NS}}/Robot/ for tiled camera.",
)
parser.add_argument(
    "--show_rgb",
    action="store_true",
    help="Open a matplotlib window and stream tactile RGB (requires --use_tactile_rgb).",
)
parser.add_argument(
    "--show_ff",
    action="store_true",
    help="Open a matplotlib window and stream force-field image (requires --use_tactile_ff).",
)
parser.add_argument("--env_index", type=int, default=0, help="Which env to visualize for --show_rgb / --show_ff.")
parser.add_argument(
    "--fps",
    type=float,
    default=20.0,
    help="Target main-loop rate when using live display (default: 20).",
)
parser.add_argument(
    "--reset-settle-steps",
    type=int,
    default=8,
    help="Physics substeps after writing default robot/contact state before get_initial_render (default: 8).",
)
parser.add_argument(
    "--no-sync-viewport-pose",
    action="store_true",
    help="Disable USD→PhysX: by default each frame reads contact object world pose from the stage and "
    "writes it to the rigid body so viewport drags take effect.",
)
parser.add_argument(
    "--sync-robot-root-from-viewport",
    action="store_true",
    help="Also read Robot root Xform from USD each frame and write articulation root pose (optional).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.usd_path.strip():
    if args_cli.asset == "ur10e_shadowhand":
        args_cli.usd_path = _default_usd_path_ur10e_shadowhand()
    else:
        args_cli.usd_path = _default_usd_path()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.timer import Timer

from ViTacLab.assets.sensor.tacsl_sensor import VisuoTactileSensorCfg, VisuoTactileSensorV2Cfg
from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import compute_tactile_shear_image
from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from isaaclab_assets.sensors import GELSIGHT_R15_CFG

_REPO_SRC_VITAC = _repo_root() / "source" / "ViTacLab"
if str(_REPO_SRC_VITAC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC_VITAC))

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10E_SHADOWHAND_LEFT_CFG,
    UR10eShadowHandTacSLSceneCfg,
    build_ur10e_shadowhand_tactile_sensor_cfgs,
)


@configclass
class Ur10eTacDebugTactileMetaCfg(UR10eShadowHandTacSLSceneCfg):
    """Only used with :func:`build_ur10e_shadowhand_tactile_sensor_cfgs` (contact prim = debug ``contact_object``)."""

    @classmethod
    def _tactile_params(cls):
        tp = super()._tactile_params()
        tp["contact_object_prim_path_expr"] = "{ENV_REGEX_NS}/contact_object"
        return tp


# Resolution must match :func:`compute_tactile_shear_image` default (``scripts/demos/sensors/tacsl_sensor.py``).
_TACTILE_SHEAR_VIZ_RESOLUTION = 30


def _tactile_shear_image_rgb_uint8(nf_hw: np.ndarray, sf_hw2: np.ndarray) -> np.ndarray:
    """Force-field image like IsaacLab ``tacsl_sensor.save_viz_helper``: ``compute_tactile_shear_image``.

    OpenCV draws in BGR; matplotlib ``imshow`` expects RGB.
    """
    img_bgr = compute_tactile_shear_image(nf_hw, sf_hw2)
    u8 = (np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)


def _rigid_body_view_prim_paths(view) -> list[str]:
    """PhysX rigid-body view: one prim path per environment instance."""
    n = int(view.count)
    raw = view.prim_paths
    out: list[str] = []
    for i in range(n):
        p = raw[i]
        if isinstance(p, (bytes, bytearray)):
            out.append(p.decode("utf-8"))
        else:
            out.append(str(p))
    return out


def _sync_contact_object_pose_from_usd(rigid: RigidObject, device: str | torch.device) -> None:
    """Read each contact rigid prim world pose from USD and push into PhysX (viewport drag)."""
    stage = sim_utils.get_current_stage()
    paths = _rigid_body_view_prim_paths(rigid.root_physx_view)
    n = len(paths)
    if n == 0 or n != rigid.num_instances:
        return
    root_pose = torch.empty((n, 7), device=device, dtype=torch.float32)
    for i, path_str in enumerate(paths):
        prim = stage.GetPrimAtPath(path_str)
        if not prim.IsValid():
            continue
        pos, quat = sim_utils.resolve_prim_pose(prim)
        root_pose[i, 0] = pos[0]
        root_pose[i, 1] = pos[1]
        root_pose[i, 2] = pos[2]
        root_pose[i, 3] = quat[0]
        root_pose[i, 4] = quat[1]
        root_pose[i, 5] = quat[2]
        root_pose[i, 6] = quat[3]
    zero_vel = torch.zeros((n, 6), device=device, dtype=torch.float32)
    rigid.write_root_pose_to_sim(root_pose, env_ids=None)
    rigid.write_root_velocity_to_sim(zero_vel, env_ids=None)


def _sync_robot_root_pose_from_usd(robot: Articulation, device: str | torch.device) -> None:
    """Read Robot articulation root Xform from USD and push root pose into PhysX."""
    stage = sim_utils.get_current_stage()
    prims = sorted(
        sim_utils.find_matching_prims(robot.cfg.prim_path, stage=stage),
        key=lambda p: p.GetPath().pathString,
    )
    n = robot.num_instances
    if len(prims) != n:
        return
    root_pose = torch.empty((n, 7), device=device, dtype=torch.float32)
    for i, prim in enumerate(prims):
        pos, quat = sim_utils.resolve_prim_pose(prim)
        root_pose[i, 0] = pos[0]
        root_pose[i, 1] = pos[1]
        root_pose[i, 2] = pos[2]
        root_pose[i, 3] = quat[0]
        root_pose[i, 4] = quat[1]
        root_pose[i, 5] = quat[2]
        root_pose[i, 6] = quat[3]
    zero_vel = torch.zeros((n, 6), device=device, dtype=torch.float32)
    robot.write_root_pose_to_sim(root_pose, env_ids=None)
    robot.write_root_velocity_to_sim(zero_vel, env_ids=None)


def _sync_viewport_poses_to_sim(scene: InteractiveScene, *, sync_contact: bool, sync_robot: bool) -> None:
    """Apply USD stage transforms to PhysX before ``write_data_to_sim`` so gizmo drags affect simulation."""
    if sync_robot:
        _sync_robot_root_pose_from_usd(scene["robot"], scene["robot"].device)
    if sync_contact and "contact_object" in scene.rigid_objects:
        _sync_contact_object_pose_from_usd(scene.rigid_objects["contact_object"], scene.rigid_objects["contact_object"].device)


# Match ur10e_shadowhand_direct_base_cfg._tactile_params / _mk (lines 110–116, 131–153).
_TACTILE_PARAMS = {
    "contact_object_prim_path_expr": "{ENV_REGEX_NS}/contact_object",
    "enable_force_field": True,
    "tactile_array_size": (20, 25),
    "tactile_margin": 0.005,
}

def _make_robot_cfg() -> ArticulationCfg:
    """GelSight finger USD as articulation (matches IsaacLab ``tacsl_sensor.py``)."""
    spawn = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=args_cli.usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        compliant_contact_stiffness=args_cli.tactile_compliance_stiffness,
        compliant_contact_damping=args_cli.tactile_compliant_damping,
        physics_material_prim_path="elastomer",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.0005),
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )


def _make_ur10e_shadowhand_robot_cfg() -> ArticulationCfg:
    """UR10e + Shadow Hand (single articulation root); USD from ``--usd-path``."""
    usd = Path(args_cli.usd_path).resolve()
    return UR10E_SHADOWHAND_LEFT_CFG.replace(
        spawn=UR10E_SHADOWHAND_LEFT_CFG.spawn.replace(usd_path=str(usd)),
    )


def _tactile_sensor_cfg_cls():
    """Which sensor class to instantiate (V2 = depth-based force field, no SDF)."""
    return VisuoTactileSensorV2Cfg if args_cli.use_visuo_tactile_v2 else VisuoTactileSensorCfg


def _maybe_upgrade_tacsl_cfg_to_v2(scfg: VisuoTactileSensorCfg) -> VisuoTactileSensorCfg | VisuoTactileSensorV2Cfg:
    """Rebuild as VisuoTactileSensorV2Cfg so ``class_type`` is VisuoTactileSensorV2."""
    if not args_cli.use_visuo_tactile_v2:
        return scfg
    d = scfg.to_dict()
    d.pop("class_type", None)
    return VisuoTactileSensorV2Cfg(**d)


def _format_visuo_tacsl_cfg_paths(
    sensor_cfg: VisuoTactileSensorCfg | VisuoTactileSensorV2Cfg, scene: InteractiveScene
) -> None:
    """Match :meth:`InteractiveScene._add_entities_from_cfg` regex formatting for TacSL."""
    ns = scene.env_regex_ns
    sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.camera_cfg is not None:
        sensor_cfg.camera_cfg.prim_path = sensor_cfg.camera_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.contact_object_prim_path_expr is not None:
        sensor_cfg.contact_object_prim_path_expr = sensor_cfg.contact_object_prim_path_expr.format(ENV_REGEX_NS=ns)


def _patch_tacsl_cfg_from_cli(sensor_cfg: VisuoTactileSensorCfg) -> VisuoTactileSensorCfg:
    # V2 force field reads depth every step; camera must be on whenever FF is on.
    enable_cam = args_cli.use_tactile_rgb or (args_cli.use_visuo_tactile_v2 and args_cli.use_tactile_ff)
    return sensor_cfg.replace(
        enable_camera_tactile=enable_cam,
        enable_force_field=args_cli.use_tactile_ff,
        normal_contact_stiffness=args_cli.normal_contact_stiffness,
        friction_coefficient=args_cli.friction_coefficient,
        tangential_stiffness=args_cli.tangential_stiffness,
        debug_vis=args_cli.debug_tactile_sensor_pts or args_cli.debug_sdf_closest_pts,
        trimesh_vis_tactile_points=args_cli.trimesh_vis_tactile_points,
        visualize_sdf_closest_pts=args_cli.debug_sdf_closest_pts,
    )


def _make_single_finger_tacsl_cfg(*, contact_object_prim_path_expr: str | None) -> VisuoTactileSensorCfg:
    """Build single-finger TacSL cfg (paths still contain ``{ENV_REGEX_NS}``; format before instantiate)."""
    return _tactile_sensor_cfg_cls()(
        prim_path="{ENV_REGEX_NS}/Robot/" + args_cli.tactile_sensor_subpath.strip().strip("/"),
        history_length=0,
        debug_vis=args_cli.debug_tactile_sensor_pts or args_cli.debug_sdf_closest_pts,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=args_cli.use_tactile_rgb
        or (args_cli.use_visuo_tactile_v2 and args_cli.use_tactile_ff),
        enable_force_field=args_cli.use_tactile_ff,
        tactile_array_size=_TACTILE_PARAMS["tactile_array_size"],
        tactile_margin=_TACTILE_PARAMS["tactile_margin"],
        contact_object_prim_path_expr=contact_object_prim_path_expr,
        normal_contact_stiffness=args_cli.normal_contact_stiffness,
        friction_coefficient=args_cli.friction_coefficient,
        tangential_stiffness=args_cli.tangential_stiffness,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + args_cli.tactile_cam_subpath.strip().strip("/"),
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        trimesh_vis_tactile_points=args_cli.trimesh_vis_tactile_points,
        visualize_sdf_closest_pts=args_cli.debug_sdf_closest_pts,
    )


def _register_single_tacsl_sensor(scene: InteractiveScene, *, contact_object_type: str) -> None:
    """Register one finger TacSL sensor after scene clone (ViTacLab cfg needs manual path formatting)."""
    contact_expr = _TACTILE_PARAMS["contact_object_prim_path_expr"] if contact_object_type != "none" else None
    scfg = _make_single_finger_tacsl_cfg(contact_object_prim_path_expr=contact_expr)
    scfg = _patch_tacsl_cfg_from_cli(scfg)
    scfg = _maybe_upgrade_tacsl_cfg_to_v2(scfg)
    if contact_object_type == "none":
        scfg = scfg.replace(contact_object_prim_path_expr=None, debug_vis=True)
    _format_visuo_tacsl_cfg_paths(scfg, scene)
    scene.sensors["tactile_sensor"] = scfg.class_type(scfg)


def _register_five_tacsl_sensors(scene: InteractiveScene) -> None:
    """After ``InteractiveScene`` + env clone — same order as :meth:`UR10eShadowHandDirectBaseEnv._setup_scene`."""
    meta = Ur10eTacDebugTactileMetaCfg(
        num_envs=scene.cfg.num_envs,
        env_spacing=scene.cfg.env_spacing,
        replicate_physics=scene.cfg.replicate_physics,
        clone_in_fabric=getattr(scene.cfg, "clone_in_fabric", False),
    )
    cfgs = build_ur10e_shadowhand_tactile_sensor_cfgs(meta)
    for name, scfg in cfgs.items():
        scfg = _patch_tacsl_cfg_from_cli(scfg)
        scfg = _maybe_upgrade_tacsl_cfg_to_v2(scfg)
        if args_cli.contact_object_type == "none":
            scfg = scfg.replace(contact_object_prim_path_expr=None)
        _format_visuo_tacsl_cfg_paths(scfg, scene)
        scene.sensors[name] = scfg.class_type(scfg)


def _refresh_articulation_physx_views_after_tacsl(scene: InteractiveScene) -> None:
    """Rebuild PhysX articulation tensor views after TacSL sensors finish initializing.

    On timeline PLAY, Isaac Lab subscribes all assets with the same callback order (~10). If the robot
    :class:`~isaaclab.assets.Articulation.Articulation` runs ``_initialize_impl`` **before** the five
    :class:`VisuoTactileSensor` instances, the sensors' subsequent ``_initialize_impl`` (cameras, SDF views,
    etc.) can invalidate the global physics simulation view — leaving the articulation's
    ``ArticulationView`` stale and causing ``get_dof_velocities`` / "Simulation view object is invalidated".

    Calling this **after** ``sim.reset()`` forces each articulation to drop and recreate PhysX views using
    the final stage graph.
    """
    for _name, art in scene.articulations.items():
        if not art.is_initialized:
            continue
        art._invalidate_initialize_callback(None)
        art._initialize_impl()
        art._is_initialized = True


@configclass
class TactileSensorsSceneCfg(InteractiveSceneCfg):
    """Scene: ground, dome light, GelSight short finger robot (TacSL registered after scene init)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = _make_robot_cfg()


def _contact_object_scale() -> float:
    return max(1e-6, float(args_cli.contact_object_scale))


def _scaled_cube_contact_object() -> RigidObjectCfg:
    """In-hand DexCube (``dex_cube_sdf.usd``) — aligned with ``UR10eShadowHandInHandEnvCfg.object_cfg`` spawn."""
    s = _contact_object_scale()
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_dex_cube_usd_path(),
            scale=(s, s, s),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _scaled_nut_contact_object() -> RigidObjectCfg:
    s = _contact_object_scale()
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            scale=(s, s, s),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class CubeTactileSceneCfg(TactileSensorsSceneCfg):
    contact_object = _scaled_cube_contact_object()


@configclass
class NutTactileSceneCfg(TactileSensorsSceneCfg):
    contact_object = _scaled_nut_contact_object()


@configclass
class Ur10eFiveFingerTacDebugSceneCfg(InteractiveSceneCfg):
    """Ground + dome + UR10e/Shadow robot. TacSL sensors are registered after :class:`InteractiveScene` init."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = _make_ur10e_shadowhand_robot_cfg()


@configclass
class Ur10eCubeTacDebugSceneCfg(Ur10eFiveFingerTacDebugSceneCfg):
    contact_object = _scaled_cube_contact_object()


@configclass
class Ur10eNutTacDebugSceneCfg(Ur10eFiveFingerTacDebugSceneCfg):
    contact_object = _scaled_nut_contact_object()


def _reset_robot_contact_and_sensors(
    scene: InteractiveScene,
    sim: sim_utils.SimulationContext,
    *,
    settle_steps: int,
    tactile_sensor_names: tuple[str, ...] = ("tactile_sensor",),
) -> None:
    """Restore defaults like :func:`isaaclab.envs.mdp.events.reset_scene_to_default`, then settle physics.

    Differences from the old loop-only root write:

    - Uses ``write_root_pose_to_sim`` + ``write_root_velocity_to_sim`` (same split as MDP reset).
    - Restores **all** joint positions/velocities for the articulation.
    - Runs several ``sim.step`` + ``scene.update`` cycles so PhysX applies the state **before**
      ``get_initial_render()``; otherwise the camera baseline can be taken while the asset is still
      mid-correction, and the sensor frame keeps drifting relative to the first run.

    We only reset **articulations and rigid bodies** (actuators + wrench composers), not
    :meth:`InteractiveScene.reset`, because the latter also resets every **sensor** including the
    tactile camera: that resets tiled-camera frame counters and timestamps and can diverge from the
    first-boot path (where the camera was never sensor-reset), so the RGB baseline keeps shifting.

    Then we write default poses like :func:`isaaclab.envs.mdp.events.reset_scene_to_default`, settle,
    and refresh tactile nominal (camera) if enabled.
    """
    sim_dt = sim.get_physics_dt()

    scene["robot"].reset()
    if "contact_object" in scene.rigid_objects:
        scene.rigid_objects["contact_object"].reset()

    # Rigid contact objects (same pattern as mdp/events.reset_scene_to_default)
    if "contact_object" in scene.rigid_objects:
        rigid = scene.rigid_objects["contact_object"]
        default_root_state = rigid.data.default_root_state.clone()
        default_root_state[:, 0:3] += scene.env_origins
        # RigidObject: env_ids must be None (all envs) or a torch index tensor — not slice(None).
        rigid.write_root_pose_to_sim(default_root_state[:, :7], env_ids=None)
        rigid.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=None)

    # Articulation robot
    robot = scene["robot"]
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=None)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=None)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
        env_ids=None,
    )
    if hasattr(robot, "set_joint_position_target"):
        try:
            robot.set_joint_position_target(robot.data.default_joint_pos.clone(), env_ids=None)
            robot.set_joint_velocity_target(robot.data.default_joint_vel.clone(), env_ids=None)
        except Exception:
            pass

    scene.write_data_to_sim()

    for name in tactile_sensor_names:
        ts = scene[name]
        # ``scene.update`` runs camera tactile and requires ``_nominal_tactile`` (set by get_initial_render).
        if getattr(ts.cfg, "enable_camera_tactile", False):
            try:
                ts.get_initial_render()
            except Exception as e:
                print(f"[WARN] get_initial_render before settle ({name}) failed: {e}", file=sys.stderr)

    for _ in range(max(1, int(settle_steps))):
        sim.step()
        scene.update(sim_dt)

    for name in tactile_sensor_names:
        ts = scene[name]
        if getattr(ts.cfg, "enable_camera_tactile", False):
            try:
                ts.get_initial_render()
            except Exception as e:
                print(f"[WARN] get_initial_render after settle ({name}) failed: {e}", file=sys.stderr)


def mkdir_helper(dir_path: str) -> tuple[str, str]:
    os.makedirs(dir_path, exist_ok=True)
    tactile_force_field_dir = os.path.join(dir_path, "tactile_force_field")
    tactile_rgb_image_dir = os.path.join(dir_path, "tactile_rgb_image")
    os.makedirs(tactile_force_field_dir, exist_ok=True)
    os.makedirs(tactile_rgb_image_dir, exist_ok=True)
    return tactile_force_field_dir, tactile_rgb_image_dir


def save_viz_helper(
    dir_path_list: tuple[str, str],
    count: int,
    tactile_data: VisuoTactileSensorData,
    num_envs: int,
    nrows: int,
    ncols: int,
) -> None:
    tactile_force_field_dir, tactile_rgb_image_dir = dir_path_list

    if tactile_data.tactile_shear_force is not None and tactile_data.tactile_normal_force is not None:
        tactile_normal_force = tactile_data.tactile_normal_force.view((num_envs, nrows, ncols))
        tactile_shear_force = tactile_data.tactile_shear_force.view((num_envs, nrows, ncols, 2))

        tactile_image = compute_tactile_shear_image(
            tactile_normal_force[0, :, :].detach().cpu().numpy(), tactile_shear_force[0, :, :].detach().cpu().numpy()
        )

        if tactile_normal_force.shape[0] > 1:
            tactile_image_1 = compute_tactile_shear_image(
                tactile_normal_force[1, :, :].detach().cpu().numpy(),
                tactile_shear_force[1, :, :].detach().cpu().numpy(),
            )
            combined_image = np.vstack([tactile_image, tactile_image_1])
            cv2.imwrite(
                os.path.join(tactile_force_field_dir, f"{count:04d}.png"), (combined_image * 255).astype(np.uint8)
            )
        else:
            cv2.imwrite(
                os.path.join(tactile_force_field_dir, f"{count:04d}.png"), (tactile_image * 255).astype(np.uint8)
            )

    if tactile_data.tactile_rgb_image is not None:
        tactile_rgb_data = tactile_data.tactile_rgb_image.cpu().numpy()
        tactile_rgb_data = np.transpose(tactile_rgb_data, axes=(0, 2, 1, 3))
        tactile_rgb_data_first_2 = tactile_rgb_data[:2] if len(tactile_rgb_data) >= 2 else tactile_rgb_data
        tactile_rgb_tiled = np.concatenate(tactile_rgb_data_first_2, axis=0)
        if tactile_rgb_tiled.dtype != np.uint8:
            tactile_rgb_tiled = (
                (tactile_rgb_tiled * 255).astype(np.uint8)
                if tactile_rgb_tiled.max() <= 1.0
                else tactile_rgb_tiled.astype(np.uint8)
            )
        cv2.imwrite(os.path.join(tactile_rgb_image_dir, f"{count:04d}.png"), tactile_rgb_tiled)


def _tactile_key_to_short(sensor_name: str) -> str:
    return sensor_name.replace("tactile_sensor_", "", 1)


def save_viz_helper_five(
    dir_path_list: tuple[str, str],
    count: int,
    scene: InteractiveScene,
    sensor_names: tuple[str, ...],
    num_envs: int,
) -> None:
    """Save one PNG per finger (prefix ``ff_``, ``lf_``, …) under the same dirs as single-finger mode."""
    tactile_force_field_dir, tactile_rgb_image_dir = dir_path_list
    for sname in sensor_names:
        short = _tactile_key_to_short(sname)
        td: VisuoTactileSensorData = scene[sname].data
        nrows = scene[sname].cfg.tactile_array_size[0]
        ncols = scene[sname].cfg.tactile_array_size[1]
        if td.tactile_shear_force is not None and td.tactile_normal_force is not None:
            tactile_normal_force = td.tactile_normal_force.view((num_envs, nrows, ncols))
            tactile_shear_force = td.tactile_shear_force.view((num_envs, nrows, ncols, 2))
            tactile_image = compute_tactile_shear_image(
                tactile_normal_force[0, :, :].detach().cpu().numpy(),
                tactile_shear_force[0, :, :].detach().cpu().numpy(),
            )
            cv2.imwrite(
                os.path.join(tactile_force_field_dir, f"{short}_{count:04d}.png"),
                (tactile_image * 255).astype(np.uint8),
            )
        if td.tactile_rgb_image is not None:
            tactile_rgb_data = td.tactile_rgb_image.cpu().numpy()
            tactile_rgb_data = np.transpose(tactile_rgb_data, axes=(0, 2, 1, 3))
            row0 = tactile_rgb_data[0]
            if row0.dtype != np.uint8:
                row0 = (np.clip(row0, 0.0, 1.0) * 255).astype(np.uint8) if row0.max() <= 1.0 else row0.astype(np.uint8)
            cv2.imwrite(os.path.join(tactile_rgb_image_dir, f"{short}_{count:04d}.png"), row0)


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    *,
    tactile_sensor_names: tuple[str, ...] = ("tactile_sensor",),
) -> None:
    sim_dt = sim.get_physics_dt()
    count = 0
    num_envs = scene.num_envs

    if args_cli.save_viz:
        save_root = os.path.abspath(os.path.expanduser(args_cli.save_viz_dir))
        print(f"[INFO] Saving tactile PNGs under: {save_root}/")
        print("    - tactile_force_field/*.png  (shear visualization)")
        print("    - tactile_rgb_image/*.png    (camera tactile RGB)")
        dir_path_list = mkdir_helper(save_root)
    else:
        dir_path_list = ("", "")

    if args_cli.show_rgb and not args_cli.use_tactile_rgb:
        print("[WARN] --show_rgb is set but --use_tactile_rgb is off; RGB window will stay blank until enabled.")
    if args_cli.show_ff and not args_cli.use_tactile_ff:
        print("[WARN] --show_ff is set but --use_tactile_ff is off; FF window will stay blank until enabled.")

    fig = None
    im_rgb = None
    im_ff = None
    im_rgb_list: list | None = None
    im_ff_list: list | None = None
    target_dt = 1.0 / max(1e-3, float(args_cli.fps))
    env_viz = max(0, min(int(args_cli.env_index), num_envs - 1))

    ts0 = scene[tactile_sensor_names[0]]
    nrows = ts0.cfg.tactile_array_size[0]
    ncols = ts0.cfg.tactile_array_size[1]
    multi_finger = len(tactile_sensor_names) > 1

    if args_cli.show_rgb or args_cli.show_ff:
        import matplotlib

        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        zh, zw = GELSIGHT_R15_CFG.image_height, GELSIGHT_R15_CFG.image_width
        zero_rgb = np.zeros((zh, zw, 3), dtype=np.uint8)
        # Match ``compute_tactile_shear_image`` output size (H*res, W*res, 3).
        zero_ff = np.zeros(
            (nrows * _TACTILE_SHEAR_VIZ_RESOLUTION, ncols * _TACTILE_SHEAR_VIZ_RESOLUTION, 3),
            dtype=np.uint8,
        )

        if multi_finger:
            nf = len(tactile_sensor_names)
            im_rgb_list = []
            im_ff_list = []
            if args_cli.show_rgb and args_cli.show_ff:
                fig, axes = plt.subplots(nf, 2, figsize=(10, 2.8 * nf))
                for i, sname in enumerate(tactile_sensor_names):
                    short = _tactile_key_to_short(sname)
                    im_rgb_list.append(axes[i, 0].imshow(zero_rgb))
                    axes[i, 0].set_title(f"{short} RGB (env {env_viz})")
                    axes[i, 0].axis("off")
                    im_ff_list.append(axes[i, 1].imshow(zero_ff))
                    axes[i, 1].set_title(f"{short} FF (env {env_viz})")
                    axes[i, 1].axis("off")
            elif args_cli.show_rgb:
                fig, axes = plt.subplots(nf, 1, figsize=(7, 2.8 * nf))
                axs = axes if nf > 1 else [axes]
                for i, sname in enumerate(tactile_sensor_names):
                    short = _tactile_key_to_short(sname)
                    im_rgb_list.append(axs[i].imshow(zero_rgb))
                    axs[i].set_title(f"{short} RGB (env {env_viz})")
                    axs[i].axis("off")
            else:
                fig, axes = plt.subplots(nf, 1, figsize=(7, 2.8 * nf))
                axs = axes if nf > 1 else [axes]
                for i, sname in enumerate(tactile_sensor_names):
                    short = _tactile_key_to_short(sname)
                    im_ff_list.append(axs[i].imshow(zero_ff))
                    axs[i].set_title(f"{short} FF (env {env_viz})")
                    axs[i].axis("off")
        elif args_cli.show_rgb and args_cli.show_ff:
            fig, (ax_r, ax_f) = plt.subplots(2, 1, figsize=(8, 10))
            im_rgb = ax_r.imshow(zero_rgb)
            ax_r.set_title(f"Tactile RGB (env {env_viz})")
            ax_r.axis("off")
            im_ff = ax_f.imshow(zero_ff)
            ax_f.set_title(f"Force field RGB (env {env_viz})")
            ax_f.axis("off")
        elif args_cli.show_rgb:
            fig, ax_r = plt.subplots(1, 1, figsize=(8, 6))
            im_rgb = ax_r.imshow(zero_rgb)
            ax_r.set_title(f"Tactile RGB (env {env_viz})")
            ax_r.axis("off")
        else:
            fig, ax_f = plt.subplots(1, 1, figsize=(8, 6))
            im_ff = ax_f.imshow(zero_ff)
            ax_f.set_title(f"Force field RGB (env {env_viz})")
            ax_f.axis("off")

        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)
        print("[INFO] Live matplotlib viewer open; close the figure or stop the app to exit.")

    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    force_tensor[:, 0, 2] = -1.0

    physics_timer = Timer()
    physics_total_time = 0.0
    physics_total_count = 0

    sync_contact = not args_cli.no_sync_viewport_pose
    sync_robot = bool(args_cli.sync_robot_root_from_viewport)
    if sync_contact:
        print(
            "[INFO] Viewport pose sync: contact object USD→PhysX each frame "
            "(drag with transform gizmo). Use --no-sync-viewport-pose to disable."
        )
    if sync_robot:
        print("[INFO] Viewport pose sync: Robot root USD→PhysX each frame (--sync-robot-root-from-viewport).")

    while simulation_app.is_running():
        t_loop = time.time()
        if count == 122:
            count = 0
            _reset_robot_contact_and_sensors(
                scene,
                sim,
                settle_steps=max(1, int(args_cli.reset_settle_steps)),
                tactile_sensor_names=tactile_sensor_names,
            )
            print("[INFO] Reset robot/contact to defaults (MDP-style), settled sim, refreshed tactile baseline.")

        if "contact_object" in scene.keys():
            if count > 20:
                env_indices = torch.arange(scene.num_envs, device=sim.device)
                odd_mask = env_indices % 2 == 1
                even_mask = env_indices % 2 == 0
                torque_tensor[odd_mask, 0, 2] = 10
                torque_tensor[even_mask, 0, 2] = -10
                scene["contact_object"].permanent_wrench_composer.set_forces_and_torques(force_tensor, torque_tensor)

        if sync_contact or sync_robot:
            _sync_viewport_poses_to_sim(scene, sync_contact=sync_contact, sync_robot=sync_robot)

        scene.write_data_to_sim()
        physics_timer.start()
        sim.step()
        physics_timer.stop()
        physics_total_time += physics_timer.total_run_time
        physics_total_count += 1
        count += 1
        scene.update(sim_dt)

        if args_cli.save_viz and dir_path_list[0]:
            if multi_finger:
                save_viz_helper_five(dir_path_list, count, scene, tactile_sensor_names, num_envs)
            else:
                tactile_data = scene[tactile_sensor_names[0]].data
                save_viz_helper(dir_path_list, count, tactile_data, num_envs, nrows, ncols)

        if fig is not None:
            import matplotlib.pyplot as plt

            if im_rgb_list is not None:
                for im, sname in zip(im_rgb_list, tactile_sensor_names):
                    td = scene[sname].data
                    img = getattr(td, "tactile_rgb_image", None)
                    if img is not None and hasattr(img, "shape") and img.ndim == 4:
                        x = img[env_viz].detach().cpu().numpy()
                        if x.shape[0] == 3 and x.ndim == 3:
                            x = np.transpose(x, (1, 2, 0))
                        im.set_data(_img_to_uint8(x))
            elif im_rgb is not None:
                tactile_data = scene[tactile_sensor_names[0]].data
                img = getattr(tactile_data, "tactile_rgb_image", None)
                if img is not None and hasattr(img, "shape") and img.ndim == 4:
                    x = img[env_viz].detach().cpu().numpy()
                    if x.shape[0] == 3 and x.ndim == 3:
                        x = np.transpose(x, (1, 2, 0))
                    im_rgb.set_data(_img_to_uint8(x))

            if im_ff_list is not None:
                for im, sname in zip(im_ff_list, tactile_sensor_names):
                    td = scene[sname].data
                    nf = getattr(td, "tactile_normal_force", None)
                    sf = getattr(td, "tactile_shear_force", None)
                    if nf is None or sf is None:
                        continue
                    nr_i = scene[sname].cfg.tactile_array_size[0]
                    nc_i = scene[sname].cfg.tactile_array_size[1]
                    nf_flat = nf[env_viz].detach().cpu().numpy().reshape(-1)
                    sf_flat = sf[env_viz].detach().cpu().numpy().reshape(-1, 2)
                    p = int(nf_flat.shape[0])
                    nr, nc = nr_i, nc_i
                    if p != nr * nc:
                        nr = int(np.sqrt(p))
                        nc = max(1, p // max(1, nr))
                    nf_img = nf_flat.reshape(nr, nc)
                    sf_img = sf_flat.reshape(nr, nc, 2)
                    im.set_data(_tactile_shear_image_rgb_uint8(nf_img, sf_img))
            elif im_ff is not None:
                tactile_data = scene[tactile_sensor_names[0]].data
                nf = getattr(tactile_data, "tactile_normal_force", None)
                sf = getattr(tactile_data, "tactile_shear_force", None)
                if nf is not None and sf is not None:
                    nf_flat = nf[env_viz].detach().cpu().numpy().reshape(-1)
                    sf_flat = sf[env_viz].detach().cpu().numpy().reshape(-1, 2)
                    p = int(nf_flat.shape[0])
                    nr, nc = nrows, ncols
                    if p != nr * nc:
                        nr = int(np.sqrt(p))
                        nc = max(1, p // max(1, nr))
                    nf_img = nf_flat.reshape(nr, nc)
                    sf_img = sf_flat.reshape(nr, nc, 2)
                    im_ff.set_data(_tactile_shear_image_rgb_uint8(nf_img, sf_img))

            fig.canvas.draw_idle()
            plt.pause(0.001)

        elapsed = time.time() - t_loop
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)

    timing_summary = scene[tactile_sensor_names[0]].get_timing_summary()
    physics_avg = physics_total_time / (physics_total_count * scene.num_envs) if physics_total_count > 0 else 0.0
    timing_summary["physics_total"] = physics_total_time
    timing_summary["physics_average"] = physics_avg
    timing_summary["physics_fps"] = 1 / physics_avg if physics_avg > 0 else 0.0
    print(timing_summary)

    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")


def main() -> None:
    usd = Path(args_cli.usd_path)
    if not usd.is_file():
        print(
            f"[ERROR] USD not found: {usd}\n"
            "  --asset finger: …/Sensors/Gelsight_finger/gelsight_r15_finger_short_v2.usd\n"
            "  --asset ur10e_shadowhand: …/Robots/ShadowHand/ur10e/"
            "ur10e_shadow_left_hand_glb_withtac_v2_no_gelsight_articulation.usd\n"
            "or pass --usd-path.",
            file=sys.stderr,
        )
        simulation_app.close()
        raise SystemExit(1)

    print(f"[INFO] Asset mode: {args_cli.asset}")
    print(f"[INFO] Using USD: {usd.resolve()}")
    if args_cli.asset == "finger":
        print("[INFO] Robot: ArticulationCfg (UsdFileWithCompliantContactCfg)")
    else:
        print("[INFO] Robot: UR10e + Shadow Hand — five TacSL sensors registered after scene clone (task-env pattern).")
    if args_cli.contact_object_type != "none":
        print(
            f"[INFO] contact_object_type={args_cli.contact_object_type}, "
            f"--contact-object-scale={args_cli.contact_object_scale} "
            "(cube: DexCube dex_cube_sdf.usd; nut: factory_nut_m16.usd; USD root scale)"
        )

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2**30),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.6, 1.0], target=[-0.1, 0.1, 0.5])

    _ur10e_scene_kwargs = dict(
        num_envs=args_cli.num_envs,
        env_spacing=0.2,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    if args_cli.asset == "ur10e_shadowhand":
        if args_cli.contact_object_type == "cube":
            scene_cfg = Ur10eCubeTacDebugSceneCfg(**_ur10e_scene_kwargs)
        elif args_cli.contact_object_type == "nut":
            scene_cfg = Ur10eNutTacDebugSceneCfg(**_ur10e_scene_kwargs)
        elif args_cli.contact_object_type == "none":
            scene_cfg = Ur10eFiveFingerTacDebugSceneCfg(**_ur10e_scene_kwargs)
        else:
            raise ValueError(f"Invalid contact_object_type: {args_cli.contact_object_type!r}")
        # Register TacSL before ``sim.reset()`` so prims exist before PLAY. After ``reset()``, refresh
        # articulation PhysX views: sensor init order vs robot can leave stale ``ArticulationView``s.
        scene = InteractiveScene(scene_cfg)
        _register_five_tacsl_sensors(scene)
        sim.reset()
        _refresh_articulation_physx_views_after_tacsl(scene)
        tactile_names: tuple[str, ...] = _TACTILE_FIVE_NAMES
    else:
        if args_cli.contact_object_type == "cube":
            scene_cfg = CubeTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        elif args_cli.contact_object_type == "nut":
            scene_cfg = NutTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        elif args_cli.contact_object_type == "none":
            scene_cfg = TactileSensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        else:
            raise ValueError(f"Invalid contact_object_type: {args_cli.contact_object_type!r}")
        scene = InteractiveScene(scene_cfg)
        _register_single_tacsl_sensor(scene, contact_object_type=args_cli.contact_object_type)
        sim.reset()
        tactile_names = ("tactile_sensor",)

    print("[INFO] Setup complete.")

    _reset_robot_contact_and_sensors(
        scene,
        sim,
        settle_steps=max(1, int(args_cli.reset_settle_steps)),
        tactile_sensor_names=tactile_names,
    )
    run_simulator(sim, scene, tactile_sensor_names=tactile_names)


if __name__ == "__main__":
    main()
    simulation_app.close()
