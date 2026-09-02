#!/usr/bin/env python3
"""Capture NF validation sim scene screenshot (weight on horizontal GelSight pad).

Usage (ViTacLab repo root)::

    # Headless auto PNG (recommended for advisor pack)
    ../IsaacLab/isaaclab.sh -p scripts/demo/capture_vitacsim_nf_scene_screenshot.py \\
        --headless --enable_cameras --device cuda:0 \\
        --weight-id W100 \\
        --out logs/vitacsim_validation/v2/setup_schematic_nf_sim.png

    # GUI: needs --enable_cameras; use isaaclab.python.kit (not headless rendering kit)
    ../IsaacLab/isaaclab.sh -p scripts/demo/capture_vitacsim_nf_scene_screenshot.py \\
        --enable_cameras --device cuda:0 --weight-id W100 --hold-seconds 120
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="NF validation sim scene screenshot.")
parser.add_argument("--weight-id", type=str, default="W100", choices=("W200", "W100", "W050", "W020", "W010"))
parser.add_argument("--settle-steps", type=int, default=180)
parser.add_argument("--finger-root-z", type=float, default=0.444)
parser.add_argument(
    "--out",
    type=str,
    default="logs/vitacsim_validation/v2/setup_schematic_nf_sim.png",
)
parser.add_argument(
    "--preset",
    type=str,
    default="panorama",
    choices=("panorama", "closeup"),
    help="panorama=NF demo third-person wide shot; closeup=detail view.",
)
parser.add_argument(
    "--camera-eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override eye (env frame). Default from --preset.",
)
parser.add_argument(
    "--camera-target",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override look-at (env frame). Default from --preset.",
)
parser.add_argument("--shot-width", type=int, default=1600)
parser.add_argument("--shot-height", type=int, default=900)
parser.add_argument(
    "--camera-aperture",
    type=float,
    default=-1.0,
    help="Pinhole horizontal aperture mm; <=0 uses preset default (wider = more panoramic).",
)
parser.add_argument(
    "--render-warmup",
    type=int,
    default=24,
    help="Extra render frames after camera pose before capture.",
)
parser.add_argument(
    "--hold-seconds",
    type=int,
    default=0,
    help="If >0 and GUI is open, keep sim alive for manual Viewport screenshot.",
)
parser.add_argument(
    "--weight-scale",
    type=float,
    default=0.75,
    help="Uniform visual scale on validation weight (schematic only; mass unchanged).",
)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ViTacLab.tasks.direct.vitacsim_validation.validation_weight_spawner_cfg import validation_weight_spawner_cfg
from ViTacLab.tasks.direct.vitacsim_validation.weight_spec import GEOMETRY, WEIGHT_MASS_KG
from ViTacLab.tasks.direct.pretraining.gelsight_finger_pretrain_base_cfg import (
    GELSIGHT_FINGER_SHORT_USD,
    build_gelsight_finger_robot_cfg,
)

_NOMINAL_WARMUP = 8
_WEIGHT_REST_Z = 0.442
_WEIGHT_CLEARANCE_Z = 0.520
_WEIGHT_DROP_OFFSET = 0.012


_CAMERA_PRESETS = {
    # Oblique third-person — slightly closer than NF demo wide shot.
    "panorama": {
        "eye": (0.20, 0.26, 0.52),
        "target": (0.0, 0.0, 0.455),
        "aperture": 20.0,
    },
    "closeup": {
        "eye": (-0.14, 0.32, 0.465),
        "target": (0.0, 0.0, 0.442),
        "aperture": 10.0,
    },
}


def _resolve_camera() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    preset = _CAMERA_PRESETS[str(args_cli.preset)]
    eye = tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else preset["eye"]
    target = tuple(args_cli.camera_target) if args_cli.camera_target is not None else preset["target"]
    aperture = float(args_cli.camera_aperture)
    if aperture <= 0.0:
        aperture = float(preset["aperture"])
    return eye, target, aperture


def _camera_ros_offset(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    from isaaclab.utils.math import (
        convert_camera_frame_orientation_convention,
        create_rotation_matrix_from_view,
        quat_from_matrix,
    )

    eyes = torch.tensor([list(eye)], dtype=torch.float32)
    targets = torch.tensor([list(target)], dtype=torch.float32)
    rotm = create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device="cpu")
    quat_opengl = quat_from_matrix(rotm)[0]
    quat_ros = convert_camera_frame_orientation_convention(
        quat_opengl.unsqueeze(0), origin="opengl", target="ros"
    )[0]
    return eye, tuple(float(x) for x in quat_ros.tolist())


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def _spawn_z() -> float:
    return _WEIGHT_REST_Z + _WEIGHT_DROP_OFFSET


def _make_weight_cfg(weight_id: str) -> RigidObjectCfg:
    scale = float(args_cli.weight_scale)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=validation_weight_spawner_cfg(weight_id, visual_scale=scale),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _WEIGHT_CLEARANCE_Z)),
    )


def _make_scene_camera_cfg(aperture: float | None = None) -> TiledCameraCfg:
    eye, target, aperture_eff = _resolve_camera()
    if aperture is not None:
        aperture_eff = float(aperture)
    cam_pos, cam_rot = _camera_ros_offset(eye, target)
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros"),
        data_types=["rgb"],
        update_period=0.0,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=float(aperture_eff),
            clipping_range=(0.05, 20.0),
        ),
        width=int(args_cli.shot_width),
        height=int(args_cli.shot_height),
    )


@configclass
class SceneShotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = build_gelsight_finger_robot_cfg().replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(args_cli.finger_root_z)),
            rot=(0.70711, -0.70711, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        )
    )
    contact_object: RigidObjectCfg = _make_weight_cfg(args_cli.weight_id)
    scene_camera: TiledCameraCfg = _make_scene_camera_cfg()


def _write_weight_pose(scene: InteractiveScene, z_root: float) -> None:
    obj = scene.rigid_objects["contact_object"]
    state = obj.data.default_root_state.clone()
    state[:, 0] = scene.env_origins[:, 0]
    state[:, 1] = scene.env_origins[:, 1]
    state[:, 2] = scene.env_origins[:, 2] + float(z_root)
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device).expand(state.shape[0], 4)
    state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(state[:, :7], env_ids=None)
    obj.write_root_velocity_to_sim(state[:, 7:], env_ids=None)


def _save_rgb_np(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    arr = rgb
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    finger_usd = Path(GELSIGHT_FINGER_SHORT_USD)
    if not finger_usd.is_file():
        finger_usd = _repo_root() / GELSIGHT_FINGER_SHORT_USD
    if not finger_usd.is_file():
        print(f"[ERROR] GelSight finger USD not found: {GELSIGHT_FINGER_SHORT_USD}", file=sys.stderr)
        return 1

    out_path = Path(args_cli.out).expanduser().resolve()
    cam_eye, cam_target, cam_aperture = _resolve_camera()
    eye = torch.tensor([list(cam_eye)], device=args_cli.device, dtype=torch.float32)
    target = torch.tensor([list(cam_target)], device=args_cli.device, dtype=torch.float32)

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(cam_eye), target=list(cam_target))

    scene = InteractiveScene(SceneShotSceneCfg(num_envs=1, env_spacing=0.3))
    scene_cam = scene["scene_camera"]

    sim.reset()
    sim_dt = sim.get_physics_dt()
    scene["robot"].reset()
    scene.rigid_objects["contact_object"].reset()

    clearance_z = _WEIGHT_CLEARANCE_Z
    _write_weight_pose(scene, clearance_z)
    for _ in range(_NOMINAL_WARMUP):
        _write_weight_pose(scene, clearance_z)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    _write_weight_pose(scene, _spawn_z())
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    for _ in range(int(args_cli.settle_steps)):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    origin = scene.env_origins[0]
    eyes_w = eye + origin.unsqueeze(0)
    targets_w = target + origin.unsqueeze(0)
    scene_cam.set_world_poses_from_view(eyes_w, targets_w)

    for _ in range(max(8, int(args_cli.render_warmup))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim.render()

    rgb = scene_cam.data.output["rgb"][0].detach().cpu().numpy()
    _save_rgb_np(out_path, rgb)
    mass_g = WEIGHT_MASS_KG[args_cli.weight_id]
    eff_r = GEOMETRY.large_radius * float(args_cli.weight_scale)
    print(
        f"[INFO] preset={args_cli.preset} eye={cam_eye} target={cam_target} "
        f"aperture={cam_aperture:.1f} weight_scale={float(args_cli.weight_scale):.2f} "
        f"weight_radius={eff_r*1000:.2f}mm -> {out_path}"
    )
    print(f"[INFO] weight={args_cli.weight_id} ({mass_g*1000:.0f}g) settled")

    hold = int(args_cli.hold_seconds)
    if hold > 0 and not getattr(args_cli, "headless", False):
        print(f"[INFO] GUI hold {hold}s — adjust Viewport, then screenshot (e.g. Viewport menu or OS capture).")
        import time

        t0 = time.time()
        while time.time() - t0 < hold:
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            sim.render()
            time.sleep(0.05)
    else:
        print("[INFO] Headless capture done. For manual framing, re-run without --headless and --hold-seconds 120.")

    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = int(main())
    except KeyboardInterrupt:
        exit_code = 130
    os._exit(exit_code)
