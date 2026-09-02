# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: :class:`~ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_v2.VisuoTactileSensorV2`.

Single GelSight short finger + factory nut (dynamic, gravity on). V2 uses depth-camera force field (no object SDF).

Run from ViTacLab repo root (Isaac Lab python)::

    conda activate env_isaaclab_510test
    cd /path/to/IssacLab_510test/IsaacLab
    ./isaaclab.sh -p ../ViTacLab/scripts/demo/demo_visuotactile_sensor_v2.py \\
        --enable_cameras --num_envs 1

With live force-field matplotlib window::

    ./isaaclab.sh -p ../ViTacLab/scripts/demo/demo_visuotactile_sensor_v2.py \\
        --enable_cameras --num_envs 1 --show_ff

Headless / no display::

    ./isaaclab.sh -p ../ViTacLab/scripts/demo/demo_visuotactile_sensor_v2.py \\
        --enable_cameras --headless --no_plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VisuoTactileSensorV2 (depth-based TacSL) demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--no_plot",
    action="store_true",
    help="Disable matplotlib force-field window.",
)
parser.add_argument(
    "--show_ff",
    action="store_true",
    help="Open matplotlib window streaming force-field shear image.",
)
parser.add_argument(
    "--show_rgb",
    action="store_true",
    help="Also stream tactile RGB in matplotlib (requires --enable_cameras).",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=60,
    help="Print force-field stats every N sim steps (0 = disable).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after N steps (0 = run until app closes).",
)
parser.add_argument(
    "--nut_torque_z",
    type=float,
    default=10.0,
    help="Z torque on nut after step 20 (0 = no external torque).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.show_ff or args_cli.show_rgb:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.sensors import GELSIGHT_R15_CFG

from ViTacLab.assets.sensor.tacsl_sensor import VisuoTactileSensorV2Cfg
from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import compute_tactile_shear_image

_TACTILE_ARRAY = (20, 25)
_TACTILE_MARGIN = 0.005
_TACTILE_SHEAR_VIZ_RES = 30
_SETTLE_STEPS = 8


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def _finger_usd() -> str:
    return str(
        (
            _repo_root()
            / "source/ViTacLab/ViTacLab/assets/data/Sensors/Gelsight_finger/gelsight_r15_finger_short_v2.usd"
        ).resolve()
    )


def _make_robot_cfg(usd_path: str) -> ArticulationCfg:
    spawn = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        physics_material_prim_path="elastomer",
        activate_contact_sensors=True,
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


def _make_nut_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _format_tacsl_paths(sensor_cfg: VisuoTactileSensorV2Cfg, scene: InteractiveScene) -> None:
    """ViTacLab TacSL cfg is not recognized by InteractiveScene isinstance check — format manually."""
    ns = scene.env_regex_ns
    sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.camera_cfg is not None:
        sensor_cfg.camera_cfg.prim_path = sensor_cfg.camera_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.contact_object_prim_path_expr is not None:
        sensor_cfg.contact_object_prim_path_expr = sensor_cfg.contact_object_prim_path_expr.format(ENV_REGEX_NS=ns)


def _make_v2_sensor_cfg() -> VisuoTactileSensorV2Cfg:
    return VisuoTactileSensorV2Cfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=_TACTILE_ARRAY,
        tactile_margin=_TACTILE_MARGIN,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        # Default deadband (2 mm) zeros most real contacts; match pour / debug TacSL scripts.
        depth_penetration_deadband=0.0,
        normal_contact_stiffness=1.0,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def _register_v2_sensor(scene: InteractiveScene) -> None:
    scfg = _make_v2_sensor_cfg()
    _format_tacsl_paths(scfg, scene)
    scene.sensors["tactile_sensor"] = scfg.class_type(scfg)


def _write_nut_default_pose(scene: InteractiveScene) -> None:
    """Reset nut to spawn pose with zero velocity (used to keep baseline no-contact)."""
    if "contact_object" not in scene.rigid_objects:
        return
    rigid = scene.rigid_objects["contact_object"]
    state = rigid.data.default_root_state.clone()
    state[:, 0:3] += scene.env_origins
    rigid.write_root_pose_to_sim(state[:, :7], env_ids=None)
    rigid.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]), env_ids=None)


def _settle_and_baseline(
    scene: InteractiveScene,
    sim: sim_utils.SimulationContext,
    *,
    settle_steps: int = _SETTLE_STEPS,
) -> None:
    """Capture nominal depth with nut **not** touching the pad (required for V2 force field)."""
    sim_dt = sim.get_physics_dt()
    scene["robot"].reset()
    if "contact_object" in scene.rigid_objects:
        scene.rigid_objects["contact_object"].reset()
        _write_nut_default_pose(scene)

    robot = scene["robot"]
    state = robot.data.default_root_state.clone()
    state[:, 0:3] += scene.env_origins
    robot.write_root_pose_to_sim(state[:, :7], env_ids=None)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=None)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone(), env_ids=None)
    scene.write_data_to_sim()

    ts = scene["tactile_sensor"]
    # Pin nut during settle so gravity does not pollute the no-contact depth baseline.
    for _ in range(max(1, settle_steps)):
        _write_nut_default_pose(scene)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    if ts.cfg.enable_camera_tactile:
        ts.get_initial_render()


def _tactile_ff_rgb(nf_hw: np.ndarray, sf_hw2: np.ndarray) -> np.ndarray:
    peak_n = max(float(np.abs(nf_hw).max()), 1e-9)
    peak_s = max(float(np.linalg.norm(sf_hw2, axis=-1).max()), 1e-9)
    img_bgr = compute_tactile_shear_image(
        nf_hw,
        sf_hw2,
        normal_force_threshold=peak_n,
        shear_force_threshold=peak_s,
    )
    u8 = (np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if x.max() <= 1.0:
        x = np.clip(x, 0.0, 1.0) * 255.0
    return x.astype(np.uint8)


@configclass
class VisuoTactileV2DemoSceneCfg(InteractiveSceneCfg):
    """GelSight short finger + nut; TacSL V2 registered after scene init."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = _make_robot_cfg(_finger_usd())
    contact_object = _make_nut_cfg()


def main() -> None:
    usd = Path(_finger_usd())
    if not usd.is_file():
        print(f"[ERROR] Finger USD not found: {usd}", file=sys.stderr)
        simulation_app.close()
        raise SystemExit(1)

    print(f"[INFO] VisuoTactileSensorV2 demo — finger USD: {usd}")
    print("[INFO] Force field: depth-based (no SDF). Camera tactile: enabled.")
    print("[INFO] Contact object (nut): dynamic rigid body, gravity enabled.")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.6, 1.0], target=[-0.1, 0.1, 0.5])

    scene = InteractiveScene(VisuoTactileV2DemoSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2))
    _register_v2_sensor(scene)
    sim.reset()
    _settle_and_baseline(scene, sim)
    print("[INFO] Setup complete. VisuoTactileSensorV2 initialized.")

    nrows, ncols = _TACTILE_ARRAY
    show_plot = (args_cli.show_ff or args_cli.show_rgb) and not args_cli.no_plot
    fig = im_ff = im_rgb = None
    if show_plot:
        import matplotlib.pyplot as plt

        plt.ion()
        if args_cli.show_ff and args_cli.show_rgb:
            fig, (ax_ff, ax_rgb) = plt.subplots(1, 2, figsize=(12, 5), num="VisuoTactileSensorV2")
            zh, zw = GELSIGHT_R15_CFG.image_height, GELSIGHT_R15_CFG.image_width
            im_ff = ax_ff.imshow(
                np.zeros((nrows * _TACTILE_SHEAR_VIZ_RES, ncols * _TACTILE_SHEAR_VIZ_RES, 3), dtype=np.uint8)
            )
            ax_ff.set_title("Force field (shear)")
            ax_ff.axis("off")
            im_rgb = ax_rgb.imshow(np.zeros((zh, zw, 3), dtype=np.uint8))
            ax_rgb.set_title("Tactile RGB")
            ax_rgb.axis("off")
        elif args_cli.show_ff:
            fig, ax = plt.subplots(figsize=(7, 6), num="VisuoTactileSensorV2 FF")
            im_ff = ax.imshow(
                np.zeros((nrows * _TACTILE_SHEAR_VIZ_RES, ncols * _TACTILE_SHEAR_VIZ_RES, 3), dtype=np.uint8)
            )
            ax.set_title("Force field (shear)")
            ax.axis("off")
        else:
            fig, ax = plt.subplots(figsize=(7, 6), num="VisuoTactileSensorV2 RGB")
            zh, zw = GELSIGHT_R15_CFG.image_height, GELSIGHT_R15_CFG.image_width
            im_rgb = ax.imshow(np.zeros((zh, zw, 3), dtype=np.uint8))
            ax.set_title("Tactile RGB")
            ax.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        print("[INFO] Matplotlib viewer open.")

    sim_dt = sim.get_physics_dt()
    step = 0
    force = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    nut = scene["contact_object"]

    while simulation_app.is_running():
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break

        if step > 20 and args_cli.nut_torque_z != 0.0:
            env_ids = torch.arange(scene.num_envs, device=sim.device)
            torque[env_ids % 2 == 1, 0, 2] = args_cli.nut_torque_z
            torque[env_ids % 2 == 0, 0, 2] = -args_cli.nut_torque_z
            nut.permanent_wrench_composer.set_forces_and_torques(force, torque)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1

        td = scene["tactile_sensor"].data
        nf = getattr(td, "tactile_normal_force", None)
        sf = getattr(td, "tactile_shear_force", None)

        if args_cli.log_interval > 0 and step % args_cli.log_interval == 0 and nf is not None:
            peak = float(nf.abs().max().detach().cpu())
            mean = float(nf.abs().mean().detach().cpu())
            pen = getattr(scene["tactile_sensor"].data, "penetration_depth", None)
            pen_peak = float(pen.abs().max().detach().cpu()) if pen is not None else 0.0
            print(
                f"[step {step:5d}] |normal_force| peak={peak:.6f} mean={mean:.6f} "
                f"|penetration| peak={pen_peak:.6f}"
            )

        if show_plot and fig is not None:
            import matplotlib.pyplot as plt

            if im_ff is not None and nf is not None and sf is not None:
                nf0 = nf[0].detach().cpu().numpy().reshape(nrows, ncols)
                sf0 = sf[0].detach().cpu().numpy().reshape(nrows, ncols, 2)
                im_ff.set_data(_tactile_ff_rgb(nf0, sf0))
            if im_rgb is not None:
                rgb = getattr(td, "tactile_rgb_image", None)
                if rgb is not None and rgb.ndim == 4:
                    x = rgb[0].detach().cpu().numpy()
                    if x.shape[0] == 3:
                        x = np.transpose(x, (1, 2, 0))
                    im_rgb.set_data(_img_to_uint8(x))
            fig.canvas.draw_idle()
            plt.pause(0.001)

    if show_plot and fig is not None:
        import matplotlib.pyplot as plt

        plt.close("all")
    print("[INFO] Demo finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()
