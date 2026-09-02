# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: grid tactile sensor with Wuji hand USD as contact pad under one or more rolling spheres.

Follows :mod:`demo_grid_tactile_sensor` (force / friction grids, optional matplotlib), but replaces the
cuboid pad with ``wujihand.usd``. Spheres are ``Sphere_0..N`` along X (see ``--num_spheres``, ``--sphere_spacing``).
Place or export ``wujihand.usd`` at the default path, or pass ``--usd``.

``wujihand.usd`` is a multi-link hand articulation. This demo loads it with :class:`~isaaclab.assets.ArticulationCfg`
and attaches :class:`~ViTacLab.assets.sensor.grid_tactile.GridTactileSensor` to a **single** link (default
``right_palm_link``). Use ``--tactile_link`` if your USD uses a different palm prim name.

With GUI (not ``--headless``), a docked **Wuji hand — root pose** panel provides sliders for root
position (m) and Euler XYZ (deg), plus **Apply once** / **Read from sim** and **Apply continuously**
(default on) so the hand tracks the sliders each physics step.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

_VI_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WUJI_USD = (
    _VI_ROOT
    / "source/ViTacLab/ViTacLab/assets/data/Robots/wuji-hand-description-main/usd/right/wujihand.usd"
)

parser = argparse.ArgumentParser(description="Grid tactile sensor demo — Wuji hand USD as pad.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--usd",
    type=str,
    default=str(_DEFAULT_WUJI_USD),
    help="Absolute or repo-relative path to wujihand.usd (or any rigid pad USD).",
)
parser.add_argument("--scale", type=float, default=1.0, help="Uniform spawn scale for the pad USD.")
parser.add_argument(
    "--pad_pos",
    type=float,
    nargs=3,
    default=(0.535, 0.5, 0.07),
    metavar=("X", "Y", "Z"),
    help="Pad root position (m), world frame.",
)
parser.add_argument(
    "--pad_quat",
    type=float,
    nargs=4,
    default=(0.70711, 0.0, -0.70711, 0.0),
    metavar=("W", "X", "Y", "Z"),
    help="Pad root orientation quaternion (w, x, y, z), world frame.",
)
parser.add_argument(
    "--sphere_pos",
    type=float,
    nargs=3,
    default=(0.5, 0.5, 0.12),
    metavar=("X", "Y", "Z"),
    help="Center of the sphere row: middle ball when num_spheres>1 (m, world).",
)
parser.add_argument(
    "--num_spheres",
    type=int,
    default=1,
    help="Number of spheres; placed along +X/--sphere_spacing, named Sphere_0..",
)
parser.add_argument(
    "--sphere_spacing",
    type=float,
    default=0.055,
    help="Center-to-center distance between adjacent spheres along X (m).",
)
parser.add_argument(
    "--sensor_body",
    type=int,
    default=0,
    help="Body index for grid output (sensor is on one link only; keep 0).",
)
parser.add_argument(
    "--tactile_link",
    type=str,
    default="right_palm_link",
    help="Rigid link name under WujiPad for GridTactileSensor (single-body contact path).",
)
parser.add_argument(
    "--no_plot",
    action="store_true",
    help="Disable matplotlib window (use on headless / no display).",
)
parser.add_argument(
    "--pose_ui",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show Isaac Sim window with sliders for hand root position (m) and Euler XYZ (deg); "
    "disabled automatically when --headless.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_usd_path = Path(args_cli.usd)
if not _usd_path.is_file():
    print(
        "[ERROR]: Pad USD not found:\n"
        f"         {_usd_path.resolve()}\n"
        "         Export or copy wujihand.usd there, or pass a valid path via --usd."
    )
    sys.exit(1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from ViTacLab.assets.sensor import GridTactileSensorCfg


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


def _apply_hand_root_pose_from_ui_models(scene: InteractiveScene, ui: dict) -> None:
    """Write root link pose from 6 float models (px,py,pz m; rx,ry,rz deg XYZ) for env ``ui['env_id']``."""
    hand = scene[ui["hand_key"]]
    env_id = int(ui["env_id"])
    dev = hand.device
    models: list = ui["models"]
    px, py, pz = (_float_ui_model_get(models[i]) for i in range(3))
    rx, ry, rz = (math.radians(_float_ui_model_get(models[i])) for i in range(3, 6))
    roll = torch.tensor([rx], device=dev, dtype=torch.float32)
    pitch = torch.tensor([ry], device=dev, dtype=torch.float32)
    yaw = torch.tensor([rz], device=dev, dtype=torch.float32)
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    pos = torch.tensor([[px, py, pz]], device=dev, dtype=torch.float32)
    pose = torch.cat([pos, quat], dim=-1)
    eids = torch.tensor([env_id], device=dev, dtype=torch.long)
    hand.write_root_link_pose_to_sim(pose, env_ids=eids)
    vel = torch.zeros(1, 6, device=dev, dtype=torch.float32)
    hand.write_root_link_velocity_to_sim(vel, env_ids=eids)


def _sync_hand_pose_ui_from_sim(scene: InteractiveScene, ui: dict) -> None:
    hand = scene[ui["hand_key"]]
    env_id = int(ui["env_id"])
    pose = hand.data.root_link_pose_w[env_id]
    pos = pose[:3].detach().cpu().tolist()
    quat = pose[3:7].unsqueeze(0).detach()
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    degs = [math.degrees(float(roll[0])), math.degrees(float(pitch[0])), math.degrees(float(yaw[0]))]
    models: list = ui["models"]
    for i in range(3):
        _float_ui_model_set(models[i], float(pos[i]))
    for i in range(3):
        _float_ui_model_set(models[i + 3], float(degs[i]))


def _try_create_wuji_hand_pose_ui(scene: InteractiveScene, *, hand_key: str, env_id: int) -> dict | None:
    """Docked ``omni.ui`` panel to edit articulation root pose (world frame). Returns state dict or None."""
    if getattr(args_cli, "headless", False) or not bool(getattr(args_cli, "pose_ui", True)):
        return None
    try:
        import omni.ui  # type: ignore
    except Exception as exc:
        print(f"[WARN]: Pose UI skipped (omni.ui unavailable: {exc}).")
        return None

    hand = scene[hand_key]
    pose0 = hand.data.root_link_pose_w[int(env_id)].detach()
    pos0 = pose0[:3].cpu().tolist()
    quat0 = pose0[3:7].unsqueeze(0)
    r0, p0, y0 = euler_xyz_from_quat(quat0)
    init_deg = [
        math.degrees(float(r0[0])),
        math.degrees(float(p0[0])),
        math.degrees(float(y0[0])),
    ]

    ui: dict = {"hand_key": hand_key, "env_id": int(env_id), "models": [], "live": None}

    win = omni.ui.Window(
        "Wuji hand — root pose",
        width=340,
        height=0,
        dock_preference=omni.ui.DockPreference.RIGHT_TOP,
    )

    labels = ("px (m)", "py (m)", "pz (m)", "rx (deg)", "ry (deg)", "rz (deg)")
    bounds = (
        (-1.5, 1.5),
        (-1.5, 1.5),
        (-0.5, 1.0),
        (-180.0, 180.0),
        (-180.0, 180.0),
        (-180.0, 180.0),
    )
    inits = [float(pos0[0]), float(pos0[1]), float(pos0[2]), init_deg[0], init_deg[1], init_deg[2]]

    with win.frame:
        with omni.ui.VStack(spacing=4):
            omni.ui.Label("Root link pose (world). Euler XYZ matches isaaclab.utils.math.", word_wrap=True)
            live_m = omni.ui.SimpleBoolModel()
            live_m.set_value(True)
            ui["live"] = live_m
            with omni.ui.HStack():
                omni.ui.Label("Apply continuously", width=140)
                omni.ui.CheckBox(model=live_m)
            for i in range(6):
                m = omni.ui.SimpleFloatModel()
                _float_ui_model_set(m, inits[i])
                ui["models"].append(m)
                lo, hi = bounds[i]
                with omni.ui.HStack(spacing=6):
                    omni.ui.Label(labels[i], width=72, alignment=omni.ui.Alignment.LEFT_CENTER)
                    omni.ui.FloatSlider(model=m, min=float(lo), max=float(hi), step=(hi - lo) / 400.0)
            with omni.ui.HStack(spacing=6):
                omni.ui.Button(
                    "Apply once",
                    clicked_fn=lambda: _apply_hand_root_pose_from_ui_models(scene, ui),
                )
                omni.ui.Button(
                    "Read from sim",
                    clicked_fn=lambda: _sync_hand_pose_ui_from_sim(scene, ui),
                )

    print("[INFO]: Opened 'Wuji hand — root pose' UI (dock right). Toggle live apply or use Apply once.")
    return ui


def _wuji_urdf_for_hand_usd(usd_path: Path) -> Path:
    """``.../usd/<side>/wujihand.usd`` → ``.../urdf/<side>.urdf``."""
    r = usd_path.resolve()
    side = r.parent.name
    return r.parent.parent.parent / "urdf" / f"{side}.urdf"


def _joint_pos_mid_from_urdf(urdf_path: Path) -> dict[str, float]:
    """For each revolute joint with limits, use midpoint so defaults lie inside [lower, upper]."""
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


def _make_force_grid_plot(
    grid_hw: tuple[int, int],
    extent_m: tuple[float, float],
):
    """Open a live matplotlib figure: normal (left) and tangential friction magnitude (right)."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.ion()
    h, w = grid_hw
    ext_u, ext_v = extent_m
    extent = (-0.5 * ext_v, 0.5 * ext_v, -0.5 * ext_u, 0.5 * ext_u)
    fig, (ax_n, ax_f) = plt.subplots(1, 2, num="Grid tactile — Wuji pad", figsize=(11, 4.5))
    z = np.zeros((h, w), dtype=np.float64)
    im_n = ax_n.imshow(
        z,
        cmap="RdBu_r",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        extent=extent,
    )
    im_n.set_clim(-1.0, 1.0)
    cbar_n = fig.colorbar(im_n, ax=ax_n, fraction=0.046, pad=0.04)
    cbar_n.set_label("signed f·n_pad (N)")
    ax_n.set_xlabel("tangent v (m)")
    ax_n.set_ylabel("tangent u (m)")
    ax_n.set_title("Normal (get_contact_data)")

    im_f = ax_f.imshow(
        z,
        cmap="viridis",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        extent=extent,
    )
    im_f.set_clim(0.0, 1.0)
    cbar_f = fig.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
    cbar_f.set_label("|f_friction,tan| (N)")
    ax_f.set_xlabel("tangent v (m)")
    ax_f.set_ylabel("tangent u (m)")
    ax_f.set_title("Friction ‖(f_u, f_v)‖ (get_friction_data)")
    fig.tight_layout()
    fig.show()
    return fig, im_n, im_f, cbar_n, cbar_f


def _update_force_grid_plot(im_n, im_f, g_normal: torch.Tensor, friction_uv: torch.Tensor | None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    g = g_normal.detach().float().cpu().numpy()
    im_n.set_data(g)
    peak = float(np.abs(g).max())
    lim = max(peak, 1e-9)
    im_n.set_clim(-lim, lim)

    if friction_uv is not None:
        fu = friction_uv[0].detach().float().cpu().numpy()
        fv = friction_uv[1].detach().float().cpu().numpy()
        mag = np.sqrt(np.maximum(fu * fu + fv * fv, 0.0))
        im_f.set_data(mag)
        im_f.set_clim(0.0, max(float(mag.max()), 1e-9))

    fig = im_n.figure
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def _build_scene_cfg() -> InteractiveSceneCfg:
    pad_rot_wxyz = tuple(float(x) for x in args_cli.pad_quat)
    s = float(args_cli.scale)
    usd_str = str(_usd_path.resolve())
    link = str(args_cli.tactile_link).strip().strip("/")
    tactile_prim = f"{{ENV_REGEX_NS}}/WujiPad/{link}"

    urdf_primary = _wuji_urdf_for_hand_usd(_usd_path)
    urdf_fallback = (
        _VI_ROOT
        / "source/ViTacLab/ViTacLab/assets/data/Robots/wuji-hand-description-main/urdf"
        / f"{_usd_path.resolve().parent.name}.urdf"
    )
    urdf_path = urdf_primary if urdf_primary.is_file() else urdf_fallback
    if not urdf_path.is_file():
        print(
            "[ERROR]: Need Wuji URDF (e.g. ``urdf/right.urdf``) to build valid ``joint_pos``; "
            "zero defaults violate joint limits.\n"
            f"         Tried: {urdf_primary}\n"
            f"         Tried: {urdf_fallback}"
        )
        sys.exit(1)
    wuji_joint_pos = _joint_pos_mid_from_urdf(urdf_path)
    if not wuji_joint_pos:
        print(f"[ERROR]: No revolute limits parsed from {urdf_path}")
        sys.exit(1)

    n_spheres = max(1, int(args_cli.num_spheres))
    spacing = float(args_cli.sphere_spacing)
    sx, sy, sz = (float(x) for x in args_cli.sphere_pos)
    filter_exprs = [f"{{ENV_REGEX_NS}}/Sphere_{i}" for i in range(n_spheres)]

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    wuji_pad = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/WujiPad",
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
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(args_cli.pad_pos),
            rot=pad_rot_wxyz,
            joint_pos=wuji_joint_pos,
            joint_vel={},
        ),
        actuators={},
    )

    sphere_spawn = sim_utils.SphereCfg(
        radius=0.0048,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0025),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 0.35), metallic=0.15),
    )

    sphere_fields: dict = {}
    sphere_ann: dict[str, type] = {}
    for i in range(n_spheres):
        ox = (i - 0.5 * (n_spheres - 1)) * spacing
        key = f"sphere_{i}"
        sphere_ann[key] = RigidObjectCfg
        sphere_fields[key] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Sphere_{i}",
            spawn=sphere_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(sx + ox, sy, sz),
                ang_vel=(1.25, 1.25, 0),
                lin_vel=(0.1, 0, 0),
            ),
        )

    tactile_pad = GridTactileSensorCfg(
        prim_path=tactile_prim,
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=filter_exprs,
        max_contact_data_count_per_prim=max(64, 24 * n_spheres),
        grid_resolution=(24, 24),
        patch_extent=(0.12, 0.12),
        # 贴片外法向为传感器刚体坐标系 +X（pressure = f·n 沿 +X）；切向网格铺在 Y–Z 平面
        pad_normal_axis=0,
        track_friction=True,
        track_pose=True,
    )

    annotations: dict[str, type] = {
        "ground": AssetBaseCfg,
        "dome_light": AssetBaseCfg,
        "wuji_pad": ArticulationCfg,
        "tactile_pad": GridTactileSensorCfg,
    }
    annotations.update(sphere_ann)
    class_body: dict = {
        "__annotations__": annotations,
        "ground": ground,
        "dome_light": dome_light,
        "wuji_pad": wuji_pad,
        "tactile_pad": tactile_pad,
    }
    class_body.update(sphere_fields)

    GridTactileWujiPadSceneCfg = configclass(
        type("GridTactileWujiPadSceneCfg", (InteractiveSceneCfg,), class_body)
    )
    return GridTactileWujiPadSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
    )


def _pose_ui_live_on(ui: dict | None) -> bool:
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


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    *,
    enable_plot: bool,
    grid_resolution: tuple[int, int],
    patch_extent: tuple[float, float],
    sensor_body: int,
    pose_ui: dict | None = None,
):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    b = max(0, int(sensor_body))

    plot_handles = None
    if enable_plot:
        try:
            plot_handles = _make_force_grid_plot(grid_resolution, patch_extent)
            print("[INFO]: Matplotlib plot window opened (close --no_plot if you see no window).")
        except Exception as exc:
            print(f"[WARN]: Could not start matplotlib plot ({exc}). Continuing without plot.")
            plot_handles = None

    while simulation_app.is_running():
        if _pose_ui_live_on(pose_ui):
            try:
                _apply_hand_root_pose_from_ui_models(scene, pose_ui)  # type: ignore[arg-type]
            except Exception:
                pass

        if count % 400 == 0:
            count = 0
            root_pad = scene["wuji_pad"].data.default_root_state.clone()
            root_pad[:, :3] += scene.env_origins
            scene["wuji_pad"].write_root_pose_to_sim(root_pad[:, :7])
            scene["wuji_pad"].write_root_velocity_to_sim(root_pad[:, 7:])

            for i in range(max(1, int(args_cli.num_spheres))):
                key = f"sphere_{i}"
                root_sphere = scene[key].data.default_root_state.clone()
                root_sphere[:, :3] += scene.env_origins
                root_sphere[:, 2] += 0.05 * torch.randn(scene.num_envs, device=sim.device)
                av = torch.tensor([0.25, 0.25, 1.5], device=sim.device, dtype=root_sphere.dtype)
                root_sphere[:, 10:13] = av.unsqueeze(0).expand(root_sphere.shape[0], 3)
                scene[key].write_root_pose_to_sim(root_sphere[:, :7])
                scene[key].write_root_velocity_to_sim(root_sphere[:, 7:])

            scene.reset()
            print("[INFO]: Reset pad and sphere(s) state.")

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        if count % 10 == 0:
            grid = scene["tactile_pad"].data.force_grid
            fuv = scene["tactile_pad"].data.friction_grid_uv
            net = scene["tactile_pad"].data.net_forces_w
            if grid is not None and b < grid.shape[1]:
                # 多球时每个 filter 一维，显示/打印对各球累加后的贴片
                g0 = grid[0, b].sum(dim=0)
                peak = g0.max().item()
                trough = g0.min().item()
                mean = g0.mean().item()
                net_n = net[0, b].norm().item() if b < net.shape[1] else float("nan")
                f_mag = 0.0
                if fuv is not None and b < fuv.shape[1]:
                    uv = fuv[0, b].sum(dim=0)
                    f_mag = torch.sqrt(torch.clamp(uv[0] ** 2 + uv[1] ** 2, min=0.0)).max().item()
                cdc = scene["tactile_pad"].data.contact_data_count
                if cdc is not None and b < cdc.shape[1]:
                    n_per = [int(cdc[0, b, f].item()) for f in range(cdc.shape[2])]
                    n_pts = sum(n_per)
                else:
                    n_per = []
                    n_pts = -1
                print(
                    f"[t={sim_time:6.3f}s] body={b} contact_pts={n_pts} (per_sphere={n_per})  "
                    f"normal_grid min={trough:.4f} max={peak:.4f} mean={mean:.6f} N  "
                    f"friction_mag_max={f_mag:.4f}  |net_forces_w|={net_n:.4f}"
                )
                if plot_handles is not None:
                    _im_n = plot_handles[1]
                    _im_f = plot_handles[2]
                    try:
                        _fuv_slice = (
                            None
                            if fuv is None or b >= fuv.shape[1]
                            else fuv[0, b].sum(dim=0)
                        )
                        _update_force_grid_plot(_im_n, _im_f, g0, _fuv_slice)
                    except Exception as exc:
                        print(f"[WARN]: Plot update failed ({exc}); disabling plot.")
                        plot_handles = None
            elif grid is not None:
                print(
                    f"[WARN]: sensor_body={b} out of range for num_bodies={grid.shape[1]}; "
                    "use --sensor_body in [0, num_bodies-1]."
                )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, 1.2, 1.0], target=[0.5, 0.5, 0.1])

    scene_cfg = _build_scene_cfg()
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    for _ in range(20):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    print(
        f"[INFO]: Grid tactile demo — Wuji pad + {max(1, int(args_cli.num_spheres))} sphere(s) "
        "(see --help for pose / spacing)."
    )
    print(f"[INFO]: Using pad USD: {_usd_path}")
    pose_ui = _try_create_wuji_hand_pose_ui(scene, hand_key="wuji_pad", env_id=0)
    enable_plot = not args_cli.no_plot
    run_simulator(
        sim,
        scene,
        enable_plot=enable_plot,
        grid_resolution=scene_cfg.tactile_pad.grid_resolution,
        patch_extent=scene_cfg.tactile_pad.patch_extent,
        sensor_body=args_cli.sensor_body,
        pose_ui=pose_ui,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
