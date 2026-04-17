# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: grid tactile sensor on a pad contacting a sphere (per-contact PhysX data binned to H×W)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grid tactile sensor demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--no_plot",
    action="store_true",
    help="Disable matplotlib window (use on headless / no display).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from ViTacLab.assets.sensor import GridTactileSensorCfg


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
    fig, (ax_n, ax_f) = plt.subplots(1, 2, num="Grid tactile — normal + friction", figsize=(11, 4.5))
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


@configclass
class GridTactileDemoSceneCfg(InteractiveSceneCfg):
    """Flat pad under a sphere; tactile sensor on the pad with the sphere as filter prim."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    pad = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pad",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.35, 0.45), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.025)),
    )

    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 0.35), metallic=0.15),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.5, 0.15),
            ang_vel=(1.25, 1.25, 0),
            lin_vel=(0.1, 0, 0),
        ),
    )

    tactile_pad = GridTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Pad",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        max_contact_data_count_per_prim=64,
        grid_resolution=(24, 24),
        patch_extent=(0.35, 0.35),
        pad_normal_axis=2,
        track_friction=True,
        track_pose=True,
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    *,
    enable_plot: bool,
    grid_resolution: tuple[int, int],
    patch_extent: tuple[float, float],
):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    plot_handles = None
    if enable_plot:
        try:
            plot_handles = _make_force_grid_plot(grid_resolution, patch_extent)
            print("[INFO]: Matplotlib plot window opened (close --no_plot if you see no window).")
        except Exception as exc:
            print(f"[WARN]: Could not start matplotlib plot ({exc}). Continuing without plot.")
            plot_handles = None

    while simulation_app.is_running():
        if count % 400 == 0:
            count = 0
            root_pad = scene["pad"].data.default_root_state.clone()
            root_pad[:, :3] += scene.env_origins
            scene["pad"].write_root_pose_to_sim(root_pad[:, :7])
            scene["pad"].write_root_velocity_to_sim(root_pad[:, 7:])

            root_sphere = scene["sphere"].data.default_root_state.clone()
            root_sphere[:, :3] += scene.env_origins
            root_sphere[:, 2] += 0.05 * torch.randn(scene.num_envs, device=sim.device)
            av = torch.tensor([0.25, 0.25, 1.5], device=sim.device, dtype=root_sphere.dtype)
            root_sphere[:, 10:13] = av.unsqueeze(0).expand(root_sphere.shape[0], 3)
            scene["sphere"].write_root_pose_to_sim(root_sphere[:, :7])
            scene["sphere"].write_root_velocity_to_sim(root_sphere[:, 7:])

            scene.reset()
            print("[INFO]: Reset pad and sphere state.")

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        if count % 10 == 0:
            grid = scene["tactile_pad"].data.force_grid
            fuv = scene["tactile_pad"].data.friction_grid_uv
            net = scene["tactile_pad"].data.net_forces_w
            if grid is not None:
                g0 = grid[0, 0, 0]
                peak = g0.max().item()
                trough = g0.min().item()
                mean = g0.mean().item()
                net_n = net[0, 0].norm().item()
                f_mag = 0.0
                if fuv is not None:
                    uv = fuv[0, 0, 0]
                    f_mag = torch.sqrt(torch.clamp(uv[0] ** 2 + uv[1] ** 2, min=0.0)).max().item()
                print(
                    f"[t={sim_time:6.3f}s] normal_grid min={trough:.4f} max={peak:.4f} mean={mean:.6f} N  "
                    f"friction_mag_max={f_mag:.4f}  |net_forces_w|={net_n:.4f}"
                )
                if plot_handles is not None:
                    _im_n = plot_handles[1]
                    _im_f = plot_handles[2]
                    try:
                        _update_force_grid_plot(_im_n, _im_f, g0, None if fuv is None else fuv[0, 0, 0])
                    except Exception as exc:
                        print(f"[WARN]: Plot update failed ({exc}); disabling plot.")
                        plot_handles = None


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, 1.2, 1.0], target=[0.5, 0.5, 0.1])

    scene_cfg = GridTactileDemoSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Warm up physics so contacts exist before the first read.
    for _ in range(20):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    print("[INFO]: Grid tactile demo — force_grid (normal) + friction_grid_uv (tangential)")
    enable_plot = not args_cli.no_plot
    run_simulator(
        sim,
        scene,
        enable_plot=enable_plot,
        grid_resolution=scene_cfg.tactile_pad.grid_resolution,
        patch_extent=scene_cfg.tactile_pad.patch_extent,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
