# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: full-hand palm-frame voxel tactile on the UR10e + Shadow Hand in-hand manipulation layout.

Scene / object / default arm pose match
:class:`ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg.UR10eShadowHandInHandEnvCfg`.

The robot USD spawn enables contact reporters so :class:`~ViTacLab.assets.sensor.shadow_hand_full_tactile.ShadowHandFullTactileSensor`
can aggregate ``get_contact_data`` / ``get_friction_data`` into ``voxel_grid``.

On startup, joint positions are written with :meth:`~isaaclab.assets.articulation.Articulation.write_joint_state_to_sim`
so the arm matches the in-hand configuration immediately (no transient from default USD pose).

Tactile visualization: matplotlib ``Axes3D.voxels`` in **palm** coordinates (meters on the axes when supported
by your matplotlib version); color encodes ``|fn| + ‖ft‖`` per cell.

Console logs include **contact_pts_mean_palm**: PhysX ``get_contact_data`` contact positions (world) transformed
into the palm body frame (same as voxel binning), averaged per env/filter; see
``ShadowHandFullTactileData.contact_normal_points_mean_palm``.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Shadow Hand full-hand voxel tactile sensor demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--no_plot",
    action="store_true",
    help="Disable matplotlib (headless / no display).",
)
parser.add_argument(
    "--debug_vis",
    action="store_true",
    help="Enable Isaac Lab contact-sensor debug visualization in the viewport.",
)
parser.add_argument(
    "--voxel_alpha",
    type=float,
    default=0.85,
    help="Opacity of occupied voxels in the 3D plot (0..1).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    spawn_factory_table,
)
from ViTacLab.assets.sensor import ShadowHandFullTactileSensorCfg
from ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg import (
    UR10eShadowHandInHandEnvCfg,
)

_INHAND_CFG = UR10eShadowHandInHandEnvCfg()
_OBJECT_CFG = _INHAND_CFG.object_cfg.replace(prim_path="{ENV_REGEX_NS}/object")


def _robot_cfg_inhand_with_contact() -> ArticulationCfg:
    """Same articulated defaults as :attr:`UR10eShadowHandInHandEnvCfg.robot_cfg`, plus contact reporters."""
    rc = _INHAND_CFG.robot_cfg
    return rc.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=rc.spawn.replace(activate_contact_sensors=True),
    )


@configclass
class ShadowHandFullTactileInHandSceneCfg(InteractiveSceneCfg):
    """UR10e Shadow Hand (same robot_cfg as in-hand RL) + DexCube + full-hand voxel tactile (vs object).

    Match :class:`UR10eShadowHandDirectBaseEnv` stage: factory table + offset ground are spawned in :func:`main`
    (not the default ``InteractiveScene`` ground plane).
    """

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: ArticulationCfg = _robot_cfg_inhand_with_contact()

    object = _OBJECT_CFG

    full_hand_tactile = ShadowHandFullTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        max_contact_data_count_per_prim=1024,
        palm_link_name_substr="palm",
        voxel_resolution=(48, 36, 8),
        voxel_min_bounds_palm=(-0.2, -0.2, -0.2),
        voxel_max_bounds_palm=(0.2, 0.2, 0.2),
        track_friction=True,
        track_pose=False,
    )


def _apply_inhand_initial_robot_state(robot) -> torch.Tensor:
    """Write default (in-hand) joint pose into the sim immediately so the arm is already correct on frame 0.

    Mirrors env reset: all DOF positions from ``default_joint_pos`` (includes merged
    ``UR10eShadowHandInHandEnvCfg.robot_cfg`` arm + USD hand defaults), zero velocities.
    """
    dof_pos = robot.data.default_joint_pos.clone()
    dof_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(dof_pos, dof_vel)
    robot.set_joint_position_target(dof_pos)
    return dof_pos


def _intensity_rgba_numpy(
    intensity: torch.Tensor,
    *,
    cmap_name: str,
    vmax: float | None,
    alpha: float,
) -> np.ndarray:
    """Map scalar ``(nx,ny,nz)`` to ``(nx,ny,nz,4)`` RGBA (numpy) in display [0,1]."""
    import matplotlib

    s_np = intensity.detach().float().cpu().numpy()
    norm_vmax = float(s_np.max()) if vmax is None else float(vmax)
    norm_vmax = max(norm_vmax, 1e-9)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(s_np / norm_vmax)
    rgba[..., 3] = alpha
    return rgba


def _apply_palm_voxel_view_limits(
    ax,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
) -> None:
    """Set limits to 2x the ``[bmin,bmax]`` span (centered); re-apply after ``voxels`` (it may autoscale)."""
    xm, ym, zm = bmin
    xM, yM, zM = bmax
    cx, cy, cz = 0.5 * (xm + xM), 0.5 * (ym + yM), 0.5 * (zm + zM)
    sx, sy, sz = xM - xm, yM - ym, zM - zm
    ax.set_xlim(cx - sx, cx + sx)
    ax.set_ylim(cy - sy, cy + sy)
    ax.set_zlim(cz - sz, cz + sz)
    ax.set_box_aspect((sx, sy, sz))


def _make_3d_voxel_figure(*, bmin: tuple[float, float, float], bmax: tuple[float, float, float]):
    """3D matplotlib figure: palm-frame voxels colored by ``|fn| + ‖ft‖`` (occupied cells)."""
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure(num="Full-hand voxel tactile — palm frame (3D)", figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    _apply_palm_voxel_view_limits(ax, bmin, bmax)
    ax.set_xlabel("palm +X (m)")
    ax.set_ylabel("palm +Y (m)")
    ax.set_zlabel("palm +Z (m)")
    ax.set_title("voxel_grid in palm frame (|fn| + ‖ft‖)")
    fig.tight_layout()
    fig.show()
    return fig, ax


def _update_3d_voxel_figure(
    ax,
    v0: torch.Tensor,
    *,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
    alpha: float,
    mean_palm: torch.Tensor | None = None,
):
    """Redraw voxels from ``v0``; optional ``mean_palm`` marks the contact-point centroid in palm frame."""
    import inspect

    import matplotlib.pyplot as plt

    nx, ny, nz, _ = v0.shape
    fn = v0[..., 0].abs()
    f1, f2 = v0[..., 1], v0[..., 2]
    ft_mag = torch.sqrt(torch.clamp(f1 * f1 + f2 * f2, min=0.0))
    intensity = fn + ft_mag

    int_np = intensity.detach().cpu().numpy()
    imax = float(np.max(int_np)) if int_np.size else 0.0
    if imax > 0.0:
        # Absolute 1e-9 hides real signal when peak is e.g. 1e-7; use small fraction of peak.
        eps = max(imax * 1e-9, np.finfo(np.float64).tiny * 1e3)
        filled = int_np > eps
    else:
        filled = np.zeros(int_np.shape, dtype=bool)

    for c in list(ax.collections):
        c.remove()

    show_voxels = bool(filled.any())
    if show_voxels:
        rgba = _intensity_rgba_numpy(
            intensity, cmap_name="magma", vmax=float(intensity.max().item()), alpha=alpha
        )
        x_e = np.linspace(bmin[0], bmax[0], nx + 1, dtype=np.float64)
        y_e = np.linspace(bmin[1], bmax[1], ny + 1, dtype=np.float64)
        z_e = np.linspace(bmin[2], bmax[2], nz + 1, dtype=np.float64)

        sig = inspect.signature(ax.voxels)
        kw = dict(facecolors=rgba, edgecolor="k", linewidth=0.06, shade=False)
        if "x" in sig.parameters:
            ax.voxels(filled, **kw, x=x_e, y=y_e, z=z_e)
        else:
            ax.voxels(filled, **kw)

    if mean_palm is not None and torch.isfinite(mean_palm).all():
        mp = mean_palm.detach().float().cpu().numpy().reshape(3)
        ax.scatter(
            [mp[0]],
            [mp[1]],
            [mp[2]],
            c="lime",
            s=120,
            depthshade=True,
            edgecolors="k",
            linewidths=0.8,
            zorder=10,
            label="contact points mean (palm)",
        )

    _apply_palm_voxel_view_limits(ax, bmin, bmax)

    ax.figure.canvas.draw_idle()
    ax.figure.canvas.flush_events()
    plt.pause(0.001)


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    *,
    enable_plot: bool,
    palm_bmin: tuple[float, float, float],
    palm_bmax: tuple[float, float, float],
    voxel_alpha: float,
):
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    targets = robot.data.default_joint_pos.clone()

    fig_ax = None
    if enable_plot:
        try:
            fig_ax = _make_3d_voxel_figure(bmin=palm_bmin, bmax=palm_bmax)
            print("[INFO] Matplotlib: 3D palm-frame voxel plot (use --no_plot to disable).")
        except Exception as exc:
            print(f"[WARN] Could not open matplotlib ({exc}). Continuing without plot.")
            fig_ax = None

    step = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1

        if step % 30 == 0:
            tactile = scene["full_hand_tactile"].data
            vox = tactile.voxel_grid
            if vox is not None:
                v0 = vox[0, 0]
                fn, f1, f2 = v0[..., 0], v0[..., 1], v0[..., 2]
                mag = torch.sqrt(torch.clamp(f1 * f1 + f2 * f2, min=0.0))
                cpp = tactile.contact_normal_points_mean_palm
                cpc = tactile.contact_normal_point_count
                print("cpc:",cpc)
                mean_for_plot: torch.Tensor | None = None
                palm_mu = ""
                if cpp is not None and cpc is not None:
                    p = cpp[0, 0]
                    ns = int(cpc[0, 0].item())
                    if ns > 0 and not bool(torch.isnan(p).any().item()):
                        mean_for_plot = p
                        palm_mu = (
                            f"  contact_pts_mean_palm(m)=({p[0].item():.4f},{p[1].item():.4f},{p[2].item():.4f})"
                            f"  n_physx_pts={ns}"
                        )
                    else:
                        palm_mu = "  contact_pts_mean_palm(m)=nan  n_physx_pts=0"
                print(
                    f"[step {step:5d}] voxel_grid[env=0,filter=0]: |fn|_max={fn.abs().max().item():.4f}  "
                    f"|ft|_max={mag.max().item():.4f}  shape={tuple(v0.shape)}{palm_mu}"
                )
                if fig_ax is not None:
                    _, ax = fig_ax
                    try:
                        _update_3d_voxel_figure(
                            ax,
                            v0,
                            bmin=palm_bmin,
                            bmax=palm_bmax,
                            alpha=voxel_alpha,
                            mean_palm=mean_for_plot,
                        )
                    except Exception as exc:
                        print(f"[WARN] 3D plot update failed ({exc}); disabling plot.")
                        fig_ax = None


def main():
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
        use_fabric=True,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(enable_ccd=True, bounce_threshold_velocity=0.2),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.2, 1.0, 0.8), target=(0.6, 0.0, 0.35))

    scene_cfg = ShadowHandFullTactileInHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
    if args_cli.debug_vis:
        scene_cfg = scene_cfg.replace(
            full_hand_tactile=scene_cfg.full_hand_tactile.replace(debug_vis=True),
        )
    scene = InteractiveScene(scene_cfg)

    spawn_factory_table()
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

    sim.reset()

    robot = scene["robot"]
    warm_targets = _apply_inhand_initial_robot_state(robot)
    scene.write_data_to_sim()

    for _ in range(20):
        robot.set_joint_position_target(warm_targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    print("[INFO] Shadow Hand full-hand voxel tactile demo (in-hand manipulation scene layout).")
    print("[INFO] Robot init matches UR10eShadowHandInHandEnvCfg.robot_cfg; table/ground match UR10eShadowHandDirectBaseEnv.")
    print("[INFO] scene['full_hand_tactile'].data.voxel_grid shape (N, F, nx, ny, nz, 3).")
    print(
        "[INFO] Logs include contact_pts_mean_palm: PhysX contact ``points`` (world) moved into the palm body "
        "frame (same transform as voxel binning), then averaged over samples."
    )
    vcfg = scene_cfg.full_hand_tactile
    bmin = vcfg.voxel_min_bounds_palm
    bmax = vcfg.voxel_max_bounds_palm
    run_simulator(
        sim,
        scene,
        enable_plot=not args_cli.no_plot,
        palm_bmin=bmin,
        palm_bmax=bmax,
        voxel_alpha=float(max(0.0, min(1.0, args_cli.voxel_alpha))),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
