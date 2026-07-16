# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: full-hand palm-frame voxel tactile on the UR10e + Shadow Hand in-hand manipulation layout.

Run (same style as ``demo_visuotactile_sensor_v2.py`` — **do not** pass ``--headless``)::

    conda activate env_isaaclab_510test
    cd /path/to/IssacLab_510test/ViTacLab
    python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1 --no_plot --diag_only

Body-name diagnostic only, then exit::

    python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1 --no_plot --diag_only

Live plot (schematic hand composite + optional 3D voxels)::

    python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1
    python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1 --plot_2d_only
    python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1 --plot_voxel_2d
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
parser.add_argument(
    "--plot_2d_only",
    action="store_true",
    help="Matplotlib: schematic composite only (no 3D voxels).",
)
parser.add_argument(
    "--plot_voxel_2d",
    action="store_true",
    help="Also show raw palm-frame max-projection heatmap (in addition to schematic).",
)
parser.add_argument(
    "--tactile-schematic-mirror-x",
    action="store_true",
    help="Flip schematic composite horizontally if thumb/pinky look swapped vs your 3D view.",
)
parser.add_argument(
    "--diag_only",
    action="store_true",
    help="Print body-name diagnostic and exit (no main simulation loop).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

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
from ViTacLab.assets.sensor import (
    UR10E_ARM_BODY_NAMES,
    UR10E_SHADOW_HAND_TACTILE_BODY_NAMES,
    ShadowHandTactilePlotCfg,
    build_shadow_hand_full_tactile_sensor_cfg,
    open_shadow_hand_tactile_live_plot,
    update_shadow_hand_tactile_live_plot,
)
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

    full_hand_tactile = build_shadow_hand_full_tactile_sensor_cfg(
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        max_contact_data_count_per_prim=1024,
        voxel_resolution=(48, 36, 8),
        voxel_min_bounds_palm=(-0.2, -0.2, -0.2),
        voxel_max_bounds_palm=(0.2, 0.2, 0.2),
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        track_friction=True,
        track_pose=False,
    )


def _apply_inhand_initial_robot_state(robot) -> torch.Tensor:
    """Write default (in-hand) joint pose into the sim immediately so the arm is already correct on frame 0."""
    dof_pos = robot.data.default_joint_pos.clone()
    dof_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(dof_pos, dof_vel)
    robot.set_joint_position_target(dof_pos)
    return dof_pos


def _print_tactile_body_diagnostic(scene: InteractiveScene, palm_link_name_substr: str) -> None:
    """Print ContactSensor body list and which name matches ``palm_link_name_substr``."""
    ts = scene["full_hand_tactile"]
    names = [str(n) for n in ts.body_names]
    palm_sub = str(palm_link_name_substr)
    matches = [n for n in names if palm_sub in n]
    print(f"[DIAG] full_hand_tactile.body_names ({len(names)} bodies under prim_path={ts.cfg.prim_path!r})")
    for i, name in enumerate(names):
        tag = "  <-- palm match" if palm_sub in name else ""
        print(f"  [{i:3d}] {name}{tag}")
    print(f"[DIAG] palm_link_name_substr={palm_sub!r} -> matches {matches!r}")
    if len(matches) == 0:
        print("[WARN] No palm body matched — sensor init will fail or pick wrong frame.")
    elif len(matches) > 1:
        print("[WARN] Multiple palm matches — sensor uses the first in body_names order.")
    hand_set = set(UR10E_SHADOW_HAND_TACTILE_BODY_NAMES)
    arm_set = set(UR10E_ARM_BODY_NAMES)
    hand_like = [n for n in names if n in hand_set]
    arm_like = [n for n in names if n in arm_set]
    extra = [n for n in names if n not in hand_set and n not in arm_set]
    print(f"[DIAG] hand bodies ({len(hand_like)}/{len(hand_set)} expected): {hand_like[:12]}{'...' if len(hand_like) > 12 else ''}")
    print(f"[DIAG] arm bodies  ({len(arm_like)}): {arm_like}")
    if extra:
        print(f"[DIAG] other bodies ({len(extra)}): {extra}")
    if set(hand_like) != hand_set:
        missing = sorted(hand_set - set(hand_like))
        print(f"[WARN] Missing expected hand bodies: {missing}")


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    *,
    enable_plot: bool,
    plot_cfg: ShadowHandTactilePlotCfg | None,
    palm_bmin: tuple[float, float, float],
    palm_bmax: tuple[float, float, float],
):
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    targets = robot.data.default_joint_pos.clone()

    plot_session = None
    if enable_plot and plot_cfg is not None:
        try:
            plot_session = open_shadow_hand_tactile_live_plot(bmin=palm_bmin, bmax=palm_bmax, cfg=plot_cfg)
            modes = []
            if plot_cfg.show_schematic:
                modes.append("schematic composite")
            if plot_cfg.show_2d:
                modes.append("raw 2D max-projection")
            if plot_cfg.show_3d:
                modes.append("3D voxels")
            print(f"[INFO] Matplotlib: {' + '.join(modes)} (use --no_plot to disable).")
        except Exception as exc:
            print(f"[WARN] Could not open matplotlib ({exc}). Continuing without plot.")
            plot_session = None

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
                if plot_session is not None:
                    try:
                        update_shadow_hand_tactile_live_plot(
                            plot_session,
                            v0,
                            bmin=palm_bmin,
                            bmax=palm_bmax,
                            mean_palm=mean_for_plot,
                        )
                    except Exception as exc:
                        print(f"[WARN] Tactile plot update failed ({exc}); disabling plot.")
                        plot_session = None


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

    _print_tactile_body_diagnostic(scene, scene_cfg.full_hand_tactile.palm_link_name_substr)

    if args_cli.diag_only:
        print("[INFO] --diag_only: exiting after body-name diagnostic.")
        return

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
    plot_cfg = None
    if not args_cli.no_plot:
        plot_cfg = ShadowHandTactilePlotCfg(
            voxel_alpha=float(max(0.0, min(1.0, args_cli.voxel_alpha))),
            show_schematic=True,
            show_2d=bool(args_cli.plot_voxel_2d),
            show_3d=not args_cli.plot_2d_only,
            schematic_mirror_x=bool(args_cli.tactile_schematic_mirror_x),
        )
    run_simulator(
        sim,
        scene,
        enable_plot=not args_cli.no_plot,
        plot_cfg=plot_cfg,
        palm_bmin=bmin,
        palm_bmax=bmax,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
