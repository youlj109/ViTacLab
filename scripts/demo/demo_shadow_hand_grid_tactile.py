# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: Shadow Hand per-link grid tactile (Wuji-style builder, one grid sensor per link).

Run examples::

    python scripts/demo/demo_shadow_hand_grid_tactile.py --num_envs 1
    python scripts/demo/demo_shadow_hand_grid_tactile.py --num_envs 1 --diag_only
    python scripts/demo/demo_shadow_hand_grid_tactile.py --num_envs 1 --include_mount_links
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Shadow Hand per-link grid tactile demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--diag_only",
    action="store_true",
    help="Print sensor keys / body diagnostics and exit.",
)
parser.add_argument(
    "--include_mount_links",
    action="store_true",
    help="Include forearm/wrist tactile sensors (default excludes them).",
)
parser.add_argument(
    "--pad_normal_axis",
    type=int,
    choices=(0, 1, 2),
    default=0,
    help="GridTactile pad normal axis in each link body frame.",
)
parser.add_argument(
    "--pad_normal_sign",
    type=int,
    choices=(-1, 1),
    default=1,
    help="GridTactile outward normal sign in each link body frame.",
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
    GridTactileSensorCfg,
    build_shadow_hand_grid_tactile_sensor_cfgs,
    shadow_hand_link_names,
)
from ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg import (
    UR10eShadowHandInHandEnvCfg,
)

_INHAND_CFG = UR10eShadowHandInHandEnvCfg()
_OBJECT_CFG = _INHAND_CFG.object_cfg.replace(prim_path="{ENV_REGEX_NS}/object")


def _robot_cfg_inhand_with_contact() -> ArticulationCfg:
    rc = _INHAND_CFG.robot_cfg
    return rc.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=rc.spawn.replace(activate_contact_sensors=True),
    )


def _build_scene_cfg() -> InteractiveSceneCfg:
    tactile_cfgs = build_shadow_hand_grid_tactile_sensor_cfgs(
        hand_root_prim_path_expr="{ENV_REGEX_NS}/Robot",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
        include_mount_links=bool(args_cli.include_mount_links),
        pad_normal_axis=int(args_cli.pad_normal_axis),
        pad_normal_sign=int(args_cli.pad_normal_sign),
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        track_friction=True,
        track_pose=False,
    )

    annotations: dict[str, type] = {
        "dome_light": AssetBaseCfg,
        "robot": ArticulationCfg,
        "object": type(_OBJECT_CFG),
    }
    for key in tactile_cfgs:
        annotations[key] = GridTactileSensorCfg

    class_body: dict = {
        "__annotations__": annotations,
        "dome_light": AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        ),
        "robot": _robot_cfg_inhand_with_contact(),
        "object": _OBJECT_CFG,
    }
    class_body.update(tactile_cfgs)

    SceneCfg = configclass(type("ShadowHandGridTactileSceneCfg", (InteractiveSceneCfg,), class_body))
    return SceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5)


def _apply_inhand_initial_robot_state(robot) -> torch.Tensor:
    dof_pos = robot.data.default_joint_pos.clone()
    dof_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(dof_pos, dof_vel)
    robot.set_joint_position_target(dof_pos)
    return dof_pos


def _print_diag(scene: InteractiveScene, sensor_keys: list[str]) -> None:
    print(f"[DIAG] grid tactile sensor count: {len(sensor_keys)}")
    for key in sensor_keys:
        ts = scene[key]
        bnames = [str(n) for n in ts.body_names]
        print(
            f"  - {key}: prim_path={ts.cfg.prim_path!r}, bodies={len(bnames)}, "
            f"grid={tuple(ts.cfg.grid_resolution)}, patch={tuple(ts.cfg.patch_extent)}"
        )
        if bnames:
            print(f"      body_names={bnames}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, sensor_keys: list[str]) -> None:
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    targets = robot.data.default_joint_pos.clone()

    step = 0
    palm_key = "shadow_grid_tactile_palm"
    report_key = palm_key if palm_key in sensor_keys else sensor_keys[0]
    while simulation_app.is_running():
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1

        if step % 30 == 0:
            per_link_peak: list[float] = []
            per_link_fric_peak: list[float] = []
            for key in sensor_keys:
                d = scene[key].data
                fg = d.force_grid
                uv = d.friction_grid_uv
                if fg is None:
                    continue
                g = fg[0, 0, 0]
                per_link_peak.append(float(g.abs().max().item()))
                if uv is not None:
                    u0 = uv[0, 0, 0, 0]
                    v0 = uv[0, 0, 0, 1]
                    mag = torch.sqrt(torch.clamp(u0 * u0 + v0 * v0, min=0.0))
                    per_link_fric_peak.append(float(mag.max().item()))

            rd = scene[report_key].data
            rg = rd.force_grid
            ru = rd.friction_grid_uv
            rpeak = float(rg[0, 0, 0].abs().max().item()) if rg is not None else 0.0
            rfric = 0.0
            if ru is not None:
                u0 = ru[0, 0, 0, 0]
                v0 = ru[0, 0, 0, 1]
                rfric = float(torch.sqrt(torch.clamp(u0 * u0 + v0 * v0, min=0.0)).max().item())
            all_peak = max(per_link_peak) if per_link_peak else 0.0
            all_fric = max(per_link_fric_peak) if per_link_fric_peak else 0.0
            print(
                f"[step {step:5d}] {report_key}: |fn|_max={rpeak:.4f}, |ft|_max={rfric:.4f}; "
                f"all_links(|fn|_max)={all_peak:.4f}, all_links(|ft|_max)={all_fric:.4f}"
            )


def main() -> None:
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
        use_fabric=True,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(enable_ccd=True, bounce_threshold_velocity=0.2),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.2, 1.0, 0.8), target=(0.6, 0.0, 0.35))

    scene_cfg = _build_scene_cfg()
    scene = InteractiveScene(scene_cfg)

    spawn_factory_table()
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
    sim.reset()

    robot = scene["robot"]
    warm_targets = _apply_inhand_initial_robot_state(robot)
    for _ in range(20):
        robot.set_joint_position_target(warm_targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    keys = [f"shadow_grid_tactile_{ln}" for ln in shadow_hand_link_names(include_mount_links=bool(args_cli.include_mount_links))]
    print(f"[INFO] Shadow Hand grid tactile demo: {len(keys)} per-link sensors.")
    _print_diag(scene, keys)
    if args_cli.diag_only:
        print("[INFO] --diag_only: exiting after diagnostic.")
        return
    run_simulator(sim, scene, keys)


if __name__ == "__main__":
    main()
    simulation_app.close()
