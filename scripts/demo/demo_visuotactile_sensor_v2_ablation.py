# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ablation demo for VisuoTactileSensorV2 normal/slip-stick reconstruction.

Run from ViTacLab repo root (Isaac Lab python)::

    ./isaaclab.sh -p ../ViTacLab/scripts/demo/demo_visuotactile_sensor_v2_ablation.py \\
        --enable_cameras --num_envs 1 --ablation_mode baseline

    ./isaaclab.sh -p ../ViTacLab/scripts/demo/demo_visuotactile_sensor_v2_ablation.py \\
        --enable_cameras --num_envs 1 --ablation_mode normal_only

    ./isaaclab.sh -p ../ViTacLab/scripts/demo/demo_visuotactile_sensor_v2_ablation.py \\
        --enable_cameras --num_envs 1 --ablation_mode full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VisuoTactileSensorV2 ablation demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--ablation_mode",
    type=str,
    choices=("baseline", "normal_only", "full"),
    default="full",
    help="baseline: no correction/reconstruction; normal_only: only normal correction; full: both enabled.",
)
parser.add_argument("--seed", type=int, default=42, help="Torch seed for reproducibility.")
parser.add_argument("--max_steps", type=int, default=3000, help="Stop after N steps (0 = run until app closes).")
parser.add_argument("--log_interval", type=int, default=60, help="Print metrics every N steps (0 = disable).")
parser.add_argument(
    "--enable_corrected_force_render",
    action="store_true",
    help="Enable Stage-C render correction using corrected normal force.",
)
parser.add_argument(
    "--corrected_force_render_blend",
    type=float,
    default=1.0,
    help="Blend for Stage-C render correction: (1-a)*depth + a*force.",
)
parser.add_argument(
    "--summary_json",
    type=str,
    default="",
    help="Optional path to dump aggregated summary JSON.",
)
parser.add_argument(
    "--nut_torque_z",
    type=float,
    default=10.0,
    help="Alternating Z torque on nut after step 20 (0 = no external torque).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.sensors import GELSIGHT_R15_CFG

from ViTacLab.assets.sensor import VisuoTactileSensorV2Cfg

_TACTILE_ARRAY = (20, 25)
_TACTILE_MARGIN = 0.005
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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.45), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}),
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


def _mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "baseline":
        return False, False
    if mode == "normal_only":
        return True, False
    return True, True


def _make_v2_sensor_cfg(mode: str) -> VisuoTactileSensorV2Cfg:
    use_normal, use_slip_stick = _mode_flags(mode)
    return VisuoTactileSensorV2Cfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=_TACTILE_ARRAY,
        tactile_margin=_TACTILE_MARGIN,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        depth_penetration_deadband=0.0,
        normal_contact_stiffness=1.0,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
        enable_normal_correction=use_normal,
        enable_slip_stick_reconstruction=use_slip_stick,
        enable_corrected_force_render=bool(args_cli.enable_corrected_force_render),
        corrected_force_render_blend=float(args_cli.corrected_force_render_blend),
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def _format_tacsl_paths(sensor_cfg: VisuoTactileSensorV2Cfg, scene: InteractiveScene) -> None:
    ns = scene.env_regex_ns
    sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.camera_cfg is not None:
        sensor_cfg.camera_cfg.prim_path = sensor_cfg.camera_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.contact_object_prim_path_expr is not None:
        sensor_cfg.contact_object_prim_path_expr = sensor_cfg.contact_object_prim_path_expr.format(ENV_REGEX_NS=ns)


def _register_v2_sensor(scene: InteractiveScene, mode: str) -> None:
    scfg = _make_v2_sensor_cfg(mode)
    _format_tacsl_paths(scfg, scene)
    scene.sensors["tactile_sensor"] = scfg.class_type(scfg)


def _write_nut_default_pose(scene: InteractiveScene) -> None:
    rigid = scene.rigid_objects["contact_object"]
    state = rigid.data.default_root_state.clone()
    state[:, 0:3] += scene.env_origins
    rigid.write_root_pose_to_sim(state[:, :7], env_ids=None)
    rigid.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]), env_ids=None)


def _settle_and_baseline(scene: InteractiveScene, sim: sim_utils.SimulationContext) -> None:
    sim_dt = sim.get_physics_dt()
    scene["robot"].reset()
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
    for _ in range(max(1, _SETTLE_STEPS)):
        _write_nut_default_pose(scene)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    ts.get_initial_render()


def _compute_metrics(td, mu: float) -> dict[str, float]:
    nf = getattr(td, "tactile_normal_force", None)
    sf = getattr(td, "tactile_shear_force", None)
    pen = getattr(td, "penetration_depth", None)
    if nf is None or sf is None:
        return {}
    fn_abs = nf.abs()
    ft_mag = torch.norm(sf, dim=-1)
    # Coulomb violation: |ft| > mu * fn
    rhs = mu * torch.clamp(nf, min=0.0)
    viol = ft_mag - rhs
    viol_mask = viol > 1e-6
    valid_contact = torch.clamp(nf, min=0.0) > 1e-8
    denom = max(int(valid_contact.sum().item()), 1)
    viol_rate = float((viol_mask & valid_contact).sum().item()) / float(denom)
    viol_peak = float(torch.clamp(viol, min=0.0).max().item())
    out = {
        "fn_peak": float(fn_abs.max().item()),
        "fn_mean": float(fn_abs.mean().item()),
        "ft_peak": float(ft_mag.max().item()),
        "ft_mean": float(ft_mag.mean().item()),
        "viol_rate": viol_rate,
        "viol_peak": viol_peak,
    }
    out["pen_peak"] = float(pen.abs().max().item()) if pen is not None else 0.0
    rgb = getattr(td, "tactile_rgb_image", None)
    rgb_corr = getattr(td, "tactile_rgb_image_corrected", None)
    h_corr = getattr(td, "tactile_height_map_corrected", None)
    if rgb is not None and rgb_corr is not None:
        d_rgb = (rgb.to(torch.float32) - rgb_corr.to(torch.float32)).abs()
        out["rgb_corr_l1"] = float(d_rgb.mean().item())
        out["rgb_corr_peak"] = float(d_rgb.max().item())
    else:
        out["rgb_corr_l1"] = 0.0
        out["rgb_corr_peak"] = 0.0
    if h_corr is not None:
        out["height_corr_peak"] = float(torch.abs(h_corr).max().item())
    else:
        out["height_corr_peak"] = 0.0
    return out


def _tri_modal_shape_str(td) -> str:
    tri = getattr(td, "tri_modal", None)
    if not isinstance(tri, dict):
        return "tri_modal=None"
    parts = []
    for key in (
        "vision_rgb",
        "vision_rgb_corrected",
        "vision_depth",
        "vision_height_corrected",
        "force_normal",
        "force_shear",
        "contact_mask",
        "slip_mask",
    ):
        val = tri.get(key)
        shape = tuple(val.shape) if hasattr(val, "shape") else None
        parts.append(f"{key}:{shape}")
    po = getattr(td, "policy_obs", None)
    if isinstance(po, dict):
        for key in ("force_token", "force_flat", "depth_flat", "rgb_flat"):
            val = po.get(key)
            shape = tuple(val.shape) if hasattr(val, "shape") else None
            parts.append(f"{key}:{shape}")
    return " ".join(parts)


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if len(rows) == 0:
        return {}
    keys = sorted(rows[0].keys())
    out: dict[str, float] = {"n_logs": float(len(rows))}
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
        out[f"{key}_min"] = float(vals.min())
        out[f"{key}_max"] = float(vals.max())
    return out


@configclass
class VisuoTactileV2AblationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = _make_robot_cfg(_finger_usd())
    contact_object = _make_nut_cfg()


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    usd = Path(_finger_usd())
    if not usd.is_file():
        print(f"[ERROR] Finger USD not found: {usd}", file=sys.stderr)
        simulation_app.close()
        raise SystemExit(1)

    use_normal, use_slip_stick = _mode_flags(args_cli.ablation_mode)
    print(f"[INFO] Ablation mode: {args_cli.ablation_mode}")
    print(f"[INFO] normal_correction={use_normal}, slip_stick_reconstruction={use_slip_stick}")
    print(
        f"[INFO] corrected_force_render={bool(args_cli.enable_corrected_force_render)}, "
        f"blend={float(args_cli.corrected_force_render_blend):.3f}"
    )
    print(f"[INFO] Finger USD: {usd}")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.6, 1.0], target=[-0.1, 0.1, 0.5])

    scene = InteractiveScene(VisuoTactileV2AblationSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2))
    _register_v2_sensor(scene, args_cli.ablation_mode)
    sim.reset()
    _settle_and_baseline(scene, sim)

    ts_cfg = scene["tactile_sensor"].cfg
    mu = float(ts_cfg.friction_coefficient)
    sim_dt = sim.get_physics_dt()
    step = 0
    force = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    nut = scene["contact_object"]
    print("[INFO] Running ablation rollout...")
    tri_shape_logged = False
    metric_rows: list[dict[str, float]] = []

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

        if args_cli.log_interval > 0 and step % args_cli.log_interval == 0:
            ts = scene["tactile_sensor"]
            m = _compute_metrics(ts.data, mu=mu)
            if m:
                sparse_ok = int(getattr(ts, "_sparse_anchor_backend_ok", False))
                sparse_cnt = int(getattr(ts, "_last_sparse_anchor_count", 0))
                sparse_contact_cnt = int(getattr(ts, "_last_sparse_contact_count", 0))
                sparse_friction_cnt = int(getattr(ts, "_last_sparse_friction_count", 0))
                sparse_used = int(getattr(ts, "_last_sparse_used", False))
                sparse_pattern = str(getattr(ts, "_sparse_anchor_sensor_pattern", "None"))
                row = {
                    "step": float(step),
                    "fn_peak": float(m["fn_peak"]),
                    "fn_mean": float(m["fn_mean"]),
                    "ft_peak": float(m["ft_peak"]),
                    "ft_mean": float(m["ft_mean"]),
                    "viol_rate": float(m["viol_rate"]),
                    "viol_peak": float(m["viol_peak"]),
                    "pen_peak": float(m["pen_peak"]),
                    "rgb_corr_l1": float(m["rgb_corr_l1"]),
                    "rgb_corr_peak": float(m["rgb_corr_peak"]),
                    "height_corr_peak": float(m["height_corr_peak"]),
                    "sparse_ok": float(sparse_ok),
                    "sparse_cnt": float(sparse_cnt),
                    "contact_cnt": float(sparse_contact_cnt),
                    "friction_cnt": float(sparse_friction_cnt),
                    "sparse_used": float(sparse_used),
                }
                metric_rows.append(row)
                if not tri_shape_logged:
                    print(f"[INFO] Tri-modal shapes: {_tri_modal_shape_str(ts.data)}")
                    tri_shape_logged = True
                print(
                    f"[step {step:5d}] fn_peak={row['fn_peak']:.6f} fn_mean={row['fn_mean']:.6f} "
                    f"ft_peak={row['ft_peak']:.6f} ft_mean={row['ft_mean']:.6f} "
                    f"viol_rate={row['viol_rate']:.4f} viol_peak={row['viol_peak']:.6f} "
                    f"pen_peak={row['pen_peak']:.6f} rgb_corr_l1={row['rgb_corr_l1']:.6f} "
                    f"rgb_corr_peak={row['rgb_corr_peak']:.3f} height_corr_peak={row['height_corr_peak']:.6f} "
                    f"sparse_ok={sparse_ok} sparse_cnt={sparse_cnt} "
                    f"contact_cnt={sparse_contact_cnt} friction_cnt={sparse_friction_cnt} "
                    f"sparse_used={sparse_used} sparse_pattern={sparse_pattern}"
                )

    summary = _aggregate_metrics(metric_rows)
    if summary:
        print(
            "[SUMMARY] "
            f"n_logs={int(summary['n_logs'])} "
            f"fn_peak_mean={summary['fn_peak_mean']:.6f} "
            f"ft_peak_mean={summary['ft_peak_mean']:.6f} "
            f"viol_rate_mean={summary['viol_rate_mean']:.6f} "
            f"rgb_corr_l1_mean={summary['rgb_corr_l1_mean']:.6f} "
            f"sparse_used_mean={summary['sparse_used_mean']:.3f}"
        )
    if args_cli.summary_json:
        out_path = Path(args_cli.summary_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ablation_mode": args_cli.ablation_mode,
            "enable_corrected_force_render": bool(args_cli.enable_corrected_force_render),
            "corrected_force_render_blend": float(args_cli.corrected_force_render_blend),
            "seed": int(args_cli.seed),
            "max_steps": int(args_cli.max_steps),
            "log_interval": int(args_cli.log_interval),
            "summary": summary,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Summary JSON saved: {out_path}")

    print("[INFO] Ablation demo finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()
