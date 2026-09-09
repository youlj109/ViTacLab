#!/usr/bin/env python3
"""Evaluate alignment between PhysX contacts and ViTac tactile outputs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PhysX-vs-ViTac alignment evaluation.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps_per_case", type=int, default=220)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--physx_contact_force_threshold", type=float, default=1e-3)
parser.add_argument("--slip_force_threshold", type=float, default=0.5)
parser.add_argument("--contact_ratio_threshold", type=float, default=0.01)
parser.add_argument("--slip_ratio_threshold", type=float, default=0.01)
parser.add_argument("--sensor_contact_force_threshold", type=float, default=1e-6)
parser.add_argument("--sensor_slip_force_threshold", type=float, default=1e-5)
parser.add_argument("--min_contact_points", type=int, default=1)
parser.add_argument("--max_corr_lag", type=int, default=5)
parser.add_argument("--corr_smooth_window", type=int, default=5)
parser.add_argument(
    "--raw_depth_contact_threshold",
    type=float,
    default=1e-5,
    help="Depth-delta threshold (m) to mark any-object contact from tactile camera.",
)
parser.add_argument(
    "--raw_depth_active_ratio_threshold",
    type=float,
    default=0.005,
    help="Per-frame active-pixel ratio threshold for raw depth contact.",
)
parser.add_argument("--summary_json", type=str, default="")
parser.add_argument("--force_exit", action="store_true")
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


@dataclass(frozen=True)
class CaseSpec:
    name: str
    pose_xy_offset: tuple[float, float] = (0.0, 0.0)
    pin_no_contact: bool = False
    torque_mode: str = "none"
    torque_z: float = 10.0
    target_contact: bool = True
    interference_contact: bool = False


CASES = (
    CaseSpec("no_contact", pin_no_contact=True, target_contact=False, interference_contact=False),
    CaseSpec("normal_press_center"),
    CaseSpec("normal_press_edge", pose_xy_offset=(0.006, 0.0)),
    CaseSpec("shear_slide_constant", torque_mode="constant", torque_z=12.0),
    CaseSpec("stick_then_slip", torque_mode="stick_then_slip", torque_z=14.0),
    CaseSpec("interference_only", torque_mode="constant", torque_z=12.0, target_contact=False, interference_contact=True),
)

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


def _make_interference_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/interference_object",
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
        depth_penetration_deadband=0.0,
        normal_contact_stiffness=1.0,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
        enable_normal_correction=True,
        enable_slip_stick_reconstruction=True,
        use_physx_sparse_anchors=True,
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


def _register_v2_sensor(scene: InteractiveScene) -> None:
    scfg = _make_v2_sensor_cfg()
    _format_tacsl_paths(scfg, scene)
    scene.sensors["tactile_sensor"] = scfg.class_type(scfg)


def _write_object_pose(
    scene: InteractiveScene,
    object_name: str,
    xy_offset: tuple[float, float] = (0.0, 0.0),
    z_offset: float = 0.0,
) -> None:
    rigid = scene.rigid_objects[object_name]
    state = rigid.data.default_root_state.clone()
    state[:, 0] += float(xy_offset[0])
    state[:, 1] += float(xy_offset[1])
    state[:, 2] += float(z_offset)
    state[:, 0:3] += scene.env_origins
    rigid.write_root_pose_to_sim(state[:, :7], env_ids=None)
    rigid.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]), env_ids=None)


def _reset_robot_and_scene(scene: InteractiveScene) -> None:
    scene["robot"].reset()
    scene.rigid_objects["contact_object"].reset()
    scene.rigid_objects["interference_object"].reset()
    robot = scene["robot"]
    state = robot.data.default_root_state.clone()
    state[:, 0:3] += scene.env_origins
    robot.write_root_pose_to_sim(state[:, :7], env_ids=None)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=None)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone(), env_ids=None)
    scene.write_data_to_sim()


def _settle_and_baseline(scene: InteractiveScene, sim: sim_utils.SimulationContext) -> None:
    sim_dt = sim.get_physics_dt()
    _reset_robot_and_scene(scene)
    _write_object_pose(scene, "contact_object", (0.0, 0.0))
    _write_object_pose(scene, "interference_object", (0.0, 0.0), z_offset=0.35)
    ts = scene["tactile_sensor"]
    for _ in range(max(1, _SETTLE_STEPS)):
        _write_object_pose(scene, "contact_object", (0.0, 0.0))
        _write_object_pose(scene, "interference_object", (0.0, 0.0), z_offset=0.35)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    ts.get_initial_render()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones((window,), dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="valid")


def _lagged_corr(a: list[float], b: list[float], *, max_lag: int, smooth_window: int) -> float:
    aa = _moving_avg(np.asarray(a, dtype=np.float64), smooth_window)
    bb = _moving_avg(np.asarray(b, dtype=np.float64), smooth_window)
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x = aa[lag:]
            y = bb[:-lag]
        elif lag < 0:
            x = aa[:lag]
            y = bb[-lag:]
        else:
            x = aa
            y = bb
        c = _safe_corr(x, y)
        if abs(c) > abs(best):
            best = c
    return float(best)


def _lagged_corr_masked(a: list[float], b: list[float], mask: list[bool], *, max_lag: int, smooth_window: int) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mm = np.asarray(mask, dtype=bool)
    if aa.size != bb.size or aa.size != mm.size:
        return 0.0
    if int(mm.sum()) < 3:
        return 0.0
    return _lagged_corr(aa[mm].tolist(), bb[mm].tolist(), max_lag=max_lag, smooth_window=smooth_window)


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = float(tp) / float(max(tp + fp, 1))
    r = float(tp) / float(max(tp + fn, 1))
    f1 = 0.0 if (p + r) <= 1e-12 else 2.0 * p * r / (p + r)
    return p, r, f1


def _extract_physx(ts, env_id: int = 0) -> dict[str, float | int | torch.Tensor]:
    view = getattr(ts, "_contact_physx_view", None)
    if view is None:
        return {
            "fn_total": 0.0,
            "ft_total": 0.0,
            "contact_count": 0,
            "friction_count": 0,
            "contact_points": torch.zeros((0, 3), dtype=torch.float32, device=ts._device),
        }
    num_filters = int(view.filter_count)
    forces, points, _normals, _seps, buffer_count, buffer_start_indices = view.get_contact_data(dt=ts._sim_physics_dt)
    friction_forces, _fp, buffer_count_f, buffer_start_indices_f = view.get_friction_data(dt=ts._sim_physics_dt)
    counts = buffer_count.view(ts._num_envs, num_filters)
    starts = buffer_start_indices.view(ts._num_envs, num_filters)
    counts_f = buffer_count_f.view(ts._num_envs, num_filters)
    starts_f = buffer_start_indices_f.view(ts._num_envs, num_filters)

    fn_total = 0.0
    ft_total = 0.0
    contact_count = 0
    friction_count = 0
    pts: list[torch.Tensor] = []
    for fi in range(num_filters):
        cnt = int(counts[env_id, fi].item())
        contact_count += cnt
        if cnt > 0:
            sl = slice(int(starts[env_id, fi].item()), int(starts[env_id, fi].item()) + cnt)
            fn_total += float(torch.clamp(forces[sl].reshape(-1), min=0.0).sum().item())
            pts.append(points[sl])
        cntf = int(counts_f[env_id, fi].item())
        friction_count += cntf
        if cntf > 0:
            slf = slice(int(starts_f[env_id, fi].item()), int(starts_f[env_id, fi].item()) + cntf)
            ft_total += float(torch.norm(friction_forces[slf], dim=-1).sum().item())
    cp = torch.cat(pts, dim=0) if len(pts) > 0 else torch.zeros((0, 3), device=ts._device, dtype=torch.float32)
    return {
        "fn_total": fn_total,
        "ft_total": ft_total,
        "contact_count": contact_count,
        "friction_count": friction_count,
        "contact_points": cp,
    }


def _sensor_metrics(ts, env_id: int = 0) -> dict[str, float | torch.Tensor]:
    d = ts.data
    nf = torch.clamp(d.tactile_normal_force[env_id], min=0.0) if d.tactile_normal_force is not None else torch.zeros(1, device=ts._device)
    sf = d.tactile_shear_force[env_id] if d.tactile_shear_force is not None else torch.zeros((nf.shape[0], 2), device=ts._device)
    ft = torch.norm(sf, dim=-1)
    contact_mask = d.contact_mask[env_id].bool() if d.contact_mask is not None else (nf > 1e-8)
    slip_mask = d.slip_mask[env_id].bool() if d.slip_mask is not None else (ft > 1e-8)
    return {
        "fn_total": float(nf.sum().item()),
        "ft_total": float(ft.sum().item()),
        "contact_ratio": float(contact_mask.float().mean().item()),
        "slip_ratio": float(slip_mask.float().mean().item()),
        "contact_count": float(contact_mask.sum().item()),
        "contact_mask": contact_mask,
        "nf": nf,
    }


def _spatial_metrics(ts, physx_points_w: torch.Tensor, contact_mask: torch.Tensor, nf: torch.Tensor, env_id: int = 0) -> tuple[float, float]:
    tp_w = ts.data.tactile_points_pos_w
    if tp_w is None:
        return 0.0, 0.0
    tactile_points_w = tp_w[env_id]
    active = tactile_points_w[contact_mask]
    if active.numel() <= 0 or physx_points_w.numel() <= 0:
        return 0.0, 0.0
    nn = torch.cdist(physx_points_w.unsqueeze(0), active.unsqueeze(0)).squeeze(0)
    nn_err = float(torch.min(nn, dim=-1).values.mean().item())
    w = torch.clamp(nf, min=0.0)
    if float(w.sum().item()) > 1e-12:
        tc = (tactile_points_w * w.unsqueeze(-1)).sum(dim=0) / w.sum()
    else:
        tc = active.mean(dim=0)
    pc = physx_points_w.mean(dim=0)
    c_err = float(torch.norm(tc - pc).item())
    return c_err, nn_err


def _case_pass(case: str, s: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if case == "interference_only":
        if s["raw_depth_contact_rate"] < 0.20:
            reasons.append("interference_contact_not_observed")
        if s["target_false_activation_rate"] > 0.10:
            reasons.append("target_false_activation_rate>0.10")
        return len(reasons) == 0, reasons
    if case == "no_contact":
        if s["contact_fpr"] > 0.05:
            reasons.append("contact_fpr>0.05")
    else:
        if s["contact_recall"] < 0.70:
            reasons.append("contact_recall<0.70")
        if s["fn_corr"] < 0.30:
            reasons.append("fn_corr<0.30")
        if s["centroid_error_mean"] > 0.03:
            reasons.append("centroid_error_mean>3cm")
        if case in ("shear_slide_constant", "stick_then_slip"):
            if s["slip_f1"] < 0.50:
                reasons.append("slip_f1<0.50")
            if s["ft_corr"] < 0.20 and s["ft_ratio_mean"] < 0.05:
                reasons.append("ft_corr<0.20_and_ft_ratio<0.05")
    return len(reasons) == 0, reasons


@configclass
class AlignmentSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot = _make_robot_cfg(_finger_usd())
    contact_object = _make_nut_cfg()
    interference_object = _make_interference_cfg()


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=[0.5, 0.6, 1.0], target=[-0.1, 0.1, 0.5])
    scene = InteractiveScene(AlignmentSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2))
    _register_v2_sensor(scene)
    sim.reset()
    _settle_and_baseline(scene, sim)
    sim_dt = sim.get_physics_dt()
    ts = scene["tactile_sensor"]
    target_obj = scene["contact_object"]
    interference_obj = scene["interference_object"]
    force = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque = torch.zeros(scene.num_envs, 1, 3, device=sim.device)

    case_rows: dict[str, dict[str, Any]] = {}
    case_pass: dict[str, bool] = {}
    print("[INFO] Starting PhysX alignment cases...")

    for case in CASES:
        print(f"[INFO] Case start: {case.name}")
        _reset_robot_and_scene(scene)
        if case.target_contact:
            _write_object_pose(scene, "contact_object", case.pose_xy_offset)
        else:
            _write_object_pose(scene, "contact_object", (0.0, 0.0), z_offset=0.35)
        if case.interference_contact:
            _write_object_pose(scene, "interference_object", case.pose_xy_offset)
        else:
            _write_object_pose(scene, "interference_object", (0.0, 0.0), z_offset=0.35)
        scene.write_data_to_sim()

        fn_p, fn_s, ft_p, ft_s = [], [], [], []
        slip_mask_s: list[bool] = []
        cen_e, nn_e, sparse_u = [], [], []
        raw_depth_contact_steps = 0
        c_tp = c_fp = c_fn = 0
        s_tp = s_fp = s_fn = 0
        neg_steps = neg_false = 0
        target_only_neg_steps = 0
        target_only_false_steps = 0

        for step in range(int(args_cli.steps_per_case)):
            torque.zero_()
            if case.pin_no_contact:
                _write_object_pose(scene, "contact_object", (0.0, 0.0), z_offset=0.35)
                _write_object_pose(scene, "interference_object", (0.0, 0.0), z_offset=0.35)
            elif case.torque_mode == "constant":
                torque[:, 0, 2] = float(case.torque_z)
                if case.interference_contact and (not case.target_contact):
                    interference_obj.permanent_wrench_composer.set_forces_and_torques(force, torque)
                else:
                    target_obj.permanent_wrench_composer.set_forces_and_torques(force, torque)
            elif case.torque_mode == "stick_then_slip" and step > int(args_cli.steps_per_case) // 2:
                ramp_mid = int(args_cli.steps_per_case) // 2 + max(int(args_cli.steps_per_case) // 6, 1)
                torque[:, 0, 2] = 0.7 * float(case.torque_z) if step < ramp_mid else float(case.torque_z)
                if case.interference_contact and (not case.target_contact):
                    interference_obj.permanent_wrench_composer.set_forces_and_torques(force, torque)
                else:
                    target_obj.permanent_wrench_composer.set_forces_and_torques(force, torque)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            p = _extract_physx(ts, 0)
            s = _sensor_metrics(ts, 0)
            physx_contact = (
                int(p["contact_count"]) > 0
                and (
                    float(p["fn_total"]) > float(args_cli.physx_contact_force_threshold)
                    or float(p["ft_total"]) > 0.25 * float(args_cli.slip_force_threshold)
                )
            )
            sensor_contact = (
                float(s["contact_ratio"]) > float(args_cli.contact_ratio_threshold)
                or float(s["fn_total"]) > float(args_cli.sensor_contact_force_threshold)
                or float(s["contact_count"]) >= float(max(int(args_cli.min_contact_points), 1))
            )
            if physx_contact and sensor_contact:
                c_tp += 1
            elif (not physx_contact) and sensor_contact:
                c_fp += 1
            elif physx_contact and (not sensor_contact):
                c_fn += 1
            if not physx_contact:
                neg_steps += 1
                if sensor_contact:
                    neg_false += 1
            if not case.target_contact:
                target_only_neg_steps += 1
                if sensor_contact:
                    target_only_false_steps += 1

            physx_slip = float(p["ft_total"]) > float(args_cli.slip_force_threshold)
            sensor_slip = (
                float(s["slip_ratio"]) > float(args_cli.slip_ratio_threshold)
                or float(s["ft_total"]) > float(args_cli.sensor_slip_force_threshold)
            )
            if physx_slip and sensor_slip:
                s_tp += 1
            elif (not physx_slip) and sensor_slip:
                s_fp += 1
            elif physx_slip and (not sensor_slip):
                s_fn += 1

            fn_p.append(float(p["fn_total"]))
            fn_s.append(float(s["fn_total"]))
            ft_p.append(float(p["ft_total"]))
            ft_s.append(float(s["ft_total"]))
            slip_mask_s.append(bool(physx_slip))
            ce, ne = _spatial_metrics(
                ts,
                cast(torch.Tensor, p["contact_points"]),
                cast(torch.Tensor, s["contact_mask"]),
                cast(torch.Tensor, s["nf"]),
                0,
            )
            cen_e.append(ce)
            nn_e.append(ne)
            sparse_u.append(1.0 if bool(getattr(ts, "_last_sparse_used", False)) else 0.0)

            if int(args_cli.log_interval) > 0 and (step + 1) % int(args_cli.log_interval) == 0:
                print(
                    f"[{case.name} step {step+1:04d}] fn_physx={fn_p[-1]:.3f} fn_sensor={fn_s[-1]:.3f} "
                    f"ft_physx={ft_p[-1]:.3f} ft_sensor={ft_s[-1]:.3f} contact_cnt={int(p['contact_count'])}"
                )

            # Camera raw depth contact (any-object): used to verify interference case truly contacts pad.
            raw_depth = ts.data.tactile_depth_image
            nominal = getattr(ts, "_nominal_tactile", None)
            if raw_depth is not None and isinstance(nominal, dict):
                depth_key = "distance_to_image_plane" if "distance_to_image_plane" in nominal else ("depth" if "depth" in nominal else None)
                if depth_key is not None and depth_key in nominal:
                    raw_delta = torch.clamp(nominal[depth_key][0].squeeze(-1) - raw_depth[0].squeeze(-1), min=0.0)
                    raw_active_ratio = float((raw_delta > float(args_cli.raw_depth_contact_threshold)).float().mean().item())
                    if raw_active_ratio > float(args_cli.raw_depth_active_ratio_threshold):
                        raw_depth_contact_steps += 1

        cp, cr, cf1 = _f1(c_tp, c_fp, c_fn)
        sp, sr, sf1 = _f1(s_tp, s_fp, s_fn)
        vf = [i for i, x in enumerate(fn_p) if x > 1e-8]
        vt = [i for i, x in enumerate(ft_p) if x > 1e-8]
        fn_ratio = float(np.mean([fn_s[i] / max(fn_p[i], 1e-8) for i in vf])) if vf else 0.0
        ft_ratio = float(np.mean([ft_s[i] / max(ft_p[i], 1e-8) for i in vt])) if vt else 0.0
        row: dict[str, Any] = {
            "contact_precision": cp,
            "contact_recall": cr,
            "contact_f1": cf1,
            "contact_fpr": float(neg_false) / float(max(neg_steps, 1)),
            "target_false_activation_rate": float(target_only_false_steps) / float(max(target_only_neg_steps, 1)),
            "raw_depth_contact_rate": float(raw_depth_contact_steps) / float(max(int(args_cli.steps_per_case), 1)),
            "slip_precision": sp,
            "slip_recall": sr,
            "slip_f1": sf1,
            "fn_corr": _lagged_corr(fn_p, fn_s, max_lag=max(int(args_cli.max_corr_lag), 0), smooth_window=max(int(args_cli.corr_smooth_window), 1)),
            "ft_corr": _lagged_corr_masked(
                ft_p,
                ft_s,
                slip_mask_s,
                max_lag=max(int(args_cli.max_corr_lag), 0),
                smooth_window=max(int(args_cli.corr_smooth_window), 1),
            ),
            "fn_ratio_mean": fn_ratio,
            "ft_ratio_mean": ft_ratio,
            "physx_fn_mean": float(np.mean(fn_p)),
            "sensor_fn_mean": float(np.mean(fn_s)),
            "physx_ft_mean": float(np.mean(ft_p)),
            "sensor_ft_mean": float(np.mean(ft_s)),
            "centroid_error_mean": float(np.mean(cen_e)),
            "nn_error_mean": float(np.mean(nn_e)),
            "sparse_used_mean": float(np.mean(sparse_u)) if sparse_u else 0.0,
        }
        ok, reasons = _case_pass(case.name, row)  # type: ignore[arg-type]
        row["pass"] = 1.0 if ok else 0.0
        row["fail_reason_count"] = float(len(reasons))
        row["fail_reasons"] = reasons
        case_rows[case.name] = row
        case_pass[case.name] = ok
        row_contact_f1 = float(row["contact_f1"])
        row_fn_corr = float(row["fn_corr"])
        row_slip_f1 = float(row["slip_f1"])
        row_ft_corr = float(row["ft_corr"])
        row_centroid_err = float(row["centroid_error_mean"])
        print(
            f"[CASE {case.name}] pass={ok} contact_f1={row_contact_f1:.3f} "
            f"fn_corr={row_fn_corr:.3f} slip_f1={row_slip_f1:.3f} "
            f"ft_corr={row_ft_corr:.3f} centroid_err={row_centroid_err:.4f} "
            f"reasons={';'.join(reasons) if reasons else 'ok'}"
        )

    overall = {
        "case_count": int(len(case_rows)),
        "pass_case_count": int(sum(1 for v in case_pass.values() if v)),
        "overall_pass": bool(all(case_pass.values()) if case_pass else False),
    }
    print("[SUMMARY] " + json.dumps({"overall": overall, "cases": case_rows}, ensure_ascii=False))

    if args_cli.summary_json:
        out = Path(args_cli.summary_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": int(args_cli.seed),
            "steps_per_case": int(args_cli.steps_per_case),
            "physx_contact_force_threshold": float(args_cli.physx_contact_force_threshold),
            "slip_force_threshold": float(args_cli.slip_force_threshold),
            "contact_ratio_threshold": float(args_cli.contact_ratio_threshold),
            "slip_ratio_threshold": float(args_cli.slip_ratio_threshold),
            "overall": overall,
            "cases": case_rows,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Summary JSON saved: {out}")

    return 0 if bool(overall["overall_pass"]) else 2


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = int(main())
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
        if bool(getattr(args_cli, "force_exit", False)):
            os._exit(exit_code)
    raise SystemExit(exit_code)
