#!/usr/bin/env python3
"""Lateral (shear) force ViTacSim validation — advisor fallback path.

Horizontal GelSight + W100 on pad (same as NF), then constant world-frame lateral
push on the weight (no gripper, no force control). Measures **tangential / shear**
tactile response while normal load stays ~ self-weight.

Usage::

    ../IsaacLab/isaaclab.sh -p scripts/demo/demo_vitacsim_lateral_force_validation.py \\
        --headless --enable_cameras --device cuda:0 \\
        --lateral-force-x 0.3 --sensor-mode vitacsim
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

_OUTPUT_SCHEMA = "sf_lateral_v2"

parser = argparse.ArgumentParser(description="ViTacSim lateral-force validation (horizontal pad, W100, no gripper).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--weight-id",
    type=str,
    default="W100",
    choices=("W200", "W100", "W050", "W020", "W010"),
    help="Default W100 per advisor fallback; other masses optional.",
)
parser.add_argument("--sensor-mode", type=str, choices=("tacsl", "vitacsim"), default="vitacsim")
parser.add_argument("--settle-steps", type=int, default=180)
parser.add_argument("--push-steps", type=int, default=60, help="Steps with lateral force before recording.")
parser.add_argument("--record-steps", type=int, default=40)
parser.add_argument("--weight-rest-z", type=float, default=0.442)
parser.add_argument("--weight-clearance-z", type=float, default=0.520)
parser.add_argument("--weight-spawn-z", type=float, default=-1.0)
parser.add_argument("--weight-drop-offset", type=float, default=0.012)
parser.add_argument("--force-render-k-ref", type=float, default=0.0, help="<=0: auto per-weight from W100 ref.")
parser.add_argument(
    "--finger-root-z",
    type=float,
    default=-1.0,
    help="GelSight finger root z. <0: validation_beta_config.DEFAULT_FINGER_ROOT_Z.",
)
parser.add_argument(
    "--enforce-weight-xy-bounds",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Clamp weight xy during push so light masses stay on pad (friction-pretrain pattern).",
)
parser.add_argument(
    "--weight-xy-bounds-half",
    type=float,
    default=0.012,
    help="Half-extent (m) of virtual xy walls when --enforce-weight-xy-bounds.",
)
parser.add_argument(
    "--lateral-force-x",
    type=float,
    default=0.0,
    help="Constant world +X lateral force on weight (N) during push/record.",
)
parser.add_argument(
    "--lateral-force-y",
    type=float,
    default=0.0,
    help="Constant world +Y lateral force on weight (N) during push/record.",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="logs/vitacsim_validation/shear_force/lateral",
    help="Output root; writes {out_dir}/Fx{..}_Fy{..}/{weight_id}/{sensor_mode}/",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--no-marker",
    action="store_true",
    help="Disable FOTS marker overlay on Taxim RGB (default: GelSight markers enabled).",
)
parser.add_argument(
    "--marker-pattern",
    type=str,
    default="gelsight",
    choices=("gelsight", "xense"),
    help="Marker layout when marker overlay is enabled.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ViTacLab.assets.sensor import VisuoTactileSensorV2Cfg
from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
from ViTacLab.tasks.direct.pretraining.gelsight_finger_pretrain_base_cfg import (
    GELSIGHT_FINGER_SHORT_USD,
    build_gelsight_finger_robot_cfg,
)
from ViTacLab.tasks.direct.vitacsim_validation.validation_weight_spawner_cfg import validation_weight_spawner_cfg
from ViTacLab.tasks.direct.vitacsim_validation.weight_spec import WEIGHT_MASS_KG
from ViTacLab.tasks.direct.vitacsim_validation.validation_beta_config import (
    DEFAULT_FINGER_ROOT_Z,
    contact_valid,
    fx_list_for_weight,
    resolve_force_render_k_ref,
)
from ViTacLab.tasks.direct.vitacsim_validation.validation_rgb_utils import (
    bg_diff_cfg_dict,
    load_bg_rgb,
    save_rgb_diff_bg,
)

_TACTILE_ARRAY = (20, 25)
_NOMINAL_WARMUP = 8


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def _mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "tacsl":
        return False, False
    return True, True


def _spawn_z() -> float:
    if float(args_cli.weight_spawn_z) >= 0.0:
        return float(args_cli.weight_spawn_z)
    return float(args_cli.weight_rest_z) + float(args_cli.weight_drop_offset)


def _force_tag(fx: float, fy: float) -> str:
    def _fmt(v: float) -> str:
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s.replace("-", "m")

    return f"Fx{_fmt(fx)}_Fy{_fmt(fy)}"


def _make_weight_cfg(weight_id: str) -> RigidObjectCfg:
    z0 = float(args_cli.weight_clearance_z)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=validation_weight_spawner_cfg(weight_id),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0)),
    )


def _finger_root_z() -> float:
    z = float(args_cli.finger_root_z)
    return DEFAULT_FINGER_ROOT_Z if z < 0.0 else z


def _force_render_k_ref() -> float:
    return resolve_force_render_k_ref(args_cli.weight_id, float(args_cli.force_render_k_ref))


def _marker_enabled() -> bool:
    return not bool(args_cli.no_marker)


def _render_cfg():
    return validation_gelsight_render_cfg(
        enable_marker=_marker_enabled(),
        marker_pattern=args_cli.marker_pattern,
    )


def _marker_stats(ts) -> dict[str, float | int | str]:
    disp = getattr(ts.data, "tactile_marker_displacement", None)
    if disp is None or disp.numel() == 0:
        return {
            "marker_pattern": "none" if not _marker_enabled() else args_cli.marker_pattern,
            "marker_count": 0,
            "marker_disp_max_px": 0.0,
            "marker_disp_mean_px": 0.0,
        }
    d = disp[0].detach().cpu()
    mag = torch.linalg.norm(d, dim=-1)
    return {
        "marker_pattern": args_cli.marker_pattern if _marker_enabled() else "none",
        "marker_count": int(d.shape[0]),
        "marker_disp_max_px": float(mag.max().item()),
        "marker_disp_mean_px": float(mag.mean().item()),
    }


def _make_sensor_cfg(mode: str) -> VisuoTactileSensorV2Cfg:
    use_normal, use_slip = _mode_flags(mode)
    render_cfg = _render_cfg()
    k_ref = 1.0e4 if mode == "tacsl" else _force_render_k_ref()
    return VisuoTactileSensorV2Cfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        render_cfg=render_cfg,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=_TACTILE_ARRAY,
        tactile_margin=0.005,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        contact_object_is_deformable=False,
        depth_penetration_deadband=0.0,
        normal_contact_stiffness=1.0,
        normal_correction_k_ref=k_ref,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
        enable_normal_correction=use_normal,
        enable_slip_stick_reconstruction=use_slip,
        enable_corrected_force_render=(mode == "vitacsim"),
        corrected_force_render_blend=1.0,
        marker_shear_from_force_field=(mode == "vitacsim"),
        marker_shear_force_gain=3.0,
        marker_shear_force_ref_n=0.05,
        require_physx_sparse_anchors=(mode == "vitacsim"),
        strict_target_contact_attribution=True,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            height=render_cfg.image_height,
            width=render_cfg.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def _format_paths(sensor_cfg: VisuoTactileSensorV2Cfg, scene: InteractiveScene) -> None:
    ns = scene.env_regex_ns
    sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.camera_cfg is not None:
        sensor_cfg.camera_cfg.prim_path = sensor_cfg.camera_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.contact_object_prim_path_expr is not None:
        sensor_cfg.contact_object_prim_path_expr = sensor_cfg.contact_object_prim_path_expr.format(ENV_REGEX_NS=ns)


@configclass
class LateralForceValidationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = build_gelsight_finger_robot_cfg().replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, _finger_root_z()),
            rot=(0.70711, -0.70711, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        )
    )
    contact_object: RigidObjectCfg = _make_weight_cfg(args_cli.weight_id)


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


def _apply_lateral_force(scene: InteractiveScene, fx: float, fy: float) -> None:
    obj = scene.rigid_objects["contact_object"]
    n = scene.num_envs
    forces = torch.zeros(n, 1, 3, device=obj.device)
    forces[:, 0, 0] = float(fx)
    forces[:, 0, 1] = float(fy)
    obj.set_external_force_and_torque(forces, torch.zeros_like(forces), env_ids=None, is_global=True)


def _clear_external_force(scene: InteractiveScene) -> None:
    _apply_lateral_force(scene, 0.0, 0.0)


def _weight_bottom_z(scene: InteractiveScene) -> float:
    obj = scene.rigid_objects["contact_object"]
    return float((obj.data.root_pos_w[0, 2] - scene.env_origins[0, 2]).item())


def _save_rgb(path: Path, rgb_u8: torch.Tensor) -> None:
    try:
        from PIL import Image
    except ImportError:
        np.save(path.with_suffix(".npy"), rgb_u8.detach().cpu().numpy())
        return
    arr = rgb_u8.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 3:
        Image.fromarray(arr.astype(np.uint8)).save(path)


def _enforce_weight_xy_bounds(scene: InteractiveScene) -> None:
    if not bool(args_cli.enforce_weight_xy_bounds):
        return
    obj = scene.rigid_objects["contact_object"]
    half = float(args_cli.weight_xy_bounds_half)
    restitution = 0.0
    pos_w = obj.data.root_pos_w
    quat_w = obj.data.root_quat_w
    d = obj.data
    lin_w = getattr(d, "root_lin_vel_w", d.root_link_lin_vel_w).clone()
    ang_w = d.root_ang_vel_w
    pos_env = pos_w - scene.env_origins
    px, py, pz = pos_env[:, 0], pos_env[:, 1], pos_env[:, 2]
    vx, vy = lin_w[:, 0], lin_w[:, 1]
    hit_x_hi, hit_x_lo = px > half, px < -half
    hit_y_hi, hit_y_lo = py > half, py < -half
    hit_any = hit_x_hi | hit_x_lo | hit_y_hi | hit_y_lo
    if not hit_any.any():
        return
    px_new = torch.clamp(px, -half, half)
    py_new = torch.clamp(py, -half, half)
    vx_new = torch.where(hit_x_hi & (vx > 0), -vx * restitution, vx)
    vx_new = torch.where(hit_x_lo & (vx < 0), -vx * restitution, vx_new)
    vy_new = torch.where(hit_y_hi & (vy > 0), -vy * restitution, vy)
    vy_new = torch.where(hit_y_lo & (vy < 0), -vy * restitution, vy_new)
    lin_w[:, 0], lin_w[:, 1] = vx_new, vy_new
    pos_env_new = torch.stack([px_new, py_new, pz], dim=-1)
    pos_w_new = pos_env_new + scene.env_origins
    root = torch.cat([pos_w_new, quat_w], dim=-1)
    obj.write_root_pose_to_sim(root, env_ids=None)
    obj.write_root_velocity_to_sim(torch.cat([lin_w, ang_w], dim=-1), env_ids=None)


def _extract_physx(ts, env_id: int = 0) -> dict[str, float | int]:
    view = getattr(ts, "_contact_physx_view", None)
    if view is None:
        return {"physx_fn_total": 0.0, "physx_ft_total": 0.0, "physx_contact_count": 0}

    num_filters = int(view.filter_count)
    forces, _points, _normals, _seps, buffer_count, buffer_start_indices = view.get_contact_data(
        dt=ts._sim_physics_dt
    )
    friction_forces, _fp, buffer_count_f, buffer_start_indices_f = view.get_friction_data(dt=ts._sim_physics_dt)
    counts = buffer_count.view(ts._num_envs, num_filters)
    starts = buffer_start_indices.view(ts._num_envs, num_filters)
    counts_f = buffer_count_f.view(ts._num_envs, num_filters)
    starts_f = buffer_start_indices_f.view(ts._num_envs, num_filters)

    fn_total = 0.0
    ft_total = 0.0
    contact_count = 0
    for fi in range(num_filters):
        cnt = int(counts[env_id, fi].item())
        contact_count += cnt
        if cnt > 0:
            st = int(starts[env_id, fi].item())
            sl = slice(st, st + cnt)
            fn_total += float(torch.clamp(forces[sl].reshape(-1), min=0.0).sum().item())
        cntf = int(counts_f[env_id, fi].item())
        if cntf > 0:
            stf = int(starts_f[env_id, fi].item())
            slf = slice(stf, stf + cntf)
            ft_total += float(torch.norm(friction_forces[slf], dim=-1).sum().item())

    return {
        "physx_fn_total": fn_total,
        "physx_ft_total": ft_total,
        "physx_contact_count": contact_count,
    }


def _shear_stats(sf: np.ndarray | None) -> dict[str, float]:
    if sf is None:
        return {"ft_peak_max": 0.0, "ft_peak_mean": 0.0, "ft_field_sum": 0.0}
    arr = np.nan_to_num(sf, nan=0.0)
    mag = np.linalg.norm(arr, axis=-1)
    return {
        "ft_peak_max": float(mag.max()),
        "ft_peak_mean": float(mag.mean()),
        "ft_field_sum": float(mag.sum()),
    }


def _record_step(ts, scene, sim, sim_dt, fx: float, fy: float) -> dict:
    _apply_lateral_force(scene, fx, fy)
    scene.write_data_to_sim()
    sim.step()
    _enforce_weight_xy_bounds(scene)
    scene.update(sim_dt)
    data = ts.data
    nf = getattr(data, "tactile_normal_force", None)
    sf = getattr(data, "tactile_shear_force", None)
    physx = _extract_physx(ts, env_id=0)
    rgb = getattr(data, "tactile_rgb_image", None)
    rgb_corr = getattr(data, "tactile_rgb_image_corrected", None)
    out = {
        "fn_peak": float(torch.nan_to_num(nf[0], nan=0.0).abs().max().item()) if nf is not None else 0.0,
        "ft_peak": float(torch.norm(torch.nan_to_num(sf[0], nan=0.0), dim=-1).max().item()) if sf is not None else 0.0,
        "physx_fn": float(physx["physx_fn_total"]),
        "physx_ft": float(physx["physx_ft_total"]),
        "physx_contact_count": int(physx["physx_contact_count"]),
        "rgb": rgb[0].detach().cpu() if rgb is not None else None,
        "rgb_corr": rgb_corr[0].detach().cpu() if rgb_corr is not None else None,
        "nf": nf[0].detach().cpu().numpy() if nf is not None else None,
        "sf": sf[0].detach().cpu().numpy() if sf is not None else None,
        "marker_disp": (
            getattr(data, "tactile_marker_displacement", None)[0].detach().cpu().numpy()
            if getattr(data, "tactile_marker_displacement", None) is not None
            else None
        ),
    }
    return out


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    finger_usd = Path(GELSIGHT_FINGER_SHORT_USD)
    if not finger_usd.is_file():
        finger_usd = _repo_root() / GELSIGHT_FINGER_SHORT_USD
    if not finger_usd.is_file():
        print(f"[ERROR] GelSight finger USD not found: {GELSIGHT_FINGER_SHORT_USD}", file=sys.stderr)
        return 1

    mass_kg = float(WEIGHT_MASS_KG[args_cli.weight_id])
    nominal_fn = mass_kg * 9.81
    fx = float(args_cli.lateral_force_x)
    fy = float(args_cli.lateral_force_y)
    allowed_fx = fx_list_for_weight(args_cli.weight_id)
    if fx not in allowed_fx and not any(abs(fx - v) < 1e-6 for v in allowed_fx):
        print(
            f"[WARN] Fx={fx} not in recommended sweep {allowed_fx} for {args_cli.weight_id}; "
            "light weights may lose contact.",
            file=sys.stderr,
        )
    use_normal, use_slip = _mode_flags(args_cli.sensor_mode)
    spawn_z = _spawn_z()

    print(f"[INFO] lateral validation weight={args_cli.weight_id} Fn_nom≈{nominal_fn:.3f} N")
    print(f"[INFO] applied lateral force (world) Fx={fx:.3f} Fy={fy:.3f} N")
    print(f"[INFO] sensor_mode={args_cli.sensor_mode} normal_corr={use_normal} slip_stick={use_slip}")
    print(
        f"[INFO] marker enabled={_marker_enabled()} pattern="
        f"{args_cli.marker_pattern if _marker_enabled() else 'none'}"
    )
    print(
        f"[INFO] finger_root_z={_finger_root_z():.4f} k_ref={_force_render_k_ref():.2f} "
        f"xy_bounds={bool(args_cli.enforce_weight_xy_bounds)} half={args_cli.weight_xy_bounds_half:.4f}"
    )

    render_cfg = _render_cfg()
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.35, 0.45, 0.75], target=[0.0, 0.0, 0.45])

    scene = InteractiveScene(LateralForceValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.3))
    scfg = _make_sensor_cfg(args_cli.sensor_mode)
    _format_paths(scfg, scene)
    scene.sensors["tactile_sensor"] = scfg.class_type(scfg)
    sim.reset()

    sim_dt = sim.get_physics_dt()
    scene["robot"].reset()
    scene.rigid_objects["contact_object"].reset()
    ts = scene["tactile_sensor"]

    clearance_z = float(args_cli.weight_clearance_z)
    _write_weight_pose(scene, clearance_z)
    for _ in range(_NOMINAL_WARMUP):
        _write_weight_pose(scene, clearance_z)
        _clear_external_force(scene)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    ts.get_initial_render()

    _write_weight_pose(scene, spawn_z)
    _clear_external_force(scene)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    for _ in range(int(args_cli.settle_steps)):
        _clear_external_force(scene)
        scene.write_data_to_sim()
        sim.step()
        _enforce_weight_xy_bounds(scene)
        scene.update(sim_dt)

    settle_px = _extract_physx(ts, env_id=0)
    settle_fn = float(settle_px["physx_fn_total"])
    if not contact_valid(
        physx_fn_mean=settle_fn,
        nominal_fn=nominal_fn,
        contact_count=int(settle_px["physx_contact_count"]),
    ):
        print(
            f"[WARN] post-settle contact weak: physx_fn={settle_fn:.3f} N "
            f"(nominal {nominal_fn:.3f} N, contacts={settle_px['physx_contact_count']})",
            file=sys.stderr,
        )

    for _ in range(int(args_cli.push_steps)):
        _record_step(ts, scene, sim, sim_dt, fx, fy)

    fn_peaks, ft_peaks, physx_fn, physx_ft, physx_contact_counts = [], [], [], [], []
    rgb_last = rgb_corr_last = None
    nf_last = sf_last = marker_disp_last = None
    for _ in range(int(args_cli.record_steps)):
        rec = _record_step(ts, scene, sim, sim_dt, fx, fy)
        fn_peaks.append(rec["fn_peak"])
        ft_peaks.append(rec["ft_peak"])
        physx_fn.append(rec["physx_fn"])
        physx_ft.append(rec["physx_ft"])
        physx_contact_counts.append(rec["physx_contact_count"])
        if rec["rgb"] is not None:
            rgb_last = rec["rgb"].detach().cpu()
        if rec["rgb_corr"] is not None:
            rgb_corr_last = rec["rgb_corr"].detach().cpu()
        if rec["nf"] is not None:
            nf_last = rec["nf"]
        if rec["sf"] is not None:
            sf_last = rec["sf"]
        if rec.get("marker_disp") is not None:
            marker_disp_last = rec["marker_disp"]

    _clear_external_force(scene)
    shear_stats = _shear_stats(sf_last)
    physx_last = _extract_physx(ts, env_id=0)
    weight_z_final = _weight_bottom_z(scene)

    tag = _force_tag(fx, fy)
    out_dir = (
        Path(args_cli.out_dir).expanduser().resolve()
        / tag
        / args_cli.weight_id
        / args_cli.sensor_mode
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_rgb = load_bg_rgb(render_cfg)
    if rgb_last is not None:
        _save_rgb(out_dir / "tactile_rgb_depth.png", rgb_last)
        save_rgb_diff_bg(out_dir / "tactile_rgb_depth_diff_bg.png", rgb_last, bg_rgb)
    if rgb_corr_last is not None and args_cli.sensor_mode == "vitacsim":
        _save_rgb(out_dir / "tactile_rgb.png", rgb_corr_last)
        _save_rgb(out_dir / "tactile_rgb_corrected.png", rgb_corr_last)
        save_rgb_diff_bg(out_dir / "tactile_rgb_diff_bg.png", rgb_corr_last, bg_rgb)
        save_rgb_diff_bg(out_dir / "tactile_rgb_corrected_diff_bg.png", rgb_corr_last, bg_rgb)
    elif rgb_last is not None:
        _save_rgb(out_dir / "tactile_rgb.png", rgb_last)
        save_rgb_diff_bg(out_dir / "tactile_rgb_diff_bg.png", rgb_last, bg_rgb)
    if nf_last is not None:
        np.save(out_dir / "tactile_normal_force.npy", nf_last)
    if sf_last is not None:
        np.save(out_dir / "tactile_shear_force.npy", sf_last)
    gt_sf = getattr(ts, "get_physx_shear_gt_tactile", None)
    if callable(gt_sf):
        gt = gt_sf(env_ids=[0])
        if gt is not None:
            np.save(out_dir / "physx_shear_gt.npy", gt[0].detach().cpu().numpy())
    if marker_disp_last is not None:
        np.save(out_dir / "tactile_marker_displacement.npy", marker_disp_last)

    physx_fn_mean = float(np.mean(physx_fn) if physx_fn else 0.0)
    physx_contact_mean = float(np.mean(physx_contact_counts) if physx_contact_counts else 0.0)
    is_valid = contact_valid(
        physx_fn_mean=physx_fn_mean,
        nominal_fn=nominal_fn,
        contact_count=int(round(physx_contact_mean)),
    )

    summary = {
        "output_schema": _OUTPUT_SCHEMA,
        "experiment": "lateral_push_no_gripper",
        "weight_id": args_cli.weight_id,
        "mass_kg": mass_kg,
        "nominal_fn_n": nominal_fn,
        "lateral_force_x_n": fx,
        "lateral_force_y_n": fy,
        "lateral_force_tag": tag,
        "sensor_mode": args_cli.sensor_mode,
        "enable_normal_correction": use_normal,
        "enable_slip_stick_reconstruction": use_slip,
        "enable_marker_simulation": _marker_enabled(),
        "marker_pattern": args_cli.marker_pattern if _marker_enabled() else "none",
        **_marker_stats(ts),
        "force_render_k_ref": _force_render_k_ref(),
        "finger_root_z": _finger_root_z(),
        "enforce_weight_xy_bounds": bool(args_cli.enforce_weight_xy_bounds),
        "weight_xy_bounds_half": float(args_cli.weight_xy_bounds_half),
        "settle_physx_fn_n": settle_fn,
        "weight_z_final": weight_z_final,
        "fn_peak_max": float(max(fn_peaks) if fn_peaks else 0.0),
        "ft_peak_max": float(max(ft_peaks) if ft_peaks else shear_stats["ft_peak_max"]),
        "ft_field_sum": shear_stats["ft_field_sum"],
        "physx_fn_total_mean": physx_fn_mean,
        "physx_ft_total_mean": float(np.mean(physx_ft) if physx_ft else 0.0),
        "physx_fn_total_last": float(physx_last["physx_fn_total"]),
        "physx_ft_total_last": float(physx_last["physx_ft_total"]),
        "physx_contact_count_mean": physx_contact_mean,
        "physx_contact_count_last": int(physx_last["physx_contact_count"]),
        "physx_fn_ratio_nominal": physx_fn_mean / nominal_fn if nominal_fn > 1e-9 else 0.0,
        "contact_valid": is_valid,
        **bg_diff_cfg_dict(),
        "applied_force_magnitude_n": float((fx * fx + fy * fy) ** 0.5),
        "recommended_fx_sweep": list(allowed_fx),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[INFO] fn_peak={summary['fn_peak_max']:.4f} ft_peak={summary['ft_peak_max']:.4f} "
        f"physx_ft={summary['physx_ft_total_mean']:.4f} physx_fn={physx_fn_mean:.4f} "
        f"valid={is_valid} (applied |F|={summary['applied_force_magnitude_n']:.3f} N)"
    )
    print(f"[INFO] saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = int(main())
    except KeyboardInterrupt:
        exit_code = 130
    os._exit(exit_code)
