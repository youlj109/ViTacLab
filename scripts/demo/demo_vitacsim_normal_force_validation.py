#!/usr/bin/env python3
"""Normal-force ViTacSim validation: horizontal GelSight + vertical validation weight (no gripper).

Each trial places the weight bottom-center on the pad center (self-weight normal load).
Outputs TacSL baseline vs ViTacSim full tactile RGB + force fields.

Usage (ViTacLab repo root, Isaac Sim python)::

    ../IsaacLab/isaaclab.sh -p scripts/demo/demo_vitacsim_normal_force_validation.py \\
        --headless --enable_cameras --device cuda:0 \\
        --weight-id W100 --sensor-mode vitacsim
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

_OUTPUT_SCHEMA = "nf_v3_beta"

parser = argparse.ArgumentParser(description="ViTacSim normal-force validation (horizontal pad, no gripper).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--profile",
    type=str,
    default="cylinder",
    choices=("cylinder", "advisor"),
    help="cylinder=legacy W* weights; advisor=M2 nut + G* mass cases (lab protocol).",
)
parser.add_argument(
    "--weight-id",
    type=str,
    default="W100",
    choices=("W200", "W100", "W050", "W020", "W010"),
    help="Cylinder validation weight preset (profile=cylinder).",
)
parser.add_argument(
    "--case-id",
    type=str,
    default="G110",
    help="Advisor case id G010..G210 (profile=advisor). Overrides --weight-id.",
)
parser.add_argument(
    "--sensor-mode",
    type=str,
    choices=("tacsl", "vitacsim"),
    default="vitacsim",
    help="tacsl=depth Taxim only; vitacsim=PhysX normal+shear reconstruction.",
)
parser.add_argument("--settle-steps", type=int, default=180, help="Physics settle steps after weight release.")
parser.add_argument("--record-steps", type=int, default=40, help="Steps to average/record after settle.")
parser.add_argument(
    "--weight-rest-z",
    type=float,
    default=-1.0,
    help="Reference/target bottom-center z (<0: advisor/cylinder default from spec).",
)
parser.add_argument(
    "--weight-clearance-z",
    type=float,
    default=0.520,
    help="Bottom-center z while capturing TacSL nominal (no pad contact).",
)
parser.add_argument(
    "--weight-spawn-z",
    type=float,
    default=-1.0,
    help="Bottom-center z when released for gravity settle (<0 => rest_z + drop_offset).",
)
parser.add_argument(
    "--weight-drop-offset",
    type=float,
    default=0.012,
    help="If weight-spawn-z<0: spawn at weight_rest_z + this offset before settle.",
)
parser.add_argument(
    "--force-render-k-ref",
    type=float,
    default=0.0,
    help="k_ref for force-corrected Taxim height (delta=fn/k_ref). "
    "<=0: auto scale from W100 reference (heavier => brighter RGB).",
)
parser.add_argument(
    "--tactile-uv-shift-u",
    type=float,
    default=-1.0,
    help="Height-map sampling shift du (px) before Taxim; <0 uses advisor default.",
)
parser.add_argument(
    "--tactile-uv-shift-v",
    type=float,
    default=-1.0,
    help="Height-map sampling shift dv (px) before Taxim; <0 uses advisor default.",
)
parser.add_argument(
    "--contact-offset-x",
    type=float,
    default=-1.0,
    help="M2 nut in-plane offset x on gel (m). <0: advisor spec default.",
)
parser.add_argument(
    "--contact-offset-y",
    type=float,
    default=-1.0,
    help="M2 nut in-plane offset y on gel (m). <0: advisor spec default.",
)
parser.add_argument(
    "--finger-root-z",
    type=float,
    default=-1.0,
    help="GelSight finger articulation root z (lower => deeper pad contact / higher load). "
    "<0: use validation_beta_config.DEFAULT_FINGER_ROOT_Z.",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="logs/vitacsim_validation/normal_force",
    help="Output root; writes {out_dir}/{weight_id}/{sensor_mode}/",
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
parser.add_argument(
    "--save-nominal-to",
    type=str,
    default="",
    help="If set, save zero-contact tactile RGB (+ marker disp) to this directory after nominal capture.",
)
parser.add_argument(
    "--nominal-only",
    action="store_true",
    help="After --save-nominal-to (or nominal capture), exit without weight drop/recording.",
)
parser.add_argument(
    "--fitted-params",
    type=str,
    default="",
    help="Optional fitted_params.json (marker gain + rgb_diff_scale applied to k_ref).",
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

from ViTacLab.tasks.direct.vitacsim_validation.validation_weight_spawner_cfg import validation_weight_spawner_cfg
from ViTacLab.tasks.direct.vitacsim_validation.validation_m2_nut_spawner_cfg import validation_m2_nut_spawner_cfg
from ViTacLab.tasks.direct.vitacsim_validation.m2_nut_spec import (
    ADVISOR_CASE_MASS_G,
    ADVISOR_FINGER_ROOT_Z,
    ADVISOR_MARKER_DEPTH_GAMMA,
    ADVISOR_MARKER_DEPTH_GAMMA_LOW_LOAD,
    ADVISOR_MARKER_DEPTH_GAMMA_LOAD_T0,
    ADVISOR_MARKER_LOAD_REF_FN_N,
    ADVISOR_MARKER_LOAD_SCALE_EXPONENT,
    ADVISOR_MARKER_SHEAR_FORCE_GAIN,
    ADVISOR_MARKER_SHEAR_FORCE_REF_N,
    ADVISOR_MARKER_SHEAR_FROM_FORCE_FIELD,
    ADVISOR_TACTILE_UV_SHIFT_PX,
    ADVISOR_WEIGHT_CLEARANCE_Z,
    ADVISOR_WEIGHT_DROP_OFFSET,
    ADVISOR_WEIGHT_REST_Z,
    M2_GEOMETRY,
    advisor_mass_kg,
    nominal_fn_n as advisor_nominal_fn_n,
)
from ViTacLab.tasks.direct.vitacsim_validation.weight_spec import GEOMETRY, WEIGHT_MASS_KG
from ViTacLab.assets.sensor import VisuoTactileSensorV2Cfg
from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
from ViTacLab.tasks.direct.pretraining.gelsight_finger_pretrain_base_cfg import (
    GELSIGHT_FINGER_SHORT_USD,
    build_gelsight_finger_robot_cfg,
)

from ViTacLab.tasks.direct.vitacsim_validation.validation_beta_config import (
    DEFAULT_FINGER_ROOT_Z,
    contact_valid,
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


def _fitted_params_path() -> Path | None:
    raw = str(getattr(args_cli, "fitted_params", "") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_file():
        p = _repo_root() / raw
    return p if p.is_file() else None


def _fitted_rgb_scale() -> float | None:
    path = _fitted_params_path()
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    scale = data.get("recommended_force_render_k_ref_scale")
    return float(scale) if scale is not None else None


def _mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "tacsl":
        return False, False
    return True, True


def _spawn_z() -> float:
    if float(args_cli.weight_spawn_z) >= 0.0:
        return float(args_cli.weight_spawn_z)
    rest_z = _weight_rest_z()
    drop = float(args_cli.weight_drop_offset)
    if _is_advisor() and float(args_cli.weight_drop_offset) == 0.012:
        drop = ADVISOR_WEIGHT_DROP_OFFSET
    return rest_z + drop


def _weight_rest_z() -> float:
    z = float(args_cli.weight_rest_z)
    if z < 0.0:
        return ADVISOR_WEIGHT_REST_Z if _is_advisor() else 0.442
    return z


def _weight_clearance_z() -> float:
    z = float(args_cli.weight_clearance_z)
    if _is_advisor() and abs(z - 0.520) < 1e-6:
        return ADVISOR_WEIGHT_CLEARANCE_Z
    return z


def _is_advisor() -> bool:
    return str(args_cli.profile) == "advisor"


def _run_case_id() -> str:
    if _is_advisor():
        cid = str(args_cli.case_id)
        if cid not in ADVISOR_CASE_MASS_G:
            raise ValueError(f"Unknown advisor case_id={cid!r}; expected one of {sorted(ADVISOR_CASE_MASS_G)}")
        return cid
    return str(args_cli.weight_id)


def _contact_offset_xy() -> tuple[float, float]:
    ox = float(args_cli.contact_offset_x)
    oy = float(args_cli.contact_offset_y)
    if ox < 0.0:
        ox = 0.0
    if oy < 0.0:
        oy = 0.0
    return ox, oy


def _tactile_uv_shift_px() -> tuple[float, float]:
    u = float(args_cli.tactile_uv_shift_u)
    v = float(args_cli.tactile_uv_shift_v)
    if _is_advisor() and u < 0.0 and v < 0.0:
        return ADVISOR_TACTILE_UV_SHIFT_PX
    return (0.0 if u < 0.0 else u, 0.0 if v < 0.0 else v)


def _make_contact_cfg(case_id: str) -> RigidObjectCfg:
    z0 = _weight_clearance_z()
    ox, oy = _contact_offset_xy()
    if _is_advisor():
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/contact_object",
            spawn=validation_m2_nut_spawner_cfg(case_id),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(ox, oy, z0)),
        )
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=validation_weight_spawner_cfg(case_id),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(ox, oy, z0)),
    )


def _finger_root_z() -> float:
    z = float(args_cli.finger_root_z)
    if z < 0.0:
        return ADVISOR_FINGER_ROOT_Z if _is_advisor() else DEFAULT_FINGER_ROOT_Z
    if _is_advisor() and abs(z - DEFAULT_FINGER_ROOT_Z) < 1e-6:
        return ADVISOR_FINGER_ROOT_Z
    return z


def _force_render_k_ref(case_id: str) -> float:
    k = resolve_force_render_k_ref(case_id, float(args_cli.force_render_k_ref))
    scale = _fitted_rgb_scale()
    if scale is not None and scale > 1e-6:
        # fit scales sim diff down when scale<1 => increase k_ref to reduce corrected height.
        k = k / scale
    return k


def _marker_enabled() -> bool:
    return not bool(args_cli.no_marker)


def _render_cfg():
    fp = _fitted_params_path()
    return validation_gelsight_render_cfg(
        enable_marker=_marker_enabled(),
        marker_pattern=args_cli.marker_pattern,
        profile=str(args_cli.profile),
        fitted_params_path=str(fp) if fp is not None else None,
    )


def _marker_stats(ts) -> dict[str, float | int | str]:
    disp = getattr(ts.data, "tactile_marker_displacement", None)
    if disp is None or disp.numel() == 0:
        return {
            "marker_pattern": "none" if not _marker_enabled() else args_cli.marker_pattern,
            "marker_count": 0,
            "marker_disp_max_px": 0.0,
            "marker_disp_p95_px": 0.0,
            "marker_disp_mean_px": 0.0,
            "marker_at_cap_count": 0,
            "marker_at_cap_frac": 0.0,
        }
    d = disp[0].detach().cpu()
    mag = torch.linalg.norm(d, dim=-1)
    cap_px = 3.0
    render_cfg = getattr(getattr(ts, "cfg", None), "render_cfg", None)
    if render_cfg is not None:
        cap_px = float(getattr(render_cfg, "marker_max_displacement_px", cap_px))
    at_cap = int((mag > cap_px - 0.05).sum().item())
    return {
        "marker_pattern": args_cli.marker_pattern if _marker_enabled() else "none",
        "marker_count": int(d.shape[0]),
        "marker_disp_max_px": float(mag.max().item()),
        "marker_disp_p95_px": float(torch.quantile(mag, 0.95).item()),
        "marker_disp_mean_px": float(mag.mean().item()),
        "marker_at_cap_count": at_cap,
        "marker_at_cap_frac": float(at_cap / max(int(d.shape[0]), 1)),
    }


def _make_sensor_cfg(mode: str) -> VisuoTactileSensorV2Cfg:
    use_normal, use_slip = _mode_flags(mode)
    render_cfg = _render_cfg()
    case_id = _run_case_id()
    # tacsl baseline uses depth*1e4 heuristic; vitacsim render maps PhysX nf (N) -> height (m).
    k_ref = 1.0e4 if mode == "tacsl" else _force_render_k_ref(case_id)
    marker_load_ref = ADVISOR_MARKER_LOAD_REF_FN_N if _is_advisor() else 0.72
    marker_load_exp = ADVISOR_MARKER_LOAD_SCALE_EXPONENT if _is_advisor() else 0.5
    marker_depth_gamma = ADVISOR_MARKER_DEPTH_GAMMA if _is_advisor() else 1.0
    marker_depth_gamma_lo = ADVISOR_MARKER_DEPTH_GAMMA_LOW_LOAD if _is_advisor() else 1.0
    marker_depth_gamma_t0 = ADVISOR_MARKER_DEPTH_GAMMA_LOAD_T0 if _is_advisor() else 0.35
    marker_shear_from_ff = (mode == "vitacsim") and (
        ADVISOR_MARKER_SHEAR_FROM_FORCE_FIELD if _is_advisor() else False
    )
    marker_shear_gain = ADVISOR_MARKER_SHEAR_FORCE_GAIN if _is_advisor() else 3.0
    marker_shear_ref = ADVISOR_MARKER_SHEAR_FORCE_REF_N if _is_advisor() else 0.05
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
        require_physx_sparse_anchors=(mode == "vitacsim"),
        strict_target_contact_attribution=True,
        tactile_uv_shift_px=_tactile_uv_shift_px(),
        marker_load_ref_fn_n=marker_load_ref,
        marker_load_scale_exponent=marker_load_exp,
        marker_depth_gamma=marker_depth_gamma,
        marker_depth_gamma_low_load=marker_depth_gamma_lo,
        marker_depth_gamma_load_t0=marker_depth_gamma_t0,
        marker_shear_from_force_field=marker_shear_from_ff,
        marker_shear_force_gain=marker_shear_gain,
        marker_shear_force_ref_n=marker_shear_ref,
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
class NormalForceValidationSceneCfg(InteractiveSceneCfg):
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
    contact_object: RigidObjectCfg = _make_contact_cfg(_run_case_id())


def _write_weight_pose(scene: InteractiveScene, z_root: float) -> None:
    obj = scene.rigid_objects["contact_object"]
    ox, oy = _contact_offset_xy()
    state = obj.data.default_root_state.clone()
    state[:, 0] = scene.env_origins[:, 0] + ox
    state[:, 1] = scene.env_origins[:, 1] + oy
    state[:, 2] = scene.env_origins[:, 2] + float(z_root)
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device).expand(state.shape[0], 4)
    state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(state[:, :7], env_ids=None)
    obj.write_root_velocity_to_sim(state[:, 7:], env_ids=None)


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


def _sensor_force_stats(nf: torch.Tensor) -> dict[str, float]:
    arr = torch.nan_to_num(nf, nan=0.0).abs()
    contact = arr > 1e-9
    n_contact = int(contact.sum().item())
    return {
        "fn_peak_max": float(arr.max().item()) if arr.numel() else 0.0,
        "fn_peak_mean": float(arr.mean().item()) if arr.numel() else 0.0,
        "fn_contact_mean": float(arr[contact].mean().item()) if n_contact > 0 else 0.0,
        "fn_field_sum": float(arr.sum().item()),
        "fn_contact_cells": float(n_contact),
    }


def _depth_stats(data) -> dict[str, float]:
    out = {
        "penetration_peak_max": 0.0,
        "penetration_peak_mean": 0.0,
        "height_corr_peak_max": 0.0,
        "rgb_corr_l1_mean": 0.0,
    }
    pen = getattr(data, "penetration_depth", None)
    if pen is not None:
        p = torch.nan_to_num(pen[0], nan=0.0).clamp(min=0.0)
        out["penetration_peak_max"] = float(p.max().item())
        out["penetration_peak_mean"] = float(p.mean().item())
    h_corr = getattr(data, "tactile_height_map_corrected", None)
    if h_corr is not None:
        out["height_corr_peak_max"] = float(torch.nan_to_num(h_corr[0], nan=0.0).abs().max().item())
    rgb = getattr(data, "tactile_rgb_image", None)
    rgb_c = getattr(data, "tactile_rgb_image_corrected", None)
    if rgb is not None and rgb_c is not None:
        d = (rgb[0].float() - rgb_c[0].float()).abs()
        out["rgb_corr_l1_mean"] = float(d.mean().item())
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

    case_id = _run_case_id()
    if _is_advisor():
        mass_kg = advisor_mass_kg(case_id)
        nominal_fn = advisor_nominal_fn_n(case_id)
        geom_height = M2_GEOMETRY.height
        geom_label = "m2_hex_nut"
    else:
        mass_kg = float(WEIGHT_MASS_KG[case_id])
        nominal_fn = mass_kg * 9.81
        geom_height = GEOMETRY.total_height
        geom_label = "chamfer_v1"

    use_normal, use_slip = _mode_flags(args_cli.sensor_mode)
    spawn_z = _spawn_z()
    print(f"[INFO] profile={args_cli.profile} case={case_id} mass={mass_kg:.3f} kg nominal_Fn≈{nominal_fn:.3f} N")
    print(f"[INFO] sensor_mode={args_cli.sensor_mode} normal_corr={use_normal} slip_stick={use_slip}")
    print(
        f"[INFO] marker enabled={_marker_enabled()} pattern="
        f"{args_cli.marker_pattern if _marker_enabled() else 'none'}"
    )
    print(
        f"[INFO] z clear/spawn/rest={_weight_clearance_z():.4f}/"
        f"{spawn_z:.4f}/{_weight_rest_z():.4f}"
    )

    render_cfg = _render_cfg()
    print(f"[INFO] GelSight render: {render_cfg.base_data_path}/{render_cfg.sensor_data_dir_name}")
    print(f"[INFO] finger_root_z={_finger_root_z():.4f} force_render_k_ref={_force_render_k_ref(case_id):.2f}")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.35, 0.45, 0.75], target=[0.0, 0.0, 0.45])

    scene = InteractiveScene(NormalForceValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.3))
    scfg = _make_sensor_cfg(args_cli.sensor_mode)
    _format_paths(scfg, scene)
    scene.sensors["tactile_sensor"] = scfg.class_type(scfg)
    sim.reset()

    sim_dt = sim.get_physics_dt()
    scene["robot"].reset()
    scene.rigid_objects["contact_object"].reset()

    ts = scene["tactile_sensor"]
    clearance_z = _weight_clearance_z()
    _write_weight_pose(scene, clearance_z)
    for _ in range(_NOMINAL_WARMUP):
        _write_weight_pose(scene, clearance_z)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    ts.get_initial_render()

    if args_cli.save_nominal_to:
        nom_dir = Path(args_cli.save_nominal_to).expanduser().resolve()
        nom_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        if ts.data.tactile_rgb_image is not None:
            _save_rgb(nom_dir / "tactile_rgb.png", ts.data.tactile_rgb_image[0])
        md = getattr(ts.data, "tactile_marker_displacement", None)
        if md is not None:
            np.save(nom_dir / "tactile_marker_displacement.npy", md[0].detach().cpu().numpy())
        (nom_dir / "summary.json").write_text(
            json.dumps({"case": "no_contact", "output_schema": _OUTPUT_SCHEMA}, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] saved nominal (no contact) tactile -> {nom_dir}")

    if args_cli.nominal_only:
        print("[INFO] nominal-only: skipping weight settle/record.")
        return 0

    _write_weight_pose(scene, spawn_z)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    for _ in range(int(args_cli.settle_steps)):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    fn_peaks: list[float] = []
    physx_fn_totals: list[float] = []
    physx_contact_counts: list[int] = []
    pen_peaks: list[float] = []
    height_corr_peaks: list[float] = []
    rgb_last = rgb_corr_last = None
    nf_last = sf_last = None
    marker_disp_last = None
    depth_last: dict[str, float] = {}

    for _ in range(int(args_cli.record_steps)):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        data = ts.data
        nf = getattr(data, "tactile_normal_force", None)
        sf = getattr(data, "tactile_shear_force", None)
        rgb = getattr(data, "tactile_rgb_image", None)
        rgb_corr = getattr(data, "tactile_rgb_image_corrected", None)
        dstat = _depth_stats(data)
        depth_last = dstat
        pen_peaks.append(dstat["penetration_peak_max"])
        height_corr_peaks.append(dstat["height_corr_peak_max"])
        if nf is not None:
            fn_peaks.append(float(torch.nan_to_num(nf[0], nan=0.0).abs().max().item()))
        px = _extract_physx(ts, env_id=0)
        physx_fn_totals.append(float(px["physx_fn_total"]))
        physx_contact_counts.append(int(px["physx_contact_count"]))
        if rgb is not None:
            rgb_last = rgb[0].detach().cpu()
        if rgb_corr is not None:
            rgb_corr_last = rgb_corr[0].detach().cpu()
        if nf is not None:
            nf_last = nf[0].detach().cpu().numpy()
        if sf is not None:
            sf_last = sf[0].detach().cpu().numpy()
        mdisp = getattr(data, "tactile_marker_displacement", None)
        if mdisp is not None:
            marker_disp_last = mdisp[0].detach().cpu().numpy()

    force_stats = _sensor_force_stats(torch.from_numpy(nf_last)) if nf_last is not None else {}
    physx_last = _extract_physx(ts, env_id=0)
    weight_z_final = _weight_bottom_z(scene)

    out_dir = Path(args_cli.out_dir).expanduser().resolve() / case_id / args_cli.sensor_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_rgb = load_bg_rgb(render_cfg)
    if rgb_last is not None:
        _save_rgb(out_dir / "tactile_rgb_depth.png", rgb_last)
        _save_rgb(out_dir / "tactile_rgb.png", rgb_last)
        save_rgb_diff_bg(out_dir / "tactile_rgb_depth_diff_bg.png", rgb_last, bg_rgb)
    if rgb_corr_last is not None:
        _save_rgb(out_dir / "tactile_rgb_corrected.png", rgb_corr_last)
        save_rgb_diff_bg(out_dir / "tactile_rgb_corrected_diff_bg.png", rgb_corr_last, bg_rgb)
        if args_cli.sensor_mode == "vitacsim":
            _save_rgb(out_dir / "tactile_rgb.png", rgb_corr_last)
            save_rgb_diff_bg(out_dir / "tactile_rgb_diff_bg.png", rgb_corr_last, bg_rgb)
    if nf_last is not None:
        np.save(out_dir / "tactile_normal_force.npy", nf_last)
    if sf_last is not None:
        np.save(out_dir / "tactile_shear_force.npy", sf_last)
    if marker_disp_last is not None:
        np.save(out_dir / "tactile_marker_displacement.npy", marker_disp_last)

    summary = {
        "output_schema": _OUTPUT_SCHEMA,
        "profile": str(args_cli.profile),
        "case_id": case_id,
        "weight_id": case_id,
        "mass_kg": mass_kg,
        "nominal_fn_n": nominal_fn,
        "sensor_mode": args_cli.sensor_mode,
        "enable_normal_correction": use_normal,
        "enable_slip_stick_reconstruction": use_slip,
        "enable_corrected_force_render": args_cli.sensor_mode == "vitacsim",
        "enable_marker_simulation": _marker_enabled(),
        "marker_pattern": args_cli.marker_pattern if _marker_enabled() else "none",
        **_marker_stats(ts),
        "force_render_k_ref": _force_render_k_ref(case_id),
        "fitted_params_path": str(_fitted_params_path()) if _fitted_params_path() else None,
        "fitted_rgb_diff_scale": _fitted_rgb_scale(),
        "normal_correction_k_ref": float(
            1.0e4 if args_cli.sensor_mode == "tacsl" else _force_render_k_ref(case_id)
        ),
        "finger_root_z": _finger_root_z(),
        "tactile_uv_shift_px": list(_tactile_uv_shift_px()),
        "contact_offset_x": _contact_offset_xy()[0],
        "contact_offset_y": _contact_offset_xy()[1],
        "weight_clearance_z": clearance_z,
        "weight_spawn_z": spawn_z,
        "weight_rest_z": _weight_rest_z(),
        "weight_z_final": weight_z_final,
        "geometry_total_height_m": geom_height,
        "contact_geometry": geom_label,
        "fn_peak_max": float(max(fn_peaks) if fn_peaks else force_stats.get("fn_peak_max", 0.0)),
        "fn_peak_mean": float(np.mean(fn_peaks) if fn_peaks else force_stats.get("fn_peak_mean", 0.0)),
        "fn_contact_mean": force_stats.get("fn_contact_mean", 0.0),
        "fn_field_sum": force_stats.get("fn_field_sum", 0.0),
        "fn_contact_cells": force_stats.get("fn_contact_cells", 0.0),
        "penetration_peak_max": float(max(pen_peaks) if pen_peaks else 0.0),
        "penetration_peak_mean": float(np.mean(pen_peaks) if pen_peaks else 0.0),
        "height_corr_peak_max": float(max(height_corr_peaks) if height_corr_peaks else 0.0),
        "rgb_corr_l1_mean": depth_last.get("rgb_corr_l1_mean", 0.0),
        "physx_fn_total_mean": float(np.mean(physx_fn_totals) if physx_fn_totals else 0.0),
        "physx_fn_total_max": float(max(physx_fn_totals) if physx_fn_totals else 0.0),
        "physx_fn_total_last": float(physx_last["physx_fn_total"]),
        "physx_ft_total_last": float(physx_last["physx_ft_total"]),
        "physx_contact_count_last": int(physx_last["physx_contact_count"]),
        "physx_contact_count_mean": float(
            np.mean(physx_contact_counts) if physx_contact_counts else int(physx_last["physx_contact_count"])
        ),
        "physx_fn_ratio_nominal": float(
            (np.mean(physx_fn_totals) if physx_fn_totals else float(physx_last["physx_fn_total"])) / nominal_fn
            if nominal_fn > 1e-9
            else 0.0
        ),
        "sparse_fn_total_n": float(getattr(ts, "_sparse_fn_total", torch.zeros(1))[0].item())
        if hasattr(ts, "_sparse_fn_total")
        else None,
        "marker_load_scale": float(
            (getattr(ts, "_sparse_fn_total", torch.zeros(1))[0].item() / 0.72) ** 0.5
        )
        if hasattr(ts, "_sparse_fn_total") and args_cli.sensor_mode == "vitacsim"
        else None,
        "contact_valid": contact_valid(
            physx_fn_mean=float(np.mean(physx_fn_totals) if physx_fn_totals else float(physx_last["physx_fn_total"])),
            nominal_fn=nominal_fn,
            contact_count=int(np.mean(physx_contact_counts) if physx_contact_counts else physx_last["physx_contact_count"]),
        ),
        **bg_diff_cfg_dict(),
        "render_base": str(render_cfg.base_data_path),
        "render_data_dir": str(render_cfg.sensor_data_dir_name),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[INFO] fn_peak={summary['fn_peak_max']:.4f} "
        f"physx_fn={summary['physx_fn_total_mean']:.4f} "
        f"pen_peak={summary['penetration_peak_max']:.6f} "
        f"z_final={weight_z_final:.4f} (nominal {nominal_fn:.3f} N)"
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
