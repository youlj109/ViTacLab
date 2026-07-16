#!/usr/bin/env python3
"""Tactile-feedback-driven insertion demo (independent case).

This is a closed-loop demo for Forge peg insertion:
- uses tactile normal/shear feedback each step
- adjusts XY alignment from shear direction
- modulates Z push based on contact and overload
- records third-person video for presentation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Forge tactile-feedback insertion demo.")
parser.add_argument("--task", type=str, default="peg_insert", choices=["peg_insert"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--fps", type=float, default=20.0)
parser.add_argument("--output_video", type=str, default="logs/tactile_feedback_demo/forge_tactile_insert.mp4")
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--with_tactile_panel", action="store_true", help="Append tactile visualization panel to video.")
parser.add_argument("--panel_width", type=int, default=880, help="Width of right-side tactile panel in output video.")
parser.add_argument("--save_tactile_npz", action="store_true", help="Save per-step tactile point-array snapshots.")
parser.add_argument(
    "--tactile_npz_path",
    type=str,
    default="logs/tactile_feedback_demo/forge_tactile_insert_tactile.npz",
    help="Output NPZ path for tactile point arrays.",
)
parser.add_argument("--tactile_dump_interval", type=int, default=1, help="Store tactile arrays every N steps.")
parser.add_argument("--geo_xy_weight", type=float, default=0.7, help="Weight of geometry guidance in XY control.")
parser.add_argument("--geo_deadband", type=float, default=0.02, help="Deadband for normalized geometry XY errors.")
parser.add_argument("--contact_active_threshold", type=float, default=0.02, help="Active ratio threshold for contact.")
parser.add_argument("--overload_threshold", type=float, default=5.0e5, help="Normal force threshold for overload backoff.")
parser.add_argument("--prealign_xy_gate", type=float, default=0.08, help="Require |geo_x|,|geo_y| below this before descending.")
parser.add_argument("--insert_xy_gate", type=float, default=0.06, help="If exceeded during insertion, backoff and re-align.")
parser.add_argument(
    "--contact_excess_force_threshold",
    type=float,
    default=2.0e4,
    help="Excess normal force above grasp baseline to treat as insertion contact.",
)
parser.add_argument(
    "--overload_excess_force_threshold",
    type=float,
    default=1.6e5,
    help="Excess normal force above grasp baseline to trigger overload backoff.",
)
parser.add_argument(
    "--baseline_ema",
    type=float,
    default=0.02,
    help="EMA factor for grasp-force baseline update in non-insertion phases.",
)
parser.add_argument(
    "--success_hold_steps",
    type=int,
    default=6,
    help="Consecutive steps satisfying success geometry to count one successful insertion.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import ViTacLab.tasks  # noqa: F401
from isaaclab_tasks.direct.factory import factory_utils
from ViTacLab.tasks.direct.simple_gripper.forge_env import ForgeEnv
from ViTacLab.tasks.direct.simple_gripper.forge_env_cfg import ForgeTaskPegInsertCfg


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype in (np.float32, np.float64):
        if float(arr.max()) <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 255.0)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def _resize_fit(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to target size."""
    if img.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _to_colormap(arr: np.ndarray, clip_max: float | None = None) -> np.ndarray:
    """Convert scalar map to JET colormap image."""
    x = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(x)) if clip_max is None else float(clip_max)
    if vmax <= 1e-8:
        norm = np.zeros_like(x, dtype=np.uint8)
    else:
        norm = np.clip((x / vmax) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def _render_shear_with_arrows(
    shear_mag: np.ndarray,
    shear_x: np.ndarray,
    shear_y: np.ndarray,
    width: int,
    height: int,
    grid_step: int = 4,
) -> np.ndarray:
    """Render tangential-force heatmap with sparse, thin direction arrows."""
    vis = _resize_fit(_to_colormap(shear_mag), width, height)
    gh, gw = shear_x.shape[:2]
    if gh <= 0 or gw <= 0:
        return vis
    mag = np.sqrt(shear_x * shear_x + shear_y * shear_y).astype(np.float32)
    max_mag = float(np.max(mag)) if mag.size > 0 else 0.0
    if max_mag <= 1e-8:
        return vis
    cell_w = float(width) / float(max(gw, 1))
    cell_h = float(height) / float(max(gh, 1))
    scale = 0.75 * min(cell_w, cell_h) / max_mag
    for r in range(0, gh, max(1, int(grid_step))):
        for c in range(0, gw, max(1, int(grid_step))):
            m = float(mag[r, c])
            if m < 0.22 * max_mag:
                continue
            x0 = int((c + 0.5) * cell_w)
            y0 = int((r + 0.5) * cell_h)
            dx = int(np.clip(shear_x[r, c] * scale, -0.65 * cell_w, 0.65 * cell_w))
            dy = int(np.clip(shear_y[r, c] * scale, -0.65 * cell_h, 0.65 * cell_h))
            p0 = (x0, y0)
            p1 = (x0 + dx, y0 + dy)
            cv2.arrowedLine(vis, p0, p1, (245, 245, 245), 1, cv2.LINE_AA, tipLength=0.28)
    return vis


def _reshape_grid(arr: np.ndarray, h: int, w: int, channels: int | None = None) -> np.ndarray:
    """Robustly reshape tactile arrays into grid form to avoid stripe artifacts."""
    x = np.asarray(arr)
    if channels is None:
        if x.ndim == 2 and x.shape == (h, w):
            return x.astype(np.float32)
        if x.ndim == 1 and x.size == h * w:
            return x.reshape(h, w).astype(np.float32)
    else:
        if x.ndim == 3 and x.shape[0] == h and x.shape[1] == w and x.shape[2] >= channels:
            return x[..., :channels].astype(np.float32)
        if x.ndim == 2 and x.shape == (h * w, channels):
            return x.reshape(h, w, channels).astype(np.float32)
        if x.ndim == 1 and x.size == h * w * channels:
            return x.reshape(h, w, channels).astype(np.float32)
    flat = x.reshape(-1).astype(np.float32)
    need = h * w if channels is None else h * w * channels
    y = np.zeros((need,), dtype=np.float32)
    y[: min(need, flat.size)] = flat[: min(need, flat.size)]
    if channels is None:
        return y.reshape(h, w)
    return y.reshape(h, w, channels)


def _collect_tactile_arrays(env: ForgeEnv, env_id: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Collect tactile arrays per finger and merged maps."""
    h, w = int(env.tactile_array_size[0]), int(env.tactile_array_size[1])
    out: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}, "merged": {}}
    rgb_list: list[np.ndarray] = []
    normal_list: list[np.ndarray] = []
    shearx_list: list[np.ndarray] = []
    sheary_list: list[np.ndarray] = []
    contact_list: list[np.ndarray] = []
    slip_list: list[np.ndarray] = []

    for key, sensor_name in (("left", "tactile_sensor_left"), ("right", "tactile_sensor_right")):
        if sensor_name not in env.scene.sensors:
            continue
        data: Any = env.scene[sensor_name].data
        rgb_t = getattr(data, "tactile_rgb_image", None)
        nf_t = getattr(data, "tactile_normal_force", None)
        sf_t = getattr(data, "tactile_shear_force", None)
        contact_t = getattr(data, "contact_mask", None)
        slip_t = getattr(data, "slip_mask", None)

        finger: dict[str, np.ndarray] = {}
        if rgb_t is not None:
            rgb_np = _to_uint8_rgb(rgb_t[env_id].detach().cpu().numpy())
            finger["rgb"] = rgb_np
            rgb_list.append(rgb_np.astype(np.float32))
        if nf_t is not None:
            normal = _reshape_grid(nf_t[env_id].detach().cpu().numpy(), h=h, w=w)
            finger["normal"] = normal
            normal_list.append(normal)
        if sf_t is not None:
            shear = _reshape_grid(sf_t[env_id].detach().cpu().numpy(), h=h, w=w, channels=2)
            finger["shear_x"] = shear[..., 0]
            finger["shear_y"] = shear[..., 1]
            finger["shear_mag"] = np.sqrt(finger["shear_x"] * finger["shear_x"] + finger["shear_y"] * finger["shear_y"])
            shearx_list.append(finger["shear_x"])
            sheary_list.append(finger["shear_y"])
        if contact_t is not None:
            contact = _reshape_grid(contact_t[env_id].detach().cpu().numpy(), h=h, w=w)
            finger["contact"] = contact
            contact_list.append(contact)
        if slip_t is not None:
            slip = _reshape_grid(slip_t[env_id].detach().cpu().numpy(), h=h, w=w)
            finger["slip"] = slip
            slip_list.append(slip)
        out[key] = finger

    merged = out["merged"]
    if rgb_list:
        merged["rgb"] = np.clip(np.mean(np.stack(rgb_list, axis=0), axis=0), 0.0, 255.0).astype(np.uint8)
    if normal_list:
        merged["normal"] = np.mean(np.stack(normal_list, axis=0), axis=0).astype(np.float32)
    if shearx_list:
        merged["shear_x"] = np.mean(np.stack(shearx_list, axis=0), axis=0).astype(np.float32)
    if sheary_list:
        merged["shear_y"] = np.mean(np.stack(sheary_list, axis=0), axis=0).astype(np.float32)
    if "shear_x" in merged and "shear_y" in merged:
        merged["shear_mag"] = np.sqrt(merged["shear_x"] * merged["shear_x"] + merged["shear_y"] * merged["shear_y"])
    if contact_list:
        merged["contact"] = np.mean(np.stack(contact_list, axis=0), axis=0).astype(np.float32)
    if slip_list:
        merged["slip"] = np.mean(np.stack(slip_list, axis=0), axis=0).astype(np.float32)
    return out


def _render_tactile_panel(
    bundle: dict[str, dict[str, np.ndarray]],
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    """Render tactile panel with per-finger RGB/contact/normal/shear maps."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    row_h = panel_h // 2
    col_w = panel_w // 4
    for row_idx, key in enumerate(("left", "right")):
        row0 = row_idx * row_h
        row1 = row0 + row_h
        finger = bundle.get(key, {})
        raw_rgb = finger.get("rgb", np.zeros((80, 80, 3), dtype=np.uint8))
        raw_bgr = raw_rgb[:, :, ::-1].copy()
        normal = finger.get("normal", np.zeros((20, 25), dtype=np.float32))
        normal_bgr = _to_colormap(normal)
        shear_mag = finger.get("shear_mag", np.zeros((20, 25), dtype=np.float32))
        shear_x = finger.get("shear_x", np.zeros((20, 25), dtype=np.float32))
        shear_y = finger.get("shear_y", np.zeros((20, 25), dtype=np.float32))
        contact = finger.get("contact", np.zeros_like(normal))
        slip = finger.get("slip", np.zeros_like(normal))
        # Overlay contact/slip on RGB so deformation-related regions are visually explicit.
        contact_up = cv2.resize(contact.astype(np.float32), (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        slip_up = cv2.resize(slip.astype(np.float32), (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = raw_bgr.copy()
        overlay[contact_up > 0.5] = (0.4 * overlay[contact_up > 0.5] + 0.6 * np.array([0, 200, 0])).astype(np.uint8)
        overlay[slip_up > 0.5] = (0.35 * overlay[slip_up > 0.5] + 0.65 * np.array([0, 80, 255])).astype(np.uint8)
        contour = (contact > 0.5).astype(np.uint8) * 255
        contour = cv2.resize(contour, (normal_bgr.shape[1], normal_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        normal_bgr[contour > 0, :] = (0, 255, 0)

        panel[row0:row1, 0:col_w] = _resize_fit(raw_bgr, col_w, row_h)
        panel[row0:row1, col_w : 2 * col_w] = _resize_fit(overlay, col_w, row_h)
        panel[row0:row1, 2 * col_w : 3 * col_w] = _resize_fit(normal_bgr, col_w, row_h)
        panel[row0:row1, 3 * col_w : 4 * col_w] = _render_shear_with_arrows(
            shear_mag=shear_mag,
            shear_x=shear_x,
            shear_y=shear_y,
            width=col_w,
            height=row_h,
            grid_step=4,
        )

        cv2.putText(panel, f"{key} rgb", (8, row0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"{key} contact/slip",
            (col_w + 8, row0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{key} normal_map",
            (2 * col_w + 8, row0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{key} tangential",
            (3 * col_w + 8, row0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return panel


def _read_tactile_feedback(env: ForgeEnv, env_id: int = 0) -> tuple[float, float, float, float]:
    """Return (normal_total, shear_mean_x, shear_mean_y, active_ratio)."""
    nf_sum = 0.0
    sx_sum = 0.0
    sy_sum = 0.0
    cnt = 0
    active_sum = 0.0
    active_cnt = 0
    for name in ("tactile_sensor_left", "tactile_sensor_right"):
        if name not in env.scene.sensors:
            continue
        data = env.scene[name].data
        nf = getattr(data, "tactile_normal_force", None)
        sf = getattr(data, "tactile_shear_force", None)
        if nf is not None:
            nf_e = nf[env_id]
            nf_sum += float(torch.clamp(nf_e, min=0.0).sum().item())
            active_sum += float((nf_e > 1e-4).float().mean().item())
            active_cnt += 1
        if sf is not None:
            sf_e = sf[env_id].reshape(-1, 2)
            sx_sum += float(sf_e[:, 0].mean().item())
            sy_sum += float(sf_e[:, 1].mean().item())
            cnt += 1
    active_ratio = (active_sum / active_cnt) if active_cnt > 0 else 0.0
    if cnt <= 0:
        return nf_sum, 0.0, 0.0, active_ratio
    return nf_sum, sx_sum / cnt, sy_sum / cnt, active_ratio


def _build_action(
    phase: str,
    normal_total: float,
    normal_excess: float,
    shear_x: float,
    shear_y: float,
    active_ratio: float,
    active_excess: float,
    geo_err_x: float,
    geo_err_y: float,
    geo_xy_weight: float,
    geo_deadband: float,
    contact_active_threshold: float,
    overload_threshold: float,
    contact_excess_force_threshold: float,
    overload_excess_force_threshold: float,
    prealign_xy_gate: float,
    insert_xy_gate: float,
    stuck: bool,
    settle_counter: int,
    xy_dist: float,
    z_disp: float,
    z_target: float,
    z_prog: float,
    center_hold: int,
    contact_hold: int,
    recovery_step: int,
) -> tuple[np.ndarray, int, str, int]:
    """Geometry-guided + tactile-corrected closed-loop policy in env action space (7D)."""
    a = np.zeros(7, dtype=np.float32)

    _ = overload_threshold  # deprecated: keep for backward compatibility
    contact = (normal_excess > contact_excess_force_threshold) or (active_excess > contact_active_threshold)
    overload = normal_excess > overload_excess_force_threshold
    shear_norm = float(np.sqrt(shear_x * shear_x + shear_y * shear_y))
    geo_x = 0.0 if abs(geo_err_x) < geo_deadband else float(geo_err_x)
    geo_y = 0.0 if abs(geo_err_y) < geo_deadband else float(geo_err_y)
    tactile_x = float(np.clip(-4.0 * shear_x, -0.20, 0.20))
    tactile_y = float(np.clip(-4.0 * shear_y, -0.20, 0.20))
    geo_cmd_x = float(np.clip(-1.2 * geo_x, -0.55, 0.55))
    geo_cmd_y = float(np.clip(-1.2 * geo_y, -0.55, 0.55))
    blend = float(np.clip(geo_xy_weight, 0.0, 1.0))
    if contact:
        # Keep geometry dominant after contact; tactile is secondary anti-jamming cue.
        blend = float(np.clip(max(geo_xy_weight, 0.75), 0.65, 0.92))
    a[0] = float(np.clip(blend * geo_cmd_x + (1.0 - blend) * tactile_x, -0.45, 0.45))
    a[1] = float(np.clip(blend * geo_cmd_y + (1.0 - blend) * tactile_y, -0.45, 0.45))

    if phase == "recover":
        if recovery_step < 12:
            a[2] = 0.08  # quick lift
        elif recovery_step < 34:
            i = recovery_step - 12
            ang = 0.55 * float(i)
            a[0] = float(np.clip(0.14 * np.cos(ang), -0.20, 0.20))
            a[1] = float(np.clip(0.14 * np.sin(ang), -0.20, 0.20))
            a[2] = 0.02
        elif recovery_step < 46:
            a[2] = -0.02
        else:
            phase = "pre_align"
            recovery_step = 0
        if phase == "recover":
            recovery_step += 1
        a[5] = 0.0
        a[6] = -1.0
        return np.clip(a, -1.0, 1.0), settle_counter, phase, recovery_step

    centered = xy_dist < 0.0025
    centered_for_insert = xy_dist < 0.0016
    insertion_ready = centered_for_insert and contact

    def _adaptive_z_cmd() -> float:
        if overload:
            return 0.07
        if (normal_excess > 0.55 * overload_excess_force_threshold) and (z_prog < 4e-5):
            return 0.04
        if insertion_ready and z_disp > z_target + 0.0002:
            # Clearly in insertion corridor: commit to a stronger downward push.
            return -0.060
        if (not centered_for_insert) and contact:
            return 0.02
        if z_disp > z_target + 0.0015:
            return -0.040
        if z_disp > z_target + 0.0006:
            return -0.025
        return -0.010

    max_geo_abs = max(abs(geo_x), abs(geo_y))
    if phase == "pre_align":
        # Do not descend until geometry is near hole center.
        a[2] = 0.03 if (max_geo_abs > prealign_xy_gate or contact) else -0.01
        if center_hold >= 8 and (not contact):
            phase = "approach"
    elif phase == "approach":
        if max_geo_abs > prealign_xy_gate:
            phase = "pre_align"
            a[2] = 0.03
        else:
            a[2] = -0.035
        if contact_hold >= 3:
            phase = "align_insert"
    elif phase == "align_insert":
        if stuck or (max_geo_abs > insert_xy_gate):
            a[2] = 0.08
            phase = "recover"
            recovery_step = 0
            settle_counter = 0
        else:
            a[2] = _adaptive_z_cmd()
        if phase not in ("recover", "pre_align"):
            if shear_norm < 0.0010 and centered_for_insert and contact:
                settle_counter += 1
            else:
                settle_counter = 0
            if settle_counter >= 10:
                phase = "insert_hold"
                settle_counter = 0
    else:  # insert_hold
        if stuck or (max_geo_abs > insert_xy_gate):
            phase = "recover"
            recovery_step = 0
            settle_counter = 0
            a[2] = 0.08
        else:
            a[0] = float(np.clip(0.7 * a[0], -0.25, 0.25))
            a[1] = float(np.clip(0.7 * a[1], -0.25, 0.25))
            if insertion_ready and z_disp > z_target + 0.0001 and (not overload):
                a[2] = -0.050
            else:
                a[2] = _adaptive_z_cmd()

    # only yaw channel among rotation is effective in ForgeEnv; keep fixed
    a[5] = 0.0
    # success_pred channel unused by controller, keep -1
    a[6] = -1.0
    return np.clip(a, -1.0, 1.0), settle_counter, phase, recovery_step


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = ForgeTaskPegInsertCfg()
    cfg.scene.num_envs = max(1, int(args_cli.num_envs))
    cfg.device = getattr(args_cli, "device", None) or "cuda:0"
    cfg.enable_cameras = True
    cfg.obs_mode = "full"

    env = ForgeEnv(cfg)
    obs, extras = env.reset()

    if "third_person_camera" not in env.scene.sensors:
        raise RuntimeError("third_person_camera is unavailable. enable_cameras must be True.")

    out_path = Path(args_cli.output_video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cam_rgb = env.scene["third_person_camera"].data.output.get("rgb", None)
    if cam_rgb is None:
        raise RuntimeError("third_person_camera has no rgb output.")
    frame0 = _to_uint8_rgb(cam_rgb[0].detach().cpu().numpy())
    h, w = int(frame0.shape[0]), int(frame0.shape[1])
    panel_w = max(320, int(args_cli.panel_width))
    with_panel = bool(args_cli.with_tactile_panel)
    out_w = w + panel_w if with_panel else w
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args_cli.fps), (out_w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create writer: {out_path}")
    save_npz = bool(args_cli.save_tactile_npz)
    npz_path = Path(args_cli.tactile_npz_path).expanduser().resolve()
    if save_npz:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
    dump_interval = max(1, int(args_cli.tactile_dump_interval))
    tactile_logs: dict[str, list[Any]] = {
        "step": [],
        "phase": [],
        "normal_total": [],
        "active_ratio": [],
        "shear_x_mean": [],
        "shear_y_mean": [],
        "geo_err_x": [],
        "geo_err_y": [],
        "xy_dist_m": [],
        "z_disp_m": [],
        "stuck_flag": [],
        "normal_map": [],
        "shear_x_map": [],
        "shear_y_map": [],
        "contact_map": [],
        "slip_map": [],
        "left_normal_map": [],
        "right_normal_map": [],
        "left_contact_map": [],
        "right_contact_map": [],
    }
    xy_scale = max(float(getattr(env.cfg_task.fixed_asset_cfg, "diameter", 0.0081)) * 0.5, 0.0025)

    phase = "pre_align"
    settle_counter = 0
    stuck_counter = 0
    prev_z_disp: float | None = None
    prev_xy_dist: float | None = None
    baseline_nf: float | None = None
    baseline_active: float | None = None
    center_hold = 0
    contact_hold = 0
    recovery_step = 0
    fixed_cfg = env.cfg_task.fixed_asset_cfg
    if env.cfg_task.name in ("peg_insert", "gear_mesh"):
        z_target = float(fixed_cfg.height) * float(env.cfg_task.success_threshold)
    elif env.cfg_task.name == "nut_thread":
        z_target = float(getattr(fixed_cfg, "thread_pitch", 0.002)) * float(env.cfg_task.success_threshold)
    else:
        z_target = 0.001
    success_count = 0
    true_success_count = 0
    success_hold = 0
    episode_success_latched = False
    try:
        for step in range(1, int(args_cli.steps) + 1):
            if not simulation_app.is_running():
                break

            nf, sx, sy, active = _read_tactile_feedback(env, env_id=0)
            if baseline_nf is None:
                baseline_nf = float(nf)
            if baseline_active is None:
                baseline_active = float(active)
            baseline_ema = float(np.clip(float(args_cli.baseline_ema), 0.001, 0.2))
            if phase in ("pre_align", "approach"):
                baseline_nf = (1.0 - baseline_ema) * baseline_nf + baseline_ema * float(nf)
                baseline_active = (1.0 - baseline_ema) * baseline_active + baseline_ema * float(active)
            nf_excess = max(0.0, float(nf) - float(baseline_nf))
            active_excess = max(0.0, float(active) - float(baseline_active))
            held_base_pos, _ = factory_utils.get_held_base_pose(
                env.held_pos, env.held_quat, env.cfg_task.name, env.cfg_task.fixed_asset_cfg, env.num_envs, env.device
            )
            target_base_pos, _ = factory_utils.get_target_held_base_pose(
                env.fixed_pos, env.fixed_quat, env.cfg_task.name, env.cfg_task.fixed_asset_cfg, env.num_envs, env.device
            )
            xy_err = (held_base_pos[0, 0:2] - target_base_pos[0, 0:2]).detach().cpu().numpy()
            xy_dist = float(np.linalg.norm(xy_err))
            z_disp = float((held_base_pos[0, 2] - target_base_pos[0, 2]).detach().cpu().item())
            geo_x = float(np.clip(xy_err[0] / xy_scale, -1.0, 1.0))
            geo_y = float(np.clip(xy_err[1] / xy_scale, -1.0, 1.0))
            z_prog = 0.0 if prev_z_disp is None else abs(z_disp - prev_z_disp)
            xy_improve = 0.0 if prev_xy_dist is None else (float(prev_xy_dist) - float(xy_dist))
            prev_z_disp = z_disp
            prev_xy_dist = xy_dist
            centered = bool(xy_dist < 0.0025)
            success_geom = bool(centered and (z_disp < z_target))
            if success_geom:
                success_hold += 1
            else:
                success_hold = 0
            if (not episode_success_latched) and (success_hold >= max(1, int(args_cli.success_hold_steps))):
                true_success_count += 1
                episode_success_latched = True
            center_hold = center_hold + 1 if centered else 0
            contact_now = (nf_excess > float(args_cli.contact_excess_force_threshold)) or (
                active_excess > float(args_cli.contact_active_threshold)
            )
            contact_hold = contact_hold + 1 if contact_now else 0
            high_force = (nf_excess > 0.6 * float(args_cli.overload_excess_force_threshold)) or (
                active_excess > max(0.15, float(args_cli.contact_active_threshold))
            )
            low_progress = z_prog < 6e-5
            poor_xy_progress = xy_improve < 8e-5
            if high_force and low_progress and poor_xy_progress and phase in ("approach", "align_insert", "insert_hold"):
                stuck_counter += 1
            else:
                stuck_counter = max(0, stuck_counter - 1)
            stuck = stuck_counter >= 6
            act_np, settle_counter, phase, recovery_step = _build_action(
                phase=phase,
                normal_total=nf,
                normal_excess=nf_excess,
                shear_x=sx,
                shear_y=sy,
                active_ratio=active,
                active_excess=active_excess,
                geo_err_x=geo_x,
                geo_err_y=geo_y,
                geo_xy_weight=float(args_cli.geo_xy_weight),
                geo_deadband=float(args_cli.geo_deadband),
                contact_active_threshold=float(args_cli.contact_active_threshold),
                overload_threshold=float(args_cli.overload_threshold),
                contact_excess_force_threshold=float(args_cli.contact_excess_force_threshold),
                overload_excess_force_threshold=float(args_cli.overload_excess_force_threshold),
                prealign_xy_gate=float(args_cli.prealign_xy_gate),
                insert_xy_gate=float(args_cli.insert_xy_gate),
                stuck=bool(stuck),
                settle_counter=settle_counter,
                xy_dist=float(xy_dist),
                z_disp=float(z_disp),
                z_target=float(z_target),
                z_prog=float(z_prog),
                center_hold=int(center_hold),
                contact_hold=int(contact_hold),
                recovery_step=int(recovery_step),
            )
            act = torch.tensor(act_np, dtype=torch.float32, device=env.device).unsqueeze(0)
            obs, rew, terminated, truncated, extras = env.step(act)
            bundle: dict[str, dict[str, np.ndarray]] = {}
            if with_panel or save_npz:
                bundle = _collect_tactile_arrays(env, env_id=0)

            done = bool((terminated[0] or truncated[0]).item()) if torch.is_tensor(terminated) else bool(terminated[0] or truncated[0])
            if done:
                success_count += 1
                obs, extras = env.reset()
                phase = "pre_align"
                settle_counter = 0
                stuck_counter = 0
                prev_z_disp = None
                prev_xy_dist = None
                baseline_nf = None
                baseline_active = None
                center_hold = 0
                contact_hold = 0
                recovery_step = 0
                success_hold = 0
                episode_success_latched = False

            cam_rgb = env.scene["third_person_camera"].data.output.get("rgb", None)
            if cam_rgb is None:
                continue
            rgb = _to_uint8_rgb(cam_rgb[0].detach().cpu().numpy())
            bgr = rgb[:, :, ::-1].copy()

            rew_v = float(rew[0].item()) if torch.is_tensor(rew) else float(np.asarray(rew)[0])
            title = f"Tactile-Feedback Insert | step={step}/{int(args_cli.steps)} | phase={phase}"
            line2 = (
                f"normal={nf:.2f} excess={nf_excess:.2f} active={active:.3f}/{active_excess:.3f} "
                f"shear=({sx:.4f},{sy:.4f}) "
                f"geo_xy=({geo_x:.3f},{geo_y:.3f}) xy_mm={xy_dist*1000.0:.2f} z_mm={z_disp*1000.0:.2f}/{z_target*1000.0:.2f} "
                f"stuck={int(stuck)} reward={rew_v:.3f} reset={int(done)}"
            )
            cv2.putText(bgr, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr, line2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"true_success={true_success_count} | resets={success_count} | success_hold={success_hold}",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            if with_panel:
                panel = _render_tactile_panel(bundle, panel_h=h, panel_w=panel_w)
                writer.write(np.concatenate([bgr, panel], axis=1))
            else:
                writer.write(bgr)

            if save_npz and (step % dump_interval == 0):
                tactile_logs["step"].append(int(step))
                tactile_logs["phase"].append(str(phase))
                tactile_logs["normal_total"].append(float(nf))
                tactile_logs["active_ratio"].append(float(active))
                tactile_logs["shear_x_mean"].append(float(sx))
                tactile_logs["shear_y_mean"].append(float(sy))
                tactile_logs["geo_err_x"].append(float(geo_x))
                tactile_logs["geo_err_y"].append(float(geo_y))
                tactile_logs["xy_dist_m"].append(float(xy_dist))
                tactile_logs["z_disp_m"].append(float(z_disp))
                tactile_logs["stuck_flag"].append(float(1.0 if stuck else 0.0))
                merged = bundle.get("merged", {})
                if "normal" in merged:
                    tactile_logs["normal_map"].append(merged["normal"].astype(np.float32))
                if "shear_x" in merged:
                    tactile_logs["shear_x_map"].append(merged["shear_x"].astype(np.float32))
                if "shear_y" in merged:
                    tactile_logs["shear_y_map"].append(merged["shear_y"].astype(np.float32))
                if "contact" in merged:
                    tactile_logs["contact_map"].append(merged["contact"].astype(np.float32))
                if "slip" in merged:
                    tactile_logs["slip_map"].append(merged["slip"].astype(np.float32))
                left = bundle.get("left", {})
                right = bundle.get("right", {})
                if "normal" in left:
                    tactile_logs["left_normal_map"].append(left["normal"].astype(np.float32))
                if "normal" in right:
                    tactile_logs["right_normal_map"].append(right["normal"].astype(np.float32))
                if "contact" in left:
                    tactile_logs["left_contact_map"].append(left["contact"].astype(np.float32))
                if "contact" in right:
                    tactile_logs["right_contact_map"].append(right["contact"].astype(np.float32))

            if int(args_cli.log_interval) > 0 and (step % int(args_cli.log_interval) == 0 or step == int(args_cli.steps)):
                print(
                    f"[TF-INSERT] step={step:04d} phase={phase} nf={nf:.4f} active={active:.4f} "
                    f"nf_ex={nf_excess:.2f} active_ex={active_excess:.4f} "
                    f"shear=({sx:.5f},{sy:.5f}) geo=({geo_x:.4f},{geo_y:.4f}) xy_mm={xy_dist*1000.0:.2f} "
                    f"z_mm={z_disp*1000.0:.2f}/{z_target*1000.0:.2f} stuck={int(stuck)} reward={rew_v:.4f} "
                    f"true_success={true_success_count} resets={success_count}"
                )
    finally:
        writer.release()
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass

    if save_npz:
        payload: dict[str, np.ndarray] = {
            "step": np.asarray(tactile_logs["step"], dtype=np.int32),
            "phase": np.asarray(tactile_logs["phase"], dtype="<U32"),
            "normal_total": np.asarray(tactile_logs["normal_total"], dtype=np.float32),
            "active_ratio": np.asarray(tactile_logs["active_ratio"], dtype=np.float32),
            "shear_x_mean": np.asarray(tactile_logs["shear_x_mean"], dtype=np.float32),
            "shear_y_mean": np.asarray(tactile_logs["shear_y_mean"], dtype=np.float32),
            "geo_err_x": np.asarray(tactile_logs["geo_err_x"], dtype=np.float32),
            "geo_err_y": np.asarray(tactile_logs["geo_err_y"], dtype=np.float32),
            "xy_dist_m": np.asarray(tactile_logs["xy_dist_m"], dtype=np.float32),
            "z_disp_m": np.asarray(tactile_logs["z_disp_m"], dtype=np.float32),
            "stuck_flag": np.asarray(tactile_logs["stuck_flag"], dtype=np.float32),
        }
        if tactile_logs["normal_map"]:
            payload["normal_map"] = np.stack(tactile_logs["normal_map"], axis=0).astype(np.float32)
        if tactile_logs["shear_x_map"]:
            payload["shear_x_map"] = np.stack(tactile_logs["shear_x_map"], axis=0).astype(np.float32)
        if tactile_logs["shear_y_map"]:
            payload["shear_y_map"] = np.stack(tactile_logs["shear_y_map"], axis=0).astype(np.float32)
        if tactile_logs["contact_map"]:
            payload["contact_map"] = np.stack(tactile_logs["contact_map"], axis=0).astype(np.float32)
        if tactile_logs["slip_map"]:
            payload["slip_map"] = np.stack(tactile_logs["slip_map"], axis=0).astype(np.float32)
        if tactile_logs["left_normal_map"]:
            payload["left_normal_map"] = np.stack(tactile_logs["left_normal_map"], axis=0).astype(np.float32)
        if tactile_logs["right_normal_map"]:
            payload["right_normal_map"] = np.stack(tactile_logs["right_normal_map"], axis=0).astype(np.float32)
        if tactile_logs["left_contact_map"]:
            payload["left_contact_map"] = np.stack(tactile_logs["left_contact_map"], axis=0).astype(np.float32)
        if tactile_logs["right_contact_map"]:
            payload["right_contact_map"] = np.stack(tactile_logs["right_contact_map"], axis=0).astype(np.float32)
        np.savez_compressed(str(npz_path), **payload)
        print(f"[DONE] tactile point-array dump: {npz_path}")

    print(f"[DONE] tactile-feedback insertion video: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
