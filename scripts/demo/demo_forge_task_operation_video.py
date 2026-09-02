#!/usr/bin/env python3
"""Record a presentation-style third-person robot operation video on Forge tasks.

This script is intended for supervisor/demo presentation:
- shows robot behavior from scene third-person camera
- avoids internal tactile heatmap-only diagnostics
- overlays concise runtime info (task/mode/step/reward)
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Forge task operation video (third-person camera).")
parser.add_argument("--task", type=str, default="peg_insert", choices=["peg_insert", "gear_mesh", "nut_thread"])
parser.add_argument("--mode", type=str, default="full", choices=["baseline", "normal_only", "full"])
parser.add_argument("--steps", type=int, default=420)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fps", type=float, default=20.0)
parser.add_argument("--log_interval", type=int, default=20, help="Print progress every N steps.")
parser.add_argument("--curve_window", type=int, default=180, help="History length for on-screen metric curves.")
parser.add_argument("--output_video", type=str, default="logs/operation_demo/forge_operation_demo.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import ViTacLab.tasks  # noqa: F401
from ViTacLab.tasks.direct.simple_gripper.forge_env import ForgeEnv
from ViTacLab.tasks.direct.simple_gripper.forge_env_cfg import ForgeTaskGearMeshCfg, ForgeTaskNutThreadCfg, ForgeTaskPegInsertCfg


def _make_cfg(task: str):
    if task == "peg_insert":
        cfg = ForgeTaskPegInsertCfg()
    elif task == "gear_mesh":
        cfg = ForgeTaskGearMeshCfg()
    elif task == "nut_thread":
        cfg = ForgeTaskNutThreadCfg()
    else:
        raise ValueError(f"Unsupported task: {task}")
    return cfg


def _configure_vitac_mode(cfg, mode: str) -> None:
    # Ensure cameras exist so third_person_camera is available.
    cfg.enable_cameras = True
    cfg.obs_mode = "full"
    for sensor_name in ("tactile_sensor_left", "tactile_sensor_right"):
        if not hasattr(cfg.scene, sensor_name):
            continue
        sensor_cfg = getattr(cfg.scene, sensor_name)
        if mode == "baseline":
            sensor_cfg.enable_normal_correction = False
            sensor_cfg.enable_slip_stick_reconstruction = False
        elif mode == "normal_only":
            sensor_cfg.enable_normal_correction = True
            sensor_cfg.enable_slip_stick_reconstruction = False
        else:  # full
            sensor_cfg.enable_normal_correction = True
            sensor_cfg.enable_slip_stick_reconstruction = True
        sensor_cfg.use_physx_sparse_anchors = True


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    x = frame
    if x.dtype in (np.float32, np.float64):
        mx = float(np.max(x)) if x.size > 0 else 0.0
        if mx <= 1.0:
            x = np.clip(x, 0.0, 1.0) * 255.0
        else:
            x = np.clip(x, 0.0, 255.0)
    else:
        x = np.clip(x.astype(np.float32), 0.0, 255.0)
    if x.ndim == 3 and x.shape[-1] >= 3:
        x = x[..., :3]
    return x.astype(np.uint8)


def _scripted_action(step: int, total_steps: int, device: torch.device, num_envs: int) -> torch.Tensor:
    # 7-D action: [x, y, z, roll, pitch, yaw, success_pred]
    # Keep motion smooth and conservative for stable visual demo.
    t = float(step) / float(max(total_steps, 1))
    a = torch.zeros((num_envs, 7), device=device, dtype=torch.float32)

    # Approach / operation / retreat phases.
    if t < 0.25:
        a[:, 2] = -0.18
    elif t < 0.75:
        tt = (t - 0.25) / 0.50
        a[:, 2] = -0.05 + 0.04 * float(np.sin(2.0 * np.pi * tt))
        a[:, 0] = 0.10 * float(np.sin(2.0 * np.pi * tt))
        a[:, 1] = 0.06 * float(np.sin(4.0 * np.pi * tt + 0.6))
        a[:, 5] = 0.10 * float(np.sin(2.0 * np.pi * tt))
    else:
        a[:, 2] = 0.14
    return torch.clamp(a, -1.0, 1.0)


def _read_tactile_metrics(env, env_id: int = 0) -> tuple[float, float]:
    fn_total = 0.0
    contact_ratio_sum = 0.0
    contact_count = 0
    for sensor_name in ("tactile_sensor_left", "tactile_sensor_right"):
        if sensor_name not in env.scene.sensors:
            continue
        data = env.scene[sensor_name].data
        nf = getattr(data, "tactile_normal_force", None)
        cm = getattr(data, "contact_mask", None)
        if nf is not None:
            fn_total += float(torch.clamp(nf[env_id], min=0.0).sum().item())
        if cm is not None:
            contact_ratio_sum += float(cm[env_id].float().mean().item())
            contact_count += 1
    contact_ratio = (contact_ratio_sum / float(contact_count)) if contact_count > 0 else 0.0
    return fn_total, contact_ratio


def _draw_progress_bar(img: np.ndarray, step: int, total: int) -> None:
    h, w = img.shape[:2]
    x0, x1 = 12, w - 12
    y0, y1 = h - 42, h - 26
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    progress = float(step) / float(max(total, 1))
    fill_x = int(round(x0 + (x1 - x0) * np.clip(progress, 0.0, 1.0)))
    cv2.rectangle(img, (x0 + 1, y0 + 1), (max(x0 + 1, fill_x - 1), y1 - 1), (80, 190, 110), -1)
    cv2.putText(
        img,
        f"progress {100.0 * progress:5.1f}%",
        (x0, y0 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def _draw_series_strip(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    data: deque[float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (85, 85, 85), 1)
    vals = np.asarray(list(data), dtype=np.float32)
    if vals.size >= 2:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        xs = np.linspace(x + 2, x + w - 2, vals.size, dtype=np.float32)
        ys = y + h - 2 - ((vals - vmin) / (vmax - vmin) * (h - 4))
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        latest = float(vals[-1])
    else:
        latest = 0.0
    cv2.putText(img, f"{label}: {latest:.3f}", (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 225, 225), 1, cv2.LINE_AA)


def _draw_metric_panel(img: np.ndarray, reward_hist: deque[float], fn_hist: deque[float], contact_hist: deque[float]) -> None:
    h, w = img.shape[:2]
    panel_w = min(390, max(260, w // 2))
    panel_h = 165
    x0, y0 = w - panel_w - 12, 12
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0.0, img)
    cv2.putText(img, "Runtime Curves", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    strip_h = 42
    gap = 8
    sx = x0 + 8
    sw = panel_w - 16
    _draw_series_strip(img, sx, y0 + 25, sw, strip_h, reward_hist, "reward", (80, 200, 255))
    _draw_series_strip(img, sx, y0 + 25 + (strip_h + gap), sw, strip_h, fn_hist, "sensor_fn", (120, 230, 120))
    _draw_series_strip(img, sx, y0 + 25 + 2 * (strip_h + gap), sw, strip_h, contact_hist, "contact_ratio", (240, 190, 90))


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = _make_cfg(args_cli.task)
    cfg.scene.num_envs = max(1, int(args_cli.num_envs))
    cfg.device = getattr(args_cli, "device", None) or "cuda:0"
    _configure_vitac_mode(cfg, str(args_cli.mode))

    env = ForgeEnv(cfg)
    obs, extras = env.reset()

    if "third_person_camera" not in env.scene.sensors:
        raise RuntimeError("third_person_camera is unavailable. Check cfg.enable_cameras and scene setup.")

    out_path = Path(args_cli.output_video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe one frame to initialize writer.
    cam_rgb = env.scene["third_person_camera"].data.output.get("rgb", None)
    if cam_rgb is None:
        raise RuntimeError("third_person_camera has no rgb output.")
    frame0 = _to_uint8_rgb(cam_rgb[0].detach().cpu().numpy())
    h, w = int(frame0.shape[0]), int(frame0.shape[1])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args_cli.fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    ep_count = 0
    reward_hist: deque[float] = deque(maxlen=max(20, int(args_cli.curve_window)))
    fn_hist: deque[float] = deque(maxlen=max(20, int(args_cli.curve_window)))
    contact_hist: deque[float] = deque(maxlen=max(20, int(args_cli.curve_window)))
    t_start = time.time()
    try:
        for step in range(1, int(args_cli.steps) + 1):
            if not simulation_app.is_running():
                break
            action = _scripted_action(step, int(args_cli.steps), env.device, env.num_envs)
            obs, rew, terminated, truncated, extras = env.step(action)

            cam_rgb = env.scene["third_person_camera"].data.output.get("rgb", None)
            if cam_rgb is None:
                continue
            frame = _to_uint8_rgb(cam_rgb[0].detach().cpu().numpy())
            bgr = frame[:, :, ::-1].copy()

            rew_v = float(rew[0].item()) if torch.is_tensor(rew) else float(np.asarray(rew)[0])
            fn_total, contact_ratio = _read_tactile_metrics(env, env_id=0)
            reward_hist.append(rew_v)
            fn_hist.append(fn_total)
            contact_hist.append(contact_ratio)
            done = bool((terminated[0] or truncated[0]).item()) if torch.is_tensor(terminated) else bool(terminated[0] or truncated[0])
            if done:
                ep_count += 1

            title = f"Forge {args_cli.task} | mode={args_cli.mode} | step={step}/{args_cli.steps}"
            subtitle = f"reward={rew_v:.3f} | sensor_fn={fn_total:.3f} | contact_ratio={contact_ratio:.3f} | done={int(done)} | episodes={ep_count}"
            cv2.putText(bgr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr, subtitle, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 2, cv2.LINE_AA)
            _draw_metric_panel(bgr, reward_hist, fn_hist, contact_hist)
            _draw_progress_bar(bgr, step, int(args_cli.steps))
            cv2.putText(
                bgr,
                "Presentation view: third-person robot operation",
                (12, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            writer.write(bgr)

            if int(args_cli.log_interval) > 0 and (step % int(args_cli.log_interval) == 0 or step == int(args_cli.steps)):
                elapsed = max(time.time() - t_start, 1e-6)
                speed = float(step) / elapsed
                remain = max(int(args_cli.steps) - step, 0)
                eta = remain / max(speed, 1e-6)
                print(
                    f"[PROGRESS] {100.0*step/max(int(args_cli.steps),1):5.1f}% "
                    f"step={step}/{int(args_cli.steps)} reward={rew_v:.3f} "
                    f"sensor_fn={fn_total:.3f} contact_ratio={contact_ratio:.3f} "
                    f"speed={speed:.2f} step/s eta={eta:.1f}s"
                )

            if done:
                obs, extras = env.reset()
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

    print(f"[DONE] operation demo saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
