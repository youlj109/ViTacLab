#!/usr/bin/env python3
"""Reset-only sanity check for dual-arm bottle cap scene layout.

Verifies robot roots match hand-over / bi-stab base and bottle sits upright on the table
before running trajectory replay.

Usage (ViTacLab repo root, Isaac Sim python):

    ./isaaclab.sh -p scripts/debug/check_unscrew_bottle_scene.py --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "source" / "ViTacLab") not in sys.path:
    sys.path.insert(0, str(_REPO / "source" / "ViTacLab"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--settle-steps", type=int, default=48)
    p.add_argument(
        "--out-dir",
        type=str,
        default="logs/demo_bottle/scene_check",
        help="Directory for optional third-person PNG snapshot.",
    )
    AppLauncher.add_app_launcher_args(p)
    return p


def main() -> int:
    args_cli, _ = _build_parser().parse_known_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import ViTacLab.tasks  # noqa: F401
    from ViTacLab.tasks.direct.medium_dexhand.unscrewing_bottle_cap.unscrewing_bottle_cap_env import (
        UR10eDualShadowHandUnscrewBottleCapEnv,
    )
    from ViTacLab.tasks.direct.medium_dexhand.unscrewing_bottle_cap.unscrewing_bottle_cap_env_cfg import (
        UR10eDualShadowHandUnscrewBottleCapEnvCfg,
    )

    cfg = UR10eDualShadowHandUnscrewBottleCapEnvCfg()
    cfg.scene.num_envs = max(1, int(args_cli.num_envs))
    cfg.device = str(getattr(args_cli, "device", "cuda:0"))
    cfg.enable_cameras = True

    env = UR10eDualShadowHandUnscrewBottleCapEnv(cfg)
    act_dim = len(env.actuated_dof_indices)
    zero = {
        "right_hand": torch.zeros(env.num_envs, act_dim, device=env.device),
        "left_hand": torch.zeros(env.num_envs, act_dim, device=env.device),
    }

    env.reset()
    for _ in range(max(0, int(args_cli.settle_steps))):
        env.step(zero)

    u = env.unwrapped
    origin = u.scene.env_origins[0].detach().cpu().numpy()
    bottle_env = (u.bottle.data.root_pos_w[0] - u.scene.env_origins[0]).detach().cpu().numpy()
    right_root = (u.right_hand.data.root_pos_w[0] - u.scene.env_origins[0]).detach().cpu().numpy()
    left_root = (u.left_hand.data.root_pos_w[0] - u.scene.env_origins[0]).detach().cpu().numpy()
    right_rot = u.right_hand.data.root_quat_w[0].detach().cpu().numpy()
    left_rot = u.left_hand.data.root_quat_w[0].detach().cpu().numpy()
    fall_h = float(getattr(u.cfg, "fall_height", 0.0))
    cfg_bottle_z = float(u.cfg.bottle_root_init_pos[2])

    print("[SCENE] env_origin_w:", np.round(origin, 4).tolist())
    print("[SCENE] bottle_pos_env:", np.round(bottle_env, 4).tolist(), f"(cfg z={cfg_bottle_z:.3f})")
    print("[SCENE] right_robot_root_env:", np.round(right_root, 4).tolist(), "rot_wxyz", np.round(right_rot, 4).tolist())
    print("[SCENE] left_robot_root_env:", np.round(left_root, 4).tolist(), "rot_wxyz", np.round(left_rot, 4).tolist())
    print("[SCENE] fall_height:", fall_h)

    ok = True
    if bottle_env[2] <= fall_h + 0.02:
        print(f"[FAIL] bottle z={bottle_env[2]:.4f} <= fall_height+margin ({fall_h + 0.02:.4f})")
        ok = False
    if abs(bottle_env[2] - cfg_bottle_z) > 0.08:
        print(f"[WARN] bottle z drift after settle: |z-cfg|={abs(bottle_env[2] - cfg_bottle_z):.4f}")

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "third_person_camera" in u.scene.sensors:
        try:
            rgb = u.scene["third_person_camera"].data.output["rgb"][0].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
            from PIL import Image

            img_path = out_dir / "bottle_cap_scene_third_person.png"
            Image.fromarray(rgb[..., :3]).save(img_path)
            print(f"[SCENE] saved {img_path}")
        except Exception as exc:
            print(f"[WARN] could not save third-person snapshot: {exc}")

    if ok:
        print("[OK] bottle scene layout check passed.")
    else:
        print("[FAIL] bottle scene layout check failed.")

    env.close()
    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
