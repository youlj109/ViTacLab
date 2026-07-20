"""Short smoke test: Pickup env boot + _build_record_dict schema check."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pickup env smoke test.")
parser.add_argument("--task", type=str, default="Isaac-UR10eShadowHand-Pickup-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ViTacLab.tasks  # noqa: F401

REQUIRED_RECORD_KEYS = (
    "joint_pos",
    "tactile_pos",
    "tactile_normal_force",
    "tactile_shear_force",
    "tactile_rgb_image",
    "third_person_camera",
    "third_person_camera_pos",
)


def main() -> int:
    print("[INFO] smoke: parsing env cfg...", flush=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.enable_cameras = True
    print("[INFO] smoke: creating env (may take ~30s)...", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    print(f"[INFO] smoke: env created: {type(unwrapped).__name__}", flush=True)

    build_record = getattr(unwrapped, "_build_record_dict", None)
    if build_record is None:
        print("[FAIL] env has no _build_record_dict")
        return 1

    env.reset()
    print("[INFO] smoke: reset done, stepping...", flush=True)
    for step in range(int(args.max_steps)):
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        env.step(actions)
        if step == int(args.max_steps) - 1:
            record = build_record()
            missing = [k for k in REQUIRED_RECORD_KEYS if k not in record]
            if missing:
                print(f"[FAIL] record missing keys: {missing}")
                return 1
            print(f"[OK] smoke passed after {args.max_steps} steps; record keys={sorted(record.keys())}")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
