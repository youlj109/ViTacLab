"""Exercise recording on unchanged local tasks, without checkpoints.

Run in Isaac Lab with --headless. This checks creation, stepping and canonical
NPZ recording, not task success or compatibility of a trained policy. An actor
observation shape mismatch is reported separately; no task methods are patched.
"""
import argparse
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Isaac-UR10eShadowHand-BlindGrasp-Direct-v0',
                    help='One registered task to instantiate.')
parser.add_argument('--steps', type=int, default=2, help='Number of zero-action control steps.')
parser.add_argument('--config-only', action='store_true', help='Check task configs without spawning assets.')
parser.add_argument('--output', type=Path, help='Optional new NPZ file; existing files are never overwritten.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.config_only:
    args.enable_cameras = True
if args.steps < 1:
    parser.error('--steps must be positive')
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
import ViTacLab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.enable_cameras = True
    cfg.seed = 0
    if args.config_only:
        print(f'[PASS] config loaded: {args.task}', flush=True)
        return

    env = gym.make(args.task, cfg=cfg)
    try:
        obs, _ = env.reset()
        frames = []
        observation_shape_matches = True
        for _ in range(args.steps):
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, reward, terminated, truncated, _ = env.step(
                action)
            expected = env.unwrapped.single_observation_space['policy'].shape
            observation_shape_matches &= obs['policy'].shape[1:] == expected
            assert torch.isfinite(obs['policy']).all()
            assert torch.isfinite(reward).all()
            record = env.unwrapped._build_record_dict()
            for key in ('joint_pos', 'tactile_pos', 'tactile_normal_force', 'tactile_shear_force',
                        'tactile_rgb_image', 'third_person_camera', 'third_person_camera_pos'):
                assert key in record, key
                assert record[key].numel() > 0 and torch.isfinite(record[key]).all(), key
            frame = {key: value[0].detach().cpu().numpy().copy() for key, value in record.items()}
            frame['action'] = action[0].detach().cpu().numpy().copy()
            frames.append(frame)
        if not observation_shape_matches:
            print(f'[WARN] unchanged local task: actor obs={tuple(obs["policy"].shape)}, '
                  f'declared={expected}; policy compatibility NOT verified.', flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('xb') as stream:
                np.savez_compressed(stream, **{key: np.stack([f[key] for f in frames]) for key in frames[0]})
            print(f'[RECORD] {args.output}', flush=True)
        print(f'[PASS recording only] {args.task}: {args.steps} steps, obs={tuple(obs["policy"].shape)}, '
              f'actor_shape_matches={observation_shape_matches}, record_keys={sorted(record)}', flush=True)
    except Exception:
        # Print before Kit shutdown; some Kit builds exit during app.close().
        print('[FAIL] runtime smoke raised an exception:', flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        env.close()


if __name__ == '__main__':
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        print('[FAIL] collection smoke did not complete.', flush=True)
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        # Fast Kit shutdown may exit directly. Tell Kit the test result first;
        # full extension teardown can segfault on this headless Docker runtime.
        import omni.kit.app
        omni.kit.app.get_app().post_quit(exit_code)
        app.close()
    sys.exit(exit_code)
