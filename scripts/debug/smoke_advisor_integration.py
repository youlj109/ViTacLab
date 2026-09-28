"""Exercise integrated task configuration and short zero-action rollouts, without checkpoints.

Run in Isaac Lab with --headless --enable_cameras. This checks runtime contracts,
not task success or the validity of a trained policy.
"""
import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Isaac-UR10eShadowHand-BlindGrasp-Direct-v0',
                    help='One registered task to instantiate.')
parser.add_argument('--steps', type=int, default=2, help='Number of zero-action control steps.')
parser.add_argument('--config-only', action='store_true', help='Check task configs without spawning assets.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.config_only:
    args.enable_cameras = True
if args.steps < 1:
    parser.error('--steps must be positive')
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import ViTacLab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import TacSLSensorPolicyCfg


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.enable_cameras = True
    cfg.seed = 0
    policy_builder = getattr(type(cfg.scene), '_tactile_params', None)
    if policy_builder is not None:
        params = policy_builder()
        print(f'[CONFIG] tactile params: {params}', flush=True)
        if 'BlindGrasp' in args.task or 'BlindRetrieval' in args.task:
            assert TacSLSensorPolicyCfg(**params).strict_target_contact_attribution is True
            if 'BlindRetrieval' in args.task:
                assert params['contact_object_prim_path_expr'].endswith('object_(cube|sphere|cone)')
    if args.config_only:
        print(f'[PASS] config loaded: {args.task}', flush=True)
        return

    env = gym.make(args.task, cfg=cfg)
    try:
        obs, _ = env.reset()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(
                torch.zeros(env.action_space.shape, device=env.unwrapped.device))
            expected = env.unwrapped.single_observation_space['policy'].shape
            assert obs['policy'].shape[1:] == expected, (obs['policy'].shape, expected)
            assert torch.isfinite(obs['policy']).all()
            assert torch.isfinite(reward).all()
        record = env.unwrapped._build_record_dict()
        for key in ('joint_pos', 'tactile_normal_force', 'tactile_shear_force', 'tactile_rgb_image'):
            assert key in record, key
            assert torch.isfinite(record[key]).all(), key
        print(f'[PASS] {args.task}: {args.steps} control steps, obs={tuple(obs["policy"].shape)}, '
              f'record_keys={sorted(record)}', flush=True)
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
        print('[FAIL] integration smoke did not complete.', flush=True)
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        # Fast Kit shutdown may exit directly. Tell Kit the test result first;
        # full extension teardown can segfault on this headless Docker runtime.
        import omni.kit.app
        omni.kit.app.get_app().post_quit(exit_code)
        app.close()
    sys.exit(exit_code)
