"""Exercise actual task methods without starting Kit or loading USD assets.

The AST loader keeps the real observation/dimension/constructor methods and
replaces only the Isaac base constructor (which records the space dimension).
Run with a Python containing PyTorch, e.g. Isaac Sim's /isaac-sim/python.sh.
This is a tensor/schema regression test, not a physics or task-success test.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "source/ViTacLab/ViTacLab/tasks/direct"


class _SpaceInitialized(Exception):
    pass


class _IsaacBase:
    def __init__(self, cfg, *args, **kwargs):
        self.initialized_space_dim = cfg.observation_space
        if getattr(cfg, "test_return_from_base", False):
            self.cfg = cfg
            self.num_envs = 2
            self.num_actions = 30
            self.device = "cpu"
            self.actions = torch.zeros(2, 30)
            class Scene:
                sensors = cfg.test_sensors

                def __getitem__(self, key):
                    return self.sensors[key]
            self.scene = Scene()
            return
        raise _SpaceInitialized


NS = {
    "torch": torch,
    "UR10eShadowHandDirectBaseEnv": _IsaacBase,
    "TACTILE_NORMAL_DIM": 2500,
    "TACTILE_SHEAR_DIM": 5000,
    "TACTILE_SENSOR_NAMES": tuple(range(5)),
    "VisualizationMarkers": lambda cfg: None,
    "_unscale": lambda x, lower, upper: (2 * x - upper - lower) / (upper - lower),
}


def load_class(path, name, methods):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.decorator_list = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods] or [ast.Pass()]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), NS)
    return NS[name]


METHODS = {"__init__", "_policy_observation_dim", "_get_observations",
           "_accumulate_episode_success_stats", "get_episode_success_rate",
           "_init_pickup_tactile_buffers", "_update_tactile_data"}
PICKUP = load_class(TASKS / "simple_dexhand/hand_pickup/hand_pickup_env.py", "UR10eShadowHandPickupEnv", METHODS)
CLASSIFY = load_class(TASKS / "difficult_dexhand/blind_classification/blind_classification_env.py", "UR10eShadowHandBlindClassificationEnv", METHODS)
RETRIEVE = load_class(TASKS / "difficult_dexhand/blind_retrieval/blind_retrieval_env.py", "UR10eShadowHandBlindRetrievalEnv", METHODS)
GRASP = load_class(TASKS / "difficult_dexhand/blind_grasp/blind_grasp_env.py", "UR10eShadowHandBlindGraspEnv", {"_get_rewards"})


def fixture(cls, geometry, full):
    env = object.__new__(cls)
    env.cfg = SimpleNamespace(include_object_state_in_policy_obs=geometry, use_full_tactile_obs=full,
                              num_shape_classes=3, vel_obs_scale=0.2)
    env._compute_intermediate_values = lambda: None
    env._update_tactile_data = lambda: None
    env.robot_dof_pos = torch.arange(60, dtype=torch.float32).reshape(2, 30) / 100
    env.robot_dof_vel = env.robot_dof_pos + 0.1
    env.robot_dof_lower_limits = -torch.ones(2, 30)
    env.robot_dof_upper_limits = torch.ones(2, 30)
    env.actuated_dof_indices = list(range(30))
    env.object_pos = torch.full((2, 3), 2.0)
    env.object_rot = torch.full((2, 4), 3.0)
    env.goal_object_pos = torch.full((2, 3), 5.0)
    env.actions = torch.full((2, 33 if cls is CLASSIFY else 30), 0.4)
    env._action_rate = torch.full((2, 30), 0.5)
    env._robot_action_dim = 30
    env._target_cmd_onehot = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    env._tactile_normal_force = torch.full((2, 2500), 0.01)
    env._tactile_shear_force = torch.full((2, 5000), 0.02)
    env._tactile_normal_mean = torch.full((2, 5), 0.01)
    env._tactile_shear_mean = torch.full((2, 10), 0.02)
    return env


class BlindObservationTests(unittest.TestCase):
    def test_constructor_buffers_after_sensor_registration_and_without_cameras(self):
        for sensors_enabled in (False, True):
            for full in (False, True):
                cfg = SimpleNamespace(
                    include_object_state_in_policy_obs=False, use_full_tactile_obs=full,
                    vel_obs_scale=0.2, goal_marker_cfg=None, test_return_from_base=True,
                    scene=SimpleNamespace(_tactile_params=lambda: {"tactile_array_size": (20, 25)}),
                    test_sensors={},
                )
                if sensors_enabled:
                    cfg.test_sensors = {
                        name: SimpleNamespace(
                            cfg=SimpleNamespace(tactile_array_size=(20, 25)),
                            data=SimpleNamespace(tactile_normal_force=torch.full((2, 500), 0.01),
                                                 tactile_shear_force=torch.full((2, 500, 2), 0.02)),
                        ) for name in range(5)
                    }
                env = PICKUP(cfg)
                self.assertEqual(env._tactile_normal_force.shape, (2, 2500))
                self.assertEqual(env._tactile_shear_force.shape, (2, 5000))
                self.assertEqual(env._num_tactile_sensors, 5)
                template = fixture(PICKUP, False, full)
                for name in ("robot_dof_pos", "robot_dof_vel", "robot_dof_lower_limits",
                             "robot_dof_upper_limits", "actuated_dof_indices"):
                    setattr(env, name, getattr(template, name))
                env._compute_intermediate_values = lambda: None
                obs = env._get_observations()["policy"]
                self.assertEqual(tuple(obs.shape), (2, cfg.observation_space))
                self.assertEqual(bool(torch.any(obs[:, 120:] != 0)), sensors_enabled)

    def test_dimensions_before_gym_initialization_and_geometry_exclusion(self):
        for cls, geom_dim, cmd_dim in ((PICKUP, 13, 0), (CLASSIFY, 7, 0), (RETRIEVE, 13, 3)):
            for geometry in (False, True):
                for full in (False, True):
                    with self.subTest(task=cls.__name__, geometry=geometry, full=full):
                        env = fixture(cls, geometry, full)
                        expected = 120 + cmd_dim + (geom_dim if geometry else 0) + (7500 if full else 15)
                        self.assertEqual(cls._policy_observation_dim(env.cfg), expected)
                        # Execute actual constructors up to the Isaac base call.
                        instance = object.__new__(cls)
                        with self.assertRaises(_SpaceInitialized):
                            cls.__init__(instance, env.cfg)
                        self.assertEqual(instance.initialized_space_dim, expected)
                        obs = env._get_observations()["policy"]
                        self.assertEqual(tuple(obs.shape), (2, expected))
                        env.object_pos += 10
                        env.object_rot += 10
                        env.goal_object_pos += 10
                        changed = env._get_observations()["policy"]
                        self.assertEqual(torch.equal(obs, changed), not geometry)

    def test_legacy_order_and_class_logits_exclusion(self):
        for cls in (PICKUP, CLASSIFY, RETRIEVE):
            env = fixture(cls, True, False)
            chunks = [env.robot_dof_pos, 0.2 * env.robot_dof_vel]
            if cls is RETRIEVE:
                chunks.append(env._target_cmd_onehot)
            chunks.extend([env.object_pos, env.object_rot])
            if cls is not CLASSIFY:
                chunks.extend([env.goal_object_pos, env.goal_object_pos - env.object_pos])
            chunks.extend([env.actions[:, :30], env._action_rate, env._tactile_normal_mean, env._tactile_shear_mean])
            expected = torch.cat(chunks, -1)
            torch.testing.assert_close(env._get_observations()["policy"], expected)
            if cls is CLASSIFY:
                env.actions[:, 30:] = 99
                torch.testing.assert_close(env._get_observations()["policy"], expected)

    def test_blind_retrieval_keeps_target_command(self):
        env = fixture(RETRIEVE, False, True)
        obs = env._get_observations()["policy"]
        torch.testing.assert_close(obs[:, 60:63], env._target_cmd_onehot)

    def test_grasp_success_needs_live_contact_and_lift_and_counts_episodes(self):
        env = object.__new__(GRASP)
        env.cfg = SimpleNamespace(
            tactile_contact_threshold=1e-4, tactile_load_threshold=5e-5,
            tactile_success_min_fingers=2, tactile_success_hold_steps=20,
            tactile_contact_reward_weight=0.5, tactile_multi_finger_reward_weight=1.,
            tactile_load_reward_weight=2., tactile_slip_penalty_weight=-0.25,
            action_l2_weight=-0.005, action_rate_l2_weight=-0.005,
            benchmark_lift_height=0.35, fall_height=0.02,
            max_consecutive_success=0, episode_success_ema_alpha=0.15,
        )
        env.num_envs = 3
        env.device = "cpu"
        env._update_tactile_data = lambda: None
        env._compute_intermediate_values = lambda: None
        env._out_of_bounds = lambda positions: torch.zeros(len(positions), dtype=torch.bool)
        # Lift without contact, live multi-finger lift, contact without lift.
        env.object_pos = torch.tensor([[0.54, 0., 0.4], [0.54, 0., 0.4], [0.54, 0., 0.1]])
        nf = torch.zeros(3, 5, 500)
        sf = torch.zeros(3, 5, 500, 2)
        nf[1:, :2, 0] = 2e-4
        sf[1:, :2, 0, 0] = 1e-4
        env._tactile_normal_force = nf.reshape(3, -1)
        env._tactile_shear_force = sf.reshape(3, -1)
        env._prev_tactile_shear_peak = torch.zeros(3, 5)
        env.actions = torch.zeros(3, 30)
        env._action_rate = torch.zeros(3, 30)
        env._tactile_success_streak = torch.zeros(3, dtype=torch.long)
        env._success_streak = torch.zeros(3, dtype=torch.long)
        # Stale tactile success must not make an airborne, unheld object succeed.
        env.tactile_successes = torch.ones(3, dtype=torch.bool)
        env.successes = torch.zeros(3)
        env.object_slot_index = torch.zeros(3)
        env.extras = {}
        env._episode_success_count = 0
        env._episode_total_count = 0
        env._episode_success_rate_ema = 0.
        env.reset_terminated = torch.zeros(3, dtype=torch.bool)
        env.reset_time_outs = torch.ones(3, dtype=torch.bool)
        reward = env._get_rewards()
        torch.testing.assert_close(env.successes, torch.tensor([0., 1., 0.]))
        # Same touch, different lift: privileged geometry must not affect reward.
        torch.testing.assert_close(reward[1], reward[2])
        env._accumulate_episode_success_stats(torch.arange(3))
        self.assertEqual((env._episode_success_count, env._episode_total_count), (1, 3))
        env._get_rewards()
        self.assertAlmostEqual(env.extras["log"]["episode_success_rate_all_time"], 1 / 3)
        self.assertAlmostEqual(env.extras["log"]["episode_success_rate"], 0.05)

    def test_actual_tactile_builder_target_and_attribution(self):
        NS["UR10eShadowHandTacSLSceneCfg"] = object
        methods = {"_tactile_params", "_tactile_policy"}
        load_class(TASKS / "simple_dexhand/hand_pickup/hand_pickup_env_cfg.py", "UR10eShadowHandPickupSceneCfg", methods)
        grasp = load_class(TASKS / "difficult_dexhand/blind_grasp/blind_grasp_env_cfg.py", "UR10eShadowHandBlindGraspSceneCfg", methods)
        retrieve = load_class(TASKS / "difficult_dexhand/blind_retrieval/blind_retrieval_env_cfg.py", "UR10eShadowHandBlindRetrievalSceneCfg", methods)
        classify = load_class(TASKS / "difficult_dexhand/blind_classification/blind_classification_env_cfg.py", "UR10eShadowHandBlindClassificationSceneCfg", methods)
        source = ROOT / "source/ViTacLab/ViTacLab/assets/robot/ur10e_shadowhand_direct_base_single/ur10e_shadowhand_direct_base_cfg.py"
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "build_ur10e_shadowhand_tactile_sensor_cfgs")
        module = ast.Module(body=[node], type_ignores=[])
        NS.update(VisuoTactileSensorCfg=SimpleNamespace, TiledCameraCfg=SimpleNamespace,
                  GELSIGHT_R15_CFG=SimpleNamespace(image_height=128, image_width=128),
                  UR10E_SHADOWHAND_TACTILE_SENSOR_SPECS=[("tactile_sensor_ff", "ff")])
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), NS)
        for cls in (grasp, retrieve, classify):
            sensor = NS["build_ur10e_shadowhand_tactile_sensor_cfgs"](cls())["tactile_sensor_ff"]
            suffix = "object" if cls is grasp else "object_(cube|sphere|cone)"
            self.assertEqual(sensor.contact_object_prim_path_expr, "/World/envs/env_.*/" + suffix)
            self.assertTrue(sensor.strict_target_contact_attribution)
            self.assertTrue(sensor.use_physx_sparse_anchors)
            self.assertFalse(sensor.require_physx_sparse_anchors)  # unchanged main default


if __name__ == "__main__":
    unittest.main(verbosity=2)
