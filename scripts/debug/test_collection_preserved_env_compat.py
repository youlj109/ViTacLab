#!/usr/bin/env python3
"""Offline collector compatibility checks; no Isaac or environment substitutions.

Extract only pure helpers so the real command-line scripts still import Isaac
normally. These checks do not assert successful physics or tactile observations.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "data_collection" / "full_trajectory"


def load_helper(filename, name, **namespace):
    tree = ast.parse((SCRIPTS / filename).read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace.setdefault("Any", Any)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SCRIPTS / filename), "exec"), namespace)
    return namespace[name]


class PreservedEnvCollectorTests(unittest.TestCase):
    def sampler(self, labels):
        resets = []
        labels = iter(labels)
        env = SimpleNamespace(shape_labels=[None])

        def reset(target_env, **kwargs):
            self.assertIs(target_env, env)
            resets.append(kwargs)
            label = next(labels)
            env.shape_labels[0] = SimpleNamespace(item=lambda: label)

        helper = load_helper("play_blind_classification.py", "_reset_for_shape", _reset_and_warmup=reset)
        return helper, env, resets

    def test_fixed_shape_selection_uses_normal_reset(self):
        helper, env, resets = self.sampler([2, 0, 1])
        self.assertEqual(helper(env, 33, 8, 1, 0), 1)
        self.assertEqual(len(resets), 3)
        self.assertEqual(resets[-1], {"act_dim": 33, "settle_steps": 8})

    def test_random_shape_preserves_actual_label(self):
        helper, env, resets = self.sampler([2])
        self.assertEqual(helper(env, 33, 8, -1, 0), 2)
        self.assertEqual(len(resets), 1)

    def test_unavailable_shape_does_not_fake_requested_label(self):
        helper, env, resets = self.sampler([0, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "after 3 resets"):
            helper(env, 33, 8, 2, 0, max_attempts=3)
        self.assertEqual(env.shape_labels[0].item(), 0)
        self.assertEqual(len(resets), 3)

    def test_invalid_shape_fails_without_running_simulation(self):
        helper, env, resets = self.sampler([])
        with self.assertRaises(ValueError):
            helper(env, 33, 8, 3, 0)
        self.assertFalse(resets)

    def test_saved_subset_is_not_claimed_as_actor_contract(self):
        meta = load_helper("play_single_phase.py", "_recording_scope_metadata")()
        self.assertFalse(meta["policy_tactile_only_enforced"])
        self.assertTrue(meta["record_subset_excludes_object_state"])
        self.assertEqual(meta["policy_observations"], "unchanged_registered_environment")
        self.assertNotIn("policy_input_contract", meta)

    def test_collectors_do_not_replace_environment_callbacks_or_finger_effort(self):
        forbidden = {"_apply_action", "_get_rewards", "_get_observations", "effort_limit_sim",
                     "episode_length_s", "act_moving_average", "goal_resample_time_range_s",
                     "object_reset_pos_x_range", "object_reset_pos_y_range", "_use_rl_control"}
        for filename in ("play_single_phase.py", "collect_blind_grasp_privileged.py",
                         "play_blind_classification.py", "play_blind_retrieval.py", "record_single.py"):
            tree = ast.parse((SCRIPTS / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    self.assertNotIn(node.attr, forbidden, f"{filename} overwrites {node.attr}")

    def test_reset_boundaries_are_detected_before_post_reset_recording(self):
        def asarray(value):
            return SimpleNamespace(any=lambda: any(value) if isinstance(value, list) else bool(value))
        helper = load_helper("play_single_phase.py", "_step_has_ended",
                             torch=SimpleNamespace(is_tensor=lambda value: False),
                             np=SimpleNamespace(asarray=asarray))
        self.assertFalse(helper(None))
        self.assertFalse(helper(({}, 0, [False], [False], {})))
        self.assertTrue(helper(({}, 0, [False], [True], {})))
        self.assertTrue(helper(({}, 0, [True], [False], {})))
        self.assertFalse(helper(({}, 0, False, {"success": True})))
        self.assertTrue(helper(({}, 0, True, {})))


if __name__ == "__main__":
    unittest.main()
