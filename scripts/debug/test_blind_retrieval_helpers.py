#!/usr/bin/env python3
"""CPU-only checks for retrieval's task-local helpers; no Isaac Sim imports."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "data_collection" / "full_trajectory"
PHASE = ast.parse((SCRIPTS / "play_single_phase.py").read_text())
RETRIEVAL = ast.parse((SCRIPTS / "play_blind_retrieval.py").read_text())


def extracted_helpers():
    names = {
        "_target_object", "_palm_pos_env", "_pick_close_frame",
        "_settle_after_object_write", "_min_fingertip_dist_to_target", "_is_pick_close_frame",
    }
    body = [node for tree in (PHASE, RETRIEVAL) for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "np": np, "_np": np.asarray, "Any": Any,
        "torch": SimpleNamespace(zeros=lambda *shape, **_: np.zeros(shape)),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "retrieval_helpers", "exec"), namespace)
    return namespace


class RetrievalHelperTests(unittest.TestCase):
    def test_shared_helper_imports_exist(self):
        exported = {node.name for node in PHASE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for node in PHASE.body:
            if isinstance(node, ast.Assign):
                exported.update(target.id for target in node.targets if isinstance(target, ast.Name))
        for filename in ("play_blind_retrieval.py", "play_blind_classification.py", "collect_blind_grasp_privileged.py"):
            tree = ast.parse((SCRIPTS / filename).read_text())
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == "play_single_phase":
                    for alias in node.names:
                        self.assertIn(alias.name, exported, f"{filename}: missing {alias.name}")

    def test_geometry_reads_target_without_mutating_primary_object(self):
        helper = extracted_helpers()
        sentinel = object()
        env = SimpleNamespace(
            object=sentinel, target_object_id=np.array([1]),
            _shape_objects=[
                SimpleNamespace(data=SimpleNamespace(root_pos_w=np.array([[100, 0, 0]]))),
                SimpleNamespace(data=SimpleNamespace(root_pos_w=np.array([[11, 0, 0]]))),
            ],
            robot=SimpleNamespace(
                body_names=["rh_palm", "rh_ffdistal", "rh_thdistal"],
                data=SimpleNamespace(body_pos_w=np.array([[[10, 1, 0], [11, 0, 0.02], [11, 0, 0.1]]])),
            ),
            scene=SimpleNamespace(env_origins=np.array([[10, 0, 0]])),
        )
        self.assertAlmostEqual(helper["_min_fingertip_dist_to_target"](env, 0), 0.02)
        self.assertIs(env.object, sentinel)
        np.testing.assert_allclose(helper["_palm_pos_env"](env, 0), [0, 1, 0])
        env.robot.body_names = ["rh_palm"]
        self.assertEqual(helper["_min_fingertip_dist_to_target"](env, 0), float("inf"))

    def test_frame_selection_and_settle_do_not_reset_calibration(self):
        helper = extracted_helpers()
        frames = [{"name": "Start01"}, {"name": "Pick02_03"}, {"name": "Finish02"}]
        self.assertIs(helper["_pick_close_frame"](frames), frames[1])
        self.assertIsNone(helper["_pick_close_frame"]([]))
        steps = []
        env = SimpleNamespace(num_envs=2, device="cpu", step=steps.append)
        helper["_settle_after_object_write"](env, 30, 3)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].shape, (2, 30))
        np.testing.assert_array_equal(steps[0], 0)

    def test_retrieval_uses_current_wrist_helper_signature(self):
        calls = [node for node in ast.walk(RETRIEVAL)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "_wrist_pos_for_frame"]
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(len(call.args), 1)
            self.assertFalse(call.keywords)


if __name__ == "__main__":
    unittest.main()
