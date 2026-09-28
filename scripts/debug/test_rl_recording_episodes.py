"""Offline regression for generic RL collection boundaries (no Isaac Sim import)."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


PLAYER = Path(__file__).resolve().parents[1] / "rsl_rl/full_rl/play.py"
TREE = ast.parse(PLAYER.read_text())
HELPER = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "_advance_episode_recording"
)
NAMESPACE = {}
exec(compile(ast.Module(body=[HELPER], type_ignores=[]), str(PLAYER), "exec"), NAMESPACE)
advance = NAMESPACE["_advance_episode_recording"]


class EpisodeRecordingTests(unittest.TestCase):
    def test_failed_attempt_does_not_enter_later_success(self):
        buffers, saved, outputs = [{"frame": ["failed"]}], [False], []
        callback = lambda i, buf: outputs.append((i, buf))
        self.assertEqual(advance(buffers, saved, [False], [True], callback, 50), 0)
        self.assertEqual(buffers, [{}])
        buffers[0] = {"frame": ["new episode"]}
        self.assertEqual(advance(buffers, saved, [True], [False], callback, 50), 1)
        self.assertEqual(outputs, [(0, {"frame": ["new episode"]})])

    def test_success_with_auto_reset_can_save_next_episode(self):
        buffers, saved, outputs = [{}], [False], []
        callback = lambda i, buf: outputs.append(buf)
        # More than one success with no explicit/global reset must work.
        for episode in range(50):
            buffers[0] = {"frame": [episode]}
            self.assertEqual(advance(buffers, saved, [True], [True], callback, 50), 1)
            self.assertEqual(saved, [False])
            self.assertEqual(buffers, [{}])
        self.assertEqual(len(outputs), 50)

    def test_persistent_success_saved_only_once_until_done(self):
        buffers, saved, outputs = [{"frame": [1]}], [False], []
        callback = lambda i, buf: outputs.append(buf)
        advance(buffers, saved, [True], [False], callback, 50)
        advance(buffers, saved, [True], [False], callback, 49)
        advance(buffers, saved, [True], [True], callback, 49)
        self.assertEqual(outputs, [{"frame": [1]}])
        self.assertEqual(saved, [False])

    def test_parallel_episode_boundaries_are_independent(self):
        buffers = [{"frame": ["failure"]}, {"frame": ["ongoing"]}]
        saved = [False, False]
        advance(buffers, saved, [False, False], [True, False], lambda *_: None, 10)
        self.assertEqual(buffers, [{}, {"frame": ["ongoing"]}])

    def test_simultaneous_success_does_not_exceed_target(self):
        buffers, saved, outputs = [{"frame": [0]}, {"frame": [1]}], [False, False], []
        count = advance(
            buffers, saved, [True, True], [False, False],
            lambda i, buf: outputs.append(i), 1,
        )
        self.assertEqual(count, 1)
        self.assertEqual(outputs, [0])

    def test_main_snapshots_before_step_not_after_auto_reset(self):
        main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        loop = next(node for node in ast.walk(main) if isinstance(node, ast.While))
        append_lines = [
            node.lineno for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_append_env_frame_to_buffer"
        ]
        step_lines = [
            node.lineno for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ]
        self.assertEqual(len(append_lines), 1)
        self.assertEqual(len(step_lines), 1)
        self.assertLess(append_lines[0], step_lines[0])


if __name__ == "__main__":
    unittest.main()
