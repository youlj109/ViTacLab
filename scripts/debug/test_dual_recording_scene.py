"""CPU-only checks that recording sensor selection retains the local scene.

The scene fixtures mirror Isaac's ordinary Python configuration classes; no
physics or simulator success is claimed by these tests.
"""

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dual_recording_scene", ROOT / "scripts/rsl_rl/full_rl/dual_recording_scene.py"
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


@dataclass
class Scene:
    num_envs: int = 3
    env_spacing: float = 1.7
    replicate_physics: bool = False
    clone_in_fabric: bool = True
    custom_asset: dict = field(default_factory=lambda: {"usd": "local_asset.usd"})

    def describe(self):
        return "local scene"


class TactileScene(Scene):
    @classmethod
    def _tactile_params(cls):
        return {"contact_object_prim_path_expr": "original", "tactile_array_size": (20, 25)}


class CustomScene(Scene):
    def describe(self):
        return "custom local scene"


class CustomTactileScene(TactileScene):
    @classmethod
    def _tactile_params(cls):
        params = super()._tactile_params()
        params["tactile_margin"] = 0.012
        return params


class DualRecordingSceneTests(unittest.TestCase):
    def test_preserves_all_scene_fields_and_custom_methods(self):
        for cls in (Scene, TactileScene, CustomScene, CustomTactileScene):
            for contact_target in ("/World/envs/env_.*/object", "/World/envs/env_.*/plate"):
                with self.subTest(scene=cls.__name__, target=contact_target):
                    original = cls()
                    original.extra_camera = object()
                    previous = dict(vars(original))
                    result = ADAPTER._recording_scene(original, contact_target, TactileScene)
                    self.assertIsNot(result, original)
                    self.assertIs(type(original), cls)
                    self.assertIsInstance(result, cls)
                    self.assertIsInstance(result, TactileScene)
                    self.assertEqual(vars(result), previous)
                    for key, value in previous.items():
                        self.assertIs(getattr(result, key), value)
                    self.assertEqual(result.describe(), original.describe())
                    expected = (original._tactile_params() if hasattr(original, "_tactile_params")
                                else TactileScene._tactile_params())
                    expected["contact_object_prim_path_expr"] = contact_target
                    self.assertEqual(type(result)._tactile_params(), expected)
                    if hasattr(original, "_tactile_params"):
                        self.assertEqual(original._tactile_params()["contact_object_prim_path_expr"], "original")

    def test_sensor_params_are_fresh_and_repeat_selection_preserves_fields(self):
        original = CustomTactileScene()
        first = ADAPTER._recording_scene(original, "object", TactileScene)
        first._tactile_params()["tactile_margin"] = 99
        second = ADAPTER._recording_scene(first, "plate", TactileScene)
        self.assertEqual(second._tactile_params()["tactile_margin"], 0.012)
        self.assertEqual(second._tactile_params()["contact_object_prim_path_expr"], "plate")
        self.assertEqual(vars(second), vars(original))


if __name__ == "__main__":
    unittest.main(verbosity=2)
