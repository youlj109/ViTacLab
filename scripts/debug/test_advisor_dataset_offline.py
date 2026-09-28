#!/usr/bin/env python3
"""Synthetic NPZ/conversion regressions; no Isaac Sim, recordings, or GPU needed.

Run from any directory with a NumPy environment. Canonical ViTacDP pose conversion
and Zarr round trips are additionally exercised when their dependencies exist.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from policy.prepare_dataset import _episode_payload, build_parser
from scripts.debug.validate_play_record_npz import validate_npz

HAS_POSES = importlib.util.find_spec("pytorch3d") is not None
HAS_ZARR = importlib.util.find_spec("zarr") is not None and importlib.util.find_spec("numcodecs") is not None


def fixture(*, canonical: bool = False) -> dict[str, np.ndarray]:
    frames, sensors = 5, 5
    values = {
        "joint_pos": np.arange(frames * 30, dtype=np.float32).reshape(frames, 30),
        # Classification has a 33D action even with 30D proprioception.
        "action": np.arange(frames * 33, dtype=np.float32).reshape(frames, 33),
        "tactile_normal_force": np.ones((frames, sensors, 2, 3, 1), dtype=np.float32),
        "tactile_shear_force": np.broadcast_to(np.array([2, 3], dtype=np.float32), (frames, sensors, 2, 3, 2)).copy(),
        "tactile_rgb_image": np.full((frames, sensors, 4, 6, 3), 127, dtype=np.uint8),
        "third_person_camera": np.full((frames, 8, 12, 4), 80, dtype=np.uint8),
    }
    if canonical:
        values["third_person_camera_pos"] = np.zeros((frames, 1, 7), dtype=np.float32)
        values["third_person_camera_pos"][..., 3] = 1
        values["tactile_pos"] = np.zeros((frames, sensors, 7), dtype=np.float32)
        values["tactile_pos"][..., 3] = 1
    return values


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vitaclab-dataset-test-")
        self.base = Path(self.temp.name)
        self.input = self.base / "episodes"
        self.input.mkdir()
        self.output = self.base / "data.zarr"
        self.path = self.input / "episode_000.npz"

    def tearDown(self):
        self.temp.cleanup()

    def save(self, values, name="episode_000.npz"):
        path = self.input / name
        np.savez(path, **values)
        return path

    def args(self, *options):
        return build_parser().parse_args([
            str(self.input), str(self.output), "--policy", "ViTacDP",
            "--profile", "tactile_only", *options,
        ])

    def cli(self, *options, ok=True):
        result = subprocess.run([
            sys.executable, str(ROOT / "policy/prepare_dataset.py"),
            str(self.input), str(self.output), "--policy", "ViTacDP",
            "--profile", "tactile_only", *options,
        ], cwd=ROOT, text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_validator_canonical_and_tactile_only(self):
        self.save(fixture(canonical=True))
        self.assertTrue(validate_npz(str(self.path))["ok"])
        self.assertFalse(validate_npz(str(self.path), profile="tactile_only")["ok"])
        self.save(fixture())
        self.assertTrue(validate_npz(str(self.path), profile="tactile_only")["ok"])

    def test_validator_detects_bad_shapes_lengths_channels_and_nonfinite(self):
        for key, bad in (
            ("joint_pos", np.array(0)),
            ("action", np.zeros((4, 33))),
            ("tactile_normal_force", np.zeros((5, 5, 2, 3, 2))),
            ("tactile_shear_force", np.zeros((5, 5, 2, 4, 2))),
            ("tactile_rgb_image", np.zeros((5, 4, 4, 6, 3))),
            ("joint_pos", np.full((5, 30), np.nan)),
        ):
            with self.subTest(key=key, shape=bad.shape):
                values = fixture()
                values[key] = bad
                self.save(values)
                report = validate_npz(str(self.path), profile="tactile_only")
                self.assertFalse(report["ok"])
                self.assertTrue(report["errors"])

    def test_validator_rejects_pickle_and_vision_geometry_leakage(self):
        for key, value in (
            ("tactile_pos", np.zeros((5, 5, 7))),
            ("object_pos", np.zeros((5, 3))),
            ("twist_camera", np.zeros((5, 8, 12, 3), dtype=np.uint8)),
            ("metadata", np.array([{"unsafe": "object"}], dtype=object)),
        ):
            with self.subTest(key=key):
                values = fixture()
                values[key] = value
                self.save(values)
                self.assertFalse(validate_npz(str(self.path), profile="tactile_only")["ok"])

    def test_tactile_only_rgb_drops_vision_geometry(self):
        values = fixture(canonical=True)
        values["object_pos"] = np.zeros((5, 3))
        self.save(values)
        cameras, payload = _episode_payload(self.path, self.args(), None)
        self.assertEqual(cameras, [])
        self.assertEqual(set(payload), {"state", "action", "tac_rgb"})
        np.testing.assert_array_equal(payload["state"], values["joint_pos"][:-1])
        np.testing.assert_array_equal(payload["action"], values["joint_pos"][1:])
        np.testing.assert_array_equal(payload["tac_rgb"], values["tactile_rgb_image"][:-1])

    def test_force_channels_recorded_actions_and_stride(self):
        values = fixture()
        self.save(values)
        _, payload = _episode_payload(self.path, self.args(
            "--tactile-type", "force", "--action-source", "recorded_action", "--stride", "2",
        ), None)
        self.assertEqual(payload["state"].shape, (2, 30))
        self.assertEqual(payload["action"].shape, (2, 33))
        np.testing.assert_array_equal(payload["action"], values["action"][[0, 2]])
        np.testing.assert_array_equal(payload["tac_force"][0, 0, 0, 0], [1, 2, 3])

    def test_stride_does_not_hide_raw_frame_mismatch(self):
        values = fixture()
        values["tactile_rgb_image"] = np.zeros((6, 5, 4, 6, 3), dtype=np.uint8)
        self.save(values)
        with self.assertRaisesRegex(ValueError, "before stride"):
            _episode_payload(self.path, self.args("--stride", "2"), None)

    def test_converter_rejects_invalid_tactile_channels_and_nan_camera(self):
        values = fixture()
        values["tactile_rgb_image"] = np.zeros((5, 5, 4, 6, 2), dtype=np.uint8)
        self.save(values)
        with self.assertRaisesRegex(ValueError, "final dimension"):
            _episode_payload(self.path, self.args(), None)
        values = fixture()
        values["third_person_camera"] = np.full((5, 8, 12, 3), np.nan)
        self.save(values)
        with self.assertRaisesRegex(ValueError, "finite"):
            _episode_payload(self.path, self.args("--policy", "Diffusion_Policy", "--profile", "canonical"), None)

    def test_camera_only_canonical_retains_alignment(self):
        values = fixture()
        self.save(values)
        cameras, payload = _episode_payload(self.path, self.args(
            "--policy", "Diffusion_Policy", "--profile", "canonical", "--action-source", "recorded_action",
        ), None)
        self.assertEqual(cameras, ["third_person_camera"])
        self.assertEqual(set(payload), {"state", "action", "head_camera"})
        self.assertEqual(payload["action"].shape, (4, 33))
        self.assertEqual(payload["head_camera"].shape, (4, 8, 12, 3))

    @unittest.skipUnless(HAS_POSES, "pytorch3d not installed; run in a policy pose environment")
    def test_canonical_vitacdp_rgb_and_force_pose_conversion(self):
        self.save(fixture(canonical=True))
        for tactile in ("rgb", "force"):
            with self.subTest(tactile=tactile):
                _, payload = _episode_payload(self.path, self.args(
                    "--profile", "canonical", "--tactile-type", tactile,
                ), None)
                self.assertEqual(payload["camera_pos"].shape, (4, 1, 9))
                self.assertEqual(payload["tac_pos"].shape, (4, 5, 9))
                np.testing.assert_allclose(payload["camera_pos"][0, 0, 3:], [1, 0, 0, 0, 1, 0])
                self.cli("--profile", "canonical", "--tactile-type", tactile, "--validate-only")

    def test_validate_only_cli_without_writing(self):
        self.save(fixture())
        for tactile in ("rgb", "force"):
            report = json.loads(self.cli("--tactile-type", tactile, "--validate-only").stdout)
            self.assertEqual(report["transitions"], 4)
            self.assertEqual(report["camera_keys"], [])
            self.assertFalse(self.output.exists())
        self.cli("--policy", "Diffusion_Policy", "--profile", "canonical", "--validate-only")
        self.cli("--policy", "Diffusion_Policy", "--validate-only", ok=False)

    def test_invalid_episodes_do_not_touch_existing_output(self):
        values = fixture()
        self.save(values)
        values["tactile_rgb_image"] = values["tactile_rgb_image"].astype(np.float32)
        self.save(values, "episode_001.npz")
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.touch()
        result = self.cli("--overwrite", ok=False)
        self.assertIn("Schema dtype changed", result.stderr)
        self.assertTrue(sentinel.exists())

    def test_overwrite_cannot_delete_source_episodes(self):
        self.save(fixture())
        self.output = self.base
        result = self.cli("--overwrite", ok=False)
        self.assertIn("must not be the input directory or its ancestor", result.stderr)
        self.assertTrue(self.path.exists())

    @unittest.skipUnless(HAS_ZARR, "zarr/numcodecs not installed")
    def test_zarr_round_trip_both_tactile_representations(self):
        import zarr

        values = fixture()
        self.save(values)
        self.save(values, "episode_001.npz")
        for tactile in ("rgb", "force"):
            with self.subTest(tactile=tactile):
                self.output = self.base / f"{tactile}.zarr"
                self.cli("--tactile-type", tactile, "--action-source", "recorded_action")
                root = zarr.open(str(self.output), mode="r")
                self.assertEqual(set(root["data"].keys()), {"state", "action", f"tac_{tactile}"})
                np.testing.assert_array_equal(root["meta/episode_ends"][:], [4, 8])
                np.testing.assert_array_equal(root["data/action"][:4], values["action"][:-1])
                self.assertEqual(root["data/state"].shape, (8, 30))
                self.assertEqual(root["data/action"].shape, (8, 33))
                self.assertEqual(json.loads(root.attrs["vitaclab_schema"])["profile"], "tactile_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
