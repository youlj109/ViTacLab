#!/usr/bin/env python3
"""CPU-only integration tests for demo output preservation and launcher wiring.

Uses a stub simulator/offline interpreter; does not claim simulation correctness.
Run: python3 scripts/debug/test_demo_launchers.py
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    "bi_blind_bin_drop", "bistab", "blind_classification", "blind_grasp",
    "blind_retrieval", "bottle", "data", "forge", "forge_franka_rl",
    "handover", "inhand", "pour",
)


def stub_main():
    args = sys.argv[2:]
    with open(os.environ["DEMO_TEST_CALLS"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps(args) + "\n")

    def option(*names):
        for name in names:
            if name in args:
                return Path(args[args.index(name) + 1])
        return None

    records = option("--record-path", "--record_path", "--data_path")
    if records is not None:
        records.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            (records / f"episode_{index:03d}_completed.npz").write_bytes(b"stub npz")
    report = option("--json-out")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}", encoding="utf-8")
    if "visualize_play_record_npz.py" in args[0]:
        videos = option("--out_dir")
        videos.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            for channel in ("demo_panel", "third_person", "tactile_rgb", "tactile_force_field"):
                (videos / f"episode_{index:03d}_{channel}.mp4").write_bytes(b"stub video")


class DemoLauncherTests(unittest.TestCase):
    def test_full_and_visualize_preserve_existing_outputs(self):
        for name in RUNNERS:
            for mode in ("full", "visualize"):
                with self.subTest(runner=name, mode=mode), tempfile.TemporaryDirectory(prefix="vitaclab-demo-test-") as temp:
                    work = Path(temp)
                    records, videos = work / "records", work / "videos"
                    records.mkdir()
                    videos.mkdir()
                    originals = {
                        records / "episode_000_completed.npz": b"original recording",
                        videos / "episode_000_demo_panel.mp4": b"original video",
                        work / "summary.txt": b"original summary",
                        work / "validation.json": b"original validation",
                    }
                    for path, content in originals.items():
                        path.write_bytes(content)
                    checkpoint = work / "model_9999.pt"
                    checkpoint.touch()
                    stub = work / "stub-python"
                    stub.write_text(
                        "#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable)
                        + " " + shlex.quote(str(Path(__file__).resolve())) + ' --stub "$@"\n',
                        encoding="utf-8",
                    )
                    stub.chmod(0o755)
                    env = dict(os.environ)
                    for variable in ("TASK", "PHASE", "SHAPE_LABEL", "TARGET_LABEL", "TRAJECTORY", "KEYFRAME_JSON"):
                        env.pop(variable, None)
                    env.update(
                        MODE=mode, SKIP_COLLECT="0", SKIP_SENSOR="1", SKIP_VALIDATE="0",
                        PYTHON_SIM=str(stub), PYTHON_OFFLINE=str(stub),
                        RECORD_DIR=str(records), VIDEO_DIR=str(videos),
                        SUMMARY_FILE=str(work / "summary.txt"),
                        VALIDATION_JSON=str(work / "validation.json"),
                        SCENE_CHECK_DIR=str(work / "scene"), CHECKPOINT=str(checkpoint),
                        DEMO_TEST_CALLS=str(work / "calls.jsonl"),
                    )
                    filename = f"demo_{name}_collection_replay.sh"
                    result = subprocess.run(
                        ["bash", str(ROOT / "bash_command" / filename)],
                        cwd=work, env=env, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    for path, content in originals.items():
                        self.assertEqual(path.read_bytes(), content, str(path))
                    calls = [json.loads(line) for line in (work / "calls.jsonl").read_text().splitlines()]
                    validate = next(args for args in calls if "validate_play_record_npz.py" in args[0])
                    used_records = Path(validate[1])
                    if mode == "full":
                        self.assertEqual(used_records.parent, records)
                        self.assertNotEqual(used_records, records)
                    else:
                        self.assertEqual(used_records, records)
                    exported = [path for path in videos.rglob("*.mp4") if path.parent != videos]
                    self.assertEqual(len(exported), 8, result.stdout)
                    self.assertTrue(any("episode_000" in path.name for path in exported))
                    self.assertTrue(any("episode_001" in path.name for path in exported))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--stub"]:
        stub_main()
    else:
        unittest.main()
