#!/usr/bin/env python3
"""Regenerate tactile_rgb_diff_bg.png with tuned bg-subtraction params (no sim re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from validation_rgb_utils import DEFAULT_BG_DIFF, bg_diff_cfg_dict, load_bg_from_path, save_rgb_diff_bg

try:
    from PIL import Image
except ImportError:
    raise SystemExit("PIL required")


def _repo_gelsight_bg() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/gelsight_r15_data/bg.jpg"
    )


def _process_trial(trial_dir: Path, bg, *, dry_run: bool) -> bool:
    summary_path = trial_dir / "summary.json"
    rgb_paths: list[tuple[str, Path]] = []
    if (trial_dir / "tactile_rgb_corrected.png").is_file():
        rgb_paths.append(("tactile_rgb_diff_bg.png", trial_dir / "tactile_rgb_corrected.png"))
        rgb_paths.append(("tactile_rgb_corrected_diff_bg.png", trial_dir / "tactile_rgb_corrected.png"))
    elif (trial_dir / "tactile_rgb.png").is_file():
        rgb_paths.append(("tactile_rgb_diff_bg.png", trial_dir / "tactile_rgb.png"))
    if (trial_dir / "tactile_rgb_depth.png").is_file():
        rgb_paths.append(("tactile_rgb_depth_diff_bg.png", trial_dir / "tactile_rgb_depth.png"))

    if not rgb_paths:
        return False

    if dry_run:
        print(f"[DRY] {trial_dir}")
        return True

    import numpy as np

    for out_name, src in rgb_paths:
        arr = np.asarray(Image.open(src).convert("RGB"))
        save_rgb_diff_bg(trial_dir / out_name, arr, bg, cfg=DEFAULT_BG_DIFF)

    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data.update(bg_diff_cfg_dict(DEFAULT_BG_DIFF))
        summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="logs/vitacsim_validation/v2")
    parser.add_argument("--bg", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    bg_path = Path(args.bg).expanduser() if args.bg else _repo_gelsight_bg()
    bg = load_bg_from_path(bg_path)
    if bg is None:
        print(f"[ERROR] bg not found: {bg_path}", file=sys.stderr)
        return 1

    n = 0
    for summary in sorted(root.glob("**/summary.json")):
        if _process_trial(summary.parent, bg, dry_run=args.dry_run):
            n += 1
    print(f"[INFO] updated diff_bg in {n} trials (cfg={DEFAULT_BG_DIFF})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
