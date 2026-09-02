#!/usr/bin/env python3
"""Live progress bar for NF validation sweep.

Usage (ViTacLab repo root, in a second terminal while sweep runs)::

    python3 scripts/demo/watch_nf_sweep_progress.py
    python3 scripts/demo/watch_nf_sweep_progress.py --root logs/vitacsim_validation/normal_force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WEIGHT_ORDER = ("W200", "W100", "W050", "W020", "W010")
MODES = ("tacsl", "vitacsim")
BAR_WIDTH = 32


def _expected_jobs(weights: tuple[str, ...], modes: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(w, m) for w in weights for m in modes]


def _is_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-f", "demo_vitacsim_normal_force_validation.py"], text=True)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def _read_progress_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _count_done(root: Path, jobs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    done: list[tuple[str, str]] = []
    for w, m in jobs:
        sp = root / w / m / "summary.json"
        if sp.is_file():
            try:
                data = json.loads(sp.read_text(encoding="utf-8"))
                # Require chamfer-aware summaries from the updated weight model.
                if data.get("output_schema") == "nf_v2_chamfer_rgb":
                    done.append((w, m))
            except json.JSONDecodeError:
                pass
    return done


def _render_bar(done: int, total: int) -> str:
    filled = int(BAR_WIDTH * done / max(total, 1))
    return "[" + "=" * filled + ">" * (1 if done < total and filled < BAR_WIDTH else 0) + " " * max(0, BAR_WIDTH - filled - (1 if done < total and filled < BAR_WIDTH else 0)) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch NF sweep progress.")
    parser.add_argument("--root", type=str, default="logs/vitacsim_validation/normal_force")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    jobs = _expected_jobs(WEIGHT_ORDER, MODES)
    total = len(jobs)
    progress_file = root / "_sweep_progress.json"
    log_file = root / "_full_sweep.log"

    while True:
        prog = _read_progress_file(progress_file)
        done_jobs = _count_done(root, jobs)
        done = len(done_jobs)
        running = _is_running()

        if prog is not None:
            done = max(done, int(prog.get("done", done)))
            current = str(prog.get("current", ""))
            status = str(prog.get("status", "running" if running else "idle"))
        else:
            current = done_jobs[-1][0] + "/" + done_jobs[-1][1] if done_jobs else "-"
            status = "running" if running else ("done" if done >= total else "idle")

        if running and prog and prog.get("current"):
            current = str(prog["current"])
            status = str(prog.get("status", "running"))

        bar = _render_bar(done, total)
        line1 = f"{bar} {done}/{total}  status={status}  current={current}"
        sys.stdout.write("\r\033[K" + line1)
        sys.stdout.flush()

        if done >= total and not running:
            print("\n[DONE] sweep complete.")
            if log_file.is_file():
                print(f"  log: {log_file}")
            print(f"  report: {root / 'SWEEP_REPORT.md'}")
            return 0

        if args.once:
            print()
            return 0

        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
        raise SystemExit(0)
