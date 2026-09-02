#!/usr/bin/env python3
"""Live progress bar for ViTacSim calibration sim sweep.

Run in a second terminal while ``run_vitacsim_calibration_sweep.sh`` is active::

    python3 scripts/calibration/watch_calibration_sweep_progress.py
    python3 scripts/calibration/watch_calibration_sweep_progress.py --root logs/vitacsim_calibration/sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BAR_WIDTH = 32
WEIGHTS = ("W200", "W100", "W050", "W020", "W010")
LATERAL_FX = (0.0, 0.05, 0.1, 0.15, 0.2)


def _fx_tag(fx: float) -> str:
    s = f"{fx:.3f}".rstrip("0").rstrip(".")
    return f"Fx{s.replace('-', 'm')}_Fy0"


def _expected_job_ids() -> list[str]:
    jobs = ["no_contact"]
    jobs.extend(f"NF/{w}" for w in WEIGHTS)
    jobs.extend(f"LAT/{_fx_tag(fx)}" for fx in LATERAL_FX)
    return jobs


def _is_sweep_running() -> bool:
    patterns = (
        "run_vitacsim_calibration_sweep.sh",
        "demo_vitacsim_normal_force_validation.py",
        "demo_vitacsim_lateral_force_validation.py",
    )
    for pat in patterns:
        try:
            out = subprocess.check_output(["pgrep", "-f", pat], text=True)
            if out.strip():
                return True
        except subprocess.CalledProcessError:
            continue
    return False


def _job_done(sim_root: Path, job_id: str, *, sensor_mode: str) -> bool:
    if job_id == "no_contact":
        p = sim_root / "normal_force" / "no_contact" / sensor_mode / "tactile_rgb.png"
        return p.is_file()
    if job_id.startswith("NF/"):
        wid = job_id.split("/", 1)[1]
        sp = sim_root / "normal_force" / wid / sensor_mode / "summary.json"
        if not sp.is_file():
            return False
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
            return data.get("output_schema") == "nf_v3_beta"
        except json.JSONDecodeError:
            return False
    if job_id.startswith("LAT/"):
        tag = job_id.split("/", 1)[1]
        sp = sim_root / "shear_force" / "lateral" / tag / "W100" / sensor_mode / "summary.json"
        if not sp.is_file():
            return False
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
            return data.get("output_schema") == "sf_lateral_v2"
        except json.JSONDecodeError:
            return False
    return False


def _render_bar(done: int, total: int) -> str:
    frac = done / max(total, 1)
    filled = int(BAR_WIDTH * frac)
    tail = 1 if done < total and filled < BAR_WIDTH else 0
    return "[" + "=" * filled + ">" * tail + " " * max(0, BAR_WIDTH - filled - tail) + "]"


def _read_progress(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch calibration sim sweep progress.")
    parser.add_argument("--root", type=str, default="logs/vitacsim_calibration/sweep")
    parser.add_argument("--sensor-mode", type=str, default="vitacsim")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    sim_root = Path(args.root).expanduser().resolve()
    progress_file = sim_root / "_sweep_progress.json"
    log_file = sim_root / "_calibration_sweep.log"
    jobs = _expected_job_ids()
    total = len(jobs)

    while True:
        prog = _read_progress(progress_file)
        done_ids = [j for j in jobs if _job_done(sim_root, j, sensor_mode=args.sensor_mode)]
        done = len(done_ids)
        running = _is_sweep_running()

        current = "-"
        status = "idle"
        if prog:
            done = max(done, int(prog.get("done", done)))
            current = str(prog.get("current", current))
            status = str(prog.get("status", status))
        if running:
            status = "running" if status not in ("finished", "failed") else status
        elif done >= total:
            status = "finished"

        bar = _render_bar(done, total)
        elapsed = ""
        if prog and prog.get("elapsed_s") is not None:
            elapsed = f"  elapsed={int(prog['elapsed_s'])}s"
        line = f"{bar} {done}/{total}  status={status}  current={current}{elapsed}"
        sys.stdout.write("\r\033[K" + line)
        sys.stdout.flush()

        if done >= total and status == "finished" and not running:
            print("\n[DONE] calibration sim sweep complete.")
            if log_file.is_file():
                print(f"  log: {log_file}")
            idx = sim_root / "sim_index.json"
            if idx.is_file():
                print(f"  index: {idx}")
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
