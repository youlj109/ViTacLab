#!/usr/bin/env python3
"""Install lab Xense Taxim polycalib after advisor ball-indent capture.

Expected layout under --data-dir (from lab):
  bg/no_contact.png   (or bg.jpg / bg/no_contact.jpg)
  ball/000.png ...    (~50 ball indent RGB images, 400x700)

Workflow (Taxim repo required for fit step):
  1. Hand-annotate contact circles -> dataPack.npz  (generateDataPack.py)
  2. Fit polynomial table -> polycalib.npz          (polyTableCalib.py)
  3. Copy polycalib (+ optional bg) -> xense_lab_data/
  4. Re-run sim sweep + Task2/3

Usage:
  # After Taxim annotation + polyTableCalib (manual or in TAXIM_REPO):
  python3 scripts/calibration/install_taxim_polycalib.py \\
      --polycalib /path/to/polycalib.npz \\
      --bg data/calibration/tactile/advisor_processed/bg_clean.jpg

  # Validate incoming lab folder only:
  python3 scripts/calibration/install_taxim_polycalib.py --data-dir data/calibration/tactile/ball_calib_raw --check-only

  # Run Taxim fit if TAXIM_REPO is set:
  TAXIM_REPO=~/Taxim python3 scripts/calibration/install_taxim_polycalib.py \\
      --data-dir data/calibration/tactile/ball_calib_raw --run-taxim
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import repo_root  # noqa: E402


def _xense_lab_data_dir() -> Path:
    return (
        repo_root()
        / "source"
        / "ViTacLab"
        / "ViTacLab"
        / "assets"
        / "sensor"
        / "tacsl_sensor"
        / "xense_lab_data"
    )


def _find_bg(data_dir: Path) -> Path | None:
    candidates = [
        data_dir / "bg" / "no_contact.png",
        data_dir / "bg" / "no_contact.jpg",
        data_dir / "bg.jpg",
        data_dir / "bg" / "bg.jpg",
        data_dir / "no_contact.png",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _count_ball_images(data_dir: Path) -> tuple[int, list[Path]]:
    ball_dir = data_dir / "ball"
    if not ball_dir.is_dir():
        return 0, []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(p for p in ball_dir.iterdir() if p.suffix.lower() in exts)
    return len(files), files


def _check_data_dir(data_dir: Path) -> dict:
    bg = _find_bg(data_dir)
    n_ball, ball_files = _count_ball_images(data_dir)
    return {
        "data_dir": str(data_dir),
        "bg": str(bg) if bg else None,
        "ball_count": n_ball,
        "ball_sample": [str(p.name) for p in ball_files[:5]],
        "ready_for_taxim": bg is not None and n_ball >= 10,
        "recommended_ball_count": 50,
    }


def _backup_if_exists(path: Path) -> None:
    if path.is_file():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"[INFO] backed up {path.name} -> {bak.name}")


def _install_polycalib(polycalib_src: Path, *, bg_src: Path | None) -> Path:
    xense_dir = _xense_lab_data_dir()
    xense_dir.mkdir(parents=True, exist_ok=True)

    _backup_if_exists(xense_dir / "polycalib.npz")
    shutil.copy2(polycalib_src, xense_dir / "polycalib.npz")
    print(f"[OK] polycalib -> {xense_dir / 'polycalib.npz'}")

    if bg_src is not None and bg_src.is_file():
        _backup_if_exists(xense_dir / "bg_clean.jpg")
        shutil.copy2(bg_src, xense_dir / "bg_clean.jpg")
        print(f"[OK] bg_clean -> {xense_dir / 'bg_clean.jpg'}")

    return xense_dir


def _run_taxim_fit(data_dir: Path, taxim_repo: Path) -> Path:
    taxim_repo = taxim_repo.expanduser().resolve()
    if not taxim_repo.is_dir():
        raise FileNotFoundError(f"TAXIM_REPO not found: {taxim_repo}")

    gen = taxim_repo / "Calibration" / "generateDataPack.py"
    poly = taxim_repo / "Calibration" / "polyTableCalib.py"
    if not gen.is_file():
        gen = taxim_repo / "generateDataPack.py"
    if not poly.is_file():
        poly = taxim_repo / "polyTableCalib.py"
    if not gen.is_file() or not poly.is_file():
        raise FileNotFoundError(f"Taxim scripts missing under {taxim_repo}")

    print("[INFO] Taxim generateDataPack (interactive annotation — complete in GUI)")
    subprocess.run([sys.executable, str(gen), "-data_path", str(data_dir)], check=True, cwd=str(taxim_repo))

    print("[INFO] Taxim polyTableCalib")
    subprocess.run([sys.executable, str(poly), "-data_path", str(data_dir)], check=True, cwd=str(taxim_repo))

    out = data_dir / "polycalib.npz"
    if not out.is_file():
        raise FileNotFoundError(f"Expected {out} after polyTableCalib")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Xense Taxim polycalib into ViTacLab.")
    parser.add_argument("--data-dir", type=str, default="", help="Lab ball calib folder (bg/ + ball/).")
    parser.add_argument("--polycalib", type=str, default="", help="Existing polycalib.npz to install.")
    parser.add_argument("--bg", type=str, default="", help="Optional bg_clean.jpg to install with polycalib.")
    parser.add_argument(
        "--run-taxim",
        action="store_true",
        help="Run Taxim generateDataPack + polyTableCalib in TAXIM_REPO (annotation is interactive).",
    )
    parser.add_argument(
        "--taxim-repo",
        type=str,
        default="",
        help="Path to cloned Robo-Touch/Taxim (or set TAXIM_REPO env).",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate --data-dir layout only.")
    args = parser.parse_args()

    if args.check_only:
        if not args.data_dir:
            print("[ERR] --check-only requires --data-dir", file=sys.stderr)
            return 1
        info = _check_data_dir(Path(args.data_dir).expanduser().resolve())
        print("[CHECK]", info)
        if not info["ready_for_taxim"]:
            print("[WARN] Need bg + >=10 ball images before Taxim fit.")
            return 2
        return 0

    polycalib_path: Path | None = None
    bg_path: Path | None = Path(args.bg).expanduser().resolve() if args.bg else None

    if args.run_taxim:
        if not args.data_dir:
            print("[ERR] --run-taxim requires --data-dir", file=sys.stderr)
            return 1
        data_dir = Path(args.data_dir).expanduser().resolve()
        default_taxim = repo_root() / "third_party" / "Taxim"
        taxim_repo = Path(
            args.taxim_repo or __import__("os").environ.get("TAXIM_REPO", str(default_taxim))
        ).expanduser()
        if not taxim_repo.is_dir():
            print("[ERR] Set --taxim-repo or TAXIM_REPO to cloned Taxim repo.", file=sys.stderr)
            return 1
        polycalib_path = _run_taxim_fit(data_dir, taxim_repo)
        if bg_path is None:
            found = _find_bg(data_dir)
            if found is not None:
                bg_path = found

    if args.polycalib:
        polycalib_path = Path(args.polycalib).expanduser().resolve()

    if polycalib_path is None or not polycalib_path.is_file():
        print("[ERR] Provide --polycalib or --run-taxim with valid --data-dir.", file=sys.stderr)
        print("      Check lab folder: python3 scripts/calibration/install_taxim_polycalib.py --data-dir ... --check-only")
        return 1

    _install_polycalib(polycalib_path, bg_src=bg_path)

    print("")
    print("[NEXT] Re-run sim sweep + Task2/3:")
    print("  SKIP_EXISTING=0 bash bash_command/run_vitacsim_calibration_sweep_dual.sh")
    print("  bash bash_command/run_task2_advisor_calibration.sh")
    print("  bash bash_command/run_task3_advisor_validation.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
