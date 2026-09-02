#!/usr/bin/env python3
"""Import advisor tactile mp4s (file-000 bg + M2 nut mass sweep) into calibration layout.

Autonomous frame pick:
  - bg: lowest center-patch temporal variance (skip warmup frames)
  - contact: max mean |rgb - bg| among video frames

Default: keep native mp4 resolution (400x700). Builds bg_clean.jpg (markers inpainted).

Usage::

    python3 scripts/calibration/import_advisor_tactile_videos.py
    python3 scripts/calibration/import_advisor_tactile_videos.py --install-bg
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from advisor_image_utils import (  # noqa: E402
    build_advisor_marker_rest,
    make_clean_background,
    measure_marker_residual,
)
from calibration_io import (  # noqa: E402
    ADVISOR_MARKER_PATTERN,
    ADVISOR_WEIGHT_CASES,
    XENSE_LAB_HW,
    default_real_root,
    real_nf_dir,
    repo_root,
    write_advisor_manifest,
)

WEIGHT_MP4 = {
    "G010": "10g.mp4",
    "G030": "30g.mp4",
    "G060": "60g.mp4",
    "G110": "110g.mp4",
    "G160": "160g.mp4",
    "G210": "210g.mp4",
}


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; required to decode advisor mp4.")


def _extract_frames(mp4: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-vsync", "0", str(out_dir / "f%04d.png")],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob("f*.png"))


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass
    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    # Fallback: ffprobe size + ffmpeg raw rgb (ffmpeg is already required).
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    w, h = [int(x) for x in probe.stdout.strip().split("x")]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(raw.stdout, dtype=np.uint8).reshape(h, w, 3)


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype(np.uint8)).save(path)
        return
    except ImportError:
        pass
    try:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))
        return
    except ImportError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape[:2]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-i",
            "-",
            str(path),
        ],
        check=True,
        input=arr.astype(np.uint8).tobytes(),
    )


def _center_patch_std(rgb: np.ndarray, *, frac: float = 0.45) -> float:
    gray = rgb.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    rh, rw = int(h * frac), int(w * frac)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    patch = gray[y0 : y0 + rh, x0 : x0 + rw]
    return float(patch.std())


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _resize_rgb(rgb: np.ndarray, *, out_w: int, out_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w == out_w and h == out_h:
        return rgb
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize((out_w, out_h), Image.Resampling.LANCZOS), dtype=np.uint8)
    except ImportError:
        pass
    try:
        import cv2

        return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        pass
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fin:
        fin_path = Path(fin.name)
    out_path = fin_path.with_suffix(".out.png")
    try:
        _save_rgb(rgb, fin_path)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(fin_path),
                "-vf",
                f"scale={out_w}:{out_h}",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        return _load_rgb(out_path)
    finally:
        fin_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def _pick_bg_frame(frames: list[Path], *, skip: int) -> tuple[Path, dict]:
    scored: list[tuple[float, Path, float]] = []
    for p in frames[skip:]:
        rgb = _load_rgb(p)
        scored.append((_center_patch_std(rgb), p, float(rgb.mean())))
    scored.sort(key=lambda x: x[0])
    best_std, best_path, best_mean = scored[0]
    return best_path, {
        "frame_file": best_path.name,
        "center_std": best_std,
        "mean": best_mean,
        "skip_warmup": skip,
        "num_frames": len(frames),
    }


def _pick_contact_frame(frames: list[Path], bg_rgb: np.ndarray) -> tuple[Path, dict]:
    scored: list[tuple[float, Path]] = []
    for p in frames:
        rgb = _load_rgb(p)
        scored.append((_mean_abs_diff(rgb, bg_rgb), p))
    scored.sort(reverse=True)
    best_diff, best_path = scored[0]
    return best_path, {"frame_file": best_path.name, "mean_abs_diff_vs_bg": best_diff, "num_frames": len(frames)}


def _install_xense_lab_assets(
    *,
    bg_clean: np.ndarray,
    marker_rest: np.ndarray,
) -> Path:
    """Install bg_clean + marker_rest into xense_lab_data/.

    polycalib.npz is NOT touched here — use build_xense_polycalib.py / install_taxim_polycalib.py
    after the lab ball-calibration video (file-000.mp4).
    """
    xense_dir = (
        repo_root()
        / "source"
        / "ViTacLab"
        / "ViTacLab"
        / "assets"
        / "sensor"
        / "tacsl_sensor"
        / "xense_lab_data"
    )
    xense_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(bg_clean, xense_dir / "bg_clean.jpg")
    np.save(xense_dir / "marker_rest.npy", marker_rest.astype(np.float32))
    return xense_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Import advisor mp4 tactile captures.")
    parser.add_argument("--file000-zip", type=str, default=str(repo_root() / "file-000.zip"))
    parser.add_argument("--correct-zip", type=str, default=str(repo_root() / "correct.zip"))
    parser.add_argument("--real-root", type=str, default=str(default_real_root()))
    parser.add_argument("--processed-dir", type=str, default=str(repo_root() / "data/calibration/tactile/advisor_processed"))
    parser.add_argument("--bg-warmup-skip", type=int, default=10)
    parser.add_argument(
        "--legacy-crop",
        action="store_true",
        help="Legacy: center-crop to GelSight R15 240x320 (not recommended).",
    )
    parser.add_argument("--install-bg", action="store_true", help="Install bg_clean + marker_rest into xense_lab_data/.")
    args = parser.parse_args()

    _require_ffmpeg()
    file000_zip = Path(args.file000_zip).expanduser().resolve()
    correct_zip = Path(args.correct_zip).expanduser().resolve()
    real_root = Path(args.real_root).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    out_w, out_h = XENSE_LAB_HW

    if not file000_zip.is_file():
        raise FileNotFoundError(file000_zip)
    if not correct_zip.is_file():
        raise FileNotFoundError(correct_zip)

    meta: dict = {
        "source": {"file000_zip": str(file000_zip), "correct_zip": str(correct_zip)},
        "output_size_wh": [out_w, out_h],
        "native_resolution": not bool(args.legacy_crop),
        "cases": {},
    }

    with tempfile.TemporaryDirectory(prefix="vitac_advisor_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(file000_zip) as zf:
            zf.extractall(tmp_path)
        with zipfile.ZipFile(correct_zip) as zf:
            zf.extractall(tmp_path)

        bg_mp4 = tmp_path / "file-000.mp4"
        if not bg_mp4.is_file():
            raise FileNotFoundError(f"Expected file-000.mp4 in {file000_zip}")

        bg_frames = _extract_frames(bg_mp4, tmp_path / "bg_frames")
        bg_frame_path, bg_pick = _pick_bg_frame(bg_frames, skip=int(args.bg_warmup_skip))
        bg_raw = _load_rgb(bg_frame_path)
        if args.legacy_crop:
            from calibration_io import GELSIGHT_R15_HW

            out_w, out_h = GELSIGHT_R15_HW
            bg_rgb = _resize_rgb(bg_raw, out_w=out_w, out_h=out_h)
        else:
            bg_rgb = _resize_rgb(bg_raw, out_w=out_w, out_h=out_h)

        marker_rest, marker_radius = build_advisor_marker_rest(
            bg_rgb,
            pattern=ADVISOR_MARKER_PATTERN,
            image_height=out_h,
            image_width=out_w,
        )
        bg_clean = make_clean_background(bg_rgb, marker_rest, radius_px=marker_radius)
        residual_raw = measure_marker_residual(bg_rgb, marker_rest)
        residual_clean = measure_marker_residual(bg_clean, marker_rest)

        processed_dir.mkdir(parents=True, exist_ok=True)
        bg_out = processed_dir / "bg.jpg"
        bg_clean_out = processed_dir / "bg_clean.jpg"
        marker_rest_out = processed_dir / "marker_rest.npy"
        _save_rgb(bg_rgb, bg_out)
        _save_rgb(bg_clean, bg_clean_out)
        np.save(marker_rest_out, marker_rest.astype(np.float32))
        meta["bg"] = {
            **bg_pick,
            "source_mp4": "file-000.mp4",
            "description": "Raw no-contact frame WITH printed markers (for real tracking only).",
        }
        meta["bg_clean"] = {
            "path": str(bg_clean_out),
            "description": "Gel-only background for Taxim (markers inpainted out; sim draws markers).",
            "residual_before": residual_raw,
            "residual_after": residual_clean,
        }
        meta["marker_count"] = int(marker_rest.shape[0])
        meta["marker_radius_px"] = float(marker_radius)

        no_dir = real_nf_dir(real_root, "no_contact")
        no_dir.mkdir(parents=True, exist_ok=True)
        _save_rgb(bg_rgb, no_dir / "rgb.png")
        np.save(no_dir / "marker_rest_detected.npy", marker_rest.astype(np.float32))
        meta["cases"]["no_contact"] = {"dir": str(no_dir), **bg_pick}

        for case_id in ADVISOR_WEIGHT_CASES:
            mp4_name = WEIGHT_MP4[case_id]
            mp4 = tmp_path / "chunk-000" / mp4_name
            if not mp4.is_file():
                raise FileNotFoundError(mp4)
            frames = _extract_frames(mp4, tmp_path / f"frames_{case_id}")
            pick_path, pick_meta = _pick_contact_frame(frames, bg_raw)
            contact_raw = _load_rgb(pick_path)
            contact_rgb = _resize_rgb(contact_raw, out_w=out_w, out_h=out_h)
            case_dir = real_nf_dir(real_root, case_id)
            case_dir.mkdir(parents=True, exist_ok=True)
            _save_rgb(contact_rgb, case_dir / "rgb.png")
            meta["cases"][case_id] = {"dir": str(case_dir), "mp4": mp4_name, **pick_meta}

    write_advisor_manifest(real_root / "manifest.json")
    meta_path = processed_dir / "import_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.install_bg:
        xense_dir = _install_xense_lab_assets(
            bg_clean=bg_clean,
            marker_rest=marker_rest,
        )
        print(f"[OK] installed xense lab assets (bg + marker_rest) -> {xense_dir}")
        print("[NOTE] polycalib.npz unchanged — install via bash_command/run_xense_polycalib.sh")

    print(f"[OK] real captures -> {real_root} ({out_w}x{out_h})")
    print(f"[OK] bg (raw, with markers) -> {bg_out}")
    print(f"[OK] bg_clean (Taxim, markers removed) -> {bg_clean_out}")
    if "bg_clean" in meta:
        rc = meta["bg_clean"]["residual_after"]
        print(
            f"[OK] bg_clean quality: residual_markers={rc['residual_markers']}/{rc['marker_count']} "
            f"({100.0 * rc['residual_ratio']:.1f}% sites still dark)"
        )
    print(f"[OK] marker_rest -> {marker_rest_out} ({marker_rest.shape[0]} pts)")
    print(f"[OK] metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
