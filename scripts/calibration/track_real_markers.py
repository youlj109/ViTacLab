#!/usr/bin/env python3
"""Estimate marker displacements from real tactile RGB (no_contact vs contact).

Writes marker_displacement.npy next to each case rgb.png under data/calibration/tactile/real/.

Usage::

    python3 scripts/calibration/track_real_markers.py
    python3 scripts/calibration/track_real_markers.py --real-root data/calibration/tactile/real --profile advisor
    python3 scripts/calibration/track_real_markers.py --profile cylinder --pattern gelsight
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parents[1]
_TACSL = _REPO / "source" / "ViTacLab" / "ViTacLab" / "assets" / "sensor" / "tacsl_sensor"

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    ADVISOR_MARKER_PATTERN,
    ADVISOR_WEIGHT_CASES,
    LATERAL_W100_FX,
    WEIGHT_CASES,
    _fx_tag,
    default_real_root,
    real_lateral_dir,
    real_nf_dir,
    repo_root,
)


def _load_marker_module():
    mod_name = "ViTacLab.assets.sensor.tacsl_sensor.visuotactile_marker"
    path = _TACSL / "visuotactile_marker.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("PIL required: pip install pillow") from e
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _rgb_path(case_dir: Path) -> Path:
    for name in ("rgb.png", "tactile_rgb.png"):
        p = case_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No rgb in {case_dir}")


def _detect_markers(gray: np.ndarray, rest_xy: np.ndarray, *, radius: int) -> np.ndarray:
    """Return (M,2) detected centers; fallback to rest grid on failure."""
    try:
        import cv2
    except ImportError:
        return rest_xy.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    pts = []
    h, w = gray.shape
    for i in range(rest_xy.shape[0]):
        cx, cy = int(round(rest_xy[i, 0].item())), int(round(rest_xy[i, 1].item()))
        r = max(3, radius)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        patch = blur[y0:y1, x0:x1]
        if patch.size == 0:
            pts.append(rest_xy[i])
            continue
        min_val, _, min_loc, _ = cv2.minMaxLoc(patch)
        dx, dy = float(min_loc[0] + x0), float(min_loc[1] + y0)
        pts.append(np.array([dx, dy], dtype=np.float32))
    return np.stack(pts, axis=0)


def _track_case(
    *,
    contact_dir: Path,
    rest_xy: np.ndarray,
    rest_detected: np.ndarray,
    radius: int,
    out_name: str = "marker_displacement.npy",
) -> np.ndarray:
    rgb = _load_rgb(_rgb_path(contact_dir))
    gray = np.mean(rgb.astype(np.float32), axis=-1)
    detected = _detect_markers(gray, rest_xy, radius=radius)
    disp = detected - rest_detected
    np.save(contact_dir / out_name, disp.astype(np.float32))
    return disp


def main() -> int:
    parser = argparse.ArgumentParser(description="Track real marker displacements vs no_contact.")
    parser.add_argument("--real-root", type=str, default=str(default_real_root()))
    parser.add_argument(
        "--pattern",
        type=str,
        default="auto",
        choices=("auto", "gelsight", "xense"),
        help="Marker rest grid (auto: xense for advisor, gelsight for cylinder).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="cylinder",
        choices=("cylinder", "advisor"),
        help="cylinder=W200..W010+lateral; advisor=G010..G210 NF only (lab Xense)",
    )
    args = parser.parse_args()

    pattern = (
        ADVISOR_MARKER_PATTERN
        if args.pattern == "auto" and args.profile == "advisor"
        else "gelsight"
        if args.pattern == "auto"
        else args.pattern
    )

    real_root = Path(args.real_root).expanduser().resolve()
    nf_ids = ADVISOR_WEIGHT_CASES if args.profile == "advisor" else WEIGHT_CASES
    marker_mod = _load_marker_module()
    MarkerSimulator = marker_mod.MarkerSimulator

    no_dir = real_nf_dir(real_root, "no_contact")
    no_rgb = _load_rgb(_rgb_path(no_dir))
    h, w = no_rgb.shape[:2]

    rest_override = None
    for candidate in (
        no_dir / "marker_rest_detected.npy",
        Path(args.real_root).parent / "advisor_processed" / "marker_rest.npy",
        repo_root() / "data/calibration/tactile/advisor_processed/marker_rest.npy",
    ):
        if candidate.is_file():
            rest_override = np.load(candidate).astype(np.float32)
            print(f"[INFO] marker rest from {candidate}")
            break

    sim = MarkerSimulator(
        pattern=pattern,
        image_height=h,
        image_width=w,
        device="cpu",
        rest_xy_override=rest_override,
    )
    rest_xy = sim.rest_xy.cpu().numpy()
    print(f"[INFO] marker_pattern={pattern} rest_markers={rest_xy.shape[0]}")
    r = int(max(2, round(sim.spec.radius_px))) if sim.spec else 4

    no_gray = np.mean(no_rgb.astype(np.float32), axis=-1)
    rest_detected = _detect_markers(no_gray, rest_xy, radius=r)
    np.save(no_dir / "marker_rest_detected.npy", rest_detected.astype(np.float32))
    np.save(no_dir / "marker_displacement.npy", np.zeros_like(rest_detected, dtype=np.float32))

    processed = 0
    for wid in nf_ids:
        d = real_nf_dir(real_root, wid)
        try:
            _rgb_path(d)
            _track_case(contact_dir=d, rest_xy=rest_xy, rest_detected=rest_detected, radius=r)
            processed += 1
            print(f"[OK] NF {wid} -> {d / 'marker_displacement.npy'}")
        except FileNotFoundError:
            print(f"[SKIP] NF {wid} (no rgb)")

    if args.profile == "cylinder":
        for fx in LATERAL_W100_FX:
            d = real_lateral_dir(real_root, "W100", fx)
            try:
                _track_case(contact_dir=d, rest_xy=rest_xy, rest_detected=rest_detected, radius=r)
                processed += 1
                print(f"[OK] LAT Fx={fx} -> {d / 'marker_displacement.npy'}")
            except FileNotFoundError:
                print(f"[SKIP] LAT Fx={fx} (no rgb)")

    if processed == 0:
        print("[WARN] No contact rgb.png found. Add captures under data/calibration/tactile/real/", file=sys.stderr)
        return 1
    print(f"[DONE] processed {processed} contact cases")
    print("[NEXT] python3 scripts/calibration/fit_vitacsim_rgb_marker.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
