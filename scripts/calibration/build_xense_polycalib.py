#!/usr/bin/env python3
"""Build Xense polycalib.npz from ball_calib_raw/ (auto-annotate + Taxim fit + install).

Pipeline:
  1. Optional: import_ball_calib_video.py
  2. Auto-generate dataPack.npz (circle detection; Taxim GUI not required)
  3. Run Taxim polyTableCalib.py with Xense sensor params
  4. Install polycalib (+ bg) into xense_lab_data/

Usage::

    python3 scripts/calibration/build_xense_polycalib.py
    python3 scripts/calibration/build_xense_polycalib.py --skip-import
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import repo_root  # noqa: E402
from import_ball_calib_video import (  # noqa: E402
    XENSE_BALL_RADIUS_MM,
    XENSE_SENSING_MM,
    _contact_blob_legacy,
    _load_rgb,
    _save_rgb,
)

TAXIM_NUM_BINS = 125


def _default_taxim_repo() -> Path:
    env = __import__("os").environ.get("TAXIM_REPO", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "third_party" / "Taxim"


def _find_bg(data_dir: Path) -> Path:
    for rel in (
        "bg/no_contact.png",
        "bg/no_contact.jpg",
        "bg.jpg",
        "no_contact.png",
    ):
        p = data_dir / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"No background image under {data_dir}")


def _list_ball_images(data_dir: Path) -> list[Path]:
    ball_dir = data_dir / "ball"
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(p for p in ball_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No ball images under {ball_dir}")
    return files


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _build_datapack(data_dir: Path, *, pixmm: float) -> Path:
    import cv2

    bg_path = _find_bg(data_dir)
    ball_paths = _list_ball_images(data_dir)

    bg_rgb = _load_rgb(bg_path)
    bg_bgr = _rgb_to_bgr(bg_rgb)

    imgs: list[np.ndarray] = []
    centers: list[list[float]] = []
    radii: list[float] = []
    names: list[str] = []
    records: list[dict] = []

    for p in ball_paths:
        frame_rgb = _load_rgb(p)
        frame_bgr = _rgb_to_bgr(frame_rgb)
        (cy, cx), radius_px, diff_mean, _circularity, _radial_norm = _contact_blob_legacy(bg_rgb, frame_rgb)

        if radius_px < 3.0:
            h, w = frame_rgb.shape[:2]
            cy, cx, radius_px = h / 2.0, w / 2.0, max(radius_px, 8.0)

        # Taxim stores touch_center as [row, col] = [y, x].
        imgs.append(frame_bgr)
        centers.append([float(cy), float(cx)])
        radii.append(float(radius_px))
        names.append(p.name)
        records.append(
            {
                "file": p.name,
                "touch_center_yx": [cy, cx],
                "touch_radius_px": radius_px,
                "diff_mean": diff_mean,
                "ball_radius_mm": XENSE_BALL_RADIUS_MM,
                "pixmm": pixmm,
            }
        )

    out = data_dir / "dataPack.npz"
    np.savez(
        out,
        f0=bg_bgr,
        imgs=np.stack(imgs, axis=0),
        touch_center=np.asarray(centers, dtype=np.float32),
        touch_radius=np.asarray(radii, dtype=np.float32),
        names=np.asarray(names),
        img_size=np.asarray(bg_bgr.shape),
    )

    ann_path = data_dir / "auto_annotation.json"
    ann_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[OK] dataPack -> {out} ({len(imgs)} frames)")
    print(f"[OK] auto_annotation -> {ann_path}")
    return out


def _run_poly_table_calib(
    data_dir: Path,
    taxim_repo: Path,
    *,
    pixmm: float,
    num_bins: int,
    deep_weight_gamma: float = 1.0,
    edge_weight_min: float = 0.20,
    edge_weight_power: float = 2.0,
    grad_table_smooth_sigma: float = 0.0,
) -> Path:
    """Run Taxim polyTableCalib with Xense params and NaN-safe polynomial fit."""
    import scipy.ndimage
    from scipy import interpolate
    from scipy.linalg import lstsq

    sys.path.insert(0, str(taxim_repo))
    import Basics.params as pr  # noqa: WPS433
    import Basics.sensorParams as psp  # noqa: WPS433
    from Basics.Geometry import Circle  # noqa: WPS433

    # Patch sensor params for this run only.
    psp.ball_radius = float(XENSE_BALL_RADIUS_MM)
    psp.pixmm = float(pixmm)
    psp.numBins = int(num_bins)
    psp.h = 700
    psp.w = 400

    data_file = np.load(data_dir / "dataPack.npz", allow_pickle=True)
    f0 = data_file["f0"]
    imgs = data_file["imgs"]
    radius_record = data_file["touch_radius"]
    touch_center_record = data_file["touch_center"]

    kscale = pr.kscale
    img_d = f0.astype("float")
    bg_proc = f0.copy().astype("float")
    for ch in range(img_d.shape[2]):
        bg_proc[:, :, ch] = scipy.ndimage.gaussian_filter(img_d[:, :, ch], kscale)
    frame_ = img_d
    diff_threshold = pr.diffThreshold
    d_i = np.mean(bg_proc - frame_, axis=2)
    idx = np.nonzero(d_i < diff_threshold)
    frame_mixing_per = pr.frameMixingPercentage
    for ch in range(bg_proc.shape[2]):
        bg_proc[:, :, ch][idx] = frame_mixing_per * bg_proc[:, :, ch][idx] + (1 - frame_mixing_per) * frame_[:, :, ch][idx]

    def _interpolate(img: np.ndarray) -> np.ndarray:
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]
        if newarr.size == 0:
            return np.zeros_like(img, dtype=np.float64)
        gd1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="nearest", fill_value=0)
        return np.nan_to_num(gd1, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_poly_params(xf: np.ndarray, yf: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        xf = np.asarray(xf, dtype=np.float64).ravel()
        yf = np.asarray(yf, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        mask = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(b)
        if int(mask.sum()) < 6:
            return np.zeros(6, dtype=np.float64)
        xf, yf, b = xf[mask], yf[mask], b[mask]
        a = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(xf.shape)]).T
        if w is not None:
            ww = np.asarray(w, dtype=np.float64).ravel()[mask]
            ww = np.nan_to_num(ww, nan=1.0, posinf=1.0, neginf=1.0)
            ww = np.clip(ww, 1.0e-6, None)
            sw = np.sqrt(ww)
            a = a * sw[:, None]
            b = b * sw
        params, *_ = lstsq(a, b)
        return np.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)

    value_list: list[np.ndarray] = []
    locx_list: list[np.ndarray] = []
    locy_list: list[np.ndarray] = []
    bins = int(psp.numBins)
    ball_radius_pix = float(psp.ball_radius) / float(psp.pixmm)

    radius_max = float(np.max(radius_record)) if np.size(radius_record) > 0 else 1.0
    frame_weights: list[float] = []
    gamma = max(float(deep_weight_gamma), 0.0)
    edge_min = float(np.clip(edge_weight_min, 0.0, 1.0))
    edge_pow = max(float(edge_weight_power), 0.0)

    for idx_i in range(int(np.shape(imgs)[0])):
        print(f"# iter {idx_i}")
        frame = imgs[idx_i, :, :, :]
        dI = frame.astype("float") - bg_proc
        circle = Circle(
            int(touch_center_record[idx_i, 0]),
            int(touch_center_record[idx_i, 1]),
            int(radius_record[idx_i]),
        )
        center = circle.center
        radius = circle.radius
        sizey, sizex = dI.shape[:2]
        xqq, yqq = np.meshgrid(range(sizex), range(sizey))
        xq = xqq - center[0]
        yq = yqq - center[1]
        rsqcoord = xq * xq + yq * yq
        rad_sq = radius * radius
        valid_rad = min(rad_sq, int(ball_radius_pix * ball_radius_pix))
        valid_mask = rsqcoord < valid_rad
        valid_id = np.nonzero(valid_mask)
        xvalid = xq[valid_id]
        yvalid = yq[valid_id]
        rvalid = np.sqrt(xvalid * xvalid + yvalid * yvalid)
        gradxseq = np.arcsin(np.clip(rvalid / ball_radius_pix, 0.0, 1.0))
        gradyseq = np.arctan2(-yvalid, -xvalid)
        binm = bins - 1
        x_binr = 0.5 * np.pi / binm
        y_binr = 2 * np.pi / binm
        idx_x = np.floor(gradxseq / x_binr).astype("int")
        idx_y = np.floor((gradyseq + np.pi) / y_binr).astype("int")

        value_map = np.zeros((bins, bins, 3))
        loc_x_map = np.zeros((bins, bins))
        loc_y_map = np.zeros((bins, bins))
        edge_ratio = np.clip(rvalid / max(float(radius), 1.0e-6), 0.0, 1.0)
        edge_weight = edge_min + (1.0 - edge_min) * ((1.0 - edge_ratio) ** edge_pow)
        valid_r = dI[:, :, 0][valid_id] * edge_weight
        valid_g = dI[:, :, 1][valid_id] * edge_weight
        valid_b = dI[:, :, 2][valid_id] * edge_weight
        valid_x = xqq[valid_id]
        valid_y = yqq[valid_id]
        value_map[idx_x, idx_y, 0] += valid_r
        value_map[idx_x, idx_y, 1] += valid_g
        value_map[idx_x, idx_y, 2] += valid_b
        loc_x_map[idx_x, idx_y] += valid_x
        loc_y_map[idx_x, idx_y] += valid_y
        loc_x_map = _interpolate(loc_x_map)
        loc_y_map = _interpolate(loc_y_map)
        value_map[:, :, 0] = _interpolate(value_map[:, :, 0])
        value_map[:, :, 1] = _interpolate(value_map[:, :, 1])
        value_map[:, :, 2] = _interpolate(value_map[:, :, 2])
        value_list.append(value_map)
        locx_list.append(loc_x_map)
        locy_list.append(loc_y_map)
        r_norm = float(radius_record[idx_i]) / max(radius_max, 1.0e-6)
        w_i = float(np.clip(r_norm, 1.0e-6, 1.0) ** gamma)
        frame_weights.append(max(w_i, 1.0e-6))

    table_v = np.array(value_list)
    table_x = np.array(locx_list)
    table_y = np.array(locy_list)
    table_w = np.asarray(frame_weights, dtype=np.float64)
    grad_r = np.zeros((bins, bins, 6))
    grad_g = np.zeros((bins, bins, 6))
    grad_b = np.zeros((bins, bins, 6))
    for i in range(table_v.shape[1]):
        for j in range(table_v.shape[2]):
            grad_r[i, j, :] = _fit_poly_params(table_x[:, i, j], table_y[:, i, j], table_v[:, i, j, 0], table_w)
            grad_g[i, j, :] = _fit_poly_params(table_x[:, i, j], table_y[:, i, j], table_v[:, i, j, 1], table_w)
            grad_b[i, j, :] = _fit_poly_params(table_x[:, i, j], table_y[:, i, j], table_v[:, i, j, 2], table_w)

    sigma = max(float(grad_table_smooth_sigma), 0.0)
    if sigma > 1.0e-6:
        # Smooth over (grad_mag_bin, grad_dir_bin) only; keep polynomial coefficient axis intact.
        for k in range(6):
            grad_r[:, :, k] = scipy.ndimage.gaussian_filter(grad_r[:, :, k], sigma=sigma, mode="nearest")
            grad_g[:, :, k] = scipy.ndimage.gaussian_filter(grad_g[:, :, k], sigma=sigma, mode="nearest")
            grad_b[:, :, k] = scipy.ndimage.gaussian_filter(grad_b[:, :, k], sigma=sigma, mode="nearest")

    out = data_dir / "polycalib.npz"
    np.savez(
        out,
        bins=bins,
        grad_r=grad_r,
        grad_g=grad_g,
        grad_b=grad_b,
        deep_weight_gamma=gamma,
        edge_weight_min=edge_min,
        edge_weight_power=edge_pow,
        grad_table_smooth_sigma=sigma,
    )
    print(f"[OK] polycalib -> {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and install Xense polycalib from ball video.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/ball_calib_raw"),
    )
    parser.add_argument("--video", type=str, default=str(repo_root() / "logs/file-000.mp4"))
    parser.add_argument("--num-ball", type=int, default=50)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--taxim-repo", type=str, default="")
    parser.add_argument(
        "--pixmm",
        type=float,
        default=0.0,
        help="mm/px for Taxim (default: Xense sensing width / 400).",
    )
    parser.add_argument("--bg-install", type=str, default="", help="Optional bg_clean.jpg for xense_lab_data.")
    parser.add_argument(
        "--deep-weight-gamma",
        type=float,
        default=1.0,
        help="Weight larger contact-radius samples higher during poly fit (1.0=linear, >1 deep-biased).",
    )
    parser.add_argument(
        "--edge-weight-min",
        type=float,
        default=0.20,
        help="Minimum per-pixel weight near contact rim in ball calibration fitting (0~1). Lower softens edge response.",
    )
    parser.add_argument(
        "--edge-weight-power",
        type=float,
        default=2.0,
        help="Power on (1-r/radius) for edge weighting. Higher suppresses rim samples more strongly.",
    )
    parser.add_argument(
        "--grad-table-smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma on fitted grad tables across bins (0 disables).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    taxim_repo = Path(args.taxim_repo or _default_taxim_repo()).expanduser().resolve()
    out_w = 400
    pixmm = float(args.pixmm) if args.pixmm > 0 else XENSE_SENSING_MM[0] / out_w

    if not args.skip_import:
        import_cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "import_ball_calib_video.py"),
            "--video",
            str(Path(args.video).expanduser().resolve()),
            "--out-dir",
            str(data_dir),
            "--num-ball",
            str(int(args.num_ball)),
        ]
        print("[RUN]", " ".join(import_cmd))
        subprocess.run(import_cmd, check=True)

    _build_datapack(data_dir, pixmm=pixmm)
    polycalib_path = _run_poly_table_calib(
        data_dir,
        taxim_repo,
        pixmm=pixmm,
        num_bins=TAXIM_NUM_BINS,
        deep_weight_gamma=float(args.deep_weight_gamma),
        edge_weight_min=float(args.edge_weight_min),
        edge_weight_power=float(args.edge_weight_power),
        grad_table_smooth_sigma=float(args.grad_table_smooth_sigma),
    )

    install_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "install_taxim_polycalib.py"),
        "--polycalib",
        str(polycalib_path),
    ]
    bg_install = args.bg_install.strip()
    if not bg_install:
        advisor_bg = repo_root() / "data/calibration/tactile/advisor_processed/bg_clean.jpg"
        if advisor_bg.is_file():
            bg_install = str(advisor_bg)
    if bg_install:
        install_cmd.extend(["--bg", bg_install])

    print("[RUN]", " ".join(install_cmd))
    subprocess.run(install_cmd, check=True)

    print("")
    print("[DONE] Xense polycalib installed.")
    print("[NEXT] SKIP_EXISTING=0 bash bash_command/run_vitacsim_calibration_sweep_dual.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
