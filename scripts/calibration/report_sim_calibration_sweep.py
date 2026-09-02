#!/usr/bin/env python3
"""Summarize sim calibration sweep: panels + markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    ADVISOR_WEIGHT_CASES,
    LATERAL_W100_FX,
    WEIGHT_CASES,
    XENSE_LAB_HW,
    _fx_tag,
    default_sim_root,
    load_advisor_sim_nf_cases,
    load_lateral_cases,
    load_nf_cases,
    repo_root,
)


def _load_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _resize_rgb(img: np.ndarray) -> np.ndarray:
    tw, th = XENSE_LAB_HW
    if img.shape[1] == tw and img.shape[0] == th:
        return img
    try:
        from PIL import Image
    except ImportError:
        return img
    return np.asarray(Image.fromarray(img).resize((tw, th)), dtype=np.uint8)


def _hstack(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return np.concatenate(images, axis=1)

    out_rows = []
    for img, lab in zip(images, labels):
        row = img.copy()
        cv2.putText(row, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 80), 2, cv2.LINE_AA)
        out_rows.append(row)
    return np.concatenate(out_rows, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report sim calibration sweep.")
    parser.add_argument("--sim-root", type=str, default=str(default_sim_root()))
    parser.add_argument("--sensor-mode", type=str, default="vitacsim")
    parser.add_argument("--profile", type=str, default="advisor", choices=("advisor", "cylinder"))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root() / "logs" / "vitacsim_calibration" / "report"),
    )
    args = parser.parse_args()

    sim_root = Path(args.sim_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.profile == "advisor":
        nf = load_advisor_sim_nf_cases(sim_root, sensor_mode=args.sensor_mode)
        nf_case_ids = ["no_contact", *ADVISOR_WEIGHT_CASES]
    else:
        nf = load_nf_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode)
        nf_case_ids = ["no_contact", *WEIGHT_CASES]
    lat = load_lateral_cases(sim_root, prefix="sim", sensor_mode=args.sensor_mode)

    index_path = sim_root / "sim_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}

    # NF panel
    nf_imgs, nf_labels = [], []
    for cid in nf_case_ids:
        s = nf.get(cid)
        if s is None or s.rgb is None:
            continue
        nf_imgs.append(_resize_rgb(s.rgb))
        disp = s.marker_disp_max_px
        fn = float(s.summary.get("fn_peak_max", 0.0)) if s.summary else 0.0
        nf_labels.append(f"{cid} fn={fn:.3f} md={disp:.2f}px")

    if nf_imgs:
        panel_nf = _hstack(nf_imgs, nf_labels)
        try:
            from PIL import Image

            Image.fromarray(panel_nf).save(out_dir / "panel_nf_rgb.png")
        except ImportError:
            np.save(out_dir / "panel_nf_rgb.npy", panel_nf)

    # Lateral panel
    lat_imgs, lat_labels = [], []
    for fx in LATERAL_W100_FX:
        cid = f"W100_{_fx_tag(fx)}"
        s = lat.get(cid)
        if s is None or s.rgb is None:
            continue
        lat_imgs.append(_resize_rgb(s.rgb))
        ft = float(s.summary.get("ft_peak_max", 0.0)) if s.summary else 0.0
        lat_labels.append(f"Fx={fx:.2f} ft={ft:.3f} md={s.marker_disp_max_px:.2f}px")

    if lat_imgs:
        panel_lat = _hstack(lat_imgs, lat_labels)
        try:
            from PIL import Image

            Image.fromarray(panel_lat).save(out_dir / "panel_lateral_rgb.png")
        except ImportError:
            np.save(out_dir / "panel_lateral_rgb.npy", panel_lat)

    lines = [
        "# ViTacSim Sim Calibration Sweep Report",
        "",
        f"- sim_root: `{sim_root}`",
        f"- profile: `{args.profile}`",
        f"- sensor_mode: `{args.sensor_mode}`",
        "",
        "## Normal force (marker + RGB)",
        "",
        "| case | fn_peak | marker_disp_max_px | rgb | marker |",
        "|------|---------|-------------------|-----|--------|",
    ]
    comp = index.get("normal_force", {}).get("completeness", {})
    fn_peaks = index.get("normal_force", {}).get("summary_fn_peak", {})
    md_peaks = index.get("normal_force", {}).get("marker_disp_max_px", {})
    for cid in nf_case_ids:
        c = comp.get(cid, {})
        lines.append(
            f"| {cid} | {fn_peaks.get(cid, 0.0):.4f} | {md_peaks.get(cid, 0.0):.3f} | "
            f"{'Y' if c.get('rgb') else 'N'} | {'Y' if c.get('marker_displacement') else 'N'} |"
        )

    lines.extend(["", "## Lateral W100", "", "| Fx (N) | marker_disp_max_px |", "|--------|-------------------|"])
    lat_md = index.get("lateral_force_W100", {}).get("marker_disp_max_px", {})
    for fx in LATERAL_W100_FX:
        key = f"W100_{_fx_tag(fx)}"
        lines.append(f"| {fx:.2f} | {lat_md.get(key, 0.0):.3f} |")

    lines.extend(
        [
            "",
            "## Next",
            "",
            "1. Capture real images into `data/calibration/tactile/real/` (same layout).",
            "2. `python3 scripts/calibration/track_real_markers.py --real-root data/calibration/tactile/real`",
            "3. `python3 scripts/calibration/fit_vitacsim_rgb_marker.py`",
            "",
            f"Panels: `{out_dir}/panel_nf_rgb.png`, `{out_dir}/panel_lateral_rgb.png`",
        ]
    )

    report_path = out_dir / "SIM_CALIBRATION_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] report -> {report_path}")
    if (out_dir / "panel_nf_rgb.png").is_file():
        print(f"[OK] panel_nf -> {out_dir / 'panel_nf_rgb.png'}")
    if (out_dir / "panel_lateral_rgb.png").is_file():
        print(f"[OK] panel_lat -> {out_dir / 'panel_lateral_rgb.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
