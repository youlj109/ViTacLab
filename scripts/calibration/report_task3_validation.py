#!/usr/bin/env python3
"""Task 3 physical validation: real advisor NF vs mapped sim (image + force metrics)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import (  # noqa: E402
    ADVISOR_MASS_G,
    ADVISOR_MARKER_PATTERN,
    ADVISOR_WEIGHT_CASES,
    LATERAL_W100_FX,
    XENSE_LAB_HW,
    _fx_tag,
    default_real_root,
    default_sim_root,
    load_advisor_real_nf_cases,
    load_bg_from_path,
    load_mapped_sim_nf_for_advisor,
    repo_root,
    rgb_diff_magnitude,
    sim_lateral_dir,
    sim_nf_dir,
)

from tangential_cosine_utils import compute_lateral_cosine_table  # noqa: E402

G = 9.81


def _fmt_cos(val: float) -> str:
    if val != val:
        return "n/a"
    return f"{val:.3f}"


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resize_rgb(img: np.ndarray, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    w = int(width if width is not None else XENSE_LAB_HW[0])
    h = int(height if height is not None else XENSE_LAB_HW[1])
    if img.shape[0] == h and img.shape[1] == w:
        return img
    try:
        from PIL import Image
    except ImportError:
        return img
    return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((width, height)), dtype=np.uint8)


def _align_rgb(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _resize_rgb(a)
    b = _resize_rgb(b)
    return a, b


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align_rgb(a, b)
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def _psnr(mse: float, *, peak: float = 255.0) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(peak) - 10.0 * math.log10(mse))


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight SSIM on luminance (no skimage dependency)."""
    a, b = _align_rgb(a, b)
    a_g = a.astype(np.float32).mean(axis=-1)
    b_g = b.astype(np.float32).mean(axis=-1)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = a_g.mean()
    mu_b = b_g.mean()
    sig_a = a_g.var()
    sig_b = b_g.var()
    sig_ab = float(((a_g - mu_a) * (b_g - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a + sig_b + c2)
    return float(num / den) if den > 1e-12 else 0.0


def _diff_vis(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    mag = rgb_diff_magnitude(rgb, bg)
    lo = float(np.percentile(mag, 1.0))
    hi = float(np.percentile(mag, 99.0))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = np.clip((mag - lo) / (hi - lo), 0.0, 1.0)
    heat = (norm * 255.0).astype(np.uint8)
    return np.stack([heat, (heat * 0.35).astype(np.uint8), (255 - heat).astype(np.uint8)], axis=-1)


def _abs_diff_vis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _align_rgb(a, b)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=-1)
    scaled = np.clip(diff * 4.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([scaled, (scaled * 0.2).astype(np.uint8), (255 - scaled).astype(np.uint8)], axis=-1)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return img
    out = img.copy()
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 70, 70), 1, cv2.LINE_AA)
    return out


def _vstack(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=0)


def _hstack(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def _save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _real_fn_nominal_n(case_id: str) -> float:
    mass_g = ADVISOR_MASS_G.get(case_id, 0)
    return float(mass_g) / 1000.0 * G


def _sim_fn_n(sim_root: Path, case_id: str, *, sensor_mode: str = "vitacsim") -> float | None:
    summary = _load_summary(sim_nf_dir(sim_root, case_id, sensor_mode=sensor_mode) / "summary.json")
    if not summary:
        return None
    if sensor_mode == "tacsl":
        # Depth-stiffness field estimate: flat across advisor loads (TacSL does not track PhysX load).
        field_sum = float(summary.get("fn_field_sum", float("nan")))
        if field_sum == field_sum:
            return field_sum / 1400.0
        return float(summary.get("fn_contact_mean", float("nan"))) / 40.0
    return float(summary.get("physx_fn_total_mean", float("nan")))


def _compute_case_metrics(
    real_rgb: np.ndarray,
    sim_rgb: np.ndarray,
    bg: np.ndarray,
) -> dict[str, float]:
    real_diff = _diff_vis(real_rgb, bg)
    sim_diff = _diff_vis(sim_rgb, bg)
    return {
        "mse_rgb": _mse(real_rgb, sim_rgb),
        "psnr_rgb": _psnr(_mse(real_rgb, sim_rgb)),
        "ssim_rgb": _ssim_gray(real_rgb, sim_rgb),
        "mse_diff": _mse(real_diff, sim_diff),
        "psnr_diff": _psnr(_mse(real_diff, sim_diff)),
        "ssim_diff": _ssim_gray(real_diff, sim_diff),
    }


_PANEL_HEADER_H = 36


def _column_header(label: str, *, width: int | None = None) -> np.ndarray:
    w = int(width if width is not None else XENSE_LAB_HW[0])
    img = np.zeros((_PANEL_HEADER_H, w, 3), dtype=np.uint8) + 40
    try:
        import cv2

        cv2.putText(
            img,
            label,
            (8, _PANEL_HEADER_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    except ImportError:
        pass
    return img


def _blank_header(label: str) -> np.ndarray:
    return _column_header(label)


def _missing_cell(label: str) -> np.ndarray:
    return _label(np.zeros((XENSE_LAB_HW[1], XENSE_LAB_HW[0], 3), dtype=np.uint8) + 40, label)


def _build_nf_panel(
    real_nf: dict,
    sim_nf: dict,
    bg: np.ndarray,
    out_path: Path,
) -> None:
    header = _hstack(
        [
            _blank_header("Real"),
            _blank_header("Sim (M2)"),
            _blank_header("|Real-Sim|"),
        ]
    )
    rows = [header]
    for cid in ADVISOR_WEIGHT_CASES:
        r = real_nf.get(cid)
        s = sim_nf.get(cid)
        if r is None or s is None or r.rgb is None or s.rgb is None:
            continue
        r_rgb = _resize_rgb(r.rgb)
        s_rgb = _resize_rgb(s.rgb)
        row = _hstack(
            [
                _label(r_rgb, f"{cid} {ADVISOR_MASS_G[cid]}g"),
                _label(s_rgb, cid),
                _abs_diff_vis(r_rgb, s_rgb),
            ]
        )
        rows.append(row)
    _save_png(out_path, _vstack(rows))


def _build_nf_three_way_panel(
    real_nf: dict,
    tacsl_nf: dict,
    vitacsim_nf: dict,
    out_path: Path,
) -> None:
    header = _hstack(
        [
            _blank_header("Real (Xense)"),
            _blank_header("TacSL"),
            _blank_header("ViTacSim"),
        ]
    )
    rows = [header]
    for cid in ADVISOR_WEIGHT_CASES:
        r = real_nf.get(cid)
        t = tacsl_nf.get(cid)
        v = vitacsim_nf.get(cid)
        if r is None or r.rgb is None:
            continue
        r_rgb = _resize_rgb(r.rgb)
        cols = [_label(r_rgb, f"{cid} {ADVISOR_MASS_G[cid]}g")]
        for tag, s in (("TacSL", t), ("ViTacSim", v)):
            if s is None or s.rgb is None:
                cols.append(_missing_cell(f"{tag} missing"))
            else:
                cols.append(_label(_resize_rgb(s.rgb), f"{cid} {tag}"))
        rows.append(_hstack(cols))
    _save_png(out_path, _vstack(rows))


def _build_real_sweep_panel(real_nf: dict, bg: np.ndarray, out_path: Path) -> None:
    imgs: list[np.ndarray] = []
    nc = real_nf.get("no_contact")
    if nc is not None and nc.rgb is not None:
        imgs.append(_label(_resize_rgb(nc.rgb), "no_contact"))
    for cid in ADVISOR_WEIGHT_CASES:
        s = real_nf.get(cid)
        if s is None or s.rgb is None:
            continue
        imgs.append(
            _label(
                _resize_rgb(s.rgb),
                f"{cid} {ADVISOR_MASS_G[cid]}g md={s.marker_disp_max_px:.2f}",
            )
        )
    if imgs:
        _save_png(out_path, _hstack(imgs))


def _draw_marker_overlay(
    rgb: np.ndarray,
    rest_xy: np.ndarray,
    disp: np.ndarray,
    *,
    scale: float = 2.0,
) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return rgb
    out = rgb.copy()
    for i in range(rest_xy.shape[0]):
        x0, y0 = int(round(rest_xy[i, 0])), int(round(rest_xy[i, 1]))
        dx, dy = disp[i] * scale
        x1, y1 = int(round(x0 + dx)), int(round(y0 + dy))
        cv2.circle(out, (x0, y0), 2, (80, 80, 80), -1, lineType=cv2.LINE_AA)
        mag = float(np.hypot(dx, dy))
        if mag > 0.35:
            cv2.arrowedLine(out, (x0, y0), (x1, y1), (0, 220, 80), 1, tipLength=0.35, line_type=cv2.LINE_AA)
    return out


def _build_marker_disp_panel(real_root: Path, real_nf: dict, out_path: Path) -> None:
    nc_dir = real_root / "normal_force" / "no_contact"
    rest_path = nc_dir / "marker_rest_detected.npy"
    if not rest_path.is_file():
        print("[WARN] skip marker overlay panel (no marker_rest_detected.npy)")
        return
    rest_xy = np.load(rest_path)
    imgs: list[np.ndarray] = []
    for cid in ADVISOR_WEIGHT_CASES:
        s = real_nf.get(cid)
        if s is None or s.rgb is None or s.source_dir is None:
            continue
        disp_path = s.source_dir / "marker_displacement.npy"
        if not disp_path.is_file():
            continue
        disp = np.load(disp_path)
        overlay = _draw_marker_overlay(_resize_rgb(s.rgb), rest_xy, disp)
        imgs.append(_label(overlay, f"{cid} {ADVISOR_MASS_G[cid]}g md={s.marker_disp_max_px:.1f}"))
    if imgs:
        _save_png(out_path, _hstack(imgs))


def _build_metrics_chart(rows: list[dict[str, Any]], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    w, h = 980, 440
    img = Image.new("RGB", (w, h), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    draw.text((16, 10), "Task 3 Normal — Real | TacSL | ViTacSim (M2 nut, 400x700)", fill=(20, 20, 20))
    draw.text((16, 34), "Rebuttal Table 2: MSE/SSIM/PSNR (bg-diff) + |Fn−mg|", fill=(90, 90, 90))

    headers = ["Case", "g", "MSE", "SSIM", "PSNR", "|Fn err|", "Fn ViT"]
    xs = [16, 72, 110, 170, 230, 300, 390]
    y = 64
    for i, hdr in enumerate(headers):
        draw.text((xs[i], y), hdr, fill=(30, 30, 120))
    y += 22
    for row in rows:
        vals = [
            row["case_id"],
            str(row["mass_g"]),
            f"{row.get('mse_diff', float('nan')):.0f}",
            f"{row['ssim_diff_vitacsim']:.3f}",
            f"{row.get('psnr_diff_vitacsim', row.get('psnr_diff', float('nan'))):.1f}",
            f"{row.get('fn_err_vitacsim_n', float('nan')):.3f}"
            if row.get("fn_err_vitacsim_n") is not None
            else "n/a",
            f"{row['fn_vitacsim_n']:.3f}" if row.get("fn_vitacsim_n") is not None else "n/a",
        ]
        for i, val in enumerate(vals):
            draw.text((xs[i], y), val, fill=(20, 20, 20))
        y += 20
        if y > h - 40:
            break

    bar_y = h - 120
    draw.text((16, bar_y - 18), "SSIM Real vs ViTacSim (bg-diff)", fill=(60, 60, 60))
    bar_w = 70
    for i, row in enumerate(rows):
        x = 40 + i * (bar_w + 16)
        val = float(row["ssim_diff_vitacsim"])
        bh = int(val * 80)
        draw.rectangle([x, bar_y + 80 - bh, x + bar_w - 8, bar_y + 80], fill=(70, 130, 220))
        draw.text((x, bar_y + 86), row["case_id"], fill=(40, 40, 40))

    _save_png(out_path, np.asarray(img))


def _build_lateral_sim_table(sim_root: Path, out_path: Path) -> list[dict[str, Any]]:
    cosine_rows = compute_lateral_cosine_table(sim_root)
    cos_by_tag = {r["tag"]: r for r in cosine_rows}

    rows: list[dict[str, Any]] = []
    for fx in LATERAL_W100_FX:
        tag = _fx_tag(fx)
        summary = _load_summary(sim_lateral_dir(sim_root, "W100", fx) / "summary.json")
        if not summary:
            continue
        cos = cos_by_tag.get(tag, {})
        rows.append(
            {
                "tag": tag,
                "fx_n": fx,
                "physx_ft_mean": float(summary.get("physx_ft_total_mean", 0.0)),
                "physx_fn_mean": float(summary.get("physx_fn_total_mean", 0.0)),
                "ft_peak_max": float(summary.get("ft_peak_max", 0.0)),
                "marker_disp_max_px": float(summary.get("marker_disp_max_px", 0.0)),
                "cos_gt_tacsl": cos.get("cos_gt_tacsl", float("nan")),
                "cos_gt_vitacsim": cos.get("cos_gt_vitacsim", float("nan")),
                "gt_present": bool(cos.get("gt_present", False)),
                "tacsl_field_present": bool(cos.get("tacsl_present", False)),
                "vitacsim_field_present": bool(cos.get("vitacsim_present", False)),
            }
        )

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (860, 240), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    draw.text((12, 8), "Task 3 Tangential (sim-only; rebuttal cosine vs PhysX GT grid)", fill=(20, 20, 20))
    draw.text(
        (12, 30),
        "W100 lateral: cos(phi_PhyX, phi_TacSL/ViT) from saved shear .npy (re-run lateral dual sweep if n/a)",
        fill=(90, 90, 90),
    )
    headers = ["Fx tag", "Fx", "cos TacSL", "cos ViT", "PhysX Ft", "ft_peak", "md_max"]
    xs = [12, 100, 170, 250, 330, 430, 530]
    y = 58
    for i, h in enumerate(headers):
        draw.text((xs[i], y), h, fill=(30, 30, 120))
    y += 22
    for r in rows:
        vals = [
            r["tag"],
            f"{r['fx_n']:.2f}",
            _fmt_cos(float(r.get("cos_gt_tacsl", float("nan")))),
            _fmt_cos(float(r.get("cos_gt_vitacsim", float("nan")))),
            f"{r['physx_ft_mean']:.3f}",
            f"{r['ft_peak_max']:.3f}",
            f"{r['marker_disp_max_px']:.2f}",
        ]
        for i, val in enumerate(vals):
            draw.text((xs[i], y), val, fill=(20, 20, 20))
        y += 20
    _save_png(out_path, np.asarray(img))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 3 advisor validation report.")
    parser.add_argument("--real-root", type=str, default=str(default_real_root()))
    parser.add_argument("--sim-root", type=str, default=str(default_sim_root()))
    parser.add_argument(
        "--bg",
        type=str,
        default=str(repo_root() / "data/calibration/tactile/advisor_processed/bg_clean.jpg"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root() / "logs/vitacsim_validation/task3"),
    )
    args = parser.parse_args()

    real_root = Path(args.real_root).expanduser().resolve()
    sim_root = Path(args.sim_root).expanduser().resolve()
    bg_path = Path(args.bg).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    real_nf = load_advisor_real_nf_cases(real_root)
    tacsl_nf = load_mapped_sim_nf_for_advisor(sim_root, sensor_mode="tacsl")
    vitacsim_nf = load_mapped_sim_nf_for_advisor(sim_root, sensor_mode="vitacsim")
    bg = load_bg_from_path(bg_path)
    if bg is None:
        print(f"[ERR] bg not found: {bg_path}")
        return 1

    tacsl_cases = sum(1 for cid in ADVISOR_WEIGHT_CASES if tacsl_nf.get(cid) and tacsl_nf[cid].rgb is not None)
    if tacsl_cases == 0:
        print("[WARN] No TacSL sim RGB found. Run: bash bash_command/run_vitacsim_calibration_sweep_dual.sh")

    metric_rows: list[dict[str, Any]] = []
    for cid in ADVISOR_WEIGHT_CASES:
        r = real_nf.get(cid)
        t = tacsl_nf.get(cid)
        v = vitacsim_nf.get(cid)
        if r is None or r.rgb is None or v is None or v.rgb is None:
            continue
        m_v = _compute_case_metrics(r.rgb, v.rgb, bg)
        m_t = _compute_case_metrics(r.rgb, t.rgb, bg) if t is not None and t.rgb is not None else {}
        fn_real = _real_fn_nominal_n(cid)
        fn_tacsl = _sim_fn_n(sim_root, cid, sensor_mode="tacsl")
        fn_vit = _sim_fn_n(sim_root, cid, sensor_mode="vitacsim")
        fn_err_v = None
        fn_err_v_n = None
        if fn_vit is not None and fn_real > 1e-9:
            fn_err_v = abs(fn_vit - fn_real) / fn_real * 100.0
            fn_err_v_n = abs(fn_vit - fn_real)
        fn_err_t = None
        fn_err_t_n = None
        if fn_tacsl is not None and fn_real > 1e-9:
            fn_err_t = abs(fn_tacsl - fn_real) / fn_real * 100.0
            fn_err_t_n = abs(fn_tacsl - fn_real)
        metric_rows.append(
            {
                "case_id": cid,
                "mass_g": ADVISOR_MASS_G[cid],
                "fn_real_n": fn_real,
                "fn_tacsl_n": fn_tacsl,
                "fn_vitacsim_n": fn_vit,
                "fn_err_tacsl_pct": fn_err_t,
                "fn_err_vitacsim_pct": fn_err_v,
                "fn_err_tacsl_n": fn_err_t_n,
                "fn_err_vitacsim_n": fn_err_v_n,
                "fn_sim_n": fn_vit,
                "fn_err_pct": fn_err_v,
                "marker_real_max_px": r.marker_disp_max_px,
                "marker_real_p95_px": r.marker_disp_p95_px,
                "marker_tacsl_max_px": t.marker_disp_max_px if t is not None else None,
                "marker_tacsl_p95_px": t.marker_disp_p95_px if t is not None else None,
                "marker_vitacsim_max_px": v.marker_disp_max_px,
                "marker_vitacsim_p95_px": v.marker_disp_p95_px,
                "marker_sim_max_px": v.marker_disp_max_px,
                "ssim_diff_tacsl": m_t.get("ssim_diff", float("nan")),
                "ssim_diff_vitacsim": m_v["ssim_diff"],
                "mse_diff": m_v["mse_diff"],
                "psnr_diff": m_v["psnr_diff"],
                "psnr_diff_tacsl": m_t.get("psnr_diff", float("nan")),
                "psnr_diff_vitacsim": m_v["psnr_diff"],
                "mse_diff_tacsl": m_t.get("mse_diff", float("nan")),
                **{k: v for k, v in m_v.items() if k not in ("ssim_diff", "psnr_diff", "mse_diff")},
            }
        )

    _build_nf_three_way_panel(real_nf, tacsl_nf, vitacsim_nf, out_dir / "panel_nf_three_way.png")
    _build_nf_panel(real_nf, vitacsim_nf, bg, out_dir / "panel_nf_real_vs_sim.png")
    _build_real_sweep_panel(real_nf, bg, out_dir / "panel_real_mass_sweep.png")
    _build_marker_disp_panel(real_root, real_nf, out_dir / f"panel_real_{ADVISOR_MARKER_PATTERN}_marker_disp.png")
    _build_metrics_chart(metric_rows, out_dir / "panel_nf_metrics_table.png")
    lat_rows = _build_lateral_sim_table(sim_root, out_dir / "panel_tangential_sim_only.png")

    # Copy reference panels for one-stop viewing
    ref_panels = {
        "ref_sim_nf_sweep.png": repo_root() / "logs/vitacsim_calibration/report/panel_nf_rgb.png",
        "ref_sim_lateral_sweep.png": repo_root() / "logs/vitacsim_calibration/report/panel_lateral_rgb.png",
        "ref_marker_synthetic.png": repo_root()
        / "logs/vitacsim_validation/marker_simulation/synthetic_panel_gelsight_xense.png",
    }
    from shutil import copy2

    for name, src in ref_panels.items():
        if src.is_file():
            copy2(src, out_dir / name)

    payload = {
        "schema": "vitacsim_task3_advisor_v2",
        "profile": "advisor",
        "note": "Real Xense vs TacSL baseline vs ViTacSim (M2 nut, matching mass, 400x700).",
        "tacsl_cases_present": tacsl_cases,
        "normal_force": metric_rows,
        "tangential_sim_only": lat_rows,
        "tangential_cosine": compute_lateral_cosine_table(sim_root),
        "real_fn_source": "mass_g * 9.81 / 1000 (suspended mass nominal)",
        "sim_fn_source": "TacSL=fn_field_sum/1400 (depth field, load-flat); ViTacSim=physx_fn_total_mean",
        "tangential_real_gt": "not required (advisor: sim-only PhysX per-cell GT)",
        "panels": {
            "three_way": str(out_dir / "panel_nf_three_way.png"),
            "real_vs_vitacsim": str(out_dir / "panel_nf_real_vs_sim.png"),
            "real_sweep": str(out_dir / "panel_real_mass_sweep.png"),
            "real_marker_disp": str(out_dir / f"panel_real_{ADVISOR_MARKER_PATTERN}_marker_disp.png"),
            "metrics": str(out_dir / "panel_nf_metrics_table.png"),
            "tangential_sim": str(out_dir / "panel_tangential_sim_only.png"),
        },
    }
    json_path = out_dir / "task3_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# ViTacSim Task 3 Validation (Advisor — Real | TacSL | ViTacSim)",
        "",
        "## Normal force — rebuttal Table 2 (Real vs ViTacSim, bg-diff image)",
        "",
        "| Case | g | MSE ↓ | SSIM ↑ | PSNR ↑ | |Fn−mg| (N) | Fn sim |",
        "|------|---|-------|--------|--------|-------------|--------|",
    ]
    for row in metric_rows:
        mse = row.get("mse_diff", float("nan"))
        psnr = row.get("psnr_diff_vitacsim", row.get("psnr_diff", float("nan")))
        fn_err_n = row.get("fn_err_vitacsim_n")
        fn_vit_s = f"{row['fn_vitacsim_n']:.3f}" if row.get("fn_vitacsim_n") is not None else "n/a"
        err_s = f"{fn_err_n:.3f}" if fn_err_n is not None else "n/a"
        md_lines.append(
            f"| {row['case_id']} | {row['mass_g']} | "
            f"{mse:.1f} | {row['ssim_diff_vitacsim']:.3f} | {psnr:.1f} | {err_s} | {fn_vit_s} |"
        )
    ssim_mean = float(np.nanmean([r["ssim_diff_vitacsim"] for r in metric_rows])) if metric_rows else float("nan")
    md_lines += [
        "",
        f"Mean bg-diff SSIM (Real vs ViT): **{ssim_mean:.3f}** (rebuttal-style image metric; bg-subtract per lab `bg_clean.jpg`).",
        "Fn reference = suspended mass × g; Fn sim = PhysX total (κ≈0.67 known limitation).",
        "",
        "## Normal force — TacSL comparison (bg-diff SSIM)",
        "",
        "| Case | SSIM R-TacSL | SSIM R-ViT | Fn TacSL | Fn ViT |",
        "|------|--------------|------------|----------|--------|",
    ]
    for row in metric_rows:
        fn_tac_s = f"{row['fn_tacsl_n']:.3f}" if row.get("fn_tacsl_n") is not None else "n/a"
        fn_vit_s = f"{row['fn_vitacsim_n']:.3f}" if row.get("fn_vitacsim_n") is not None else "n/a"
        ssim_t = row.get("ssim_diff_tacsl", float("nan"))
        md_lines.append(
            f"| {row['case_id']} | "
            f"{ssim_t:.3f} | {row['ssim_diff_vitacsim']:.3f} | "
            f"{fn_tac_s} | {fn_vit_s} |"
        )
    md_lines += [
        "",
        "## Normal force — legacy detail",
        "",
        "| Case | mass (g) | SSIM R-TacSL | SSIM R-ViT | Fn real | Fn TacSL | Fn ViTac |",
        "|------|----------|--------------|------------|---------|----------|----------|",
    ]
    for row in metric_rows:
        fn_tac_s = f"{row['fn_tacsl_n']:.3f}" if row.get("fn_tacsl_n") is not None else "n/a"
        fn_vit_s = f"{row['fn_vitacsim_n']:.3f}" if row.get("fn_vitacsim_n") is not None else "n/a"
        ssim_t = row.get("ssim_diff_tacsl", float("nan"))
        md_lines.append(
            f"| {row['case_id']} | {row['mass_g']} | "
            f"{ssim_t:.3f} | {row['ssim_diff_vitacsim']:.3f} | "
            f"{row['fn_real_n']:.3f} | {fn_tac_s} | {fn_vit_s} |"
        )
    md_lines += [
        "",
        "## Marker displacement (p95 px — load-sensitive; max often pegs at cap)",
        "",
        "| Case | Real max | TacSL p95 | ViT p95 | Real p95 |",
        "|------|----------|-----------|---------|----------|",
    ]
    for row in metric_rows:
        m_r = row.get("marker_real_max_px")
        m_rp = row.get("marker_real_p95_px")
        m_tp = row.get("marker_tacsl_p95_px")
        m_vp = row.get("marker_vitacsim_p95_px")
        r_s = f"{m_r:.2f}" if m_r is not None else "n/a"
        rp_s = f"{m_rp:.2f}" if m_rp is not None else "n/a"
        t_s = f"{m_tp:.2f}" if m_tp is not None else "n/a"
        v_s = f"{m_vp:.2f}" if m_vp is not None else "n/a"
        md_lines.append(f"| {row['case_id']} | {r_s} | {t_s} | {v_s} | {rp_s} |")
    md_lines += [
        "",
        "## Marker displacement (max px)",
        "",
        "| Case | Real | TacSL | ViTacSim |",
        "|------|------|-------|----------|",
    ]
    for row in metric_rows:
        m_r = row.get("marker_real_max_px")
        m_t = row.get("marker_tacsl_max_px")
        m_v = row.get("marker_vitacsim_max_px")
        r_s = f"{m_r:.2f}" if m_r is not None else "n/a"
        t_s = f"{m_t:.2f}" if m_t is not None else "n/a"
        v_s = f"{m_v:.2f}" if m_v is not None else "n/a"
        md_lines.append(f"| {row['case_id']} | {r_s} | {t_s} | {v_s} |")
    md_lines += [
        "",
        "## Tangential — sim-only (rebuttal Table 1 cosine)",
        "",
        "| Fx tag | Fx (N) | cos(GT,TacSL) | cos(GT,ViT) | PhysX Ft |",
        "|--------|--------|---------------|-------------|----------|",
    ]
    for r in lat_rows:
        md_lines.append(
            f"| {r['tag']} | {r['fx_n']:.2f} | "
            f"{_fmt_cos(float(r.get('cos_gt_tacsl', float('nan'))))} | "
            f"{_fmt_cos(float(r.get('cos_gt_vitacsim', float('nan'))))} | "
            f"{r['physx_ft_mean']:.3f} |"
        )
    md_lines += [
        "",
        "GT = PhysX friction anchors IDW-interpolated to 20×25 tactile grid. "
        "Re-run `bash bash_command/run_lateral_cosine_sweep.sh` after marker/shear update if cosine is n/a.",
        "",
        "See `panel_tangential_sim_only.png` and calibration lateral panel `ref_sim_lateral_sweep.png`.",
        "",
        "## Visual outputs",
        "",
        f"- `{out_dir / 'panel_nf_three_way.png'}`",
        f"- `{out_dir / 'panel_nf_real_vs_sim.png'}` (Real vs ViTacSim legacy)",
        f"- `{out_dir / 'panel_real_mass_sweep.png'}`",
        f"- `{out_dir / 'panel_nf_metrics_table.png'}`",
    ]
    if tacsl_cases == 0:
        md_lines += [
            "",
            "> TacSL column empty — run `bash bash_command/run_vitacsim_calibration_sweep_dual.sh` first.",
        ]
    report = out_dir / "TASK3_VALIDATION_REPORT.md"
    report.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] panels + metrics -> {out_dir}")
    print(f"[OK] json -> {json_path}")
    print(f"[OK] report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
