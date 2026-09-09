#!/usr/bin/env python3
"""Aggregate NF validation sweep outputs into tables and comparison panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None  # type: ignore


WEIGHT_ORDER = ("W200", "W100", "W050", "W020", "W010")
MODES = ("tacsl", "vitacsim")
OUTPUT_SCHEMA = "nf_v3_beta"
LEGACY_SCHEMA = "nf_v2_chamfer_rgb"


def _load_summary(root: Path, weight_id: str, mode: str) -> dict | None:
    p = root / weight_id / mode / "summary.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("output_schema") not in (OUTPUT_SCHEMA, LEGACY_SCHEMA):
        return None
    return data


def _rgb_path(root: Path, weight_id: str, mode: str, *, depth: bool = False, diff_bg: bool = False) -> Path | None:
    sub = root / weight_id / mode
    if diff_bg:
        if mode == "vitacsim":
            p = sub / "tactile_rgb_diff_bg.png"
            if p.is_file():
                return p
        p = sub / "tactile_rgb_depth_diff_bg.png"
        return p if p.is_file() else None
    if depth:
        p = sub / "tactile_rgb_depth.png"
        if p.is_file():
            return p
    p = sub / "tactile_rgb.png"
    return p if p.is_file() else None


def _monotonic(values: list[float]) -> bool:
    if len(values) < 2:
        return True
    return all(values[i] >= values[i + 1] - 1e-6 for i in range(len(values) - 1))


def _build_table(root: Path) -> str:
    lines = [
        "# Normal Force Validation Sweep",
        "",
        f"Schema: `{OUTPUT_SCHEMA}`",
        "",
        "| Weight | mode | physx_fn | nominal | ratio | valid | pen_peak | fn_peak | rgb_l1 | z_final |",
        "|--------|------|----------|---------|-------|-------|----------|---------|--------|---------|",
    ]
    vitacsim_physx: list[float] = []
    vitacsim_peak: list[float] = []

    for wid in WEIGHT_ORDER:
        for mode in MODES:
            s = _load_summary(root, wid, mode)
            if s is None:
                lines.append(f"| {wid} | {mode} | *missing* | - | - | - | - | - | - |")
                continue
            nom = float(s.get("nominal_fn_n", 0.0))
            physx = float(s.get("physx_fn_total_mean", 0.0))
            ratio = float(s.get("physx_fn_ratio_nominal", physx / nom if nom > 1e-9 else 0.0))
            valid = s.get("contact_valid", ratio >= 0.5)
            pen = float(s.get("penetration_peak_max", 0.0))
            peak = float(s.get("fn_peak_max", 0.0))
            rgb_l1 = float(s.get("rgb_corr_l1_mean", 0.0))
            zf = float(s.get("weight_z_final", 0.0))
            lines.append(
                f"| {wid} | {mode} | {physx:.3f} | {nom:.3f} | {ratio:.2f} | {valid} | "
                f"{pen:.5f} | {peak:.3f} | {rgb_l1:.1f} | {zf:.4f} |"
            )
            if mode == "vitacsim":
                vitacsim_physx.append(physx)
                vitacsim_peak.append(peak)

    lines += [
        "",
        "## Monotonicity (vitacsim, W200→W010)",
        "",
        f"- physx_fn_total: {'PASS' if _monotonic(vitacsim_physx) else 'FAIL'} {vitacsim_physx}",
        f"- fn_peak_max: {'PASS' if _monotonic(vitacsim_peak) else 'FAIL'} {vitacsim_peak}",
        "",
        "Notes:",
        "- **tactile_rgb.png**: vitacsim=force-corrected RGB; tacsl=depth Taxim RGB.",
        "- **tactile_rgb_depth.png**: depth-only Taxim (both modes).",
        "- **ratio**: physx_fn_total_mean / nominal (target ~1.0). fn_peak is grid max, not total load.",
        "- **valid**: contact_valid (physx_fn >= 50% nominal & contacts > 0).",
        "- **rgb_l1**: mean |depth_rgb - corrected_rgb| (vitacsim only).",
    ]
    return "\n".join(lines)


def _build_panel(root: Path, out_path: Path, *, depth_panel: bool, diff_bg: bool = False) -> None:
    if Image is None:
        print("[WARN] PIL unavailable; skip panel generation.")
        return

    rows: list[list[Image.Image]] = []
    row_labels: list[str] = []
    for wid in WEIGHT_ORDER:
        imgs: list[Image.Image] = []
        for mode in MODES:
            if diff_bg:
                p = _rgb_path(root, wid, mode, diff_bg=True)
            elif depth_panel:
                p = _rgb_path(root, wid, mode, depth=True)
            else:
                p = _rgb_path(root, wid, mode, depth=False)
            if p is None:
                continue
            s = _load_summary(root, wid, mode) or {}
            imgs.append(Image.open(p).convert("RGB"))
        if imgs:
            rows.append(imgs)
            row_labels.append(wid)

    if not rows:
        print(f"[WARN] No RGB for panel depth={depth_panel}")
        return

    w, h = rows[0][0].size
    gap, header = 6, 28
    ncols = max(len(r) for r in rows)
    canvas = Image.new("RGB", (gap + ncols * (w + gap), gap + len(rows) * (h + header + gap)), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    title = "sim - bg (enhanced)" if diff_bg else ("depth Taxim RGB" if depth_panel else "primary RGB (vitacsim=corrected)")
    draw.text((gap + 4, 2), title, fill=(200, 200, 200))
    for ri, (imgs, wid) in enumerate(zip(rows, row_labels)):
        y = gap + ri * (h + header + gap)
        draw.text((gap + 4, y + 4), wid, fill=(220, 220, 220))
        for ci, img in enumerate(imgs):
            x = gap + ci * (w + gap)
            canvas.paste(img, (x, y + header))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[INFO] panel -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="logs/vitacsim_validation/normal_force")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    report = _build_table(root)
    report_path = root / "SWEEP_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"[INFO] report -> {report_path}")

    _build_panel(root, root / "panel_sweep_rgb.png", depth_panel=False)
    _build_panel(root, root / "panel_sweep_rgb_depth.png", depth_panel=True)
    _build_panel(root, root / "panel_sweep_rgb_diff_bg.png", depth_panel=False, diff_bg=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
