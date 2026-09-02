#!/usr/bin/env python3
"""Plot NF/SF validation curves and rebattle figure set (valid points only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from validation_beta_config import (
    CONTACT_VALID_FN_RATIO,
    MODES,
    NF_SCHEMA,
    SF_BAR_FX,
    SF_PANEL_FX,
    SF_PANEL_WEIGHTS,
    SF_PRIMARY_WEIGHTS,
    SF_SCHEMA,
    WEIGHT_MASS_G,
    WEIGHT_ORDER,
)

NF_LEGACY = "nf_v2_chamfer_rgb"
SF_LEGACY = "sf_lateral_v1"


def _load_nf(root: Path) -> list[dict]:
    rows: list[dict] = []
    for wid in WEIGHT_ORDER:
        for mode in MODES:
            p = root / wid / mode / "summary.json"
            if not p.is_file():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("output_schema") not in (NF_SCHEMA, NF_LEGACY):
                continue
            rows.append(d)
    return rows


def _load_sf(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.glob("**/summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("output_schema") not in (SF_SCHEMA, SF_LEGACY):
            continue
        rows.append(d)
    return rows


def _contact_valid(row: dict) -> bool:
    if "contact_valid" in row:
        return bool(row["contact_valid"])
    nom = float(row.get("nominal_fn_n", 0))
    fn = float(row.get("physx_fn_total_mean", 0))
    cnt = float(row.get("physx_contact_count_mean", row.get("physx_contact_count_last", 0)))
    return fn >= CONTACT_VALID_FN_RATIO * nom and cnt > 0


def _pick_sf(rows: list[dict], *, weight_id: str, mode: str, fx: float) -> dict | None:
    for r in rows:
        if (
            r.get("weight_id") == weight_id
            and r.get("sensor_mode") == mode
            and abs(float(r.get("lateral_force_x_n", 0)) - fx) < 1e-6
        ):
            return r
    return None


def _plot_nf_curves(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib missing; skip NF curves")
        return

    masses = [WEIGHT_MASS_G[w] for w in WEIGHT_ORDER]
    nom = [WEIGHT_MASS_G[w] / 1000.0 * 9.81 for w in WEIGHT_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ys_physx, ys_ratio = [], []
    for wid in WEIGHT_ORDER:
        r = next((x for x in rows if x.get("weight_id") == wid and x.get("sensor_mode") == "vitacsim"), None)
        if r is None:
            ys_physx.append(np.nan)
            ys_ratio.append(np.nan)
            continue
        physx = float(r.get("physx_fn_total_mean", 0))
        ys_physx.append(physx)
        ys_ratio.append(float(r.get("physx_fn_ratio_nominal", physx / float(r.get("nominal_fn_n", 1)))))

    axes[0].plot(masses, ys_physx, "o-", label="vitacsim PhysX Fn")
    axes[0].plot(masses, nom, "k--", alpha=0.4, label="nominal mg")
    axes[0].set_xlabel("mass (g)")
    axes[0].set_ylabel("PhysX Fn total (N)")
    axes[0].set_title("NF: PhysX total vs mass")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(range(len(WEIGHT_ORDER)), ys_ratio, color="steelblue", alpha=0.85)
    axes[1].axhline(1.0, color="k", ls="--", alpha=0.4, label="target ratio=1")
    axes[1].set_xticks(range(len(WEIGHT_ORDER)), WEIGHT_ORDER)
    axes[1].set_ylabel("physx_fn / nominal")
    axes[1].set_title("NF: load ratio (not fn_peak)")
    axes[1].set_ylim(0, max(1.2, np.nanmax(ys_ratio) * 1.1 if ys_ratio else 1.2))
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend()

    fig.tight_layout()
    p = out_dir / "curve_nf_fn_vs_mass.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] curve -> {p}")


def _plot_sf_curves(
    rows: list[dict],
    out_dir: Path,
    *,
    weights: tuple[str, ...],
    filename: str,
    title_suffix: str,
    valid_only: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib missing; skip SF curves")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(weights)))

    for ci, wid in enumerate(weights):
        for mode, ls, alpha in (("vitacsim", "-", 1.0), ("tacsl", "--", 0.35)):
            pts = [r for r in rows if r.get("weight_id") == wid and r.get("sensor_mode") == mode]
            if not pts:
                continue
            pts = sorted(pts, key=lambda r: float(r.get("lateral_force_x_n", 0)))
            fx, ft = [], []
            for r in pts:
                if valid_only and not _contact_valid(r):
                    continue
                fx.append(float(r.get("lateral_force_x_n", 0)))
                ft.append(float(r.get("physx_ft_total_mean", 0)))
            if not fx:
                continue
            label = f"{wid} {mode}" if mode == "vitacsim" else None
            ax.plot(fx, ft, ls, color=colors[ci], alpha=alpha, label=label)

    ax.set_xlabel("applied Fx (N)")
    ax.set_ylabel("PhysX Ft total (N)")
    ax.set_title(f"SF lateral: PhysX friction vs Fx{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / filename
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] curve -> {p}")


def _plot_sf_bar(rows: list[dict], out_dir: Path, *, fx: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(WEIGHT_ORDER))
    w = 0.35
    vit_vals, tac_vals, valid_flags = [], [], []
    for wid in WEIGHT_ORDER:
        rv = _pick_sf(rows, weight_id=wid, mode="vitacsim", fx=fx)
        rt = _pick_sf(rows, weight_id=wid, mode="tacsl", fx=fx)
        vit_vals.append(float(rv.get("physx_ft_total_mean", 0)) if rv and _contact_valid(rv) else np.nan)
        tac_vals.append(float(rt.get("physx_ft_total_mean", 0)) if rt and _contact_valid(rt) else np.nan)
        valid_flags.append(rv is not None and _contact_valid(rv))

    ax.bar(x - w / 2, vit_vals, w, label="vitacsim", color="steelblue")
    ax.bar(x + w / 2, tac_vals, w, label="tacsl", color="darkorange", alpha=0.7)
    ax.set_xticks(x, WEIGHT_ORDER)
    ax.set_ylabel("PhysX Ft (N)")
    ax.set_title(f"SF @ Fx={fx}N (valid points only; blank=invalid contact)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = out_dir / f"bar_sf_ft_at_fx{fx}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] bar -> {p}")


def _resolve_panel_fx(sf_rows: list[dict], preferred: float) -> float:
    if any(abs(float(r.get("lateral_force_x_n", 0)) - preferred) < 1e-6 for r in sf_rows):
        return preferred
    for fx in (0.2, 0.1, 0.15, 0.05, 0.03, 0.02):
        if any(abs(float(r.get("lateral_force_x_n", 0)) - fx) < 1e-6 for r in sf_rows):
            return fx
    return preferred


def _force_tag(fx: float, fy: float = 0.0) -> str:
    def _fmt(v: float) -> str:
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s.replace("-", "m")

    return f"Fx{_fmt(fx)}_Fy{_fmt(fy)}"


def _sf_trial_dir(lateral_root: Path, wid: str, mode: str, fx: float) -> Path:
    return lateral_root / _force_tag(fx, 0.0) / wid / mode


def _render_npy_heatmap(arr: np.ndarray, cmap: str = "inferno"):
    from PIL import Image

    mag = np.abs(np.nan_to_num(arr, nan=0.0))
    if mag.ndim == 3:
        mag = np.linalg.norm(mag, axis=-1)
    vmax = float(mag.max()) if mag.size else 1.0
    if vmax < 1e-9:
        vmax = 1.0
    norm = np.clip(mag / vmax, 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        rgba = (cm.get_cmap(cmap)(norm) * 255).astype(np.uint8)[..., :3]
    except ImportError:
        rgba = (norm * 255).astype(np.uint8)
        rgba = np.stack([rgba, rgba, rgba], axis=-1)
    return Image.fromarray(rgba).resize((320, 240))


def _panel_image_path(base: Path, mode: str, *, diff_bg: bool, heatmap_npy: Path | None) -> "Image.Image | None":
    from PIL import Image

    if diff_bg:
        p = base / ("tactile_rgb_diff_bg.png" if mode == "vitacsim" else "tactile_rgb_depth_diff_bg.png")
        if not p.is_file():
            p = base / "tactile_rgb_diff_bg.png"
    else:
        p = base / "tactile_rgb.png"
    if p.is_file():
        return Image.open(p).convert("RGB")
    if heatmap_npy is not None and heatmap_npy.is_file():
        return _render_npy_heatmap(np.load(heatmap_npy))
    return None


def _build_nf_panels(nf_root: Path, out_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN] PIL missing; skip NF panels")
        return

    for suffix, diff_bg, npy_name in (
        ("panel_sweep_rgb_diff_bg.png", True, "tactile_normal_force.npy"),
        ("panel_sweep_pen_heatmap.png", False, "tactile_normal_force.npy"),
    ):
        rows_imgs: list[list] = []
        row_labels: list[str] = []
        for wid in WEIGHT_ORDER:
            imgs = []
            for mode in MODES:
                base = nf_root / wid / mode
                npy = base / npy_name
                if diff_bg:
                    img = _panel_image_path(base, mode, diff_bg=True, heatmap_npy=None)
                else:
                    img = _render_npy_heatmap(np.load(npy)) if npy.is_file() else None
                if img is not None:
                    imgs.append(img)
            if imgs:
                rows_imgs.append(imgs)
                row_labels.append(wid)
        if not rows_imgs:
            continue
        w, h = rows_imgs[0][0].size
        gap, header = 6, 28
        title = "NF diff vs bg (vitacsim corrected)" if diff_bg else "NF fn field heatmap"
        canvas = Image.new("RGB", (gap + len(MODES) * (w + gap), gap + len(rows_imgs) * (h + header + gap)), (20, 20, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((gap + 4, 2), title, fill=(200, 200, 200))
        for ci, mode in enumerate(MODES):
            draw.text((gap + ci * (w + gap) + 4, gap + 2), mode, fill=(180, 180, 180))
        for ri, (imgs, wid) in enumerate(zip(rows_imgs, row_labels)):
            y = gap + ri * (h + header + gap)
            draw.text((gap + 4, y + 4), wid, fill=(220, 220, 220))
            for ci, img in enumerate(imgs):
                canvas.paste(img, (gap + ci * (w + gap), y + header))
        p = nf_root / suffix if "diff" in suffix else out_dir / suffix
        p.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(p)
        print(f"[INFO] panel -> {p}")


def _build_sf_panel(lateral_root: Path, out_path: Path, *, fx: float, weights: tuple[str, ...]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN] PIL missing; skip SF panel")
        return

    rows_imgs: list[list] = []
    row_labels: list[str] = []
    for wid in weights:
        imgs = []
        for mode in MODES:
            base = _sf_trial_dir(lateral_root, wid, mode, fx)
            npy = base / "tactile_shear_force.npy"
            img = _panel_image_path(base, mode, diff_bg=True, heatmap_npy=npy)
            if img is None:
                img = _panel_image_path(base, mode, diff_bg=False, heatmap_npy=npy)
            if img is not None:
                imgs.append(img)
        if imgs:
            rows_imgs.append(imgs)
            row_labels.append(wid)

    if not rows_imgs:
        print("[WARN] no SF RGB for panel")
        return

    w, h = rows_imgs[0][0].size
    gap, header = 6, 28
    canvas = Image.new("RGB", (gap + len(MODES) * (w + gap), gap + len(rows_imgs) * (h + header + gap)), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap + 4, 2), f"SF diff-bg / shear @ Fx={fx}N ({', '.join(weights)})", fill=(200, 200, 200))
    for ci, mode in enumerate(MODES):
        draw.text((gap + ci * (w + gap) + 4, gap + 2), mode, fill=(180, 180, 180))
    for ri, (imgs, wid) in enumerate(zip(rows_imgs, row_labels)):
        y = gap + ri * (h + header + gap)
        draw.text((gap + 4, y + 4), wid, fill=(220, 220, 220))
        for ci, img in enumerate(imgs):
            canvas.paste(img, (gap + ci * (w + gap), y + header))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[INFO] panel -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nf-root", type=str, default="logs/vitacsim_validation/normal_force")
    parser.add_argument("--lateral-root", type=str, default="logs/vitacsim_validation/shear_force/lateral")
    parser.add_argument("--out-dir", type=str, default="logs/vitacsim_validation")
    parser.add_argument("--sf-panel-fx", type=float, default=SF_PANEL_FX)
    parser.add_argument("--sf-bar-fx", type=float, default=SF_BAR_FX)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    nf_root = Path(args.nf_root).expanduser().resolve()
    lat_root = Path(args.lateral_root).expanduser().resolve()

    nf_rows = _load_nf(nf_root)
    sf_rows = _load_sf(lat_root)

    _plot_nf_curves(nf_rows, out_dir)
    _build_nf_panels(nf_root, out_dir)

    _plot_sf_curves(
        sf_rows,
        out_dir,
        weights=SF_PRIMARY_WEIGHTS,
        filename="rebattle_sf_main.png",
        title_suffix=" [rebattle main: W100+W200, valid only]",
        valid_only=True,
    )
    _plot_sf_curves(
        sf_rows,
        out_dir,
        weights=WEIGHT_ORDER,
        filename="curve_sf_ft_vs_fx_all_valid.png",
        title_suffix=" [all weights, valid only]",
        valid_only=True,
    )
    fx_panel = _resolve_panel_fx(sf_rows, float(args.sf_panel_fx))
    bar_fx = _resolve_panel_fx(sf_rows, float(args.sf_bar_fx))
    _plot_sf_bar(sf_rows, out_dir, fx=bar_fx)

    _build_sf_panel(lat_root, lat_root / "panel_sweep_rgb_diff_bg.png", fx=fx_panel, weights=SF_PANEL_WEIGHTS)
    _build_sf_panel(lat_root, out_dir / "panel_sf_lateral_rgb.png", fx=fx_panel, weights=SF_PANEL_WEIGHTS)
    _build_sf_panel(lat_root, out_dir / "rebattle_sf_panel.png", fx=fx_panel, weights=SF_PANEL_WEIGHTS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
