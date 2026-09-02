#!/usr/bin/env python3
"""[Deprecated for advisor] Matplotlib engineering diagram — use sim screenshot instead.

Prefer::

    ../IsaacLab/isaaclab.sh -p scripts/demo/capture_vitacsim_nf_scene_screenshot.py \\
        --headless --enable_cameras --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_weight_spec():
    import sys

    spec_path = (
        Path(__file__).resolve().parents[2]
        / "source/ViTacLab/ViTacLab/tasks/direct/vitacsim_validation/weight_spec.py"
    )
    mod_name = "validation_weight_spec"
    spec = importlib.util.spec_from_file_location(mod_name, spec_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.GEOMETRY, mod.LAYOUT, mod.WEIGHT_MASS_KG


GEOMETRY, LAYOUT, WEIGHT_MASS_KG = _load_weight_spec()


def render(out_path: Path, *, weight_id: str = "W100") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle, Rectangle, Wedge

    mass_g = WEIGHT_MASS_KG[weight_id] * 1000.0
    g = GEOMETRY
    lay = LAYOUT

    fig = plt.figure(figsize=(11, 5.5), facecolor="#1a1a1e")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.08)
    ax_side = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])

    for ax in (ax_side, ax_top):
        ax.set_facecolor("#1a1a1e")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#444")

    # --- Side view (x-z), pad horizontal, +z up ---
    pad_y = 0.0
    pad_thick = 0.006
    pad_len = 0.055
    finger_h = 0.028
    bracket_w = 0.012

    # Finger bracket + pad (GelSight short finger, horizontal)
    ax_side.add_patch(Rectangle((-pad_len * 0.55, pad_y), pad_len, pad_thick, fc="#5a8fd4", ec="#9ec5ff", lw=1.2))
    ax_side.add_patch(Rectangle((-pad_len * 0.55 - bracket_w, pad_y + pad_thick), bracket_w, finger_h, fc="#555", ec="#888"))
    ax_side.add_patch(Rectangle((pad_len * 0.55, pad_y + pad_thick), bracket_w, finger_h, fc="#555", ec="#888"))
    ax_side.text(0, pad_y - 0.008, "GelSight elastomer pad (horizontal)", ha="center", color="#ccc", fontsize=9)

    # Weight composite (simplified side silhouette)
    wx = 0.0
    wz = pad_y + pad_thick
    r = lay.large_radius
    h_main = lay.main_cylinder_height
    chamfer = lay.chamfer

    # bottom chamfer wedge
    ax_side.add_patch(Wedge((wx, wz + chamfer), r, 180, 360, width=r - lay.flat_radius, fc="#c49a6c", ec="#e8c9a8", lw=0.8))
    ax_side.add_patch(Rectangle((wx - lay.flat_radius, wz + chamfer), 2 * lay.flat_radius, h_main, fc="#c49a6c", ec="#e8c9a8", lw=0.8))
    ax_side.add_patch(Wedge((wx, wz + lay.large_height - chamfer), r, 0, 180, width=r - lay.flat_radius, fc="#c49a6c", ec="#e8c9a8", lw=0.8))
    # stem
    sr = g.stem_diameter * 0.5
    ax_side.add_patch(Rectangle((wx - sr, lay.stem_bottom_z + wz), 2 * sr, g.stem_height, fc="#b8895a", ec="#e8c9a8", lw=0.6))
    # sphere
    ax_side.add_patch(Circle((wx, lay.sphere_center_z + wz), g.sphere_radius, fc="#d4aa7d", ec="#e8c9a8", lw=0.6))

    ax_side.annotate(
        "",
        xy=(wx + 0.022, wz + 0.004),
        xytext=(wx + 0.022, wz + lay.large_height + g.sphere_radius + 0.018),
        arrowprops=dict(arrowstyle="-|>", color="#ff6b6b", lw=2),
    )
    ax_side.text(wx + 0.026, wz + lay.total_height * 0.55, r"$F_n=m g$", color="#ff8a8a", fontsize=11, rotation=90, va="center")

    ax_side.plot([wx - r, wx + r], [wz, wz], color="#7ee787", lw=1.5, ls="--")
    ax_side.text(wx, wz - 0.006, "contact (bottom center = origin)", ha="center", color="#7ee787", fontsize=8)

    ax_side.set_xlim(-0.05, 0.05)
    ax_side.set_ylim(-0.015, 0.05)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("x (m)", color="#bbb")
    ax_side.set_ylabel("z (m)", color="#bbb")
    ax_side.set_title(f"NF setup (side): {weight_id} ({mass_g:.0f} g) on pad, no gripper", color="#eee", fontsize=11)

    # --- Top view (x-y) ---
    ax_top.add_patch(Circle((0, 0), lay.large_radius, fill=False, ec="#e8c9a8", lw=1.2))
    ax_top.add_patch(Circle((0, 0), lay.flat_radius, fill=False, ec="#7ee787", ls="--", lw=1.0))
    ax_top.add_patch(Circle((0, 0), g.stem_diameter * 0.5, fc="#b8895a", ec="#e8c9a8", lw=0.6))
    pad_r = 0.028
    ax_top.add_patch(
        mpatches.FancyBboxPatch(
            (-pad_r, -pad_r),
            2 * pad_r,
            2 * pad_r,
            boxstyle="round,pad=0.004",
            fill=False,
            ec="#5a8fd4",
            lw=1.5,
            ls="-.",
        )
    )
    ax_top.text(0, -pad_r - 0.006, "tactile pad footprint (approx.)", ha="center", color="#5a8fd4", fontsize=8)
    ax_top.text(lay.large_radius + 0.004, 0, "Ø25 mm", color="#ccc", fontsize=8, va="center")
    ax_top.text(lay.flat_radius + 0.004, 0.008, "flat Ø23 mm", color="#7ee787", fontsize=7, va="center")

    ax_top.set_xlim(-0.04, 0.04)
    ax_top.set_ylim(-0.04, 0.04)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("x (m)", color="#bbb")
    ax_top.set_ylabel("y (m)", color="#bbb")
    ax_top.set_title("Top view (coaxial stack)", color="#eee", fontsize=11)

    fig.suptitle(
        "ViTacSim validation — normal force (horizontal GelSight + standard weight)",
        color="#f0f0f0",
        fontsize=12,
        y=0.98,
    )
    notes = (
        "Same geometry for W200…W010; mass only differs. "
        "Gravity settle → measure PhysX Fn & tactile RGB (sim − bg.jpg)."
    )
    fig.text(0.5, 0.02, notes, ha="center", color="#999", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] schematic -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="logs/vitacsim_validation/v2/setup_schematic_nf.png")
    parser.add_argument("--weight-id", type=str, default="W100")
    args = parser.parse_args()
    render(Path(args.out).expanduser().resolve(), weight_id=args.weight_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
