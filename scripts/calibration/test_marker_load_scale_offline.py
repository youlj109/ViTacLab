#!/usr/bin/env python3
"""Offline sanity check: ViTacSim marker displacement scales with PhysX load proxy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
import importlib.util

_MARKER_PY = (
    _REPO / "source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/visuotactile_marker.py"
)
_spec = importlib.util.spec_from_file_location("visuotactile_marker_offline", _MARKER_PY)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_mod.__name__ = "visuotactile_marker_offline"
sys.modules["visuotactile_marker_offline"] = _mod
_spec.loader.exec_module(_mod)
MarkerSimulator = _mod.MarkerSimulator

MM_PER_PX = 17.5 / 400.0
DEPTH_M = 0.002894
REF_FN = 0.72
EXP = 0.48
MHS = 0.0033
GAMMA_HI = 1.2
GAMMA_LO = 1.0
LOAD_T0 = 0.35
CAP = 3.0
PHYSX_FN = {"G010": 0.065, "G110": 0.72, "G210": 1.37}
REAL_MAX = {"G010": 2.24, "G110": 2.83, "G210": 3.0}


def _eff_gamma(load_scale: float) -> float:
    t = max(0.0, min(1.0, (load_scale - LOAD_T0) / max(1.0 - LOAD_T0, 1e-6)))
    return GAMMA_LO * (1.0 - t) + GAMMA_HI * t


def main() -> int:
    rest = np.load(
        _REPO
        / "source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data/marker_rest.npy"
    ).astype(np.float32)
    sim = MarkerSimulator(
        pattern="xense",
        image_height=700,
        image_width=400,
        device="cpu",
        displacement_gain=0.45,
        shear_gain=0.0,
        deadband_mm=0.008,
        max_displacement_px=CAP,
        rest_xy_override=rest,
    )
    h, w = 700, 400
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = 348.0, 200.0
    blob = torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 35.0**2)))
    depth_base = DEPTH_M * blob

    print("case  tacsl_max  vit_max  real_max  vit/real  gamma_eff")
    tacsl_max = []
    vit_max = []
    for cid, fn in PHYSX_FN.items():
        scale = (fn / REF_FN) ** EXP
        gamma = _eff_gamma(scale)
        h_tac = depth_base**GAMMA_HI
        h_vit = (depth_base**gamma) * scale
        for hm, bucket in ((h_tac, tacsl_max), (h_vit, vit_max)):
            taxim_mm = (hm * MHS * 1000.0 / MM_PER_PX).clamp(max=80.0)
            disp = sim.displacements_from_height_mm(taxim_mm)
            bucket.append(float(torch.linalg.norm(disp, dim=-1).max().item()))

    for i, cid in enumerate(PHYSX_FN):
        t, v, r = tacsl_max[i], vit_max[i], REAL_MAX[cid]
        ge = _eff_gamma((list(PHYSX_FN.values())[i] / REF_FN) ** EXP)
        print(f"{cid}  {t:8.3f}  {v:7.3f}  {r:8.2f}  {v/r:6.2f}  {ge:.2f}")

    spread_tac = max(tacsl_max) - min(tacsl_max)
    spread_vit = max(vit_max) - min(vit_max)
    if spread_tac > 0.15:
        print("[FAIL] TacSL markers should be load-flat (depth-only)")
        return 1
    if spread_vit < 0.15:
        print("[FAIL] ViTacSim markers should increase with load (p95/max spread)")
        return 1
    if not (vit_max[0] <= vit_max[1] <= vit_max[2]):
        print("[FAIL] ViTacSim marker order G010<=G110<=G210 expected")
        return 1
    if vit_max[0] < 1.5:
        print("[FAIL] ViTacSim G010 should be visible (>1.5 px)")
        return 1
    if vit_max[1] >= CAP - 0.05:
        print("[FAIL] ViTacSim G110 should stay below cap")
        return 1
    print("[OK] marker load scaling looks physical")

    # Shear force field should add +x directional marker motion (ViT path).
    sim_lo = MarkerSimulator(
        pattern="xense",
        image_height=700,
        image_width=400,
        device="cpu",
        displacement_gain=0.25,
        shear_gain=0.0,
        deadband_mm=0.008,
        max_displacement_px=CAP,
        rest_xy_override=rest,
    )
    shear_n = torch.zeros(h, w, 2)
    cy_i, cx_i = int(cy), int(cx)
    for yi in range(h):
        for xi in range(w):
            if ((xi - cx_i) ** 2 + (yi - cy_i) ** 2) ** 0.5 < 45.0:
                shear_n[yi, xi, 0] = 0.08
    shear_px = shear_n * (3.0 / 0.05)
    taxim_mm = (depth_base**GAMMA_HI * MHS * 1000.0 / MM_PER_PX).clamp(max=80.0)
    disp_nf = sim_lo.displacements_from_height_mm(taxim_mm)
    disp_sh = sim_lo.displacements_from_height_mm(taxim_mm, shear_disp_px=shear_px)
    sh_bias_x = float(disp_sh[:, 0].mean().item() - disp_nf[:, 0].mean().item())
    if sh_bias_x <= 0.05:
        print("[FAIL] +x shear field should bias marker motion in +x")
        return 1
    print(f"[OK] shear field bias_x={sh_bias_x:.3f}px (NF mean disp unchanged by cap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
