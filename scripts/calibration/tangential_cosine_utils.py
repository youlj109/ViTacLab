#!/usr/bin/env python3
"""Rebuttal-style tangential cosine similarity (sim-only, PhysX GT vs constructed fields)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from calibration_io import LATERAL_W100_FX, _fx_tag, sim_lateral_dir


def load_shear_field_npy(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        # legacy flat (P, 2)
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr
    return None


def shear_field_to_flat(field: np.ndarray | None) -> np.ndarray | None:
    if field is None:
        return None
    arr = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
    if arr.ndim == 3:
        return arr.reshape(-1, 2)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr
    return None


def cosine_similarity_shear(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity between two shear fields (any shape ending in 2)."""
    fa = shear_field_to_flat(a)
    fb = shear_field_to_flat(b)
    if fa is None or fb is None or fa.shape != fb.shape:
        return float("nan")
    va = fa.reshape(-1)
    vb = fb.reshape(-1)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(va, vb) / (na * nb))


def _case_dir(sim_root: Path, fx: float, *, sensor_mode: str) -> Path:
    return sim_lateral_dir(sim_root, "W100", fx, sensor_mode=sensor_mode)


def compute_lateral_cosine_row(
    sim_root: Path,
    fx: float,
    *,
    weight_id: str = "W100",
) -> dict[str, Any]:
    """One lateral Fx row: cos(GT, TacSL) and cos(GT, ViTacSim)."""
    tag = _fx_tag(fx)
    gt_path = _case_dir(sim_root, fx, sensor_mode="vitacsim") / "physx_shear_gt.npy"
    # GT is mode-independent; fall back to tacsl dir if saved there first.
    if not gt_path.is_file():
        gt_path = _case_dir(sim_root, fx, sensor_mode="tacsl") / "physx_shear_gt.npy"

    gt = load_shear_field_npy(gt_path)
    tac_path = _case_dir(sim_root, fx, sensor_mode="tacsl") / "tactile_shear_force.npy"
    vit_path = _case_dir(sim_root, fx, sensor_mode="vitacsim") / "tactile_shear_force.npy"
    tac = load_shear_field_npy(tac_path)
    vit = load_shear_field_npy(vit_path)

    return {
        "tag": tag,
        "fx_n": float(fx),
        "weight_id": weight_id,
        "gt_present": gt is not None,
        "tacsl_present": tac is not None,
        "vitacsim_present": vit is not None,
        "cos_gt_tacsl": cosine_similarity_shear(gt, tac),
        "cos_gt_vitacsim": cosine_similarity_shear(gt, vit),
        "gt_path": str(gt_path) if gt is not None else None,
        "tacsl_path": str(tac_path) if tac is not None else None,
        "vitacsim_path": str(vit_path) if vit is not None else None,
    }


def compute_lateral_cosine_table(
    sim_root: Path,
    *,
    fx_values: tuple[float, ...] = LATERAL_W100_FX,
) -> list[dict[str, Any]]:
    sim_root = Path(sim_root).expanduser().resolve()
    return [compute_lateral_cosine_row(sim_root, fx) for fx in fx_values]
