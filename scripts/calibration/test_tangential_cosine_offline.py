#!/usr/bin/env python3
"""Offline unit test for rebuttal-style shear-field cosine similarity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tangential_cosine_utils import cosine_similarity_shear  # noqa: E402


def main() -> int:
    gt = np.zeros((4, 5, 2), dtype=np.float64)
    gt[1, 2] = (1.0, 0.0)
    gt[2, 3] = (0.5, 0.0)

    tac = gt * 0.1
    vit = gt.copy()
    vit[0, 0] = (0.2, 0.1)

    c_tac = cosine_similarity_shear(gt, tac)
    c_vit = cosine_similarity_shear(gt, vit)
    c_zero = cosine_similarity_shear(gt, np.zeros_like(gt))

    if not (0.97 < c_vit < 1.01):
        print(f"[FAIL] expected cos(GT,ViT) ~ 1.0, got {c_vit}")
        return 1
    if not (0.99 < c_tac < 1.01):
        print(f"[FAIL] expected cos(GT,scaled) ~ 1.0, got {c_tac}")
        return 1
    if not np.isnan(c_zero):
        print(f"[FAIL] zero field cosine should be nan, got {c_zero}")
        return 1

    print(f"[OK] cosine similarity: tac={c_tac:.3f} vit={c_vit:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
