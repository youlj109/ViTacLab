#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Offline USD edit: remove nested PhysX articulation APIs from GelSight prims so that
# ``ArticulationCfg`` can resolve a **single** articulation under ``/World/envs/env_*/Robot``.
#
# Symptom without this fix::
#
#   RuntimeError: Failed to find a single articulation when resolving '.../Robot'.
#   Found multiple '[.../root_joint>, .../gelsight_lfdistal>, ...]'
#
# Run with Isaac Sim / Isaac Lab Python (provides ``pxr``), e.g. from ViTacLab repo root::
#
#     ./isaaclab.sh -p scripts/debug/strip_gelsight_articulation_from_usd.py --in-place
#     ./isaaclab.sh -p scripts/debug/strip_gelsight_articulation_from_usd.py -o /tmp/out.usd
#
# References: Isaac Lab ``check_floating_base_made_fixed.py`` (RemoveAPI pattern).

"""Strip GelSight nested articulation APIs from a robot USD (UsdPhysics + PhysxSchema)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(16):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _default_input() -> Path:
    return (
        _repo_root()
        / "source/ViTacLab/ViTacLab/assets/data/Robots/Franka/Franka_R15/franka_mimic_edit.usd"
    )


def _strip_gelsight_articulation(
    stage,
    pattern: str,
    dry_run: bool,
) -> list[tuple[str, list[str]]]:
    """Return list of (prim_path, removed_api_names)."""
    from pxr import Usd, UsdPhysics

    try:
        from pxr import PhysxSchema  # type: ignore[attr-defined]
    except ImportError:
        PhysxSchema = None  # type: ignore[assignment]

    removed: list[tuple[str, list[str]]] = []
    pat = pattern.lower()

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsValid():
            continue
        path_str = str(prim.GetPath())
        if pat not in path_str.lower():
            continue

        apis_removed: list[str] = []
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            apis_removed.append("UsdPhysics.ArticulationRootAPI")
            if not dry_run:
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            apis_removed.append("PhysxSchema.PhysxArticulationAPI")
            if not dry_run:
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        if apis_removed:
            removed.append((path_str, apis_removed))

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove UsdPhysics.ArticulationRootAPI + PhysxSchema.PhysxArticulationAPI "
        "from prims whose path contains a pattern (default: gelsight)."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Input USD path (default: ur10e_shadow_left_hand_glb_withtac_v2.usd under repo).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output USD path. Default: <input_stem>_no_gelsight_articulation.usd (unless --in-place).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write back to --input (same file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prims that would be edited; do not save.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="gelsight",
        help="Case-insensitive substring matched against prim path (default: gelsight).",
    )

    args = parser.parse_args()
    in_path = args.input if args.input is not None else _default_input()
    in_path = in_path.expanduser().resolve()

    if not in_path.is_file():
        print(f"[ERROR] Input USD not found: {in_path}", file=sys.stderr)
        return 1

    try:
        from pxr import Usd
    except ImportError:
        print(
            "[ERROR] Requires pxr (run with Isaac Sim / Isaac Lab Python, e.g. ./isaaclab.sh -p ...).",
            file=sys.stderr,
        )
        return 1

    stage = Usd.Stage.Open(str(in_path))
    if not stage:
        print(f"[ERROR] Failed to open stage: {in_path}", file=sys.stderr)
        return 1

    removed = _strip_gelsight_articulation(stage, args.pattern, dry_run=args.dry_run)

    if not removed:
        print(f"[INFO] No prims matched pattern {args.pattern!r} with articulation APIs; nothing to do.")
        return 0

    verb = "would strip" if args.dry_run else "stripped"
    print(f"[INFO] Matched {len(removed)} prim(s); {verb} API(s):")
    for path_str, apis in removed:
        print(f"  {path_str}")
        print(f"    -> {', '.join(apis)}")

    if args.dry_run:
        print("[INFO] Dry run: not saving.")
        return 0

    if args.in_place:
        out_path = in_path
    elif args.output is not None:
        out_path = args.output.expanduser().resolve()
    else:
        out_path = in_path.parent / f"{in_path.stem}_no_gelsight_articulation{in_path.suffix}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = stage.GetRootLayer()
    root.Export(str(out_path))
    print(f"[INFO] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
