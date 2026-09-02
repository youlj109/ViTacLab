#!/usr/bin/env python3
"""
Inspect USD file prim hierarchy. Run in Isaac Sim Python or any env with pxr (e.g. usd-core).

Usage:
  python scripts/debug/inspect_usd_structure.py [path_to.usd]
  Default: source/ViTacLab/ViTacLab/assets/data/Objects/Bottle/3517/mobility.usd
"""
import os
import sys

# This file lives in scripts/debug/ — repo root is two levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_USD = os.path.join(
    REPO_ROOT,
    "source/ViTacLab/ViTacLab/assets/data/Objects/Bottle/3517/mobility.usd",
)


def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USD
    if not os.path.isfile(usd_path):
        print(f"File not found: {usd_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        print("Requires pxr (e.g. run in Isaac Sim Python or: pip install usd-core)", file=sys.stderr)
        sys.exit(1)

    stage = Usd.Stage.Open(usd_path)
    print(f"# USD: {usd_path}\n")

    def print_prim(prim, indent=0):
        prefix = "  " * indent
        kind = prim.GetTypeName()
        path = prim.GetPath()
        print(f"{prefix}{path} [{kind}]")
        for child in sorted(prim.GetChildren(), key=lambda p: str(p.GetPath())):
            print_prim(child, indent + 1)

    for prim in stage.GetPseudoRoot().GetChildren():
        print_prim(prim)


if __name__ == "__main__":
    main()
