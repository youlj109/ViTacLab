#!/usr/bin/env python3
"""Count True/False validation results written by policy rollout scripts.

Usage:
    python scripts/data_collection/tools/count_validation_success.py \
        data/validation/Isaac-UR10eShadowHand-BlindGrasp-Direct-v0/ViTacDP
"""

from __future__ import annotations

import argparse
from pathlib import Path


def count_success_results(base_path: str) -> int:
    base_dir = Path(base_path).expanduser()
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"ERROR: directory not found: {base_dir}")
        return 1

    print(f"{'subdir':<48} | {'True':<8} | {'False':<8}")
    print("-" * 72)
    total_true = 0
    total_false = 0
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        target_file = subdir / "all_success.txt"
        if not target_file.exists():
            print(f"{subdir.name:<48} | missing all_success.txt")
            continue
        content = target_file.read_text(encoding="utf-8", errors="ignore")
        true_count = content.count("True")
        false_count = content.count("False")
        total_true += true_count
        total_false += false_count
        print(f"{subdir.name:<48} | {true_count:<8} | {false_count:<8}")
    print("=" * 72)
    print(f"{'TOTAL':<48} | {total_true:<8} | {total_false:<8}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Count True/False entries in validation all_success.txt files.")
    parser.add_argument("base_path", help="Directory containing validation subfolders with all_success.txt.")
    args = parser.parse_args()
    return count_success_results(args.base_path)


if __name__ == "__main__":
    raise SystemExit(main())
