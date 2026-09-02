#!/usr/bin/env bash
# Build Xense polycalib from logs/file-000.mp4 and install into xense_lab_data/.
#
# Usage (ViTacLab repo root):
#   bash bash_command/run_xense_polycalib.sh
#   bash bash_command/run_xense_polycalib.sh --skip-import

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/calibration/build_xense_polycalib.py "$@"
