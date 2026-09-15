#!/usr/bin/env bash
# Build Xense polycalib from ball-indent frames and install into xense_lab_data/.
# The true no-contact pair defaults to advisor_processed/bg.jpg + bg_clean.jpg;
# ball_calib_raw/bg/no_contact.png is intentionally not used as the fit baseline.
#
# Usage (ViTacLab repo root):
#   bash bash_command/run_xense_polycalib.sh
#   bash bash_command/run_xense_polycalib.sh --skip-import

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/calibration/build_xense_polycalib.py "$@"
