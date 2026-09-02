#!/usr/bin/env bash
# Task 3: physical validation — Real | TacSL | ViTacSim (advisor NF).
# Prerequisite: bash bash_command/run_vitacsim_calibration_sweep_dual.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/calibration/report_task3_validation.py "$@"

echo ""
echo "Open panels under: logs/vitacsim_validation/task3/"
