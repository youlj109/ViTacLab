#!/usr/bin/env bash
# Task 3 prep: run advisor NF sweep for TacSL baseline + ViTacSim full (same scene, both modes).
#
# Usage (ViTacLab repo root):
#   bash bash_command/run_vitacsim_calibration_sweep_dual.sh
#   SKIP_EXISTING=0 DEVICE=cuda:0 bash bash_command/run_vitacsim_calibration_sweep_dual.sh
#
# Outputs:
#   logs/vitacsim_calibration/sweep/normal_force/G*/tacsl/
#   logs/vitacsim_calibration/sweep/normal_force/G*/vitacsim/
#
# Next: bash bash_command/run_task3_advisor_validation.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_SH="$ROOT_DIR/bash_command/run_vitacsim_calibration_sweep.sh"
PROFILE="${PROFILE:-advisor}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

for MODE in tacsl vitacsim; do
  echo "======== dual sweep: SENSOR_MODE=$MODE profile=$PROFILE ========"
  PROFILE="$PROFILE" SENSOR_MODE="$MODE" SKIP_EXISTING="$SKIP_EXISTING" bash "$SWEEP_SH"
done

echo "[DONE] dual sweep (tacsl + vitacsim) profile=$PROFILE"
echo "[NEXT] bash bash_command/run_task3_advisor_validation.sh"
