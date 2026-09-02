#!/usr/bin/env bash
# Task 2: advisor mp4 -> real rgb -> marker track -> joint fit -> report
#
# Usage (ViTacLab repo root):
#   bash bash_command/run_task2_advisor_calibration.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${CONDA_PREFIX:-}" ]] && [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate env_isaaclab_510test 2>/dev/null || true
fi

REAL_ROOT="${REAL_ROOT:-data/calibration/tactile/real}"
SIM_ROOT="${SIM_ROOT:-logs/vitacsim_calibration/sweep}"
OUT_JSON="${OUT_JSON:-data/calibration/tactile/fitted_params.json}"
BG_PROCESSED="${BG_PROCESSED:-data/calibration/tactile/advisor_processed/bg_clean.jpg}"
MARKER_PATTERN="${MARKER_PATTERN:-xense}"

echo "======== Task 2 Step 1/4: Import advisor mp4 ========"
python3 scripts/calibration/import_advisor_tactile_videos.py --install-bg

echo "======== Task 2 Step 2/4: Track real markers (lab Xense) ========"
python3 scripts/calibration/track_real_markers.py \
  --real-root "$REAL_ROOT" \
  --profile advisor \
  --pattern "$MARKER_PATTERN"

echo "======== Task 2 Step 3/4: Joint fit (advisor profile) ========"
python3 scripts/calibration/fit_vitacsim_rgb_marker.py \
  --profile advisor \
  --sim-root "$SIM_ROOT" \
  --real-root "$REAL_ROOT" \
  --bg-path "$BG_PROCESSED" \
  --out "$OUT_JSON"

echo "======== Task 2 Step 4/4: Report ========"
python3 scripts/calibration/report_task2_calibration.py \
  --real-root "$REAL_ROOT" \
  --sim-root "$SIM_ROOT" \
  --fitted "$OUT_JSON"

echo "[DONE] Task 2 pipeline complete."
echo "  fitted_params: $OUT_JSON"
echo "  report: logs/vitacsim_calibration/task2/TASK2_CALIBRATION_REPORT.md"
echo "  real marker_pattern: $MARKER_PATTERN (lab Xense, native 400x700, M2 nut sim)"
echo "  sim sweep: PROFILE=advisor MARKER_PATTERN=xense bash bash_command/run_vitacsim_calibration_sweep.sh"
