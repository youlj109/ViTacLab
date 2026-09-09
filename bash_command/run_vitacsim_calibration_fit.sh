#!/usr/bin/env bash
# After real captures are in data/calibration/tactile/real/: track markers + joint fit.
#
# Usage:
#   bash bash_command/run_vitacsim_calibration_fit.sh
#   REAL_ROOT=data/calibration/tactile/real bash bash_command/run_vitacsim_calibration_fit.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SIM_ROOT="${SIM_ROOT:-logs/vitacsim_calibration/sweep}"
REAL_ROOT="${REAL_ROOT:-data/calibration/tactile/real}"
OUT_JSON="${OUT_JSON:-data/calibration/tactile/fitted_params.json}"

echo "[1/3] Track marker displacements from real rgb.png ..."
if python3 scripts/calibration/track_real_markers.py --real-root "$REAL_ROOT"; then
  echo "[OK] marker tracking"
else
  echo "[WARN] No real rgb.png yet — add captures under $REAL_ROOT then re-run."
  echo "       See sim_reference/: bash -c 'python3 scripts/calibration/export_sim_reference.py'"
  exit 1
fi

echo "[2/3] Joint fit RGB + marker ..."
python3 scripts/calibration/fit_vitacsim_rgb_marker.py \
  --sim-root "$SIM_ROOT" \
  --real-root "$REAL_ROOT" \
  --out "$OUT_JSON"

echo "[3/3] Done. Apply marker_displacement_gain from:"
echo "  $OUT_JSON"
echo "  validation_gelsight_render_cfg(fitted_params_path=\"$OUT_JSON\")"
