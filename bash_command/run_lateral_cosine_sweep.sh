#!/usr/bin/env bash
# Rebuttal Table 1 prep: W100 lateral Fx sweep for TacSL + ViTacSim with shear-field .npy exports.
#
# Usage (ViTacLab repo root):
#   bash bash_command/run_lateral_cosine_sweep.sh
#   SKIP_EXISTING=0 DEVICE=cuda:0 bash bash_command/run_lateral_cosine_sweep.sh
#
# Outputs:
#   logs/vitacsim_calibration/sweep/shear_force/lateral/Fx*/W100/{tacsl,vitacsim}/
#     tactile_shear_force.npy, physx_shear_gt.npy, summary.json
#
# Next:
#   python3 scripts/calibration/report_task3_validation.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ISAACLAB_SH="${ISAACLAB_SH:-../IsaacLab/isaaclab.sh}"
DEVICE="${DEVICE:-cuda:0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LAT_OUT="${LAT_OUT:-logs/vitacsim_calibration/sweep/shear_force/lateral}"
FINGER_ROOT_Z="${FINGER_ROOT_Z:-0.444}"
WEIGHT_REST_Z="${WEIGHT_REST_Z:-0.442}"
FX_VALUES=(0.0 0.05 0.1 0.15 0.2)

fmt_fx() {
  python3 - <<PY
fx=float("$1")
s=f"{fx:.3f}".rstrip("0").rstrip(".")
print(f"Fx{s.replace('-', 'm')}_Fy0")
PY
}

run_one() {
  local mode="$1"
  local fx="$2"
  local tag
  tag="$(fmt_fx "$fx")"
  local sub="$LAT_OUT/${tag}/W100/$mode"
  if [[ "$SKIP_EXISTING" == "1" && -f "$sub/summary.json" && -f "$sub/tactile_shear_force.npy" ]]; then
    echo "[SKIP] lateral $mode Fx=$fx"
    return 0
  fi
  echo "======== lateral $mode W100 Fx=$fx ========"
  PYTHONUNBUFFERED=1 "$ISAACLAB_SH" -p scripts/demo/demo_vitacsim_lateral_force_validation.py \
    --headless --enable_cameras --device "$DEVICE" \
    --weight-id W100 --sensor-mode "$mode" \
    --lateral-force-x "$fx" --lateral-force-y 0.0 \
    --out-dir "$LAT_OUT" \
    --finger-root-z "$FINGER_ROOT_Z" \
    --weight-rest-z "$WEIGHT_REST_Z"
}

for mode in tacsl vitacsim; do
  for fx in "${FX_VALUES[@]}"; do
    run_one "$mode" "$fx"
  done
done

echo "[DONE] lateral cosine sweep -> $LAT_OUT"
echo "[NEXT] python3 scripts/calibration/report_task3_validation.py"
