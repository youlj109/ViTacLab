#!/usr/bin/env bash
# Lateral (shear) force validation — per-weight Fx caps + contact validity (sf_lateral_v2).
#
# Usage:
#   bash bash_command/demo_vitacsim_lateral_force_validation.sh
#   WEIGHT=W100 MODES=vitacsim bash bash_command/demo_vitacsim_lateral_force_validation.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
OUT_DIR="${OUT_DIR:-logs/vitacsim_validation/shear_force/lateral}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS="${WEIGHTS:-W200 W100 W050 W020 W010}"
WEIGHT="${WEIGHT:-}"
MODES="${MODES:-tacsl vitacsim}"
LATERAL_FY="${LATERAL_FY:-0.0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
FINGER_ROOT_Z="${FINGER_ROOT_Z:-0.444}"

fx_list_for_weight() {
  case "$1" in
    W200) echo "0.0 0.1 0.2 0.3 0.5" ;;
    W100) echo "0.0 0.05 0.1 0.15 0.2" ;;
    W050) echo "0.0 0.03 0.05 0.08 0.1" ;;
    W020) echo "0.0 0.01 0.02 0.03 0.05" ;;
    W010) echo "0.0 0.005 0.01 0.015 0.02" ;;
    *) echo "0.0 0.1 0.2" ;;
  esac
}

run_one() {
  local wid="$1"
  local fx="$2"
  local mode="$3"
  local out_sub="$OUT_DIR/Fx${fx}_Fy${LATERAL_FY}/$wid/$mode"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_sub/summary.json" ]]; then
    schema="$(python3 - <<PY
import json
from pathlib import Path
p = Path("$out_sub/summary.json")
print(json.loads(p.read_text()).get("output_schema", ""))
PY
)"
    if [[ "$schema" == "sf_lateral_v2" ]]; then
      echo "[SKIP] weight=$wid Fx=$fx mode=$mode (v2 summary exists)"
      return 0
    fi
  fi
  echo "============================================================"
  echo "[RUN] weight=$wid Fx=$fx Fy=$LATERAL_FY mode=$mode finger_z=$FINGER_ROOT_Z"
  echo "============================================================"
  PYTHONUNBUFFERED=1 "$ISAACLAB_SH" -p scripts/demo/demo_vitacsim_lateral_force_validation.py \
    --headless --enable_cameras --device "$DEVICE" \
    --weight-id "$wid" --sensor-mode "$mode" \
    --lateral-force-x "$fx" --lateral-force-y "$LATERAL_FY" \
    --finger-root-z "$FINGER_ROOT_Z" \
    --out-dir "$OUT_DIR"
}

weight_list() {
  if [[ -n "$WEIGHT" ]]; then
    echo "$WEIGHT"
  else
    echo "$WEIGHTS"
  fi
}

for wid in $(weight_list); do
  for fx in $(fx_list_for_weight "$wid"); do
    for mode in $MODES; do
      run_one "$wid" "$fx" "$mode"
    done
  done
done

python3 scripts/demo/summarize_vitacsim_lateral_force_validation.py --root "$OUT_DIR"
python3 scripts/demo/plot_vitacsim_validation_beta.py --lateral-root "$OUT_DIR" 2>/dev/null || true

echo "[DONE] outputs under $OUT_DIR"
