#!/usr/bin/env bash
# Gripper shear validation sweep (Franka parallel jaw + W100 + lateral EE action).
#
# Usage:
#   bash bash_command/demo_vitacsim_gripper_shear_validation.sh
#   SHEAR_ACTION=0.75 MODES=vitacsim bash bash_command/demo_vitacsim_gripper_shear_validation.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
OUT_DIR="${OUT_DIR:-logs/vitacsim_validation/shear_force/gripper}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHT="${WEIGHT:-W100}"
MODES="${MODES:-tacsl vitacsim}"
SHEAR_ACTIONS="${SHEAR_ACTIONS:-0.0 0.25 0.5 0.75 1.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_one() {
  local shear="$1"
  local mode="$2"
  local tag
  tag="$(python3 - <<PY
a = float("${shear}")
s = f"{a:.3f}".rstrip("0").rstrip(".")
print("S" + s.replace("-", "m"))
PY
)"
  local out_sub="$OUT_DIR/$tag/$WEIGHT/$mode"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_sub/summary.json" ]]; then
    echo "[SKIP] shear=$shear mode=$mode"
    return 0
  fi
  echo "============================================================"
  echo "[RUN] weight=$WEIGHT shear_action=$shear mode=$mode"
  echo "============================================================"
  PYTHONUNBUFFERED=1 "$ISAACLAB_SH" -p scripts/demo/demo_vitacsim_gripper_shear_validation.py \
    --headless --enable_cameras --device "$DEVICE" \
    --weight-id "$WEIGHT" --sensor-mode "$mode" \
    --shear-action "$shear" --out-dir "$OUT_DIR"
}

for shear in $SHEAR_ACTIONS; do
  for mode in $MODES; do
    run_one "$shear" "$mode"
  done
done

python3 scripts/demo/summarize_vitacsim_gripper_shear_validation.py --root "$OUT_DIR" 2>/dev/null || true

echo "[DONE] outputs under $OUT_DIR"
