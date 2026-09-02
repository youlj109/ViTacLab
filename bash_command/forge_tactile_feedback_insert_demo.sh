#!/usr/bin/env bash
# Run tactile-feedback-driven insertion demo (independent from expert replay case).
#
# Usage:
#   bash bash_command/forge_tactile_feedback_insert_demo.sh
#
# Optional env vars:
#   STEPS=900
#   SEED=42
#   OUT_VIDEO=logs/tactile_feedback_demo/forge_tactile_insert.mp4
#   WITH_TACTILE_PANEL=1
#   SAVE_TACTILE_NPZ=1
#   OUT_NPZ=logs/tactile_feedback_demo/forge_tactile_insert_tactile.npz
#   EXTRA_DEMO_ARGS="--geo_xy_weight 0.9 --prealign_xy_gate 0.06"

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

STEPS="${STEPS:-900}"
SEED="${SEED:-42}"
OUT_VIDEO="${OUT_VIDEO:-logs/tactile_feedback_demo/forge_tactile_insert.mp4}"
WITH_TACTILE_PANEL="${WITH_TACTILE_PANEL:-1}"
SAVE_TACTILE_NPZ="${SAVE_TACTILE_NPZ:-1}"
OUT_NPZ="${OUT_NPZ:-logs/tactile_feedback_demo/forge_tactile_insert_tactile.npz}"
EXTRA_DEMO_ARGS="${EXTRA_DEMO_ARGS:-}"

echo "============================================================"
echo "[RUN] tactile-feedback insertion demo"
echo "[STEPS] $STEPS"
echo "[OUT]   $OUT_VIDEO"
echo "[PANEL] $WITH_TACTILE_PANEL"
echo "[NPZ]   $SAVE_TACTILE_NPZ"
echo "============================================================"

EXTRA_ARGS=()
if [[ "$WITH_TACTILE_PANEL" == "1" ]]; then
  EXTRA_ARGS+=(--with_tactile_panel)
fi
if [[ "$SAVE_TACTILE_NPZ" == "1" ]]; then
  EXTRA_ARGS+=(--save_tactile_npz --tactile_npz_path "$OUT_NPZ")
fi

python scripts/demo/demo_forge_tactile_feedback_insert.py \
  --headless \
  --enable_cameras \
  --task peg_insert \
  --steps "$STEPS" \
  --seed "$SEED" \
  --output_video "$OUT_VIDEO" \
  "${EXTRA_ARGS[@]}" \
  ${EXTRA_DEMO_ARGS}

echo "[DONE] tactile-feedback insertion demo finished"
echo "[VIDEO] $(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_VIDEO")"
if [[ "$SAVE_TACTILE_NPZ" == "1" ]]; then
  echo "[NPZ]   $(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_NPZ")"
fi
