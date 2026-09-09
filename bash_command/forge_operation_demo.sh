#!/usr/bin/env bash
# Generate third-person robot-operation demo video for Forge tasks.
#
# Usage:
#   bash bash_command/forge_operation_demo.sh
#
# Optional env vars:
#   TASK=peg_insert          # peg_insert | gear_mesh | nut_thread
#   MODE=full                # baseline | normal_only | full
#   STEPS=420
#   SEED=42
#   OUT_VIDEO=logs/operation_demo/forge_peg_full.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

TASK="${TASK:-peg_insert}"
MODE="${MODE:-full}"
STEPS="${STEPS:-420}"
SEED="${SEED:-42}"
OUT_VIDEO="${OUT_VIDEO:-logs/operation_demo/forge_${TASK}_${MODE}.mp4}"

python scripts/demo/demo_forge_task_operation_video.py \
  --headless \
  --enable_cameras \
  --task "$TASK" \
  --mode "$MODE" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --output_video "$OUT_VIDEO"

echo "[DONE] operation demo video: $OUT_VIDEO"
