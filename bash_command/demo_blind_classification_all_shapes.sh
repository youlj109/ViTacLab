#!/usr/bin/env bash
# Run Blind Classification demo for cube / sphere / cone sequentially.
#
# Usage:
#   bash bash_command/demo_blind_classification_all_shapes.sh
#   SAVE_OUTCOME=completed DEVICE=cuda:0 bash bash_command/demo_blind_classification_all_shapes.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAVE_OUTCOME="${SAVE_OUTCOME:-completed}"
DEVICE="${DEVICE:-cuda:0}"
SHAPE_LABELS="${SHAPE_LABELS:-0 1 2}"

for label in $SHAPE_LABELS; do
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo "Blind Classification shape_label=$label"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  SHAPE_LABEL="$label" \
    SAVE_OUTCOME="$SAVE_OUTCOME" \
    DEVICE="$DEVICE" \
    MODE=full \
    bash bash_command/demo_blind_classification_collection_replay.sh
done

echo ""
echo "[DONE] All Blind Classification shape demos finished under logs/demo_blind_classification/"
