#!/usr/bin/env bash
# Run ViTacBench bimanual tactile demos: Blind Drop (phase 1) + Blind Targeted Drop (phase 2).
#
# Usage:
#   bash bash_command/demo_bi_blind_bin_drop_all_phases.sh
#   PHASES="1 2" SAVE_OUTCOME=completed bash bash_command/demo_bi_blind_bin_drop_all_phases.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAVE_OUTCOME="${SAVE_OUTCOME:-completed}"
DEVICE="${DEVICE:-cuda:0}"
OBJECT_INIT_INDEX="${OBJECT_INIT_INDEX:-0}"
PHASES="${PHASES:-1 2}"

for phase in $PHASES; do
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo "BiBlindBinDrop phase $phase"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  PHASE="$phase" \
    SAVE_OUTCOME="$SAVE_OUTCOME" \
    DEVICE="$DEVICE" \
    OBJECT_INIT_INDEX="$OBJECT_INIT_INDEX" \
    MODE=full \
    bash bash_command/demo_bi_blind_bin_drop_collection_replay.sh
done

echo ""
echo "[DONE] BiBlindBinDrop phases finished under logs/demo_bi_blind_bin_drop/"
