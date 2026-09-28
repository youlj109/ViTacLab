#!/usr/bin/env bash
# Run Blind Grasp demo for phases 1–3 sequentially (collect + validate + panel videos).
#
# Phase 3 uses Isaac-UR10eShadowHand-BlindGraspReplay-Direct-v0 (alias env of BlindGrasp).
#
# Usage:
#   bash bash_command/demo_blind_grasp_all_phases.sh
#   SAVE_OUTCOME=completed DEVICE=cuda:0 bash bash_command/demo_blind_grasp_all_phases.sh
#   PHASES="1 3" bash bash_command/demo_blind_grasp_all_phases.sh   # rerun subset only

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAVE_OUTCOME="${SAVE_OUTCOME:-completed}"
DEVICE="${DEVICE:-cuda:0}"
OBJECT_INIT_INDEX="${OBJECT_INIT_INDEX:-auto}"
PHASES="${PHASES:-1 2 3}"

for phase in $PHASES; do
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo "Blind Grasp phase $phase"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  PHASE="$phase" \
    SAVE_OUTCOME="$SAVE_OUTCOME" \
    DEVICE="$DEVICE" \
    OBJECT_INIT_INDEX="${OBJECT_INIT_INDEX/auto/}" \
    MODE=full \
    bash bash_command/demo_blind_grasp_collection_replay.sh
done

echo ""
echo "[DONE] All Blind Grasp phases finished under logs/demo_blind_grasp/"
