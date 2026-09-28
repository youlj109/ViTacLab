#!/usr/bin/env bash
# Run all ViTacBench tasks that were missing one-command demo pipelines.
#
# Paper tasks covered here (tactile-dominant gaps):
#   - Blind Targeted Grasp  -> demo_blind_retrieval_collection_replay.sh
#   - Blind Drop            -> demo_bi_blind_bin_drop phase 1
#   - Blind Targeted Drop   -> demo_bi_blind_bin_drop phase 2
#
# Usage (ViTacLab repo root, env_isaaclab_510test):
#   bash bash_command/demo_vitacbench_missing.sh
#
# Optional:
#   SKIP_RETRIEVAL=1 bash bash_command/demo_vitacbench_missing.sh
#   SKIP_BIN_DROP=1 bash bash_command/demo_vitacbench_missing.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAVE_OUTCOME="${SAVE_OUTCOME:-completed}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_LABEL="${TARGET_LABEL:-0}"
LOG_DIR="${LOG_DIR:-logs/demo_vitacbench_missing}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "[ViTacBench missing demos] log: $RUN_LOG"
echo "[ViTacBench missing demos] started: $(date -Iseconds)"

if [[ "${SKIP_RETRIEVAL:-0}" != "1" ]]; then
  echo ""
  echo "========== [1/2] Blind Targeted Grasp (Retrieval) =========="
  TARGET_LABEL="$TARGET_LABEL" \
    SAVE_OUTCOME="$SAVE_OUTCOME" \
    DEVICE="$DEVICE" \
    MODE=full \
    bash bash_command/demo_blind_retrieval_collection_replay.sh
fi

if [[ "${SKIP_BIN_DROP:-0}" != "1" ]]; then
  echo ""
  echo "========== [2/2] Blind Drop + Blind Targeted Drop =========="
  SAVE_OUTCOME="$SAVE_OUTCOME" \
    DEVICE="$DEVICE" \
    PHASES="1 2" \
    bash bash_command/demo_bi_blind_bin_drop_all_phases.sh
fi

echo ""
echo "[DONE] ViTacBench missing demo pipelines finished."
echo "  Blind Retrieval panel: logs/demo_blind_retrieval/*/videos/*_demo_panel.mp4"
echo "  BiBlindBinDrop panels: logs/demo_bi_blind_bin_drop/phase*/videos/*_demo_panel.mp4"
echo "  Log: $RUN_LOG"
