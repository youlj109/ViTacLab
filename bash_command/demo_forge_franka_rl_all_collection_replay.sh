#!/usr/bin/env bash
# Run Franka RL Forge demos for peg / gear / nut (with nut fallback to Shadow Hand keyframes).
#
# Usage:
#   bash bash_command/demo_forge_franka_rl_all_collection_replay.sh
#
# Nut thread: if no Franka checkpoint with step >= MIN_CHECKPOINT_STEP (default 100),
# falls back to UR10e Shadow Hand pose-keyframe demo (mentor dexhand trajectory).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEMO_FRANKA="$ROOT_DIR/bash_command/demo_forge_franka_rl_collection_replay.sh"
DEMO_SHADOW="$ROOT_DIR/bash_command/demo_forge_collection_replay.sh"
MIN_CHECKPOINT_STEP="${MIN_CHECKPOINT_STEP:-100}"
ALL_SUMMARY="${ALL_SUMMARY:-logs/demo_forge_franka/all_tasks_summary.txt}"
source "$ROOT_DIR/bash_command/demo_runtime.sh"
mkdir -p "$(dirname "$ALL_SUMMARY")"
ALL_SUMMARY="$(mktemp "${ALL_SUMMARY%.txt}-XXXXXX.txt")"

run_franka() {
  local task="$1"
  TASK="$task" MIN_CHECKPOINT_STEP="$MIN_CHECKPOINT_STEP" bash "$DEMO_FRANKA"
}

run_nut_with_fallback() {
  local ckpt=""
  ckpt="$(find "$ROOT_DIR/logs/rsl_rl/Isaac-Forge-NutThread-Direct-v0" -name 'model_*.pt' 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$ckpt" ]]; then
    local step="${ckpt##*/model_}"
    step="${step%.pt}"
    if [[ "$step" -ge "$MIN_CHECKPOINT_STEP" ]]; then
      echo "[INFO] Franka Nut RL checkpoint OK: $ckpt"
      TASK=nut_thread CHECKPOINT="$ckpt" bash "$DEMO_FRANKA"
      return 0
    fi
  fi

  echo ""
  echo "[WARN] No usable Franka Nut RL checkpoint (need model_${MIN_CHECKPOINT_STEP}+)."
  echo "[WARN] Falling back to UR10e Shadow Hand nut keyframe demo (mentor dexhand trajectory)."
  FALLBACK_DIR="$ROOT_DIR/logs/demo_forge_franka/forge_nut_thread/shadow_fallback"
  demo_fresh_directory FALLBACK_DIR
  TASK=forge_nut RECORD_DIR="$FALLBACK_DIR/collection" VIDEO_DIR="$FALLBACK_DIR/videos" \
    SUMMARY_FILE="$FALLBACK_DIR/summary.txt" bash "$DEMO_SHADOW"
  printf '%s\n' \
    "Franka Nut checkpoint unavailable; these are Shadow Hand keyframe results." \
    "Outputs: $FALLBACK_DIR" > "$FALLBACK_DIR/FALLBACK_shadow_hand.txt"
}

run_franka peg_insert
run_franka gear_mesh
run_nut_with_fallback

{
  echo "ViTacLab Forge peg/gear/nut demo summary"
  echo "generated: $(date -Iseconds)"
  echo ""
  echo "Available panel videos (including earlier runs; shadow_fallback is not Franka):"
  find logs/demo_forge_franka -type f -name '*_demo_panel.mp4' -print
} | tee "$ALL_SUMMARY"

echo ""
echo "[DONE] Peg / Gear / Nut demos finished. Summary: $ALL_SUMMARY"
