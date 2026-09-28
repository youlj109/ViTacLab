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
  TASK=forge_nut bash "$DEMO_SHADOW"
  mkdir -p "$ROOT_DIR/logs/demo_forge_franka/forge_nut_thread"
  cat > "$ROOT_DIR/logs/demo_forge_franka/forge_nut_thread/FALLBACK_shadow_hand.txt" <<EOF
Franka Nut Thread RL checkpoint unavailable on this machine.
Used Shadow Hand keyframe replay instead:
  bash_command/demo_forge_collection_replay.sh TASK=forge_nut
Outputs linked from logs/demo_forge/forge_nut_thread/
EOF
  ln -sfn "$ROOT_DIR/logs/demo_forge/forge_nut_thread/collection" \
    "$ROOT_DIR/logs/demo_forge_franka/forge_nut_thread/collection" 2>/dev/null || true
  ln -sfn "$ROOT_DIR/logs/demo_forge/forge_nut_thread/videos" \
    "$ROOT_DIR/logs/demo_forge_franka/forge_nut_thread/videos" 2>/dev/null || true
  cp -f "$ROOT_DIR/logs/demo_forge/forge_nut_thread/summary.txt" \
    "$ROOT_DIR/logs/demo_forge_franka/forge_nut_thread/summary_shadow_fallback.txt" 2>/dev/null || true
}

run_franka peg_insert
run_franka gear_mesh
run_nut_with_fallback

{
  echo "ViTacLab Forge peg/gear/nut demo summary"
  echo "generated: $(date -Iseconds)"
  echo ""
  echo "Franka RL panel videos (peg/gear):"
  for slug in forge_peg_insert forge_gear_mesh; do
    panel="logs/demo_forge_franka/${slug}/videos/${slug}_demo_panel.mp4"
    if [[ -f "$panel" ]]; then
      ls -lh "$panel"
    else
      echo "  (missing) $panel"
    fi
  done
  echo ""
  echo "Nut (Franka RL or Shadow Hand fallback):"
  for panel in \
    logs/demo_forge_franka/forge_nut_thread/videos/forge_nut_thread_demo_panel.mp4 \
    logs/demo_forge/forge_nut_thread/videos/forge_nut_thread_demo_panel.mp4; do
    if [[ -f "$panel" ]]; then
      ls -lh "$panel"
      break
    fi
  done
} | tee "$ALL_SUMMARY"

echo ""
echo "[DONE] Peg / Gear / Nut demos finished. Summary: $ALL_SUMMARY"
