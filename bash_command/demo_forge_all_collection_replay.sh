#!/usr/bin/env bash
# Run Forge Peg Insert, Gear Mesh, and Nut Thread demos sequentially.
#
# Usage:
#   bash bash_command/demo_forge_all_collection_replay.sh
#
# Optional env vars (passed through to demo_forge_collection_replay.sh):
#   MODE=full|collect|visualize
#   SKIP_COLLECT=0
#   FPS=10
#   SEED=42

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEMO_SH="$ROOT_DIR/bash_command/demo_forge_collection_replay.sh"

ALL_SUMMARY="${ALL_SUMMARY:-logs/demo_forge/all_tasks_summary.txt}"
mkdir -p "$(dirname "$ALL_SUMMARY")"
ALL_SUMMARY="$(mktemp "${ALL_SUMMARY%.txt}-XXXXXX.txt")"

run_one() {
  local task="$1"
  echo ""
  echo "################################################################"
  echo "# Forge demo task: $task"
  echo "################################################################"
  TASK="$task" bash "$DEMO_SH"
}

run_one forge_insert
run_one forge_gear
run_one forge_nut

{
  echo "ViTacLab Forge all-tasks demo summary"
  echo "generated: $(date -Iseconds)"
  echo ""
  echo "Available panel videos and per-task summaries (including earlier runs):"
  find logs/demo_forge -type f \
    \( -name '*_demo_panel.mp4' -o -name summary.txt \) -print
} | tee "$ALL_SUMMARY"

echo ""
echo "[DONE] All Forge demos finished. Summary: $ALL_SUMMARY"
