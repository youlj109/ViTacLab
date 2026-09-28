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
if [[ ! -x "$DEMO_SH" ]]; then
  chmod +x "$DEMO_SH" 2>/dev/null || true
fi

ALL_SUMMARY="${ALL_SUMMARY:-logs/demo_forge/all_tasks_summary.txt}"
mkdir -p "$(dirname "$ALL_SUMMARY")"

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
  echo "Panel videos:"
  for slug in forge_peg_insert forge_gear_mesh forge_nut_thread; do
    panel="logs/demo_forge/${slug}/videos/${slug}_demo_panel.mp4"
    if [[ -f "$panel" ]]; then
      ls -lh "$panel"
    else
      echo "  (missing) $panel"
    fi
  done
  echo ""
  echo "Per-task summaries:"
  for slug in forge_peg_insert forge_gear_mesh forge_nut_thread; do
    s="logs/demo_forge/${slug}/summary.txt"
    if [[ -f "$s" ]]; then
      echo "  $s"
    fi
  done
} | tee "$ALL_SUMMARY"

echo ""
echo "[DONE] All Forge demos finished. Summary: $ALL_SUMMARY"
