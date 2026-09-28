#!/usr/bin/env bash
# Pour task demo: scripted full-IK pour + canonical NPZ + replay videos.
#
# Pour uses deformable cup/water; collection path is play_full_ik_single (not play_single JSON replay).
# Config: scripts/data_collection/ik/configs/full_ik/full_ik_pour.yaml
#
# Usage (ViTacLab repo root, env_isaaclab_510test):
#   bash bash_command/demo_pour_collection_replay.sh
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#   smoke      — quick env boot + record schema (no mp4)
#
# Common env vars:
#   RECORD_DIR=logs/demo_pour/collection
#   VIDEO_DIR=logs/demo_pour/videos
#   MAX_PLAY_STEPS=560   (just under one Pour episode; avoids timeout reset mid-NPZ)
#   SKIP_COLLECT=0
#   FPS=10
#
# Outputs:
#   $RECORD_DIR/episode_*_completed.npz
#   $VIDEO_DIR/*_demo_panel.mp4
#   logs/demo_pour/summary.txt

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-full}"
RECORD_DIR="${RECORD_DIR:-logs/demo_pour/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_pour/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_pour/summary.txt}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
MAX_PLAY_STEPS="${MAX_PLAY_STEPS:-560}"
FPS="${FPS:-10}"
SEED="${SEED:-42}"
TASK="${TASK:-Isaac-UR10eShadowHand-PourDeformable-Direct-v0}"
FULL_IK_CONFIG="${FULL_IK_CONFIG:-scripts/data_collection/ik/configs/full_ik/full_ik_pour_demo.yaml}"
GRASP_YAML="${GRASP_YAML:-scripts/data_collection/manual/config/pour_cup_relative_grasp.yaml}"

ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
if [[ -x "$ISAACLAB_SH" ]]; then
  PY_SIM=( "$ISAACLAB_SH" -p )
else
  PY_SIM=( python )
fi
PY_OFFLINE=( python3 )

mkdir -p "$(dirname "$SUMMARY_FILE")" "$RECORD_DIR" "$VIDEO_DIR"

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

preflight() {
  if [[ ! -f "$FULL_IK_CONFIG" ]]; then
    echo "[ERROR] missing full-IK config: $FULL_IK_CONFIG"
    exit 2
  fi
  if [[ ! -f "$GRASP_YAML" ]]; then
    echo "[ERROR] missing grasp YAML: $GRASP_YAML"
    exit 2
  fi
}

run_smoke() {
  log_section "[smoke] Pour env + canonical record schema"
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/debug/smoke_pour_record.py \
    --headless \
    --enable_cameras \
    --task "$TASK" \
    --num_envs 1 \
    --max_steps 25
}

run_collect() {
  log_section "[1/3] Pour scripted collection (full-IK, no checkpoint)"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi
  preflight
  echo "[RUN] play_full_ik_single"
  echo "  task: $TASK"
  echo "  config: $FULL_IK_CONFIG"
  echo "  record: $RECORD_DIR"
  echo "  max_play_steps: $MAX_PLAY_STEPS"

  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/data_collection/ik/play_full_ik_single.py \
    --headless \
    --enable_cameras \
    --task "$TASK" \
    --full-ik-config "$FULL_IK_CONFIG" \
    --num_envs 1 \
    --device cuda:0 \
    --no-checkpoint \
    --seed "$SEED" \
    --record_data \
    --record_path "$RECORD_DIR" \
    --record_env_index 0 \
    --record_max_episodes 1 \
    --record_save_on_stop \
    --record_demo_mode \
    --max_play_steps "$MAX_PLAY_STEPS"

  echo "[OK] collection finished under $RECORD_DIR"
}

run_validate() {
  log_section "[2/3] Validate NPZ schema"
  "${PY_OFFLINE[@]}" scripts/debug/validate_play_record_npz.py "$RECORD_DIR" \
    --json-out "logs/demo_pour/validation.json"
}

run_visualize() {
  log_section "[3/3] Export replay videos"
  "${PY_OFFLINE[@]}" scripts/debug/visualize_play_record_npz.py \
    "$RECORD_DIR" \
    --out_dir "$VIDEO_DIR" \
    --fps "$FPS" \
    --ff_mode heatmap \
    --rgb_mode diff_boost \
    --rgb_boost 12
}

write_summary() {
  {
    echo "ViTacLab Pour demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo ""
    echo "Collection path: play_full_ik_single + full_ik_pour_demo.yaml (GPU IK + hand grasp gate)"
    echo "Task: $TASK"
    echo "Config: $FULL_IK_CONFIG"
    echo "Grasp YAML: $GRASP_YAML"
    echo ""
    echo "Recorded NPZ:"
    ls -lh "$RECORD_DIR"/*.npz 2>/dev/null || echo "  (none)"
    echo ""
    echo "Replay videos:"
    ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || echo "  (none)"
    echo ""
    panel="$(ls "$VIDEO_DIR"/*_demo_panel.mp4 2>/dev/null | head -1 || true)"
    if [[ -n "$panel" ]]; then
      echo "Main panel video: $panel"
    fi
    echo ""
    echo "Re-run:"
    echo "  bash bash_command/demo_pour_collection_replay.sh"
    echo "  MODE=visualize SKIP_COLLECT=1 bash bash_command/demo_pour_collection_replay.sh"
    echo ""
    echo "GUI trajectory recording (optional, for play_single JSON path later):"
    echo "  python scripts/data_collection/full_trajectory/record_single.py \\"
    echo "    --task pour --num_envs 1 --enable_cameras --show_rgb --show_ff"
  } | tee "$SUMMARY_FILE"
}

case "$MODE" in
  full)
    run_collect
    run_validate
    run_visualize
    write_summary
    ;;
  collect)
    run_collect
    run_validate
    write_summary
    ;;
  visualize)
    run_validate
    run_visualize
    write_summary
    ;;
  smoke)
    run_smoke
    write_summary
    ;;
  *)
    echo "[ERROR] unknown MODE=$MODE (use full|collect|visualize|smoke)"
    exit 2
    ;;
esac

log_section "[DONE] Pour demo finished"
echo "Summary: $SUMMARY_FILE"
panel="$(ls "$VIDEO_DIR"/*_demo_panel.mp4 2>/dev/null | head -1 || true)"
if [[ -n "$panel" ]]; then
  echo "Main replay video: $panel"
fi
