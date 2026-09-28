#!/usr/bin/env bash
# Shadow Hand Blind Classification demo: keyframe replay + NPZ + tactile panel videos.
#
# Task: opaque bin with cube / sphere / cone; policy classifies via last 3 action logits.
# Demo uses open-loop keyframes + optional fixed shape label for tactile showcase.
#
# Usage (ViTacLab repo root, an Isaac Sim-enabled Python environment):
#   bash bash_command/demo_blind_classification_collection_replay.sh
#   SHAPE_LABEL=1 bash bash_command/demo_blind_classification_collection_replay.sh   # sphere
#   bash bash_command/demo_blind_classification_all_shapes.sh                          # cube+sphere+cone
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   SHAPE_LABEL=0              # 0=cube, 1=sphere, 2=cone, -1=random
#   SAVE_OUTCOME=completed     # completed | success | all
#   RECORD_DIR=logs/demo_blind_classification/cube/collection
#   VIDEO_DIR=logs/demo_blind_classification/cube/videos
#   SKIP_COLLECT=0
#   FPS=10
#   DEVICE=cuda:0
#
# Outputs:
#   $RECORD_DIR/episode_*_{cube|sphere|cone|random}_{completed|success}.npz
#   $VIDEO_DIR/blind_classification_${SHAPE_SLUG}_demo_panel.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/bash_command/demo_runtime.sh"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

MODE="${MODE:-full}"
SHAPE_LABEL="${SHAPE_LABEL:-0}"
SAVE_OUTCOME="${SAVE_OUTCOME:-completed}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SEED="${SEED:-42}"

case "$SHAPE_LABEL" in
  0) SHAPE_SLUG="cube"; SHAPE_NAME="Cube" ;;
  1) SHAPE_SLUG="sphere"; SHAPE_NAME="Sphere" ;;
  2) SHAPE_SLUG="cone"; SHAPE_NAME="Cone" ;;
  -1) SHAPE_SLUG="random"; SHAPE_NAME="Random shape" ;;
  *)
    echo "[ERROR] unknown SHAPE_LABEL=$SHAPE_LABEL (use 0|1|2|-1)"
    exit 2
    ;;
esac

TASK_SLUG="blind_classification_${SHAPE_SLUG}"
TASK_LABEL="Blind Classification (${SHAPE_NAME})"
KEYFRAME_JSON="${KEYFRAME_JSON:-scripts/data_collection/full_trajectory/pose_keyframes/UR10eShadowHandBlindClassificationEnv__UR10eShadowHandBlindClassificationEnvCfg.json}"
RECORD_DIR="${RECORD_DIR:-logs/demo_blind_classification/${SHAPE_SLUG}/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_blind_classification/${SHAPE_SLUG}/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_blind_classification/${SHAPE_SLUG}/summary.txt}"
VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_blind_classification/${SHAPE_SLUG}/validation.json}"

demo_setup_python

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

preflight() {
  if [[ ! -f "$KEYFRAME_JSON" ]]; then
    echo "[ERROR] missing keyframes: $KEYFRAME_JSON"
    exit 2
  fi
}

run_collect() {
  log_section "[1/3] ${TASK_LABEL}: keyframe replay + NPZ collection"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi
  preflight

  demo_prepare_reports
  demo_fresh_directory RECORD_DIR

  echo "[RUN] play_blind_classification.py"
  echo "  shape_label: $SHAPE_LABEL ($SHAPE_NAME)"
  echo "  keyframes: $KEYFRAME_JSON"
  echo "  save_outcome: $SAVE_OUTCOME"
  echo "  record: $RECORD_DIR"
  echo "  device: $DEVICE"

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/data_collection/full_trajectory/play_blind_classification.py \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --keyframe-json "$KEYFRAME_JSON" \
    --shape-label "$SHAPE_LABEL" \
    --seed "$SEED" \
    --num-envs 1 \
    --num-episodes "$NUM_EPISODES" \
    --save-outcome "$SAVE_OUTCOME" \
    --record-path "$RECORD_DIR" \
    --record-step-interval 1 \
    --stable-steps 15 \
    --post-arm-reached-steps 30 \
    --tactile-reset-settle-steps 8
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_blind_classification failed (exit=$play_rc)."
    echo "[HINT] If Isaac Sim hangs, identify and stop only this launcher's process."
    exit "$play_rc"
  fi

  if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
    echo "[ERROR] simulation finished but no episode_*.npz under $RECORD_DIR"
    echo "[HINT] SAVE_OUTCOME=completed SHAPE_LABEL=$SHAPE_LABEL bash bash_command/demo_blind_classification_collection_replay.sh"
    exit 3
  fi
  echo "[OK] collection finished under $RECORD_DIR"
}

run_validate() {
  demo_prepare_reports
  log_section "[2/3] Validate NPZ schema"
  "${PY_OFFLINE[@]}" scripts/debug/validate_play_record_npz.py "$RECORD_DIR" \
    --json-out "$VALIDATION_JSON"
}

run_visualize() {
  log_section "[3/3] Export replay videos (${TASK_SLUG}_*.mp4)"
  demo_fresh_directory VIDEO_DIR

  "${PY_OFFLINE[@]}" scripts/debug/visualize_play_record_npz.py \
    "$RECORD_DIR" \
    --out_dir "$VIDEO_DIR" \
    --fps "$FPS" \
    --ff_mode heatmap \
    --rgb_mode diff_boost \
    --rgb_boost 12

  demo_prefix_videos
}

write_summary() {
  demo_prepare_reports
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  {
    echo "ViTacLab Blind Classification demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "shape_label: $SHAPE_LABEL"
    echo "shape_name: $SHAPE_NAME"
    echo "task_slug: $TASK_SLUG"
    echo "save_outcome: $SAVE_OUTCOME"
    echo ""
    echo "Keyframes: $KEYFRAME_JSON"
    echo "Collection: play_blind_classification.py"
    echo ""
    echo "Recorded NPZ:"
    ls -lh "$RECORD_DIR"/*.npz 2>/dev/null || echo "  (none)"
    echo ""
    echo "Replay videos:"
    ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || echo "  (none)"
    echo ""
    panel="$(find "$VIDEO_DIR" -maxdepth 1 -name '*_demo_panel.mp4' -print -quit)"
    if [[ -f "$panel" ]]; then
      echo "Main panel video: $panel"
    fi
    echo ""
    echo "Re-run:"
    echo "  SHAPE_LABEL=$SHAPE_LABEL bash bash_command/demo_blind_classification_collection_replay.sh"
    echo "  MODE=visualize SKIP_COLLECT=1 SHAPE_LABEL=$SHAPE_LABEL bash bash_command/demo_blind_classification_collection_replay.sh"
    echo "  bash bash_command/demo_blind_classification_all_shapes.sh"
  } | tee "$SUMMARY_FILE"
}

mkdir -p "$(dirname "$SUMMARY_FILE")" "$RECORD_DIR" "$VIDEO_DIR"

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
  *)
    echo "[ERROR] unknown MODE=$MODE (use full|collect|visualize)"
    exit 2
    ;;
esac

log_section "[DONE] ${TASK_LABEL} demo finished"
echo "Summary: $SUMMARY_FILE"
panel="$(find "$VIDEO_DIR" -maxdepth 1 -name '*_demo_panel.mp4' -print -quit)"
if [[ -f "$panel" ]]; then
  echo "Main replay video: $panel"
fi
