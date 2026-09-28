#!/usr/bin/env bash
# Blind Targeted Grasp (Blind Retrieval) demo: keyframe replay + NPZ + tactile panel videos.
#
# Task: Isaac-UR10eShadowHand-BlindRetrieval-Direct-v0
# Three shapes in bin; retrieve the commanded target to the goal region.
#
# Usage (ViTacLab repo root, an Isaac Sim-enabled Python environment):
#   bash bash_command/demo_blind_retrieval_collection_replay.sh
#   TARGET_LABEL=1 bash bash_command/demo_blind_retrieval_collection_replay.sh   # sphere
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   TARGET_LABEL=0             # 0=cube, 1=sphere, 2=cone, -1=random
#   ACTION_SMOOTHING=0.0       # must be low; default 0.75 prevents hand closure / grasp
#   BENCH_MODE=1               # ViTacBench: tactile gate, no palm-carry (default 1)
#   SAVE_OUTCOME=success       # completed | success | all
#   RECORD_DIR=logs/demo_blind_retrieval/cube/collection
#   VIDEO_DIR=logs/demo_blind_retrieval/cube/videos
#   SKIP_COLLECT=0
#   FPS=10
#   DEVICE=cuda:0
#
# Outputs:
#   $VIDEO_DIR/blind_retrieval_${TARGET_SLUG}_demo_panel.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/bash_command/demo_runtime.sh"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

MODE="${MODE:-full}"
TARGET_LABEL="${TARGET_LABEL:-0}"
ACTION_SMOOTHING="${ACTION_SMOOTHING:-0.0}"
BENCH_MODE="${BENCH_MODE:-1}"
SAVE_OUTCOME="${SAVE_OUTCOME:-success}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SEED="${SEED:-42}"

case "$TARGET_LABEL" in
  0) TARGET_SLUG="cube"; TARGET_NAME="Cube" ;;
  1) TARGET_SLUG="sphere"; TARGET_NAME="Sphere" ;;
  2) TARGET_SLUG="cone"; TARGET_NAME="Cone" ;;
  -1) TARGET_SLUG="random"; TARGET_NAME="Random target" ;;
  *)
    echo "[ERROR] unknown TARGET_LABEL=$TARGET_LABEL (use 0|1|2|-1)"
    exit 2
    ;;
esac

TASK_SLUG="blind_retrieval_${TARGET_SLUG}"
TASK_LABEL="Blind Targeted Grasp / Retrieval (${TARGET_NAME})"
KEYFRAME_JSON="${KEYFRAME_JSON:-scripts/data_collection/full_trajectory/pose_keyframes/UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json}"
RECORD_DIR="${RECORD_DIR:-logs/demo_blind_retrieval/${TARGET_SLUG}/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_blind_retrieval/${TARGET_SLUG}/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_blind_retrieval/${TARGET_SLUG}/summary.txt}"
VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_blind_retrieval/${TARGET_SLUG}/validation.json}"

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

  echo "[RUN] play_blind_retrieval.py"
  echo "  gym task: Isaac-UR10eShadowHand-BlindRetrieval-Direct-v0"
  echo "  target_label: $TARGET_LABEL ($TARGET_NAME)"
  echo "  keyframes: $KEYFRAME_JSON"
  echo "  action_smoothing: $ACTION_SMOOTHING"
  echo "  bench_mode: $BENCH_MODE"
  echo "  save_outcome: $SAVE_OUTCOME"
  echo "  record: $RECORD_DIR"
  echo "  device: $DEVICE"

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/data_collection/full_trajectory/play_blind_retrieval.py \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --keyframe-json "$KEYFRAME_JSON" \
    --target-label "$TARGET_LABEL" \
    --seed "$SEED" \
    --num-envs 1 \
    --num-episodes "$NUM_EPISODES" \
    --action-smoothing "$ACTION_SMOOTHING" \
    --save-outcome "$SAVE_OUTCOME" \
    --record-path "$RECORD_DIR" \
    --record-step-interval 5 \
    --stable-steps 20 \
    --max-steps-per-frame 200 \
    --post-arm-reached-steps 40 \
    --pick-dwell-steps 200 \
    --hand-noise 0 \
    --snap-at-pick-close \
    --pick-hand-close-scale 1.35 \
    --object-settle-steps 60 \
    --pick-object-align-offset ${PICK_OBJECT_ALIGN_OFFSET:--0.018 0.0 -0.018} \
    --tactile-reset-settle-steps 8 \
    $( [[ "$BENCH_MODE" == "1" ]] && echo "--bench-mode --tactile-contact-threshold 1e-4 --no-carry-object-after-grasp" || echo "--no-bench-mode --tactile-contact-threshold 0 --pick-geom-contact-dist 0.046 --carry-object-after-grasp" )
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_blind_retrieval failed (exit=$play_rc)."
    exit "$play_rc"
  fi

  if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
    echo "[ERROR] no episode_*.npz under $RECORD_DIR"
    echo "[HINT] SAVE_OUTCOME=completed TARGET_LABEL=$TARGET_LABEL bash bash_command/demo_blind_retrieval_collection_replay.sh"
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
    echo "ViTacLab Blind Retrieval (Targeted Grasp) demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "target_label: $TARGET_LABEL"
    echo "target_name: $TARGET_NAME"
    echo "task_slug: $TASK_SLUG"
    echo "save_outcome: $SAVE_OUTCOME"
    echo ""
    echo "Keyframes: $KEYFRAME_JSON"
    echo "Collection: play_blind_retrieval.py"
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
