#!/usr/bin/env bash
# BiBlindBinDrop demo: dual-arm blind drop / targeted drop + NPZ + panel videos.
#
# Task: Isaac-UR10e-Dual-Shadow-Hand-BiBlindBinDrop-Direct-v0
# Phases (paper mapping):
#   1 — Blind Drop (Pick01 branch)
#   2 — Blind Targeted Drop (Pick02 branch)
#
# Usage (ViTacLab repo root, an Isaac Sim-enabled Python environment):
#   PHASE=1 bash bash_command/demo_bi_blind_bin_drop_collection_replay.sh
#   PHASE=2 bash bash_command/demo_bi_blind_bin_drop_collection_replay.sh
#   bash bash_command/demo_bi_blind_bin_drop_all_phases.sh
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   PHASE=1                    # 1=blind drop, 2=blind targeted drop
#   OBJECT_INIT_INDEX=0        # cube spawn preset (0..2)
#   SAVE_OUTCOME=completed     # completed | success | all
#   RECORD_DIR=logs/demo_bi_blind_bin_drop/phase1/collection
#   SKIP_COLLECT=0
#   FPS=10
#   DEVICE=cuda:0
#
# Outputs:
#   $VIDEO_DIR/bi_blind_bin_drop_phase${PHASE}_demo_panel.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/bash_command/demo_runtime.sh"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

MODE="${MODE:-full}"
PHASE="${PHASE:-1}"
OBJECT_INIT_INDEX="${OBJECT_INIT_INDEX:-0}"
SAVE_OUTCOME="${SAVE_OUTCOME:-completed}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
NUM_EPISODES="${NUM_EPISODES:-1}"
RECORD_STEP_INTERVAL="${RECORD_STEP_INTERVAL:-1}"

case "$PHASE" in
  1)
    TASK_LABEL="Blind Drop (BiBlindBinDrop phase 1)"
    TASK_SLUG="bi_blind_bin_drop_phase1"
    ;;
  2)
    TASK_LABEL="Blind Targeted Drop (BiBlindBinDrop phase 2)"
    TASK_SLUG="bi_blind_bin_drop_phase2"
    ;;
  3)
    TASK_LABEL="BiBlindBinDrop phase 3 (replay cone x=0.15)"
    TASK_SLUG="bi_blind_bin_drop_phase3"
    ;;
  4)
    TASK_LABEL="BiBlindBinDrop phase 4 (replay cone x=0.30)"
    TASK_SLUG="bi_blind_bin_drop_phase4"
    ;;
  *)
    echo "[ERROR] unknown PHASE=$PHASE (use 1|2|3|4)"
    exit 2
    ;;
esac

KEYFRAME_JSON="${KEYFRAME_JSON:-scripts/data_collection/full_trajectory/pose_keyframes_dual.json}"
RECORD_DIR="${RECORD_DIR:-logs/demo_bi_blind_bin_drop/phase${PHASE}/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_bi_blind_bin_drop/phase${PHASE}/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_bi_blind_bin_drop/phase${PHASE}/summary.txt}"
VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_bi_blind_bin_drop/phase${PHASE}/validation.json}"

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
  log_section "[1/3] ${TASK_LABEL}: dual phase replay + NPZ collection"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi
  preflight

  demo_prepare_reports
  demo_fresh_directory RECORD_DIR

  echo "[RUN] play_dual_phase.py"
  echo "  task: bi_blind_bin_drop (Isaac-UR10e-Dual-Shadow-Hand-BiBlindBinDrop-Direct-v0)"
  echo "  phase: $PHASE"
  echo "  object_init_index: $OBJECT_INIT_INDEX"
  echo "  save_outcome: $SAVE_OUTCOME"
  echo "  record: $RECORD_DIR"
  echo "  device: $DEVICE"

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/data_collection/full_trajectory/play_dual_phase.py \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --task bi_blind_bin_drop \
    --phase "$PHASE" \
    --keyframe-json "$KEYFRAME_JSON" \
    --num-envs 1 \
    --num-episodes "$NUM_EPISODES" \
    --object-init-index "$OBJECT_INIT_INDEX" \
    --save-outcome "$SAVE_OUTCOME" \
    --record-path "$RECORD_DIR" \
    --record-step-interval "$RECORD_STEP_INTERVAL" \
    --stable-steps 12 \
    --post-arm-reached-steps 30 \
    --tactile-reset-settle-steps 16
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_dual_phase failed (exit=$play_rc)."
    exit "$play_rc"
  fi

  if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
    echo "[ERROR] no episode_*.npz under $RECORD_DIR"
    echo "[HINT] SAVE_OUTCOME=completed PHASE=$PHASE bash bash_command/demo_bi_blind_bin_drop_collection_replay.sh"
    exit 3
  fi
  echo "[OK] collection finished under $RECORD_DIR"
}

run_validate() {
  demo_prepare_reports
  log_section "[2/3] Validate NPZ schema"
  set +e
  "${PY_OFFLINE[@]}" scripts/debug/validate_play_record_npz.py "$RECORD_DIR" \
    --json-out "$VALIDATION_JSON"
  val_rc=$?
  set -e
  if [[ "$val_rc" -ne 0 ]]; then
    echo "[WARN] validate_play_record_npz reported issues (dual-arm); continuing."
  fi
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
    echo "ViTacLab BiBlindBinDrop demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "phase: $PHASE"
    echo "task_label: $TASK_LABEL"
    echo "task_slug: $TASK_SLUG"
    echo "object_init_index: $OBJECT_INIT_INDEX"
    echo "save_outcome: $SAVE_OUTCOME"
    echo ""
    echo "Collection: play_dual_phase.py --task bi_blind_bin_drop --phase $PHASE"
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
