#!/usr/bin/env bash
# Dual-arm Bottle Cap demo: in-package IK waypoints (+ optional hand RL checkpoint) + NPZ + videos.
#
# Task: Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0
# Runner: scripts/rsl_rl/full_rl/play_bottle_ik_record.py
# IK defaults: scripts/rsl_rl/ik_rl/configs/ik_rl_unscrew_dual.yaml
#
# Usage (ViTacLab repo root, an Isaac Sim-enabled Python environment):
#   bash bash_command/demo_bottle_collection_replay.sh
#   MODE=scene bash bash_command/demo_bottle_collection_replay.sh
#   CHECKPOINT=/path/model.pt bash bash_command/demo_bottle_collection_replay.sh
#
# Outputs: logs/demo_bottle/
#
# Modes (env var MODE):
#   full       — scene check + collect + mp4  [default]
#   collect    — only Isaac Sim collection
#   visualize  — only mp4 from existing RECORD_DIR
#   scene      — reset-only layout check (no rollout)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/bash_command/demo_runtime.sh"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

MODE="${MODE:-full}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-1}"
SEED="${SEED:-42}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-750}"
MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SAVE_MODE="${SAVE_MODE:-first_episode}"
RECORD_STEP_INTERVAL="${RECORD_STEP_INTERVAL:-1}"
TACTILE_SETTLE_STEPS="${TACTILE_SETTLE_STEPS:-16}"

TASK_SLUG="bottle_cap"
TASK_LABEL="Bottle Cap (dual UR10e + Shadow Hand)"
GYM_TASK="Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0"
DEFAULT_CKPT_DIR="logs/rsl_rl/${GYM_TASK}"

RECORD_DIR="${RECORD_DIR:-logs/demo_bottle/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_bottle/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_bottle/summary.txt}"
VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_bottle/validation.json}"
SCENE_CHECK_SCRIPT="${SCENE_CHECK_SCRIPT:-scripts/debug/check_unscrew_bottle_scene.py}"
SCENE_CHECK_DIR="${SCENE_CHECK_DIR:-logs/demo_bottle/scene_check}"
PLAY_SCRIPT="${PLAY_SCRIPT:-scripts/rsl_rl/full_rl/play_bottle_ik_record.py}"

demo_setup_python

_find_best_checkpoint() {
  local search_dir="$1"
  if [[ ! -d "$search_dir" ]]; then
    return 1
  fi
  find "$search_dir" -name 'model_*.pt' 2>/dev/null | sort -V | tail -1
}

resolve_checkpoint() {
  if [[ -n "${CHECKPOINT:-}" ]]; then
    RESOLVED_CHECKPOINT="$(readlink -f "$CHECKPOINT" 2>/dev/null || realpath "$CHECKPOINT" 2>/dev/null || echo "$CHECKPOINT")"
  else
    best="$(_find_best_checkpoint "$ROOT_DIR/$DEFAULT_CKPT_DIR" || true)"
    RESOLVED_CHECKPOINT="${best:-}"
  fi
}

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_scene_check() {
  log_section "[scene] Bottle reset layout check"
  if [[ ! -f "$SCENE_CHECK_SCRIPT" ]]; then
    echo "[ERROR] missing scene check script: $SCENE_CHECK_SCRIPT"
    exit 2
  fi
  demo_fresh_directory SCENE_CHECK_DIR
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" "$SCENE_CHECK_SCRIPT" \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --num-envs 1 \
    --settle-steps 48 \
    --out-dir "$SCENE_CHECK_DIR"
}

run_collect() {
  log_section "[1/3] ${TASK_LABEL}: IK rollout + NPZ collection"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi
  if [[ ! -f "$PLAY_SCRIPT" ]]; then
    echo "[ERROR] missing play script: $PLAY_SCRIPT"
    exit 2
  fi

  resolve_checkpoint
  demo_prepare_reports
  demo_fresh_directory RECORD_DIR

  CKPT_ARGS=()
  if [[ -n "${RESOLVED_CHECKPOINT:-}" && -f "$RESOLVED_CHECKPOINT" ]]; then
    CKPT_ARGS=( --checkpoint "$RESOLVED_CHECKPOINT" )
    echo "[RUN] hand policy checkpoint: $RESOLVED_CHECKPOINT"
  else
    echo "[RUN] no checkpoint found under $DEFAULT_CKPT_DIR — IK-only (zero hand actions)."
    echo "      set CHECKPOINT=/path/model.pt to enable learned hand policy."
  fi

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" "$PLAY_SCRIPT" \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --task "$GYM_TASK" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --save_data \
    --data_path "$RECORD_DIR" \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --max_total_steps "$MAX_TOTAL_STEPS" \
    --save-mode "$SAVE_MODE" \
    --record_env_index 0 \
    --record_step_interval "$RECORD_STEP_INTERVAL" \
    --tactile-settle-steps "$TACTILE_SETTLE_STEPS" \
    "${CKPT_ARGS[@]}"
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_bottle_ik_record failed (exit=$play_rc)."
    exit "$play_rc"
  fi

  if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
    echo "[ERROR] no episode_*.npz under $RECORD_DIR"
    exit 3
  fi
  echo "[OK] collection finished under $RECORD_DIR"
}

run_validate() {
  demo_prepare_reports
  if [[ "$SKIP_VALIDATE" == "1" ]]; then
    log_section "[2/3] Validate NPZ schema (skipped)"
    return 0
  fi
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
    --rgb_boost 12 \
    --ff_cell_px 10 \
    --panel_max_width 1920 \
    --panel_max_height 2400

  demo_prefix_videos
}

write_summary() {
  demo_prepare_reports
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  {
    echo "ViTacLab Bottle Cap demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "task: $GYM_TASK"
    echo "runner: $PLAY_SCRIPT"
    echo "task_slug: $TASK_SLUG"
    echo "record_dir: $RECORD_DIR"
    echo "video_dir: $VIDEO_DIR"
    echo "save_mode: $SAVE_MODE"
    echo "max_steps: $MAX_STEPS"
    echo "checkpoint: ${RESOLVED_CHECKPOINT:-(none — IK only)}"
    echo ""
    ls -lh "$RECORD_DIR"/*.npz 2>/dev/null || echo "  (no npz)"
    echo ""
    ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || echo "  (no mp4)"
  } | tee "$SUMMARY_FILE"
}

mkdir -p "$(dirname "$SUMMARY_FILE")" "$RECORD_DIR" "$VIDEO_DIR"
RESOLVED_CHECKPOINT=""

case "$MODE" in
  scene)
    run_scene_check
    write_summary
    ;;
  full)
    run_scene_check
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
    echo "[ERROR] unknown MODE=$MODE (use full|collect|visualize|scene)"
    exit 2
    ;;
esac

log_section "[DONE] ${TASK_LABEL} demo finished"
panel="$(find "$VIDEO_DIR" -maxdepth 1 -name '*_demo_panel.mp4' -print -quit)"
if [[ -f "$panel" ]]; then
  echo "Main replay video: $panel"
fi
