#!/usr/bin/env bash
# Dual-arm HandOver demo: RL checkpoint rollout + NPZ + tactile panel videos.
#
# Per mentor guidance, HandOver (Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0) uses trained
# RSL-RL checkpoints — not open-loop pose trajectories.
#
# Usage (ViTacLab repo root, an Isaac Sim-enabled Python environment):
#   bash bash_command/demo_handover_collection_replay.sh
#   MODE=visualize RECORD_DIR=/path/to/collection/run bash bash_command/demo_handover_collection_replay.sh
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim RL collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   CHECKPOINT=/path/model.pt
#   RECORD_DIR=logs/demo_handover/collection
#   NUM_ENVS=1                 # use 1 for clean demo video (4 clones extra tables/arms in FOV)
#   NUM_ENVS=4
#   MAX_STEPS=500
#   MAX_TOTAL_STEPS=8000
#   SAVE_MODE=best_episode
#   SUCCESS_GOAL_DIST=0.06
#   Object: current local banana asset; checkpoints trained on the donor sphere
#   scene are not assumed compatible. This collector does not replace the object.
#   SUCCESS_STREAK=8
#   SKIP_COLLECT=0
#   FPS=10
#   DEVICE=cuda:0
#
# Outputs:
#   $RECORD_DIR/episode_*.npz
#   $VIDEO_DIR/hand_over_demo_panel.mp4
#   $VIDEO_DIR/hand_over_third_person.mp4
#   $VIDEO_DIR/hand_over_tactile_rgb.mp4
#   $VIDEO_DIR/hand_over_tactile_force_field.mp4

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
SEED="${SEED:-42}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS:-8000}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SAVE_MODE="${SAVE_MODE:-best_episode}"
SUCCESS_GOAL_DIST="${SUCCESS_GOAL_DIST:-0.06}"
SUCCESS_STREAK="${SUCCESS_STREAK:-8}"
TACTILE_SETTLE_STEPS="${TACTILE_SETTLE_STEPS:-16}"

TASK_SLUG="hand_over"
TASK_LABEL="HandOver (Dual UR10e Shadow RL)"
GYM_TASK="Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0"
DEFAULT_CKPT_DIR="logs/rsl_rl/Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0"
DEFAULT_CHECKPOINT="${DEFAULT_CHECKPOINT:-}"

RECORD_DIR="${RECORD_DIR:-logs/demo_handover/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_handover/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_handover/summary.txt}"
VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_handover/validation.json}"

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
  elif [[ -f "$ROOT_DIR/$DEFAULT_CHECKPOINT" ]]; then
    RESOLVED_CHECKPOINT="$ROOT_DIR/$DEFAULT_CHECKPOINT"
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

preflight() {
  if [[ -z "${RESOLVED_CHECKPOINT:-}" || ! -f "$RESOLVED_CHECKPOINT" ]]; then
    echo "[ERROR] missing RL checkpoint for HandOver"
    echo "  expected under: $DEFAULT_CKPT_DIR"
    echo "  set CHECKPOINT=/absolute/path/model_XXXX.pt"
    exit 2
  fi
}

run_collect() {
  log_section "[1/3] ${TASK_LABEL}: RL checkpoint rollout + NPZ collection"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi
  preflight

  demo_prepare_reports
  demo_fresh_directory RECORD_DIR

  echo "[RUN] play_handover_rl_record.py"
  echo "  gym task: $GYM_TASK"
  echo "  checkpoint: $RESOLVED_CHECKPOINT"
  echo "  record: $RECORD_DIR"
  echo "  max_steps: $MAX_STEPS  max_total_steps: $MAX_TOTAL_STEPS"
  echo "  save_mode: $SAVE_MODE"
  echo "  num_envs: $NUM_ENVS  num_episodes: $NUM_EPISODES"

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/rsl_rl/full_rl/play_handover_rl_record.py \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --task "$GYM_TASK" \
    --checkpoint "$RESOLVED_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --save_data \
    --data_path "$RECORD_DIR" \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --max_total_steps "$MAX_TOTAL_STEPS" \
    --save-mode "$SAVE_MODE" \
    --success-goal-dist "$SUCCESS_GOAL_DIST" \
    --success-streak "$SUCCESS_STREAK" \
    --tactile-settle-steps "$TACTILE_SETTLE_STEPS" \
    --record_env_index 0 \
    --record_step_interval 1
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_handover_rl_record failed (exit=$play_rc)."
    echo "[HINT] Try SAVE_MODE=best_episode NUM_ENVS=8 SEED=0 MAX_TOTAL_STEPS=8000"
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
    echo "ViTacLab HandOver RL demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "task_label: $TASK_LABEL"
    echo "task_slug: $TASK_SLUG"
    echo "gym_task: $GYM_TASK"
    echo ""
    echo "Checkpoint: $RESOLVED_CHECKPOINT"
    echo "Collection: play_handover_rl_record.py"
    echo "save_mode: $SAVE_MODE"
    echo "success_goal_dist: $SUCCESS_GOAL_DIST"
    echo "success_streak: $SUCCESS_STREAK"
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
    echo "  bash bash_command/demo_handover_collection_replay.sh"
    echo "  MODE=visualize RECORD_DIR=/path/to/collection/run bash bash_command/demo_handover_collection_replay.sh"
  } | tee "$SUMMARY_FILE"
}

resolve_checkpoint
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
