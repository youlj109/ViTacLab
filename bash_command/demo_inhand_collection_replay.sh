#!/usr/bin/env bash
# In-Hand demo: RL checkpoint rollout + NPZ + tactile panel videos (single UR10e Shadow Hand).
#
# Task: Isaac-Repose-Cube-Shadow-Tactile-Direct-v0
# Policy: privileged proprio + object/goal state (161-dim); TacSL recorded for demo video only.
#
# Usage (ViTacLab repo root, env_isaaclab_510test):
#   bash bash_command/demo_inhand_collection_replay.sh
#   MODE=visualize SKIP_COLLECT=1 bash bash_command/demo_inhand_collection_replay.sh
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim RL collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   CHECKPOINT=/path/model.pt
#   RECORD_DIR=logs/demo_inhand/collection
#   NUM_ENVS=4
#   MAX_STEPS=600
#   MAX_TOTAL_STEPS=12000
#   SAVE_MODE=success
#   SUCCESS_TOLERANCE=0.1
#   SUCCESS_STREAK=2
#   Third-person camera: tuned in UR10eShadowHandInHandEnvCfg (palm + DexCube framing)
#   SKIP_COLLECT=0
#   FPS=10
#   DEVICE=cuda:0
#
# Outputs:
#   $RECORD_DIR/episode_*.npz
#   $VIDEO_DIR/inhand_demo_panel.mp4
#   $VIDEO_DIR/inhand_third_person.mp4
#   $VIDEO_DIR/inhand_tactile_rgb.mp4
#   $VIDEO_DIR/inhand_tactile_force_field.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

MODE="${MODE:-full}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SEED="${SEED:-42}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-600}"
MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS:-8000}"
NUM_ENVS="${NUM_ENVS:-4}"
LOG_DIR="${LOG_DIR:-logs/demo_inhand}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SAVE_MODE="${SAVE_MODE:-success}"
SUCCESS_TOLERANCE="${SUCCESS_TOLERANCE:-0.1}"
SUCCESS_STREAK="${SUCCESS_STREAK:-2}"
TACTILE_SETTLE_STEPS="${TACTILE_SETTLE_STEPS:-16}"

TASK_SLUG="inhand"
TASK_LABEL="InHand (UR10e Shadow RL)"
GYM_TASK="Isaac-Repose-Cube-Shadow-Tactile-Direct-v0"
DEFAULT_CKPT_DIR="logs/rsl_rl/shadow_hand"
DEFAULT_CHECKPOINT="${DEFAULT_CKPT_DIR}/2026-03-21_01-54-19/model_55250.pt"

RECORD_DIR="${RECORD_DIR:-logs/demo_inhand/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_inhand/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_inhand/summary.txt}"
VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_inhand/validation.json}"

ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
if [[ -x "$ISAACLAB_SH" ]]; then
  PY_SIM=( "$ISAACLAB_SH" -p )
else
  PY_SIM=( python )
fi
PY_OFFLINE=( python3 )

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
    best="$(_find_best_checkpoint "$ROOT_DIR/$DEFAULT_CKPT_DIR")"
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
    echo "[ERROR] missing RL checkpoint for In-Hand"
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

  rm -f "$RECORD_DIR"/episode_*.npz
  mkdir -p "$RECORD_DIR"

  echo "[RUN] play_inhand_rl_record.py"
  echo "  gym task: $GYM_TASK"
  echo "  checkpoint: $RESOLVED_CHECKPOINT"
  echo "  record: $RECORD_DIR"
  echo "  max_steps: $MAX_STEPS  max_total_steps: $MAX_TOTAL_STEPS"
  echo "  save_mode: $SAVE_MODE"
  echo "  success: |rot_err|<=$SUCCESS_TOLERANCE streak=$SUCCESS_STREAK"
  echo "  num_envs: $NUM_ENVS  num_episodes: $NUM_EPISODES"
  mkdir -p "$LOG_DIR"
  RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
  echo "  log: $RUN_LOG"

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/rsl_rl/full_rl/play_inhand_rl_record.py \
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
    --success-tolerance "$SUCCESS_TOLERANCE" \
    --success-streak "$SUCCESS_STREAK" \
    --tactile-settle-steps "$TACTILE_SETTLE_STEPS" \
    --record-any-env \
    --max-record-frames 320 \
    --record_env_index 0 \
    --record_step_interval 1 \
    2>&1 | tee "$RUN_LOG"
  play_rc=${PIPESTATUS[0]}
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_inhand_rl_record failed (exit=$play_rc)."
    echo "[HINT] Try SAVE_MODE=first_episode NUM_ENVS=8 SEED=0 MAX_TOTAL_STEPS=12000"
    exit "$play_rc"
  fi

  if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
    echo "[ERROR] no episode_*.npz under $RECORD_DIR"
    exit 3
  fi
  echo "[OK] collection finished under $RECORD_DIR"
}

run_validate() {
  log_section "[2/3] Validate NPZ schema"
  "${PY_OFFLINE[@]}" scripts/debug/validate_play_record_npz.py "$RECORD_DIR" \
    --json-out "$VALIDATION_JSON"
}

run_visualize() {
  log_section "[3/3] Export replay videos (${TASK_SLUG}_*.mp4)"
  mkdir -p "$VIDEO_DIR"
  rm -f "$VIDEO_DIR"/*.mp4

  "${PY_OFFLINE[@]}" scripts/debug/visualize_play_record_npz.py \
    "$RECORD_DIR" \
    --out_dir "$VIDEO_DIR" \
    --fps "$FPS" \
    --ff_mode heatmap \
    --rgb_mode diff_boost \
    --rgb_boost 12

  shopt -s nullglob
  for f in "$VIDEO_DIR"/*.mp4; do
    base="$(basename "$f")"
    if [[ "$base" != "${TASK_SLUG}_"* ]]; then
      case "$base" in
        *_demo_panel.mp4) mv "$f" "$VIDEO_DIR/${TASK_SLUG}_demo_panel.mp4" ;;
        *_third_person.mp4) mv "$f" "$VIDEO_DIR/${TASK_SLUG}_third_person.mp4" ;;
        *_tactile_rgb.mp4) mv "$f" "$VIDEO_DIR/${TASK_SLUG}_tactile_rgb.mp4" ;;
        *_tactile_force_field.mp4) mv "$f" "$VIDEO_DIR/${TASK_SLUG}_tactile_force_field.mp4" ;;
        *) mv "$f" "$VIDEO_DIR/${TASK_SLUG}_${base}" ;;
      esac
    fi
  done
}

write_summary() {
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  {
    echo "ViTacLab In-Hand RL demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "task_label: $TASK_LABEL"
    echo "task_slug: $TASK_SLUG"
    echo "gym_task: $GYM_TASK"
    echo ""
    echo "Checkpoint: $RESOLVED_CHECKPOINT"
    echo "Collection: play_inhand_rl_record.py"
    echo "save_mode: $SAVE_MODE"
    echo "success_tolerance: $SUCCESS_TOLERANCE"
    echo "success_streak: $SUCCESS_STREAK (training max_consecutive_success=2)"
    if [[ -f "$RECORD_DIR/meta.json" ]]; then
      saved_via="$(python3 -c "import json; print(json.load(open('$RECORD_DIR/meta.json')).get('saved_via','unknown'))" 2>/dev/null || echo unknown)"
      best_err="$(python3 -c "import json; v=json.load(open('$RECORD_DIR/meta.json')).get('best_ep_min_rot_err'); print(f'{v:.4f}' if v is not None else 'n/a')" 2>/dev/null || echo n/a)"
      echo "collection_result: $saved_via (best_ep_min_rot_err=$best_err)"
    fi
    echo ""
    echo "Recorded NPZ:"
    ls -lh "$RECORD_DIR"/*.npz 2>/dev/null || echo "  (none)"
    echo ""
    echo "Replay videos:"
    ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || echo "  (none)"
    echo ""
    panel="$VIDEO_DIR/${TASK_SLUG}_demo_panel.mp4"
    if [[ -f "$panel" ]]; then
      echo "Main panel video: $panel"
    fi
    echo ""
    echo "Re-run:"
    echo "  bash bash_command/demo_inhand_collection_replay.sh"
    echo "  MODE=visualize SKIP_COLLECT=1 bash bash_command/demo_inhand_collection_replay.sh"
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
panel="$VIDEO_DIR/${TASK_SLUG}_demo_panel.mp4"
if [[ -f "$panel" ]]; then
  echo "Main replay video: $panel"
fi
