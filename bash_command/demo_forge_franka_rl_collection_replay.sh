#!/usr/bin/env bash
# Franka two-finger FORGE demo: RL checkpoint rollout + NPZ + tactile panel videos.
#
# Per mentor guidance, Peg / Gear / Nut (Isaac-Forge-*-Direct-v0) use trained RSL-RL
# checkpoints — not UR10e Shadow Hand pose keyframes.
#
# Usage (ViTacLab repo root, env_isaaclab_510test):
#   bash bash_command/demo_forge_franka_rl_collection_replay.sh
#   TASK=gear_mesh bash bash_command/demo_forge_franka_rl_collection_replay.sh
#   bash bash_command/demo_forge_franka_rl_all_collection_replay.sh
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim RL collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   TASK=peg_insert              # peg_insert | gear_mesh | nut_thread
#   CHECKPOINT=/path/model.pt    # override default checkpoint for TASK
#   RECORD_DIR=logs/demo_forge_franka/forge_peg_insert/collection
#   MAX_STEPS=800                # per-attempt step cap (usually hit by episode_length instead)
#   MAX_TOTAL_STEPS=0            # 0 = unlimited; e.g. 25000 to cap wall-clock search
#   NUM_ENVS=4                   # parallel envs (faster success search; uses more GPU RAM)
#   NUM_EPISODES=1
#   SKIP_COLLECT=0
#   MIN_CHECKPOINT_STEP=100      # nut_thread skips Franka RL if best ckpt step below this

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

MODE="${MODE:-full}"
TASK="${TASK:-peg_insert}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SEED="${SEED:-42}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-800}"
MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS:-0}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MIN_CHECKPOINT_STEP="${MIN_CHECKPOINT_STEP:-100}"

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

_checkpoint_step() {
  local ckpt="$1"
  local base
  base="$(basename "$ckpt" .pt)"
  echo "${base#model_}"
}

resolve_task_paths() {
  case "$TASK" in
    peg_insert)
      TASK_SLUG="forge_peg_insert"
      TASK_LABEL="Forge Peg Insert (Franka RL)"
      GYM_TASK="Isaac-Forge-PegInsert-Direct-v0"
      DEFAULT_CKPT_DIR="logs/rsl_rl/forge_peg_insert_direct"
      DEFAULT_CHECKPOINT="${DEFAULT_CKPT_DIR}/2026-02-10_11-42-29/model_1200.pt"
      ;;
    gear_mesh)
      TASK_SLUG="forge_gear_mesh"
      TASK_LABEL="Forge Gear Mesh (Franka RL)"
      GYM_TASK="Isaac-Forge-GearMesh-Direct-v0"
      DEFAULT_CKPT_DIR="logs/rsl_rl/forge_gear_mesh_direct"
      DEFAULT_CHECKPOINT="${DEFAULT_CKPT_DIR}/2026-03-06_23-30-00/model_1850.pt"
      ;;
    nut_thread)
      TASK_SLUG="forge_nut_thread"
      TASK_LABEL="Forge Nut Thread (Franka RL)"
      GYM_TASK="Isaac-Forge-NutThread-Direct-v0"
      DEFAULT_CKPT_DIR="logs/rsl_rl/Isaac-Forge-NutThread-Direct-v0"
      DEFAULT_CHECKPOINT="${DEFAULT_CKPT_DIR}/2026-04-18_04-46-59/model_0.pt"
      ;;
    *)
      echo "[ERROR] unknown TASK=$TASK (use peg_insert|gear_mesh|nut_thread)"
      exit 2
      ;;
  esac

  if [[ -n "${CHECKPOINT:-}" ]]; then
    RESOLVED_CHECKPOINT="$(readlink -f "$CHECKPOINT" 2>/dev/null || realpath "$CHECKPOINT" 2>/dev/null || echo "$CHECKPOINT")"
  elif [[ -f "$DEFAULT_CHECKPOINT" ]]; then
    RESOLVED_CHECKPOINT="$ROOT_DIR/$DEFAULT_CHECKPOINT"
  else
    best="$(_find_best_checkpoint "$ROOT_DIR/$DEFAULT_CKPT_DIR")"
    RESOLVED_CHECKPOINT="${best:-}"
  fi

  RECORD_DIR="${RECORD_DIR:-logs/demo_forge_franka/${TASK_SLUG}/collection}"
  VIDEO_DIR="${VIDEO_DIR:-logs/demo_forge_franka/${TASK_SLUG}/videos}"
  SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_forge_franka/${TASK_SLUG}/summary.txt}"
  VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_forge_franka/${TASK_SLUG}/validation.json}"
}

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

preflight() {
  if [[ -z "${RESOLVED_CHECKPOINT:-}" || ! -f "$RESOLVED_CHECKPOINT" ]]; then
    echo "[ERROR] missing RL checkpoint for TASK=$TASK"
    echo "  expected under: $DEFAULT_CKPT_DIR"
    echo "  set CHECKPOINT=/absolute/path/model_XXXX.pt"
    exit 2
  fi

  local step
  step="$(_checkpoint_step "$RESOLVED_CHECKPOINT")"
  if [[ "$TASK" == "nut_thread" && "$step" -lt "$MIN_CHECKPOINT_STEP" ]]; then
    echo "[ERROR] Franka Nut Thread checkpoint too early: $(basename "$RESOLVED_CHECKPOINT")"
    echo "  Only model_${step} found — train a usable checkpoint first, e.g.:"
    echo "  ${PY_SIM[*]} scripts/rsl_rl/full_rl/train.py --task Isaac-Forge-NutThread-Direct-v0 --headless --num_envs 128"
    echo "  Or run Shadow Hand nut demo instead:"
    echo "  TASK=forge_nut bash bash_command/demo_forge_collection_replay.sh"
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

  echo "[RUN] play_forge_rl_record.py"
  echo "  gym task: $GYM_TASK"
  echo "  checkpoint: $RESOLVED_CHECKPOINT"
  echo "  record: $RECORD_DIR"
  echo "  max_steps: $MAX_STEPS  max_total_steps: ${MAX_TOTAL_STEPS:-unlimited}"
  echo "  num_envs: $NUM_ENVS  num_episodes: $NUM_EPISODES"

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/rsl_rl/full_rl/play_forge_rl_record.py \
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
    --record_env_index 0 \
    --record_step_interval 1
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_forge_rl_record failed (exit=$play_rc)."
    echo "[HINT] Verify checkpoint matches task and increase MAX_STEPS=$MAX_STEPS"
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
    echo "ViTacLab Franka Forge RL demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "task: $TASK"
    echo "task_label: $TASK_LABEL"
    echo "task_slug: $TASK_SLUG"
    echo "gym_task: $GYM_TASK"
    echo ""
    echo "Checkpoint: $RESOLVED_CHECKPOINT"
    echo "Collection: play_forge_rl_record.py"
    echo "max_steps: $MAX_STEPS"
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
    echo "  TASK=$TASK bash bash_command/demo_forge_franka_rl_collection_replay.sh"
    echo "  MODE=visualize SKIP_COLLECT=1 TASK=$TASK bash bash_command/demo_forge_franka_rl_collection_replay.sh"
  } | tee "$SUMMARY_FILE"
}

resolve_task_paths
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
