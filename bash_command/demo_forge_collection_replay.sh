#!/usr/bin/env bash
# Shadow Hand Forge demo: trajectory replay + canonical NPZ + tactile panel videos.
#
# Tasks (UR10e + ShadowHand, pose_keyframes replay via play_full_tra_single_v5):
#   forge_insert  — Forge Peg Insert
#   forge_gear    — Forge Gear Mesh
#   forge_nut     — Forge Nut Thread
#
# Usage (ViTacLab repo root, env_isaaclab_510test):
#   bash bash_command/demo_forge_collection_replay.sh
#   TASK=forge_gear bash bash_command/demo_forge_collection_replay.sh
#   bash bash_command/demo_forge_all_collection_replay.sh   # all three tasks
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   TASK=forge_insert          # forge_insert | forge_gear | forge_nut
#   RECORD_DIR=logs/demo_forge/forge_peg_insert/collection
#   VIDEO_DIR=logs/demo_forge/forge_peg_insert/videos
#   SKIP_COLLECT=0
#   FPS=10
#   SEED=42
#
# Outputs (video names include task slug):
#   $RECORD_DIR/episode_*.npz
#   $VIDEO_DIR/${TASK_SLUG}_demo_panel.mp4
#   $VIDEO_DIR/${TASK_SLUG}_third_person.mp4
#   $VIDEO_DIR/${TASK_SLUG}_tactile_rgb.mp4
#   $VIDEO_DIR/${TASK_SLUG}_tactile_force_field.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-full}"
TASK="${TASK:-forge_insert}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SEED="${SEED:-42}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"

# Demo defaults: relax strict gates so showcase recording is easier.
# Set REQUIRE_TACTILE_SUCCESS=1 for stricter success matching (forge_task_success_demo style).
STRICT_XY="${STRICT_XY:-0}"
STRICT_Z_RATIO="${STRICT_Z_RATIO:-0}"
REQUIRE_TACTILE_SUCCESS="${REQUIRE_TACTILE_SUCCESS:-0}"
MIN_TACTILE_NORMAL_TOTAL="${MIN_TACTILE_NORMAL_TOTAL:-0.5}"
MIN_TACTILE_ACTIVE_RATIO="${MIN_TACTILE_ACTIVE_RATIO:-0.01}"
TACTILE_GATE_WINDOW="${TACTILE_GATE_WINDOW:-10}"
SUCCESS_HOLD_STEPS="${SUCCESS_HOLD_STEPS:-12}"

ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
if [[ -x "$ISAACLAB_SH" ]]; then
  PY_SIM=( "$ISAACLAB_SH" -p )
else
  PY_SIM=( python )
fi
PY_OFFLINE=( python3 )

resolve_task_paths() {
  case "$TASK" in
    forge_insert)
      TASK_SLUG="forge_peg_insert"
      TASK_LABEL="Forge Peg Insert"
      TRAJECTORY="${TRAJECTORY:-scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgePegInsertEnvCfg.json}"
      ;;
    forge_gear)
      TASK_SLUG="forge_gear_mesh"
      TASK_LABEL="Forge Gear Mesh"
      TRAJECTORY="${TRAJECTORY:-scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeGearMeshEnvCfg.json}"
      ;;
    forge_nut)
      TASK_SLUG="forge_nut_thread"
      TASK_LABEL="Forge Nut Thread"
      TRAJECTORY="${TRAJECTORY:-scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeNutThreadEnvCfg.json}"
      ;;
    *)
      echo "[ERROR] unknown TASK=$TASK (use forge_insert|forge_gear|forge_nut)"
      exit 2
      ;;
  esac

  RECORD_DIR="${RECORD_DIR:-logs/demo_forge/${TASK_SLUG}/collection}"
  VIDEO_DIR="${VIDEO_DIR:-logs/demo_forge/${TASK_SLUG}/videos}"
  SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_forge/${TASK_SLUG}/summary.txt}"
  VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_forge/${TASK_SLUG}/validation.json}"
}

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

preflight() {
  if [[ ! -f "$TRAJECTORY" ]]; then
    echo "[ERROR] missing trajectory/keyframes: $TRAJECTORY"
    exit 2
  fi
}

run_collect() {
  log_section "[1/3] ${TASK_LABEL}: trajectory replay + NPZ collection"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi
  preflight

  rm -f "$RECORD_DIR"/episode_*.npz
  mkdir -p "$RECORD_DIR"

  echo "[RUN] play_full_tra_single_v5"
  echo "  task: $TASK ($TASK_LABEL)"
  echo "  trajectory: $TRAJECTORY"
  echo "  record: $RECORD_DIR"
  echo "  device: $DEVICE"

  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi -L >/dev/null 2>&1; then
      echo "[WARN] nvidia-smi failed — Isaac Sim may crash with CUDA error 804/805"
    fi
  fi

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/rsl_rl/full_tra/play_full_tra_single_v5.py \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --task "$TASK" \
    --trajectory-file "$TRAJECTORY" \
    --num_envs 1 \
    --seed "$SEED" \
    --record-data \
    --record-path "$RECORD_DIR" \
    --record-max-episodes 1 \
    --record-step-interval 1 \
    --max-success-xy-dist "$STRICT_XY" \
    --max-success-z-disp-ratio "$STRICT_Z_RATIO" \
    $( [[ "$REQUIRE_TACTILE_SUCCESS" == "1" ]] && printf '%s' "--require-tactile-success" ) \
    --min-tactile-normal-total "$MIN_TACTILE_NORMAL_TOTAL" \
    --min-tactile-active-ratio "$MIN_TACTILE_ACTIVE_RATIO" \
    --tactile-gate-window "$TACTILE_GATE_WINDOW" \
    --success-hold-steps "$SUCCESS_HOLD_STEPS" \
    --post-arm-reached-steps 30 \
    --stable-steps 30
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_full_tra_single_v5 failed (exit=$play_rc) before/during simulation."
    echo "[HINT] Scroll up for RuntimeError / CUDA 804. Common fixes:"
    echo "  1) nvidia-smi          # verify GPU visible"
    echo "  2) Close other Isaac Sim / Omniverse processes"
    echo "  3) Re-run: DEVICE=cuda:0 TASK=$TASK bash bash_command/demo_forge_collection_replay.sh"
    exit "$play_rc"
  fi

  if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
    echo "[ERROR] simulation finished but no episode_*.npz under $RECORD_DIR"
    echo "[HINT] v5 saves only when forge task reports success. Try:"
    echo "  REQUIRE_TACTILE_SUCCESS=0 STRICT_XY=0 STRICT_Z_RATIO=0 TASK=$TASK bash bash_command/demo_forge_collection_replay.sh"
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

  # Prefix exported mp4 names with task slug for multi-task demos.
  shopt -s nullglob
  for f in "$VIDEO_DIR"/*.mp4; do
    base="$(basename "$f")"
    if [[ "$base" != "${TASK_SLUG}_"* ]]; then
      case "$base" in
        *_demo_panel.mp4)
          mv "$f" "$VIDEO_DIR/${TASK_SLUG}_demo_panel.mp4"
          ;;
        *_third_person.mp4)
          mv "$f" "$VIDEO_DIR/${TASK_SLUG}_third_person.mp4"
          ;;
        *_tactile_rgb.mp4)
          mv "$f" "$VIDEO_DIR/${TASK_SLUG}_tactile_rgb.mp4"
          ;;
        *_tactile_force_field.mp4)
          mv "$f" "$VIDEO_DIR/${TASK_SLUG}_tactile_force_field.mp4"
          ;;
        *)
          mv "$f" "$VIDEO_DIR/${TASK_SLUG}_${base}"
          ;;
      esac
    fi
  done
}

write_summary() {
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  {
    echo "ViTacLab Forge demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "task: $TASK"
    echo "task_label: $TASK_LABEL"
    echo "task_slug: $TASK_SLUG"
    echo ""
    echo "Trajectory: $TRAJECTORY"
    echo "Collection: play_full_tra_single_v5"
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
    echo "  TASK=$TASK bash bash_command/demo_forge_collection_replay.sh"
    echo "  MODE=visualize SKIP_COLLECT=1 TASK=$TASK bash bash_command/demo_forge_collection_replay.sh"
    echo "  bash bash_command/demo_forge_all_collection_replay.sh"
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
