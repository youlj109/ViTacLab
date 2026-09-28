#!/usr/bin/env bash
# Shadow Hand Blind Grasp demo: phase keyframe replay + tactile-only NPZ + tactile panel videos.
#
# Uses play_single_phase.py (ordered keyframe sequences; play_full_tra_single_v5 does not
# support blind_grasp presets and replays JSON array order, not business phase order).
#
# Recorded NPZ keeps joint proprioception, tactile arrays, and third-person RGB
# for demo videos. Object/world poses stay out of control and the IL contract.
# Validate with ``--profile tactile_only``. Success means the cube is physically
# lifted out of the bin after a tactile grasp, not a geometry teleport/carry.
#
# Phases:
#   1 — Touch01 + Pick01 + Finish01  (default showcase: explore bin, grasp cube)
#   2 — Touch01/02 + Pick02 + Finish02
#   3 — BlindGraspReplay env (same sequence as phase 2)
#
# Usage (ViTacLab repo root, an Isaac Sim-enabled Python environment):
#   bash bash_command/demo_blind_grasp_collection_replay.sh
#   PHASE=2 bash bash_command/demo_blind_grasp_collection_replay.sh
#   OBJECT_INIT_INDEX=1 SAVE_OUTCOME=success bash bash_command/demo_blind_grasp_collection_replay.sh
#
# Modes (env var MODE):
#   full       — collect + validate + export mp4  [default]
#   collect    — only Isaac Sim collection
#   visualize  — only validate + mp4 from existing RECORD_DIR
#
# Common env vars:
#   PHASE=1                    # 1 | 2 | 3
#   OBJECT_INIT_INDEX=auto     # auto: phase1->0 (Pick01@0.62), phase2/3->1 (Pick02@0.50); or 0..2
#   ACTION_SMOOTHING=0.0       # must be low; default 0.75 prevents hand closure / grasp
#   BENCH_MODE=1               # strict: no object pose/distance/alignment/carry (default 1)
#   SAVE_OUTCOME=completed     # completed | success | all  (completed = demo-friendly)
#   RECORD_DIR=logs/demo_blind_grasp/phase1/collection
#   VIDEO_DIR=logs/demo_blind_grasp/phase1/videos
#   SKIP_COLLECT=0
#   FPS=10
#   DEVICE=cuda:0
#
# Outputs:
#   $RECORD_DIR/episode_*_{completed|success}.npz
#   $VIDEO_DIR/blind_grasp_phase${PHASE}_demo_panel.mp4
#   $VIDEO_DIR/blind_grasp_phase${PHASE}_tactile_rgb.mp4
#   $VIDEO_DIR/blind_grasp_phase${PHASE}_tactile_force_field.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/bash_command/demo_runtime.sh"

# isaaclab.sh runs `tabs 4`, which fails when TERM=dumb (Cursor/CI shells).
if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

MODE="${MODE:-full}"
PHASE="${PHASE:-1}"
# Keyframe JSON: Pick01 branch uses object_init_index=0 (0.62); Pick02 uses index=1 (0.50).
case "$PHASE" in
  1) _DEFAULT_OBJECT_INIT_INDEX=0 ;;
  2|3) _DEFAULT_OBJECT_INIT_INDEX=1 ;;
  *) _DEFAULT_OBJECT_INIT_INDEX=0 ;;
esac
OBJECT_INIT_INDEX="${OBJECT_INIT_INDEX:-$_DEFAULT_OBJECT_INIT_INDEX}"
ACTION_SMOOTHING="${ACTION_SMOOTHING:-0.0}"
BENCH_MODE="${BENCH_MODE:-1}"
if [[ "$BENCH_MODE" != "1" ]]; then
  echo "[ERROR] Blind Grasp supports strict tactile BENCH_MODE=1 only." >&2
  exit 2
fi
SAVE_OUTCOME="${SAVE_OUTCOME:-success}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
FPS="${FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
NUM_EPISODES="${NUM_EPISODES:-1}"

TASK_SLUG="blind_grasp_phase${PHASE}"
RECORD_DIR="${RECORD_DIR:-logs/demo_blind_grasp/phase${PHASE}/collection}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_blind_grasp/phase${PHASE}/videos}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_blind_grasp/phase${PHASE}/summary.txt}"
VALIDATION_JSON="${VALIDATION_JSON:-logs/demo_blind_grasp/phase${PHASE}/validation.json}"

case "$PHASE" in
  1)
    TASK_LABEL="Blind Grasp Phase 1 (Touch01 + Pick01)"
    KEYFRAME_JSON="${KEYFRAME_JSON:-scripts/data_collection/full_trajectory/pose_keyframes/UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json}"
    ;;
  2)
    TASK_LABEL="Blind Grasp Phase 2 (Touch01/02 + Pick02)"
    KEYFRAME_JSON="${KEYFRAME_JSON:-scripts/data_collection/full_trajectory/pose_keyframes/UR10eShadowHandBlindGraspEnv__UR10eShadowHandBlindGraspEnvCfg.json}"
    ;;
  3)
    TASK_LABEL="Blind Grasp Phase 3 (Replay env)"
    KEYFRAME_JSON="${KEYFRAME_JSON:-scripts/data_collection/full_trajectory/pose_keyframes/UR10eShadowHandBlindGraspReplayEnv__UR10eShadowHandBlindGraspReplayEnvCfg.json}"
    ;;
  *)
    echo "[ERROR] unknown PHASE=$PHASE (use 1|2|3)"
    exit 2
    ;;
esac

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
  log_section "[1/3] ${TASK_LABEL}: phase replay + NPZ collection"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi
  preflight

  demo_prepare_reports
  demo_fresh_directory RECORD_DIR

  echo "[RUN] play_single_phase.py"
  echo "  phase: $PHASE ($TASK_LABEL)"
  echo "  keyframes: $KEYFRAME_JSON"
  echo "  object_init_index: $OBJECT_INIT_INDEX"
  echo "  action_smoothing: $ACTION_SMOOTHING"
  echo "  bench_mode: $BENCH_MODE"
  echo "  save_outcome: $SAVE_OUTCOME"
  echo "  record: $RECORD_DIR"
  echo "  device: $DEVICE"

  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi -L >/dev/null 2>&1; then
      echo "[WARN] nvidia-smi failed — Isaac Sim may crash with CUDA error 804/805"
    fi
  fi

  set +e
  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/data_collection/full_trajectory/play_single_phase.py \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --phase "$PHASE" \
    --keyframe-json "$KEYFRAME_JSON" \
    --num-envs 1 \
    --num-episodes "$NUM_EPISODES" \
    --object-init-index "$OBJECT_INIT_INDEX" \
    --action-smoothing "$ACTION_SMOOTHING" \
    --save-outcome "$SAVE_OUTCOME" \
    --record-path "$RECORD_DIR" \
    --record-step-interval 5 \
    --stable-steps 20 \
    --max-steps-per-frame 400 \
    --arm-pos-tol 0.02 \
    --post-arm-reached-steps 40 \
    --pick-dwell-steps 250 \
    --hand-noise 0 \
    --object-xy-noise 0 \
    --pick-hand-close-scale 1.0 \
    --thumb-oppose-step-rad 0.02 \
    --lift-arm-step-rad 0.008 \
    --success-z-threshold 0.35 \
    --object-settle-steps 60 \
    --bench-mode \
    --tactile-contact-threshold 1e-4
  play_rc=$?
  set -e

  if [[ "$play_rc" -ne 0 ]]; then
    echo "[ERROR] play_single_phase failed (exit=$play_rc)."
    echo "[HINT] If Isaac Sim hangs, identify and stop only this launcher's process."
    echo "[HINT] SAVE_OUTCOME=completed PHASE=$PHASE bash bash_command/demo_blind_grasp_collection_replay.sh"
    exit "$play_rc"
  fi

  if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
    echo "[ERROR] simulation finished but no episode_*.npz under $RECORD_DIR"
    echo "[HINT] Open-loop keyframes may not trigger env success. Try:"
    echo "  SAVE_OUTCOME=completed PHASE=$PHASE bash bash_command/demo_blind_grasp_collection_replay.sh"
    echo "  SAVE_OUTCOME=all PHASE=$PHASE bash bash_command/demo_blind_grasp_collection_replay.sh"
    exit 3
  fi
  echo "[OK] collection finished under $RECORD_DIR"
}

run_validate() {
  demo_prepare_reports
  log_section "[2/3] Validate NPZ schema (tactile_only)"
  "${PY_OFFLINE[@]}" scripts/debug/validate_play_record_npz.py "$RECORD_DIR" \
    --profile tactile_only \
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
    echo "ViTacLab Blind Grasp demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo "phase: $PHASE"
    echo "task_label: $TASK_LABEL"
    echo "task_slug: $TASK_SLUG"
    echo "object_init_index: $OBJECT_INIT_INDEX"
    echo "save_outcome: $SAVE_OUTCOME"
    echo ""
    echo "Keyframes: $KEYFRAME_JSON"
    echo "Collection: play_single_phase.py --phase $PHASE"
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
    echo "  PHASE=$PHASE bash bash_command/demo_blind_grasp_collection_replay.sh"
    echo "  MODE=visualize SKIP_COLLECT=1 PHASE=$PHASE bash bash_command/demo_blind_grasp_collection_replay.sh"
    echo "  bash bash_command/demo_blind_grasp_all_phases.sh"
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
