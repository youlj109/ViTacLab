#!/usr/bin/env bash
# Generate a task-success demo video from expert trajectory replay records.
#
# This uses play_full_tra_single_v5.py (UR10e + ShadowHand forge task replay),
# records one successful episode, then renders third-person video from npz.
#
# Usage:
#   bash bash_command/forge_task_success_demo.sh
#
# Optional env vars:
#   TASK=forge_insert                 # forge_insert | forge_gear | forge_nut
#   SEED=42
#   FPS=20
#   RECORD_DIR=logs/task_success_replay/forge_insert
#   OUT_VIDEO=logs/task_success_replay/forge_insert_success_demo.mp4
#   TRAJECTORY_FILE=<path override>
#   FOCUS_TAIL_FRAMES=120
#   SLOWMO_TAIL_FRAMES=60
#   SLOWMO_REPEAT=2
#   STRICT_XY=0.0015               # <=0 to disable strict visual gate
#   STRICT_Z_RATIO=0.90            # <=0 to disable strict visual gate
#   REQUIRE_TACTILE_SUCCESS=1      # 1=require tactile gate for success saving
#   MIN_TACTILE_NORMAL_TOTAL=0.5
#   MIN_TACTILE_ACTIVE_RATIO=0.01
#   TACTILE_GATE_WINDOW=10
#   SUCCESS_HOLD_STEPS=12

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

TASK="${TASK:-forge_insert}"
SEED="${SEED:-42}"
FPS="${FPS:-20}"

case "$TASK" in
  forge_insert)
    default_traj="scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgePegInsertEnvCfg.json"
    ;;
  forge_gear)
    default_traj="scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeGearMeshEnvCfg.json"
    ;;
  forge_nut)
    default_traj="scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeNutThreadEnvCfg.json"
    ;;
  *)
    echo "[ERROR] Unsupported TASK=$TASK (expected forge_insert|forge_gear|forge_nut)"
    exit 2
    ;;
esac

TRAJECTORY_FILE="${TRAJECTORY_FILE:-$default_traj}"
RECORD_DIR="${RECORD_DIR:-logs/task_success_replay/${TASK}}"
OUT_VIDEO="${OUT_VIDEO:-logs/task_success_replay/${TASK}_success_demo.mp4}"
FOCUS_TAIL_FRAMES="${FOCUS_TAIL_FRAMES:-120}"
SLOWMO_TAIL_FRAMES="${SLOWMO_TAIL_FRAMES:-60}"
SLOWMO_REPEAT="${SLOWMO_REPEAT:-2}"
STRICT_XY="${STRICT_XY:-0.0015}"
STRICT_Z_RATIO="${STRICT_Z_RATIO:-0.90}"
REQUIRE_TACTILE_SUCCESS="${REQUIRE_TACTILE_SUCCESS:-1}"
MIN_TACTILE_NORMAL_TOTAL="${MIN_TACTILE_NORMAL_TOTAL:-0.5}"
MIN_TACTILE_ACTIVE_RATIO="${MIN_TACTILE_ACTIVE_RATIO:-0.01}"
TACTILE_GATE_WINDOW="${TACTILE_GATE_WINDOW:-10}"
SUCCESS_HOLD_STEPS="${SUCCESS_HOLD_STEPS:-12}"

mkdir -p "$RECORD_DIR"
# Ensure replay record is fresh for this run (avoid stale episode reuse).
rm -f "$RECORD_DIR"/episode_*.npz

run_with_heartbeat() {
  local label="$1"
  shift
  local start_ts
  start_ts="$(date +%s)"
  "$@" &
  local child_pid=$!
  local tick=0
  while kill -0 "$child_pid" >/dev/null 2>&1; do
    sleep 20
    tick=$((tick + 1))
    local now elapsed
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    printf '[PROGRESS][%s] elapsed=%ss (tick=%d, pid=%d)\n' "$label" "$elapsed" "$tick" "$child_pid"
  done
  wait "$child_pid"
}

echo "============================================================"
echo "[RUN] Replay trajectory for success episode"
echo "[TASK] $TASK"
echo "[TRAJ] $TRAJECTORY_FILE"
echo "[RECORD_DIR] $RECORD_DIR"
echo "============================================================"

run_with_heartbeat "replay" python scripts/rsl_rl/full_tra/play_full_tra_single_v5.py \
  --headless \
  --enable_cameras \
  --task "$TASK" \
  --trajectory-file "$TRAJECTORY_FILE" \
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

if ! ls "$RECORD_DIR"/episode_*.npz >/dev/null 2>&1; then
  echo "[ERROR] Replay did not produce episode_*.npz in $RECORD_DIR"
  exit 3
fi

echo "============================================================"
echo "[RUN] Render replay record to video"
echo "[OUT] $OUT_VIDEO"
echo "============================================================"

python scripts/demo/render_replay_record_video.py \
  --record_dir "$RECORD_DIR" \
  --output_video "$OUT_VIDEO" \
  --fps "$FPS" \
  --title "Forge Success Replay - ${TASK}" \
  --focus_tail_frames "$FOCUS_TAIL_FRAMES" \
  --slowmo_tail_frames "$SLOWMO_TAIL_FRAMES" \
  --slowmo_repeat "$SLOWMO_REPEAT"

echo "[DONE] task success demo video: $OUT_VIDEO"
