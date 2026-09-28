#!/usr/bin/env bash
# One-command demo: sensor showcase + Pickup data collection + NPZ validation + replay videos.
#
# Usage (from ViTacLab repo root, inside env_isaaclab_510test or any Isaac Sim python env):
#   bash bash_command/demo_data_collection_replay.sh
#
# Modes (env var MODE):
#   full       — sensor video (optional) + collect Pickup NPZ + validate + export mp4  [default]
#   collect    — only run play_single recording
#   visualize  — only validate + export mp4 from existing RECORD_DIR
#   sensor     — only run visuotactile alignment visual demo
#
# Common env vars:
#   RECORD_DIR=logs/demo_data_collection_replay/pickup_npz
#   VIDEO_DIR=logs/demo_data_collection_replay/videos
#   TRAJECTORY=scripts/data_collection/full_trajectory/records/pickup.json
#   SKIP_SENSOR=1          # skip alignment visual (faster; use existing mp4 if present)
#   SKIP_COLLECT=0         # set 1 to reuse existing NPZ under RECORD_DIR
#   SEED=42
#   FPS=10                 # replay video export fps
#   FORCE_EXIT=1           # avoid Isaac Sim hang after sensor demo
#
# Outputs:
#   $RECORD_DIR/episode_*_completed.npz   — canonical tactile + camera + action data
#   $VIDEO_DIR/*_demo_panel.mp4           — stacked third-person / tactile RGB / force field
#   logs/demo_data_collection_replay/summary.txt
#
# Pour task (deformable cup): use bash_command/demo_pour_collection_replay.sh instead.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-full}"
RECORD_DIR="${RECORD_DIR:-logs/demo_data_collection_replay/pickup_npz}"
VIDEO_DIR="${VIDEO_DIR:-logs/demo_data_collection_replay/videos}"
TRAJECTORY="${TRAJECTORY:-scripts/data_collection/full_trajectory/records/pickup.json}"
SKIP_SENSOR="${SKIP_SENSOR:-0}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SEED="${SEED:-42}"
FPS="${FPS:-10}"
FORCE_EXIT="${FORCE_EXIT:-1}"
SENSOR_VIDEO="${SENSOR_VIDEO:-logs/alignment_visuotactile_v2/interference_visual_demo_present.mp4}"
SENSOR_STEPS="${SENSOR_STEPS:-180}"
SUMMARY_FILE="${SUMMARY_FILE:-logs/demo_data_collection_replay/summary.txt}"

# Isaac Sim scripts (play_single, sensor demos).
ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
if [[ -x "$ISAACLAB_SH" ]]; then
  PY_SIM=( "$ISAACLAB_SH" -p )
else
  PY_SIM=( python )
fi

# Offline tools (validate / mp4 export) — no Isaac Sim required.
PY_OFFLINE=( python3 )

mkdir -p "$(dirname "$SUMMARY_FILE")" "$RECORD_DIR" "$VIDEO_DIR"

log_section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_sensor_demo() {
  log_section "[1/4] ViTacSim sensor visual demo"
  if [[ "$SKIP_SENSOR" == "1" ]]; then
    echo "[SKIP] SKIP_SENSOR=1"
    if [[ -f "$SENSOR_VIDEO" ]]; then
      echo "[INFO] existing sensor video: $SENSOR_VIDEO"
    else
      echo "[WARN] no sensor video at $SENSOR_VIDEO (run MODE=sensor to generate)"
    fi
    return 0
  fi

  if [[ -f "$SENSOR_VIDEO" ]]; then
    echo "[INFO] reuse existing sensor video: $SENSOR_VIDEO"
    return 0
  fi

  loose_video="logs/alignment_visuotactile_v2/interference_visual_loose.mp4"
  strict_video="logs/alignment_visuotactile_v2/interference_visual_strict.mp4"

  STEPS="$SENSOR_STEPS" SEED="$SEED" STRICT=0 OUT_VIDEO="$loose_video" FORCE_EXIT="$FORCE_EXIT" \
    bash bash_command/visuotactile_alignment_visual.sh || true
  STEPS="$SENSOR_STEPS" SEED="$SEED" STRICT=1 OUT_VIDEO="$strict_video" FORCE_EXIT="$FORCE_EXIT" \
    bash bash_command/visuotactile_alignment_visual.sh || true

  if [[ -f "$loose_video" && -f "$strict_video" ]]; then
    LOOSE_VIDEO="$loose_video" STRICT_VIDEO="$strict_video" OUT_VIDEO="$SENSOR_VIDEO" \
      bash bash_command/visuotactile_alignment_demo.sh || true
  fi

  if [[ -f "$SENSOR_VIDEO" ]]; then
    echo "[OK] sensor presentation video: $SENSOR_VIDEO"
  else
    echo "[WARN] sensor video not generated; continuing with data collection"
  fi
}

run_collect() {
  log_section "[2/4] Pickup trajectory replay + NPZ collection"
  if [[ "$SKIP_COLLECT" == "1" ]]; then
    echo "[SKIP] SKIP_COLLECT=1, using RECORD_DIR=$RECORD_DIR"
    return 0
  fi

  if [[ ! -f "$TRAJECTORY" ]]; then
    echo "[ERROR] trajectory not found: $TRAJECTORY"
    exit 2
  fi

  echo "[RUN] play_single pickup"
  echo "  trajectory: $TRAJECTORY"
  echo "  record dir: $RECORD_DIR"

  PYTHONUNBUFFERED=1 "${PY_SIM[@]}" scripts/data_collection/full_trajectory/play_single.py \
    --headless \
    --enable_cameras \
    --task pickup \
    --trajectory-file "$TRAJECTORY" \
    --seed "$SEED" \
    --num_envs 1 \
    --record-data \
    --save-outcome completed \
    --record-path "$RECORD_DIR" \
    --record-max-episodes 1 \
    --record-step-interval 2 \
    --max-steps-per-trajectory 1200

  echo "[OK] collection finished under $RECORD_DIR"
}

run_validate() {
  log_section "[3/4] Validate NPZ schema"
  "${PY_OFFLINE[@]}" scripts/debug/validate_play_record_npz.py "$RECORD_DIR" \
    --json-out "logs/demo_data_collection_replay/validation.json"
}

run_visualize() {
  log_section "[4/4] Export replay videos (third-person + tactile RGB + force field)"
  "${PY_OFFLINE[@]}" scripts/debug/visualize_play_record_npz.py \
    "$RECORD_DIR" \
    --out_dir "$VIDEO_DIR" \
    --fps "$FPS"
}

write_summary() {
  {
    echo "ViTacLab data collection & replay demo summary"
    echo "generated: $(date -Iseconds)"
    echo "mode: $MODE"
    echo ""
    echo "Sensor showcase:"
    echo "  $SENSOR_VIDEO"
    echo ""
    echo "Recorded NPZ:"
    ls -lh "$RECORD_DIR"/*.npz 2>/dev/null || echo "  (none)"
    echo ""
    echo "Replay videos:"
    ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || echo "  (none)"
    echo ""
    echo "Quick replay (panel video):"
    panel="$(ls "$VIDEO_DIR"/*_demo_panel.mp4 2>/dev/null | head -1 || true)"
    if [[ -n "$panel" ]]; then
      echo "  $panel"
    fi
    echo ""
    echo "Re-run options:"
    echo "  bash bash_command/demo_data_collection_replay.sh                    # full pipeline"
    echo "  MODE=visualize SKIP_COLLECT=1 bash bash_command/demo_data_collection_replay.sh"
    echo "  MODE=sensor bash bash_command/demo_data_collection_replay.sh"
    echo ""
    echo "Live tactile GUI (needs DISPLAY):"
    echo "  python scripts/data_collection/manual/record_observations.py \\"
    echo "    --task pickup --num_envs 1 --show_rgb --show_ff --random_actions --enable_cameras"
  } | tee "$SUMMARY_FILE"
}

case "$MODE" in
  full)
    run_sensor_demo
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
  sensor)
    SKIP_SENSOR=0 run_sensor_demo
    write_summary
    ;;
  *)
    echo "[ERROR] unknown MODE=$MODE (use full|collect|visualize|sensor)"
    exit 2
    ;;
esac

log_section "[DONE] Demo finished"
echo "Summary: $SUMMARY_FILE"
panel="$(ls "$VIDEO_DIR"/*_demo_panel.mp4 2>/dev/null | head -1 || true)"
if [[ -n "$panel" ]]; then
  echo "Main replay video: $panel"
fi
