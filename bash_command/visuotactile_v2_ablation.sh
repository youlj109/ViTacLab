#!/usr/bin/env bash
# VisuoTactileSensorV2 ablation runs (Stage D matrix):
#   modes: baseline / normal_only / full
#   render: raw depth render vs corrected-force render
#
# Usage:
#   bash bash_command/visuotactile_v2_ablation.sh
#
# Notes:
# - Run from ViTacLab repo root.
# - Uses the same seed/steps across modes for fair comparison.
# - Logs are written to logs/ablation_visuotactile_v2/*.log
# - Per-run summary json is written to logs/ablation_visuotactile_v2/*.json
# - Uses unbuffered Python stdout so step logs appear in terminal immediately.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure local ViTacLab package import works (ViTacLab/... under source/ViTacLab).
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="logs/ablation_visuotactile_v2"
mkdir -p "$OUT_DIR"

# Common options shared by all ablation runs.
COMMON_ARGS=(
  --enable_cameras
  --num_envs 1
  --max_steps 3000
  --log_interval 60
  --seed 42
)

# Helper: run one case and tee to log.
run_case() {
  local mode="$1"
  local render_tag="$2"
  local corrected_flag="$3"
  local log_file="$OUT_DIR/${mode}_${render_tag}.log"
  local summary_file="$OUT_DIR/${mode}_${render_tag}_summary.json"
  echo "============================================================"
  echo "[RUN] mode=${mode}, render=${render_tag} -> ${log_file}"
  echo "============================================================"
  local extra_args=()
  if [[ "$corrected_flag" == "1" ]]; then
    extra_args+=(--enable_corrected_force_render --corrected_force_render_blend 1.0)
  fi
  PYTHONUNBUFFERED=1 python -u scripts/demo/demo_visuotactile_sensor_v2_ablation.py \
    "${COMMON_ARGS[@]}" \
    --ablation_mode "$mode" \
    --summary_json "$summary_file" \
    "${extra_args[@]}" | tee "$log_file"
}

# Stage D matrix: three modes x two render settings.
for mode in baseline normal_only full; do
  run_case "$mode" "raw" "0"
  run_case "$mode" "corr" "1"
done

echo
echo "[DONE] Ablation logs + summaries saved in: $OUT_DIR"
