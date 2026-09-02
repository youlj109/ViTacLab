#!/usr/bin/env bash
# ViTacSim calibration sweep.
#   PROFILE=advisor  — M2 nut + G010..G210 @ 400x700 Xense (default for Task2/3)
#   PROFILE=cylinder — legacy W* cylinder weights + lateral
#
# Usage (ViTacLab repo root):
#   bash bash_command/run_vitacsim_calibration_sweep.sh
#   PROFILE=advisor DEVICE=cuda:0 SKIP_EXISTING=0 bash bash_command/run_vitacsim_calibration_sweep.sh
#
# Progress (second terminal):
#   python3 scripts/calibration/watch_calibration_sweep_progress.py
#   tail -f logs/vitacsim_calibration/sweep/_calibration_sweep.log

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

# Isaac Sim requires env_isaaclab_510test (torch + isaacsim); avoid base python.
if [[ -z "${CONDA_PREFIX:-}" ]] && [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate env_isaaclab_510test 2>/dev/null || true
fi

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
DEVICE="${DEVICE:-cuda:0}"
SIM_ROOT="${SIM_ROOT:-logs/vitacsim_calibration/sweep}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
FINGER_ROOT_Z="${FINGER_ROOT_Z:-0.441}"
WEIGHT_REST_Z="${WEIGHT_REST_Z:-0.442}"
SENSOR_MODE="${SENSOR_MODE:-vitacsim}"
PROFILE="${PROFILE:-advisor}"
MARKER_PATTERN="${MARKER_PATTERN:-xense}"
FITTED_PARAMS="${FITTED_PARAMS:-data/calibration/tactile/fitted_params.json}"

NF_OUT="$SIM_ROOT/normal_force"
LAT_OUT="$SIM_ROOT/shear_force/lateral"
NOMINAL_DIR="$NF_OUT/no_contact/$SENSOR_MODE"
PROGRESS_FILE="$SIM_ROOT/_sweep_progress.json"
LOG_FILE="$SIM_ROOT/_calibration_sweep.log"
SWEEP_START_TS="$(date +%s)"

if [[ "$PROFILE" == "advisor" ]]; then
  TOTAL_JOBS=7
  NF_CASES=(G210 G160 G110 G060 G030 G010)
  DEMO_PROFILE_ARGS=(--profile advisor)
  MARKER_PATTERN="${MARKER_PATTERN:-xense}"
else
  TOTAL_JOBS=11
  NF_CASES=(W200 W100 W050 W020 W010)
  DEMO_PROFILE_ARGS=(--profile cylinder)
  MARKER_PATTERN="${MARKER_PATTERN:-gelsight}"
fi

DONE_JOBS=0
mkdir -p "$SIM_ROOT"

marker_args=()
if [[ "${NO_MARKER:-0}" == "1" ]]; then
  marker_args+=(--no-marker)
else
  marker_args+=(--marker-pattern "$MARKER_PATTERN")
fi

fitted_args=()
if [[ -f "$ROOT_DIR/$FITTED_PARAMS" ]]; then
  fitted_args+=(--fitted-params "$ROOT_DIR/$FITTED_PARAMS")
  echo "[INFO] using fitted params: $FITTED_PARAMS"
fi

write_progress() {
  local done="$1"
  local current="$2"
  local status="$3"
  local now
  now="$(date +%s)"
  local elapsed=$((now - SWEEP_START_TS))
  mkdir -p "$SIM_ROOT"
  printf '{"done":%s,"total":%s,"current":"%s","status":"%s","elapsed_s":%s,"sensor_mode":"%s","profile":"%s"}\n' \
    "$done" "$TOTAL_JOBS" "$current" "$status" "$elapsed" "$SENSOR_MODE" "$PROFILE" > "$PROGRESS_FILE"
}

on_fail() {
  write_progress "$DONE_JOBS" "${CURRENT_JOB:-unknown}" "failed"
}
trap on_fail ERR

write_progress 0 "-" "starting"
echo "[INFO] calibration sweep profile=$PROFILE -> $SIM_ROOT (total jobs=$TOTAL_JOBS)"
echo "[INFO] progress: python3 scripts/calibration/watch_calibration_sweep_progress.py --root $SIM_ROOT"
echo "[INFO] log: tail -f $LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

run_nf() {
  local cid="$1"
  CURRENT_JOB="NF/${cid}"
  local sub="$NF_OUT/$cid/$SENSOR_MODE"
  if [[ "$SKIP_EXISTING" == "1" && -f "$sub/summary.json" ]]; then
    echo "[SKIP] NF $cid"
    DONE_JOBS=$((DONE_JOBS + 1))
    write_progress "$DONE_JOBS" "$CURRENT_JOB" "skipped"
    return 0
  fi
  write_progress "$DONE_JOBS" "$CURRENT_JOB" "running"
  echo "======== [$(date -Iseconds)] NF case=$cid mode=$SENSOR_MODE ($((DONE_JOBS + 1))/${TOTAL_JOBS}) ========"
  local case_args=(--case-id "$cid")
  if [[ "$PROFILE" == "cylinder" ]]; then
    case_args=(--weight-id "$cid")
  fi
  if ! PYTHONUNBUFFERED=1 "$ISAACLAB_SH" -p scripts/demo/demo_vitacsim_normal_force_validation.py \
    --headless --enable_cameras --device "$DEVICE" \
    "${DEMO_PROFILE_ARGS[@]}" \
    "${case_args[@]}" --sensor-mode "$SENSOR_MODE" \
    --out-dir "$NF_OUT" \
    --finger-root-z "$FINGER_ROOT_Z" \
    --weight-rest-z "$WEIGHT_REST_Z" \
    "${marker_args[@]}" "${fitted_args[@]}"; then
    write_progress "$DONE_JOBS" "$CURRENT_JOB" "failed"
    exit 1
  fi
  DONE_JOBS=$((DONE_JOBS + 1))
  write_progress "$DONE_JOBS" "$CURRENT_JOB" "done"
}

capture_nominal() {
  CURRENT_JOB="no_contact"
  if [[ "$SKIP_EXISTING" == "1" && -f "$NOMINAL_DIR/tactile_rgb.png" ]]; then
    echo "[SKIP] no_contact nominal"
    DONE_JOBS=$((DONE_JOBS + 1))
    write_progress "$DONE_JOBS" "$CURRENT_JOB" "skipped"
    return 0
  fi
  write_progress "$DONE_JOBS" "$CURRENT_JOB" "running"
  echo "======== [$(date -Iseconds)] no_contact nominal ($((DONE_JOBS + 1))/${TOTAL_JOBS}) ========"
  local nom_case=(--case-id G110)
  local nom_weight=(--weight-id W100)
  if [[ "$PROFILE" == "cylinder" ]]; then
    nom_case=()
  else
    nom_weight=()
  fi
  if ! PYTHONUNBUFFERED=1 "$ISAACLAB_SH" -p scripts/demo/demo_vitacsim_normal_force_validation.py \
    --headless --enable_cameras --device "$DEVICE" \
    "${DEMO_PROFILE_ARGS[@]}" \
    "${nom_case[@]}" "${nom_weight[@]}" --sensor-mode "$SENSOR_MODE" \
    --out-dir "$NF_OUT" \
    --finger-root-z "$FINGER_ROOT_Z" \
    --save-nominal-to "$NOMINAL_DIR" \
    --nominal-only \
    "${marker_args[@]}"; then
    write_progress "$DONE_JOBS" "$CURRENT_JOB" "failed"
    exit 1
  fi
  DONE_JOBS=$((DONE_JOBS + 1))
  write_progress "$DONE_JOBS" "$CURRENT_JOB" "done"
}

run_lateral() {
  local fx="$1"
  local fmt
  fmt="$(python3 - <<PY
fx=float("$fx")
s=f"{fx:.3f}".rstrip("0").rstrip(".")
print(f"Fx{s.replace('-', 'm')}_Fy0")
PY
)"
  CURRENT_JOB="LAT/${fmt}"
  local sub="$LAT_OUT/${fmt}/W100/$SENSOR_MODE"
  if [[ "$SKIP_EXISTING" == "1" && -f "$sub/summary.json" ]]; then
    echo "[SKIP] lateral Fx=$fx"
    DONE_JOBS=$((DONE_JOBS + 1))
    write_progress "$DONE_JOBS" "$CURRENT_JOB" "skipped"
    return 0
  fi
  write_progress "$DONE_JOBS" "$CURRENT_JOB" "running"
  echo "======== [$(date -Iseconds)] Lateral W100 Fx=$fx ($((DONE_JOBS + 1))/${TOTAL_JOBS}) ========"
  if ! PYTHONUNBUFFERED=1 "$ISAACLAB_SH" -p scripts/demo/demo_vitacsim_lateral_force_validation.py \
    --headless --enable_cameras --device "$DEVICE" \
    --weight-id W100 --sensor-mode "$SENSOR_MODE" \
    --lateral-force-x "$fx" --lateral-force-y 0.0 \
    --out-dir "$LAT_OUT" \
    --finger-root-z "$FINGER_ROOT_Z" \
    --weight-rest-z "$WEIGHT_REST_Z" \
    "${marker_args[@]}" "${fitted_args[@]}"; then
    write_progress "$DONE_JOBS" "$CURRENT_JOB" "failed"
    exit 1
  fi
  DONE_JOBS=$((DONE_JOBS + 1))
  write_progress "$DONE_JOBS" "$CURRENT_JOB" "done"
}

capture_nominal

for cid in "${NF_CASES[@]}"; do
  run_nf "$cid"
done

if [[ "$PROFILE" == "cylinder" ]]; then
  for fx in 0.0 0.05 0.1 0.15 0.2; do
    run_lateral "$fx"
  done
fi

write_progress "$TOTAL_JOBS" "index" "indexing"
python3 scripts/calibration/collect_sim_calibration_index.py \
  --sim-root "$SIM_ROOT" --sensor-mode "$SENSOR_MODE" --profile "$PROFILE"

write_progress "$TOTAL_JOBS" "-" "finished"
echo "[DONE] sim calibration sweep (profile=$PROFILE) -> $SIM_ROOT"
echo "[NEXT] bash bash_command/run_task2_advisor_calibration.sh"
