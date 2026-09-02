#!/usr/bin/env bash
# Normal-force ViTacSim validation sweep (5 masses × tacsl/vitacsim).
#
# Usage (ViTacLab repo root):
#   bash bash_command/demo_vitacsim_normal_force_validation.sh
#   WEIGHT=W050 MODES=vitacsim bash bash_command/demo_vitacsim_normal_force_validation.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

ISAACLAB_SH="${ISAACLAB_SH:-$ROOT_DIR/../IsaacLab/isaaclab.sh}"
OUT_DIR="${OUT_DIR:-logs/vitacsim_validation/normal_force}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS="${WEIGHTS:-W200 W100 W050 W020 W010}"
MODES="${MODES:-tacsl vitacsim}"
WEIGHT="${WEIGHT:-}"  # optional single weight override
SKIP_EXISTING="${SKIP_EXISTING:-0}"  # skip if summary.json exists (v3 schema)
FINGER_ROOT_Z="${FINGER_ROOT_Z:-0.444}"
PROGRESS_FILE="$OUT_DIR/_sweep_progress.json"

write_progress() {
  local done="$1"
  local total="$2"
  local current="$3"
  local status="$4"
  mkdir -p "$OUT_DIR"
  printf '{"done":%s,"total":%s,"current":"%s","status":"%s"}\n' \
    "$done" "$total" "$current" "$status" > "$PROGRESS_FILE"
}

count_planned() {
  if [[ -n "$WEIGHT" ]]; then
    echo "$MODES" | wc -w
  else
    echo "$WEIGHTS" | wc -w | awk -v m="$(echo "$MODES" | wc -w)" '{print $1*m}'
  fi
}

TOTAL_JOBS="$(count_planned)"
DONE_JOBS=0
write_progress 0 "$TOTAL_JOBS" "-" "starting"

run_one() {
  local wid="$1"
  local mode="$2"
  local out_sub="$OUT_DIR/$wid/$mode"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_sub/summary.json" ]]; then
    schema="$(python3 - <<PY
import json
from pathlib import Path
p = Path("$out_sub/summary.json")
print(json.loads(p.read_text()).get("output_schema", ""))
PY
)"
    if [[ "$schema" == "nf_v3_beta" ]]; then
      echo "[SKIP] weight=${wid} mode=${mode} (v3 summary exists)"
      DONE_JOBS=$((DONE_JOBS + 1))
      write_progress "$DONE_JOBS" "$TOTAL_JOBS" "${wid}/${mode}" "skipped"
      return 0
    fi
  fi
  write_progress "$DONE_JOBS" "$TOTAL_JOBS" "${wid}/${mode}" "running"
  echo "============================================================"
  echo "[RUN $((DONE_JOBS + 1))/${TOTAL_JOBS}] weight=${wid} mode=${mode}"
  echo "============================================================"
  extra_args=()
  if [[ -n "${WEIGHT_REST_Z:-}" ]]; then
    extra_args+=(--weight-rest-z "$WEIGHT_REST_Z")
  fi
  if [[ -n "${WEIGHT_DROP_OFFSET:-}" ]]; then
    extra_args+=(--weight-drop-offset "$WEIGHT_DROP_OFFSET")
  fi
  if [[ -n "${WEIGHT_SPAWN_Z:-}" ]]; then
    extra_args+=(--weight-spawn-z "$WEIGHT_SPAWN_Z")
  fi
  if [[ -n "${FORCE_RENDER_K_REF:-}" ]]; then
    extra_args+=(--force-render-k-ref "$FORCE_RENDER_K_REF")
  fi
  if [[ -n "${FINGER_ROOT_Z:-}" ]]; then
    extra_args+=(--finger-root-z "$FINGER_ROOT_Z")
  fi
  PYTHONUNBUFFERED=1 "$ISAACLAB_SH" -p scripts/demo/demo_vitacsim_normal_force_validation.py \
    --headless --enable_cameras --device "$DEVICE" \
    --weight-id "$wid" --sensor-mode "$mode" \
    --out-dir "$OUT_DIR" \
    "${extra_args[@]}"
  DONE_JOBS=$((DONE_JOBS + 1))
  write_progress "$DONE_JOBS" "$TOTAL_JOBS" "${wid}/${mode}" "done"
}

if [[ -n "$WEIGHT" ]]; then
  for mode in $MODES; do
    run_one "$WEIGHT" "$mode"
  done
else
  for wid in $WEIGHTS; do
    for mode in $MODES; do
      run_one "$wid" "$mode"
    done
  done
fi

python3 scripts/demo/summarize_vitacsim_normal_force_validation.py --root "$OUT_DIR"
python3 scripts/demo/plot_vitacsim_validation_beta.py --nf-root "$OUT_DIR" 2>/dev/null || true

write_progress "$TOTAL_JOBS" "$TOTAL_JOBS" "-" "finished"
echo "[DONE] outputs under $OUT_DIR"
echo "[PROGRESS] watch: python3 scripts/demo/watch_nf_sweep_progress.py --root $OUT_DIR"
