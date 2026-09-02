#!/usr/bin/env bash
# PhysX-vs-ViTacSim alignment evaluation.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

NUM_ENVS="${NUM_ENVS:-1}"
STEPS_PER_CASE="${STEPS_PER_CASE:-220}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-logs/alignment_visuotactile_v2}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-3600}"

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/alignment.log"
SUMMARY_JSON="$OUT_DIR/alignment_summary.json"
rm -f "$LOG_FILE" "$SUMMARY_JSON"

set +e
python - <<'PY'
from PIL import Image  # noqa: F401
print("[CHECK] PIL import OK")
PY
pil_check_exit=$?
set -e
if [[ $pil_check_exit -ne 0 ]]; then
  echo "[ERROR] Python Pillow(PIL) is not importable in current environment."
  echo "[HINT] Run:"
  echo "  python -m pip uninstall -y Pillow pillow"
  echo "  python -m pip install --no-cache-dir --force-reinstall Pillow==11.3.0"
  exit 2
fi

echo "============================================================"
echo "[RUN] PhysX alignment evaluation"
echo "[LOG] $LOG_FILE"
echo "[SUMMARY] $SUMMARY_JSON"
echo "============================================================"

PYTHONUNBUFFERED=1 python -u scripts/demo/eval_visuotactile_physx_alignment.py \
  --headless \
  --enable_cameras \
  --num_envs "$NUM_ENVS" \
  --steps_per_case "$STEPS_PER_CASE" \
  --seed "$SEED" \
  --physx_contact_force_threshold 0.001 \
  --max_corr_lag 5 \
  --corr_smooth_window 5 \
  --summary_json "$SUMMARY_JSON" \
  --force_exit >"$LOG_FILE" 2>&1 &
run_pid=$!

start_ts="$(date +%s)"
last_ping_ts="$start_ts"
while kill -0 "$run_pid" 2>/dev/null; do
  if [[ -f "$SUMMARY_JSON" ]]; then
    kill -TERM "$run_pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$run_pid" 2>/dev/null || true
    break
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts > MAX_WAIT_SEC )); then
    echo "[ERROR] Timeout (${MAX_WAIT_SEC}s). Terminating alignment process..."
    kill -TERM "$run_pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$run_pid" 2>/dev/null || true
    break
  fi
  if (( now_ts - last_ping_ts >= 20 )); then
    echo "[WAIT] alignment running... $(( now_ts - start_ts ))s elapsed"
    last_ping_ts="$now_ts"
  fi
  sleep 2
done
wait "$run_pid" 2>/dev/null || true

if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "[ERROR] Missing summary JSON: $SUMMARY_JSON"
  echo "[ERROR] Last 80 lines from $LOG_FILE:"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

python - "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
doc = json.loads(path.read_text(encoding="utf-8"))
overall = doc.get("overall", {})
print(
    f"[RESULT] overall_pass={bool(overall.get('overall_pass', False))} "
    f"pass_case_count={int(overall.get('pass_case_count', 0))}/{int(overall.get('case_count', 0))}"
)
cases = doc.get("cases", {})
for name in sorted(cases.keys()):
    c = cases[name]
    print(
        f"  - {name}: "
        f"pass={bool(int(c.get('pass', 0)))} "
        f"contact_f1={float(c.get('contact_f1', 0.0)):.3f} "
        f"fn_corr={float(c.get('fn_corr', 0.0)):.3f} "
        f"slip_f1={float(c.get('slip_f1', 0.0)):.3f} "
        f"ft_corr={float(c.get('ft_corr', 0.0)):.3f} "
        f"centroid_error_mean={float(c.get('centroid_error_mean', 0.0)):.4f} "
        f"reasons={';'.join(c.get('fail_reasons', [])) if isinstance(c.get('fail_reasons', []), list) else ''}"
    )
if not bool(overall.get("overall_pass", False)):
    raise SystemExit(2)
PY

echo
echo "[DONE] Alignment evaluation finished."
