#!/usr/bin/env bash
# Build a mentor-facing ViTacSim demo package in one command.
#
# Outputs:
#   1) task success demo video (expert replay, third-person robot behavior)
#   2) attribution comparison demo video (STRICT=0 vs STRICT=1)
#   3) concise markdown notes for presentation
#
# Usage:
#   bash bash_command/vitacsim_report_pack.sh
#
# Optional env vars:
#   TASK=forge_insert               # forge_insert | forge_gear | forge_nut
#   ALIGN_STEPS=140
#   SEED=42
#   OUT_DIR=logs/report_pack_vitacsim
#   SKIP_EXISTING=1                 # 1: reuse existing outputs
#   INCLUDE_ATTRIBUTION=0           # 0: skip strict/loose attribution videos

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

TASK="${TASK:-forge_insert}"
ALIGN_STEPS="${ALIGN_STEPS:-140}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-logs/report_pack_vitacsim}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
INCLUDE_ATTRIBUTION="${INCLUDE_ATTRIBUTION:-0}"

mkdir -p "$OUT_DIR"

TASK_VIDEO="$OUT_DIR/task_success_${TASK}.mp4"
TASK_RECORD_DIR="$OUT_DIR/task_replay_records_${TASK}"
ALIGN_LOOSE_RAW="$OUT_DIR/interference_loose_raw.mp4"
ALIGN_STRICT_RAW="$OUT_DIR/interference_strict_raw.mp4"
ATTR_DEMO_VIDEO="$OUT_DIR/attribution_compare_present.mp4"
README_MD="$OUT_DIR/README_report.md"

run_if_needed() {
  local target_file="$1"
  shift
  local need_run=1
  if [[ "$SKIP_EXISTING" == "1" && -f "$target_file" ]]; then
    if [[ "$target_file" == *.mp4 ]]; then
      if python - "$target_file" <<'PY'
import sys
from pathlib import Path
import cv2

p = Path(sys.argv[1])
if (not p.exists()) or p.stat().st_size <= 0:
    raise SystemExit(1)
cap = cv2.VideoCapture(str(p))
ok_open = cap.isOpened()
ok_read, frame = cap.read() if ok_open else (False, None)
cap.release()
raise SystemExit(0 if (ok_open and ok_read and frame is not None) else 1)
PY
      then
        echo "[SKIP] valid existing video: $target_file"
        need_run=0
      else
        echo "[REBUILD] invalid/corrupted video detected: $target_file"
        rm -f "$target_file"
      fi
    else
      echo "[SKIP] exists: $target_file"
      need_run=0
    fi
  fi
  if [[ "$need_run" == "1" ]]; then
    "$@"
  fi
}

echo "============================================================"
echo "[1/4] Task success demo (expert replay)"
echo "============================================================"
run_if_needed "$TASK_VIDEO" env \
  TASK="$TASK" SEED="$SEED" RECORD_DIR="$TASK_RECORD_DIR" OUT_VIDEO="$TASK_VIDEO" \
  bash bash_command/forge_task_success_demo.sh

echo "============================================================"
echo "[2/4] Attribution raw videos (STRICT=0 / STRICT=1)"
echo "============================================================"
if [[ "$INCLUDE_ATTRIBUTION" == "1" ]]; then
  run_if_needed "$ALIGN_LOOSE_RAW" env \
    STEPS="$ALIGN_STEPS" SEED="$SEED" STRICT=0 FORCE_EXIT=1 OUT_VIDEO="$ALIGN_LOOSE_RAW" \
    bash bash_command/visuotactile_alignment_visual.sh

  run_if_needed "$ALIGN_STRICT_RAW" env \
    STEPS="$ALIGN_STEPS" SEED="$SEED" STRICT=1 FORCE_EXIT=1 OUT_VIDEO="$ALIGN_STRICT_RAW" \
    bash bash_command/visuotactile_alignment_visual.sh
else
  echo "[SKIP] attribution generation disabled (INCLUDE_ATTRIBUTION=0)"
fi

echo "============================================================"
echo "[3/4] Attribution comparison composition"
echo "============================================================"
if [[ "$INCLUDE_ATTRIBUTION" == "1" ]]; then
  run_if_needed "$ATTR_DEMO_VIDEO" env \
    LOOSE_VIDEO="$ALIGN_LOOSE_RAW" STRICT_VIDEO="$ALIGN_STRICT_RAW" OUT_VIDEO="$ATTR_DEMO_VIDEO" \
    bash bash_command/visuotactile_alignment_demo.sh
else
  echo "[SKIP] attribution composition disabled (INCLUDE_ATTRIBUTION=0)"
fi

echo "============================================================"
echo "[4/4] Write report notes"
echo "============================================================"
date_str="$(date '+%Y-%m-%d %H:%M:%S %Z')"
cat > "$README_MD" <<EOF
# ViTacSim Demo Pack

Generated: $date_str

## Files
- Task success demo (expert replay): \`$TASK_VIDEO\`
- Attribution comparison demo: \`$ATTR_DEMO_VIDEO\` (generated only when INCLUDE_ATTRIBUTION=1)
- Raw attribution (STRICT=0): \`$ALIGN_LOOSE_RAW\` (generated only when INCLUDE_ATTRIBUTION=1)
- Raw attribution (STRICT=1): \`$ALIGN_STRICT_RAW\` (generated only when INCLUDE_ATTRIBUTION=1)

## How To Explain
1. **Task success demo** shows replayed expert trajectory in Forge \`$TASK\`, from third-person camera.
2. **Attribution comparison demo** uses interference-only scenario (when enabled):
   - \`STRICT=0\`: non-target contact can produce false tactile activation.
   - \`STRICT=1\`: false activation is suppressed while raw contact still exists.
3. Together, they answer two questions:
   - "Can the system run task-level manipulation?"
   - "Is the attribution bug fixed under interference contact?"

## Run Config
- TASK=$TASK
- ALIGN_STEPS=$ALIGN_STEPS
- SEED=$SEED
- INCLUDE_ATTRIBUTION=$INCLUDE_ATTRIBUTION
EOF

echo "[DONE] report package ready:"
echo "  - $TASK_VIDEO"
echo "  - $ATTR_DEMO_VIDEO"
echo "  - $README_MD"
