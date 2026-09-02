#!/usr/bin/env bash
# Generate visual-effect demo video for ViTacSim attribution behavior.
#
# Runs interference-only scenario and exports a panel video.
#
# Usage:
#   bash bash_command/visuotactile_alignment_visual.sh
#
# Optional env vars:
#   STEPS=260
#   SEED=42
#   STRICT=1          # 1=strict attribution on, 0=off
#   OUT_VIDEO=logs/alignment_visuotactile_v2/interference_visual.mp4
#   FORCE_EXIT=1      # 1=force os._exit after writer release (avoid Isaac close hang)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"

STEPS="${STEPS:-260}"
SEED="${SEED:-42}"
STRICT="${STRICT:-1}"
OUT_VIDEO="${OUT_VIDEO:-logs/alignment_visuotactile_v2/interference_visual.mp4}"
FORCE_EXIT="${FORCE_EXIT:-1}"

if [[ "$STRICT" == "1" ]]; then
  strict_flag=(--strict_target_attribution)
else
  strict_flag=()
fi

if [[ "$FORCE_EXIT" == "1" ]]; then
  force_exit_flag=(--force_exit)
else
  force_exit_flag=()
fi

set +e
python - <<'PY'
from PIL import Image  # noqa: F401
print("[CHECK] PIL import OK")
PY
pil_check_exit=$?
set -e
if [[ $pil_check_exit -ne 0 ]]; then
  echo "[ERROR] Python Pillow(PIL) is not importable in current environment."
  exit 2
fi

echo "============================================================"
echo "[RUN] ViTacSim visual demo"
echo "[OUT] $OUT_VIDEO"
echo "============================================================"

PYTHONUNBUFFERED=1 python -u scripts/demo/demo_visuotactile_alignment_visual.py \
  --headless \
  --enable_cameras \
  --steps "$STEPS" \
  --seed "$SEED" \
  "${strict_flag[@]}" \
  "${force_exit_flag[@]}" \
  --output_video "$OUT_VIDEO"

echo "[DONE] Visual demo finished."
