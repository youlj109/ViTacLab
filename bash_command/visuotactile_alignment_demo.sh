#!/usr/bin/env bash
# Compose a presentation-style side-by-side video from loose/strict runs.
#
# Usage:
#   bash bash_command/visuotactile_alignment_demo.sh
#
# Optional env vars:
#   LOOSE_VIDEO=logs/alignment_visuotactile_v2/interference_visual_loose_fix3.mp4
#   STRICT_VIDEO=logs/alignment_visuotactile_v2/interference_visual_strict_fix3.mp4
#   OUT_VIDEO=logs/alignment_visuotactile_v2/interference_visual_demo_present.mp4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOOSE_VIDEO="${LOOSE_VIDEO:-logs/alignment_visuotactile_v2/interference_visual_loose_fix3.mp4}"
STRICT_VIDEO="${STRICT_VIDEO:-logs/alignment_visuotactile_v2/interference_visual_strict_fix3.mp4}"
OUT_VIDEO="${OUT_VIDEO:-logs/alignment_visuotactile_v2/interference_visual_demo_present.mp4}"

if [[ ! -f "$LOOSE_VIDEO" ]]; then
  echo "[ERROR] Loose video not found: $LOOSE_VIDEO"
  exit 2
fi
if [[ ! -f "$STRICT_VIDEO" ]]; then
  echo "[ERROR] Strict video not found: $STRICT_VIDEO"
  exit 2
fi

python scripts/demo/make_visuotactile_alignment_demo_video.py \
  --loose_video "$LOOSE_VIDEO" \
  --strict_video "$STRICT_VIDEO" \
  --output_video "$OUT_VIDEO"

echo "[DONE] presentation video: $OUT_VIDEO"
