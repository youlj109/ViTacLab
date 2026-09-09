#!/usr/bin/env bash
# Overlay the published runtime assets onto a ViTacLab Git checkout and verify them.
#
# Usage:
#   bash bash_command/download_hf_assets.sh <hf-user>/ViTacLab-assets

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_ID="${1:-${VITACLAB_HF_REPO:-}}"
if [[ -z "$REPO_ID" || "$REPO_ID" != */* ]]; then
  echo "Usage: $0 <hf-user-or-org>/ViTacLab-assets" >&2
  exit 2
fi

HF_CLI="${HF_CLI:-$(command -v hf || true)}"
if [[ -z "$HF_CLI" && -x "$ROOT_DIR/.venv_hf/bin/hf" ]]; then
  HF_CLI="$ROOT_DIR/.venv_hf/bin/hf"
fi
if [[ -z "$HF_CLI" ]]; then
  echo "Missing 'hf' CLI. Create .venv_hf and install huggingface_hub there." >&2
  exit 2
fi

echo "[DOWNLOAD] https://huggingface.co/datasets/$REPO_ID"
"$HF_CLI" download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$ROOT_DIR" \
  --include "source/ViTacLab/ViTacLab/assets/data/**" \
  --include "source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data/**" \
  --include "data/calibration/tactile/**"

MANIFEST_PATH="$("$HF_CLI" download "$REPO_ID" MANIFEST.json --repo-type dataset --quiet)"
python - "$MANIFEST_PATH" "$ROOT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors = []

for entry in manifest["files"]:
    path = root / entry["path"]
    if not path.is_file():
        errors.append(f"missing: {entry['path']}")
        continue
    if path.stat().st_size != entry["bytes"]:
        errors.append(f"size mismatch: {entry['path']}")
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        errors.append(f"sha256 mismatch: {entry['path']}")

if errors:
    print("\n".join(f"[ERROR] {message}" for message in errors), file=sys.stderr)
    raise SystemExit(f"Asset verification failed ({len(errors)} errors)")

print(
    f"[VERIFIED] {manifest['file_count']} files, "
    f"{manifest['total_bytes'] / (1024**3):.2f} GiB"
)
PY

echo "[DONE] ViTacLab runtime assets are ready."
