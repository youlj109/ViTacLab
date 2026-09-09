#!/usr/bin/env bash
# Create and upload the curated ViTacLab runtime assets to a public HF dataset.
#
# Usage:
#   bash bash_command/upload_hf_assets.sh <hf-user>/ViTacLab-assets
#
# Authentication is intentionally not accepted as a command-line argument.
# Run `hf auth login` first so the token does not enter shell history.

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
HF_PYTHON="${HF_PYTHON:-$(dirname "$HF_CLI")/python}"
if [[ ! -x "$HF_PYTHON" ]]; then
  echo "Missing Python interpreter next to HF CLI: $HF_PYTHON" >&2
  exit 2
fi

"$HF_CLI" auth whoami >/dev/null

"$HF_PYTHON" - "$REPO_ID" <<'PY'
import sys

from huggingface_hub import HfApi

repo_id = sys.argv[1]
url = HfApi().create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
print(f"[HF] dataset repository: {url}")
PY

CARD="docs/huggingface/ASSET_REPOSITORY_README.md"
"$HF_CLI" upload "$REPO_ID" "$CARD" README.md \
  --repo-type dataset \
  --commit-message "Add ViTacLab asset repository card"

# Preserve repository-relative paths so a download can overlay a Git checkout.
"$HF_CLI" upload "$REPO_ID" \
  source/ViTacLab/ViTacLab/assets/data \
  source/ViTacLab/ViTacLab/assets/data \
  --repo-type dataset \
  --exclude "Scene/**" \
  --exclude "Robots/Franka/Franka_R15/franka_mimic_edit.usd" \
  --exclude "**/*.zip" \
  --exclude "**/.thumbs/**" \
  --exclude "**/*（复件）*" \
  --commit-message "Upload curated simulation assets"

# The current working asset has a known self-sublayer cycle. Publish the
# validated pre-regression copy under the canonical destination instead.
GOOD_FRANKA_USD="vendor/ViTac_0.1/ViTacLab/source/ViTacLab/ViTacLab/assets/data/Robots/Franka/Franka_R15/franka_mimic_edit.usd"
FRANKA_USD_DEST="source/ViTacLab/ViTacLab/assets/data/Robots/Franka/Franka_R15/franka_mimic_edit.usd"
if [[ ! -f "$GOOD_FRANKA_USD" ]]; then
  echo "Missing corrected Franka USD: $GOOD_FRANKA_USD" >&2
  exit 2
fi
"$HF_CLI" upload "$REPO_ID" "$GOOD_FRANKA_USD" "$FRANKA_USD_DEST" \
  --repo-type dataset \
  --commit-message "Upload corrected Franka R15 USD"

"$HF_CLI" upload "$REPO_ID" \
  source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data \
  source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data \
  --repo-type dataset \
  --exclude "*.bak" \
  --commit-message "Upload Xense render calibration"

"$HF_CLI" upload "$REPO_ID" \
  data/calibration/tactile \
  data/calibration/tactile \
  --repo-type dataset \
  --exclude "**/*.bak" \
  --commit-message "Upload tactile calibration data"

MANIFEST="$(mktemp --suffix=.json)"
trap 'rm -f "$MANIFEST"' EXIT
python - "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

roots = (
    Path("source/ViTacLab/ViTacLab/assets/data/Objects"),
    Path("source/ViTacLab/ViTacLab/assets/data/Robots"),
    Path("source/ViTacLab/ViTacLab/assets/data/Sensors"),
    Path("source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data"),
    Path("data/calibration/tactile"),
)
franka_target = (
    "source/ViTacLab/ViTacLab/assets/data/Robots/Franka/"
    "Franka_R15/franka_mimic_edit.usd"
)
franka_source = Path(
    "vendor/ViTac_0.1/ViTacLab/source/ViTacLab/ViTacLab/assets/data/"
    "Robots/Franka/Franka_R15/franka_mimic_edit.usd"
)

entries = []
for root in roots:
    if not root.is_dir():
        raise SystemExit(f"Required asset directory is missing: {root}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.as_posix()
        if (
            path.suffix.lower() == ".zip"
            or ".thumbs/" in relative
            or relative.endswith(".bak")
            or "（复件）" in relative
        ):
            continue
        published_path = relative
        if published_path == franka_target:
            path = franka_source
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        entries.append(
            {
                "path": published_path,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )

payload = {
    "schema_version": 1,
    "file_count": len(entries),
    "total_bytes": sum(item["bytes"] for item in entries),
    "files": entries,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[MANIFEST] files={payload['file_count']} bytes={payload['total_bytes']}")
PY

"$HF_CLI" upload "$REPO_ID" "$MANIFEST" MANIFEST.json \
  --repo-type dataset \
  --commit-message "Add SHA256 asset manifest"

echo "[DONE] https://huggingface.co/datasets/$REPO_ID"
