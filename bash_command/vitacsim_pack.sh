#!/usr/bin/env bash
# Build a portable ViTacSim code+asset+docs archive for deployment on a new machine.
#
# Usage (from ViTacLab repo root):
#   bash bash_command/vitacsim_pack.sh
#
# Optional env vars:
#   OUT_DIR=dist/vitacsim_pack          # staging directory (default)
#   ARCHIVE=dist/vitacsim_pack.tar.gz   # final tarball path (default)
#   SKIP_TAR=0                          # 1: only stage, do not compress
#   INCLUDE_FULL_SOURCE=0               # 1: copy entire source/ViTacLab (minus Scene/Objects)
#
# Output layout mirrors repo paths under OUT_DIR/, plus:
#   README_PACK.md          quick start on new machine
#   MANIFEST.json           file count + sha256 per top-level bundle

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${OUT_DIR:-dist/vitacsim_pack_${STAMP}}"
ARCHIVE="${ARCHIVE:-dist/vitacsim_pack_${STAMP}.tar.gz}"
SKIP_TAR="${SKIP_TAR:-0}"
INCLUDE_FULL_SOURCE="${INCLUDE_FULL_SOURCE:-0}"
FAST_MANIFEST="${FAST_MANIFEST:-1}"
PATHS_FILE="${PATHS_FILE:-docs/vitacsim_pack/PATHS.include}"

if [[ "$OUT_DIR" == /tmp/* ]]; then
  echo "[WARN] OUT_DIR under /tmp can hang in some IDE sandboxes; prefer dist/vitacsim_pack_*"
fi

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR"/*
mkdir -p "$(dirname "$ARCHIVE")"

copy_path() {
  local rel="$1"
  if [[ ! -e "$rel" ]]; then
    echo "[WARN] missing (skipped): $rel"
    return 0
  fi
  local dest="$OUT_DIR/$rel"
  mkdir -p "$(dirname "$dest")"
  if [[ -d "$rel" ]]; then
    # Use cp for reliability (rsync to /tmp has hung on some sandbox setups).
    rm -rf "$dest"
    cp -a "$rel/." "$dest/"
  else
    cp -a "$rel" "$dest"
  fi
  echo "[COPY] $rel"
}

echo "============================================================"
echo "[ViTacSim Pack] staging -> $OUT_DIR"
echo "[ROOT] $ROOT_DIR"
echo "============================================================"

if [[ "$INCLUDE_FULL_SOURCE" == "1" ]]; then
  echo "[MODE] full source/ViTacLab (exclude Scene + Objects)"
  rsync -a \
    --exclude 'assets/data/Scene/' \
    --exclude 'assets/data/Objects/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "source/ViTacLab/" "$OUT_DIR/source/ViTacLab/"
else
  if [[ ! -f "$PATHS_FILE" ]]; then
    echo "[ERROR] PATHS file not found: $PATHS_FILE"
    exit 2
  fi
  echo "[MODE] manifest paths from $PATHS_FILE"
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs || true)"
    [[ -z "$line" ]] && continue
    copy_path "$line"
  done < "$PATHS_FILE"
fi

# Always ship pack docs at bundle root.
for doc in docs/VITACSIM_PRINCIPLES.md docs/VITACSIM_PHYSX_VALIDATION.md README_send.md; do
  if [[ -f "$doc" ]]; then
    mkdir -p "$OUT_DIR/docs"
    rsync -a "$doc" "$OUT_DIR/$doc"
  fi
done
rsync -a docs/vitacsim_pack/ "$OUT_DIR/docs/vitacsim_pack/" 2>/dev/null || true

# README for recipients.
cat > "$OUT_DIR/README_PACK.md" <<'EOF'
# ViTacSim Portable Pack

This archive contains **ViTacSim (VisuoTactileSensor V1/V2)** source code, robot/sensor USD assets, demo scripts, and documentation — enough to reproduce tactile demos and PhysX validation on a **new machine** that already has Isaac Sim + Isaac Lab installed.

## 1. Prerequisites (not included in this pack)

| Component | Notes |
|-----------|--------|
| **Isaac Sim 5.1.x** | Same major version as development machine |
| **Isaac Lab** | Sibling checkout, e.g. `../IsaacLab/` |
| **Conda env** | Python 3.10+, PyTorch matching Isaac Sim |
| **Isaac Lab Nucleus TacSL data** | GelSight render calibs at `{ISAACLAB_NUCLEUS_DIR}/TacSL/` (downloaded on first run) |
| **Factory task USD** | Forge peg/gear/nut assets from `isaaclab_tasks` Factory extension (Isaac Lab install) |

## 2. Install on new machine

```bash
# 1) Unpack
tar -xzf vitacsim_pack_YYYYMMDD_HHMMSS.tar.gz
cd vitacsim_pack_YYYYMMDD_HHMMSS

# 2) Merge into ViTacLab checkout (or use as standalone tree)
# Option A — overlay into existing ViTacLab repo:
VITACLAB=/path/to/ViTacLab
rsync -a ./ "$VITACLAB/"
cd "$VITACLAB"

# Option B — use staged tree directly as ViTacLab root:
cd /path/to/vitacsim_pack_YYYYMMDD_HHMMSS

# 3) Install extension
cd source/ViTacLab
python -m pip install -e .

# 4) Register with Isaac Lab (if not already)
# From IsaacLab root:
#   ./isaaclab.sh -p -m pip install -e /path/to/ViTacLab/source/ViTacLab
```

Set `PYTHONPATH` when running scripts without install:

```bash
export PYTHONPATH="/path/to/ViTacLab/source/ViTacLab${PYTHONPATH:+:$PYTHONPATH}"
```

Launch Python via Isaac Lab launcher:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/demo/eval_visuotactile_physx_alignment.py --headless --enable_cameras
```

## 3. Read documentation first

| Document | Content |
|----------|---------|
| `docs/VITACSIM_PRINCIPLES.md` | **原理说明**：V1/V2 架构、相机渲染、力场重建、任务集成 |
| `docs/VITACSIM_PHYSX_VALIDATION.md` | PhysX 对齐验证、无 silent fallback 说明 |
| `README_send.md` | 模块清单与快速 demo |

## 4. Smoke tests (recommended order)

```bash
# Quantitative PhysX alignment gate (exit 0 = pass)
bash bash_command/visuotactile_physx_alignment_check.sh

# Forge tactile-feedback insertion demo (video + NPZ)
bash bash_command/forge_tactile_feedback_insert_demo.sh

# Optional: mentor report pack (task success replay video)
bash bash_command/vitacsim_report_pack.sh
```

## 5. What is intentionally excluded

- `assets/data/Scene/` — living-room/kitchen scenes (~4 GB), not needed for ViTacSim core
- `assets/data/Objects/` — general object library (~1.7 GB); Forge uses Isaac Lab Factory assets
- Training logs, checkpoints, recorded NPZ/videos
- Full ViTacBench task suite beyond Forge + pretraining glue

See `docs/vitacsim_pack/PATHS.include` for the exact manifest.

## 6. Support

Pack built by `bash_command/vitacsim_pack.sh`. Regenerate after sensor/task changes.
EOF

# Manifest (FAST_MANIFEST=1: file list + sizes only; 0: full sha256).
python - "$OUT_DIR" "$FAST_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fast = str(sys.argv[2]).strip() not in ("0", "false", "False")
files = sorted(p for p in root.rglob("*") if p.is_file())
manifest = {
    "root": str(root.resolve()),
    "file_count": len(files),
    "total_bytes": sum(p.stat().st_size for p in files),
    "fast_manifest": fast,
    "files": [],
}
for p in files:
    rel = p.relative_to(root).as_posix()
    entry = {"path": rel, "bytes": p.stat().st_size}
    if not fast:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        entry["sha256"] = h.hexdigest()
    manifest["files"].append(entry)
out = root / "MANIFEST.json"
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
mode = "fast" if fast else "sha256"
print(f"[MANIFEST/{mode}] {out} ({manifest['file_count']} files, {manifest['total_bytes']} bytes)")
PY

if [[ "$SKIP_TAR" == "1" ]]; then
  echo "[DONE] staged only (SKIP_TAR=1): $OUT_DIR"
  exit 0
fi

echo "[TAR] $ARCHIVE"
tar -czf "$ARCHIVE" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
ls -lh "$ARCHIVE"
echo "[DONE] ViTacSim pack ready:"
echo "  dir:     $OUT_DIR"
echo "  archive: $ARCHIVE"
