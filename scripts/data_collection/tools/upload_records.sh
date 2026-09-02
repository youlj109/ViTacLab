#!/usr/bin/env bash
# Upload selected *_success.npz files with retry/resume-friendly numbering.
# Required env vars: UPLOAD_SSH_HOST, UPLOAD_SSH_PORT, REMOTE_DIR, LOCAL_DIR1.
# Optional env vars: LOCAL_DIR2, MAX_FILES_PER_DIR.
set -euo pipefail

: "${UPLOAD_SSH_HOST:?Set UPLOAD_SSH_HOST, for example user@host}"
: "${UPLOAD_SSH_PORT:?Set UPLOAD_SSH_PORT, for example 22}"
: "${REMOTE_DIR:?Set REMOTE_DIR, for example /data/ViTacLab/Dual_Blind}"
: "${LOCAL_DIR1:?Set LOCAL_DIR1 to a local play_records folder}"
LOCAL_DIR2="${LOCAL_DIR2:-}"
MAX_FILES_PER_DIR="${MAX_FILES_PER_DIR:-100}"

TMP_ALL="$(mktemp)"
trap 'rm -f "$TMP_ALL"' EXIT

retry_scp() {
  local src="$1"
  local dst="$2"
  local tries=0
  local max_tries=20
  while true; do
    if scp -O -P "${UPLOAD_SSH_PORT}" "$src" "${UPLOAD_SSH_HOST}:$dst"; then
      return 0
    fi
    tries=$((tries + 1))
    if [[ "$tries" -ge "$max_tries" ]]; then
      echo "upload failed after ${max_tries} tries: $src -> $dst" >&2
      return 1
    fi
    echo "upload failed; retrying (${tries}/${max_tries})..." >&2
    sleep 3
  done
}

collect_success_npz() {
  local base="$1"
  local limit="$2"
  python3 - "$base" "$limit" <<'PY'
import os
import sys
base, limit_s = sys.argv[1], sys.argv[2]
limit = int(limit_s)
if not os.path.isdir(base):
    raise SystemExit(f"directory not found: {base}")
files = []
for name in sorted(os.listdir(base)):
    path = os.path.join(base, name)
    if os.path.isfile(path) and name.endswith("_success.npz"):
        files.append(path)
for path in files[:limit]:
    print(path)
PY
}

ssh -p "${UPLOAD_SSH_PORT}" "${UPLOAD_SSH_HOST}" "mkdir -p '${REMOTE_DIR}'"
REMOTE_MAX=$(ssh -p "${UPLOAD_SSH_PORT}" "${UPLOAD_SSH_HOST}" "python3 - <<'PY'
import os, re
remote_dir = '${REMOTE_DIR}'
max_id = 0
if os.path.isdir(remote_dir):
    for name in os.listdir(remote_dir):
        m = re.match(r'^episode_(\d+)_success\.npz$', name)
        if m:
            max_id = max(max_id, int(m.group(1)))
print(max_id)
PY")

collect_success_npz "$LOCAL_DIR1" "$MAX_FILES_PER_DIR" >> "$TMP_ALL"
if [[ -n "$LOCAL_DIR2" ]]; then
  collect_success_npz "$LOCAL_DIR2" "$MAX_FILES_PER_DIR" >> "$TMP_ALL"
fi

idx=$((REMOTE_MAX + 1))
while IFS= read -r src; do
  [[ -z "$src" ]] && continue
  remote_name=$(printf "episode_%04d_success.npz" "$idx")
  retry_scp "$src" "${REMOTE_DIR}/${remote_name}"
  idx=$((idx + 1))
done < "$TMP_ALL"

echo "uploaded through index $((idx - 1))"
