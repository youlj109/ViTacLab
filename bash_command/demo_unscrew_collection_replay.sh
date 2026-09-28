#!/usr/bin/env bash
# Deprecated: use demo_bottle_collection_replay.sh (outputs under logs/demo_bottle/).
echo "[WARN] demo_unscrew_collection_replay.sh is deprecated; forwarding to demo_bottle_collection_replay.sh" >&2
exec bash "$(dirname "${BASH_SOURCE[0]}")/demo_bottle_collection_replay.sh" "$@"
