#!/usr/bin/env bash
# Shared by the ViTacBench demo launchers. No conda environment name is assumed.
# Override PYTHON_SIM=/path/to/python.sh or ISAACLAB_SH=/path/to/isaaclab.sh;
# PYTHON_OFFLINE=/path/to/python selects the offline NumPy/OpenCV interpreter.
# Each collection/export gets a fresh child directory under RECORD_DIR/VIDEO_DIR.
# For replay, set MODE=visualize RECORD_DIR=/path/to/the/actual/collection/run.

demo_setup_python() {
  if [[ -n "${PYTHON_SIM:-}" ]]; then
    PY_SIM=( "$PYTHON_SIM" )
  elif [[ -n "${ISAACLAB_SH:-}" ]]; then
    PY_SIM=( "$ISAACLAB_SH" -p )
  elif [[ -x "$ROOT_DIR/../IsaacLab/isaaclab.sh" ]]; then
    PY_SIM=( "$ROOT_DIR/../IsaacLab/isaaclab.sh" -p )
  elif [[ -x /workspace/isaaclab/isaaclab.sh ]]; then
    PY_SIM=( /workspace/isaaclab/isaaclab.sh -p )
  elif [[ -x /isaac-sim/python.sh ]]; then
    PY_SIM=( /isaac-sim/python.sh )
  elif command -v python >/dev/null 2>&1; then
    PY_SIM=( python )
  else
    PY_SIM=( python3 )
  fi
  PY_OFFLINE=( "${PYTHON_OFFLINE:-python3}" )
}

demo_fresh_directory() {
  local variable="$1" base="${!1}" result
  mkdir -p -- "$base"
  result="$(mktemp -d "$base/run-$(date +%Y%m%d_%H%M%S)-XXXXXX")"
  printf -v "$variable" '%s' "$result"
  echo "[OUTPUT] $variable=$result"
}

demo_prepare_reports() {
  if [[ "${DEMO_REPORTS_READY:-0}" == 1 ]]; then
    return 0
  fi
  local report_root
  report_root="$(dirname "$SUMMARY_FILE")"
  mkdir -p -- "$report_root"
  DEMO_REPORT_DIR="$(mktemp -d "$report_root/run-$(date +%Y%m%d_%H%M%S)-XXXXXX")"
  SUMMARY_FILE="$DEMO_REPORT_DIR/$(basename "$SUMMARY_FILE")"
  # Reports belong to this run, even when replaying an older collection.
  VALIDATION_JSON="$DEMO_REPORT_DIR/$(basename "${VALIDATION_JSON:-validation.json}")"
  DEMO_REPORTS_READY=1
  echo "[OUTPUT] reports=$DEMO_REPORT_DIR"
}

demo_prefix_videos() {
  local file base
  # Retain the episode identifier: stripping it overwrites earlier episodes.
  while IFS= read -r -d '' file; do
    base="${file##*/}"
    if [[ "$base" != "${TASK_SLUG}_"* ]]; then
      mv -n -- "$file" "$VIDEO_DIR/${TASK_SLUG}_${base}"
    fi
  done < <(find "$VIDEO_DIR" -maxdepth 1 -type f -name '*.mp4' -print0)
}
