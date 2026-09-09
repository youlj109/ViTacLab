#!/usr/bin/env bash
# Dependency-free preflight for a ViTacLab checkout.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${VITACLAB_ENV:-$ROOT_DIR/config/local.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

VITACLAB_ROOT="${VITACLAB_ROOT:-$ROOT_DIR}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "$ROOT_DIR/.." 2>/dev/null && pwd)/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_SH:-$ISAACLAB_ROOT/isaaclab.sh}"
errors=0

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    echo "[OK] $label"
  else
    echo "[MISSING] $label: $path" >&2
    errors=$((errors + 1))
  fi
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "$path" ]]; then
    echo "[OK] $label"
  else
    echo "[MISSING] $label: $path" >&2
    errors=$((errors + 1))
  fi
}

echo "[ViTacLab] $VITACLAB_ROOT"
check_file "$ISAACLAB_SH" "Isaac Lab launcher"
check_file "$VITACLAB_ROOT/source/ViTacLab/pyproject.toml" "ViTacLab Python package"
check_dir "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/data/Objects" "task object assets"
check_dir "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/data/Robots" "robot assets"
check_dir "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/data/Sensors" "sensor assets"
check_file \
  "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_left_hand_glb_withtac_v2_no_gelsight_articulation.usd" \
  "single-arm UR10e + Shadow Hand USD"
check_file \
  "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_right_hand_glb_withtac.usd" \
  "dual-arm right UR10e + Shadow Hand USD"
check_file \
  "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/data/Sensors/Gelsight_finger/gelsight_r15_finger_short_v2.usd" \
  "GelSight finger validation USD"
check_file \
  "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data/polycalib.npz" \
  "Xense polycalib"
check_file \
  "$VITACLAB_ROOT/source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data/marker_rest.npy" \
  "Xense rest marker positions"
check_file "$VITACLAB_ROOT/data/calibration/tactile/fitted_params.json" "fitted tactile parameters"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "[OK] NVIDIA driver is visible"
else
  echo "[MISSING] NVIDIA driver is not visible to nvidia-smi" >&2
  errors=$((errors + 1))
fi

if python3 "$VITACLAB_ROOT/scripts/audit_project.py" \
  --root "$VITACLAB_ROOT" --repro-only --verbose; then
  echo "[OK] static repository audit"
else
  echo "[FAILED] static repository audit" >&2
  errors=$((errors + 1))
fi

if ((errors)); then
  echo "[FAILED] reproduction preflight found $errors problem(s)." >&2
  exit 1
fi

echo "[PASS] Run the per-environment smoke tests in docs/REPRODUCIBILITY.md."
