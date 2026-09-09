#!/usr/bin/env bash
set -euo pipefail

# Batch validation wrapper for the canonical policy runner.
#
# Usage:
#   bash scripts/policy/batch_validate_policy.sh
#
# Common overrides:
#   TASK=Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 \
#   CHECKPOINT_TASK_NAME=BlindSingleTarget \
#   DATA_NUM=100 CKPTS="1000 1500 2000" \
#   SEED_START=42 SEED_END=51 \
#   VITACDP_PROFILES="rgb force" \
#   POLICY_OUTPUT=action \
#   bash scripts/policy/batch_validate_policy.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-20}"
MAX_STEPS="${MAX_STEPS:-500}"
DATA_NUM="${DATA_NUM:-100}"
CHECKPOINT_TASK_NAME="${CHECKPOINT_TASK_NAME:-BottleCup}"
POLICY_OUTPUT="${POLICY_OUTPUT:-action}"
RUN_DIFFUSION="${RUN_DIFFUSION:-1}"
SEED_START="${SEED_START:-42}"
SEED_END="${SEED_END:-51}"
CKPTS="${CKPTS:-2000}"
VITACDP_PROFILES="${VITACDP_PROFILES:-}"
LOG_DIR="${LOG_DIR:-logs/batch_validate_policy}"
TEE_LOG="${TEE_LOG:-1}"

mkdir -p "${LOG_DIR}"

short_task="${TASK##*:}"
ckpt_prefix="${CHECKPOINT_TASK_NAME:-${short_task}}"

echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] task=${TASK}"
echo "[INFO] checkpoint_task_name=${ckpt_prefix}"
echo "[INFO] num_envs=${NUM_ENVS} max_steps=${MAX_STEPS} data_num=${DATA_NUM}"
echo "[INFO] policy_output=${POLICY_OUTPUT} seeds=${SEED_START}..${SEED_END} ckpts=${CKPTS}"
echo "[INFO] logs=${LOG_DIR} tee=${TEE_LOG}"

run_logged() {
  local log="$1"
  shift
  if [[ "${TEE_LOG}" == "1" ]]; then
    "$@" 2>&1 | tee "${log}"
    local status="${PIPESTATUS[0]}"
    [[ "${status}" -eq 0 ]] || exit "${status}"
  else
    "$@" >"${log}" 2>&1
  fi
}

run_vitacdp() {
  local seed="$1"
  local ckpt="$2"
  local profile="$3"
  local profile_args=()
  local checkpoint_args=()
  local profile_label="default"
  if [[ -n "${profile}" ]]; then
    profile_args=(--observation-profile="${profile}")
    profile_label="${profile}"
  fi
  if [[ -n "${CHECKPOINT_TASK_NAME}" ]]; then
    checkpoint_args=(--checkpoint_task_name="${CHECKPOINT_TASK_NAME}")
  fi
  local log="${LOG_DIR}/${ckpt_prefix}_ViTacDP_${profile_label}_ckpt${ckpt}_seed${seed}.log"
  echo "[RUN] ViTacDP profile=${profile_label} seed=${seed} ckpt=${ckpt}"
  run_logged "${log}" python scripts/policy/play_policy.py \
    --task="${TASK}" \
    --num_envs="${NUM_ENVS}" \
    --max_steps="${MAX_STEPS}" \
    --data_num="${DATA_NUM}" \
    --checkpoint_num="${ckpt}" \
    --policy_name=ViTacDP \
    --policy-output="${POLICY_OUTPUT}" \
    "${profile_args[@]}" \
    "${checkpoint_args[@]}" \
    --headless \
    --seed="${seed}"
}

run_diffusion() {
  local seed="$1"
  local ckpt="$2"
  local checkpoint_args=()
  if [[ -n "${CHECKPOINT_TASK_NAME}" ]]; then
    checkpoint_args=(--checkpoint_task_name="${CHECKPOINT_TASK_NAME}")
  fi
  local log="${LOG_DIR}/${ckpt_prefix}_Diffusion_Policy_ckpt${ckpt}_seed${seed}.log"
  echo "[RUN] Diffusion_Policy seed=${seed} ckpt=${ckpt}"
  run_logged "${log}" python scripts/policy/play_policy.py \
    --task="${TASK}" \
    --num_envs="${NUM_ENVS}" \
    --max_steps="${MAX_STEPS}" \
    --data_num="${DATA_NUM}" \
    --checkpoint_num="${ckpt}" \
    --policy_name=Diffusion_Policy \
    --policy-output="${POLICY_OUTPUT}" \
    "${checkpoint_args[@]}" \
    --headless \
    --seed="${seed}"
}

for ((seed = SEED_START; seed <= SEED_END; seed++)); do
  for ckpt in ${CKPTS}; do
    if [[ -n "${VITACDP_PROFILES// }" ]]; then
      for profile in ${VITACDP_PROFILES}; do
        run_vitacdp "${seed}" "${ckpt}" "${profile}"
      done
    else
      run_vitacdp "${seed}" "${ckpt}" ""
    fi
    if [[ "${RUN_DIFFUSION}" == "1" ]]; then
      run_diffusion "${seed}" "${ckpt}"
    fi
  done
done

echo "[INFO] batch validation complete"
echo "[INFO] validation outputs: data/validation/${short_task}/..."
