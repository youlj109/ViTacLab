#!/usr/bin/env bash

# Examples:
# bash train_multi.sh Isaac-Forge-PegInsert-Direct-v0_42v58 200 42 0,1 False robot_dp

task_name=${1}
expert_data_num=${2}
seed=${3}
gpu_ids=${4}
DEBUG=${5} # True or False
alg_name=${6}

if [ -z "${task_name}" ] || [ -z "${expert_data_num}" ] || [ -z "${seed}" ] || [ -z "${gpu_ids}" ] || [ -z "${DEBUG}" ] || [ -z "${alg_name}" ]; then
  echo "Usage: bash train_multi.sh <task_name> <expert_data_num> <seed> <gpu_ids_comma_sep> <DEBUG> <alg_name>"
  exit 1
fi

config_name=${alg_name}
addition_info=train
exp_name=${task_name}-${alg_name}-${addition_info}

if [ "${DEBUG}" = "True" ]; then
  wandb_mode=offline
  echo -e "\033[33mDebug mode!\033[0m"
else
  wandb_mode=online
  echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_ids}

# e.g. "0,1,2,3" -> 4
nproc_per_node=$(echo "${gpu_ids}" | awk -F',' '{print NF}')
echo -e "\033[33mUsing GPUs: ${gpu_ids} (nproc_per_node=${nproc_per_node})\033[0m"

torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} train.py --config-name=${config_name}.yaml \
  task.name=${task_name}_${expert_data_num} \
  task.dataset.zarr_path="data/${task_name}_${expert_data_num}.zarr" \
  training.debug=${DEBUG} \
  training.seed=${seed} \
  training.device="cuda:0" \
  exp_name=${exp_name} \
  logging.mode=${wandb_mode}
