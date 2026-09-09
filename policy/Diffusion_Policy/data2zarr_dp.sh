# Examples:
# bash data2zarr_dp.sh Hang_Tops 1 100

# 'task_name' e.g.Fold_Dress, Hang_Tops, Fling_Trousers, etc.
# 'stage_index' e.g. 1, 2, 3, etc.
# 'train_data_num' means number of training data, e.g. 100, 200, 300, etc.

task_name=${1}
stage_str=${2}
train_data_num=${3}
# isaac_path=~/isaacsim_4.5.0/python.sh

# export ISAAC_PATH=$isaac_path

python data2zarr_dp.py ${task_name} ${stage_str} ${train_data_num}
