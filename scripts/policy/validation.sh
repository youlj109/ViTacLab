validation() {
    seed=$1
    max_steps=$2
    data_num=$3
    checkpoint_num=$4
    num_episodes=$5

    seeds=()
    for ((i=0; i<num_episodes; i++)); do
        seeds+=($((seed + i)))
    done

    for seed in ${seeds[@]}; do
        # python scripts/policy/play_policy.py --task=Isaac-Forge-GearMesh-Direct-v0 --num_envs=20 --max_steps=200 --data_num=$data_num --checkpoint_num=$checkpoint_num --seed=$seed --policy_name Diffusion_Policy --version 42
        python scripts/policy/play_policy.py --task=Isaac-Forge-PegInsert-Direct-v0 --num_envs=20 --max_steps=30 --data_num=$data_num --checkpoint_num=$checkpoint_num --seed=$seed --policy_name ViTacDP --version 42fixtwist_v40
    done
}

validation 42 30 200 500 10
validation 42 30 200 1000 10
validation 42 30 200 1500 10
validation 42 30 200 2000 10