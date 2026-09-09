python scripts/diffusion_policy/rollout_forge_dp.py \
  --task forge_nut \
  --task_name NutThread \
  --data_num 100 \
  --checkpoint_num 500 \
  --num_envs 2 \
  --max_episodes 100 \
  --max_steps 0 \
  --env_spacing 3.0 \
  --index_middle_j3_delta_rad 0.1 \
  --enable_cameras

python scripts/diffusion_policy/rollout_forge_vitacdp.py \
  --task forge_nut \
  --task_name NutThread_force \
  --data_num 100 \
  --checkpoint_num 500 \
  --num_envs 1 \
  --max_episodes 100 \
  --max_steps 0 \
  --max_episode_steps 500 \
  --env_spacing 3.0 \
  --policy_action_mode normalized_action \
  --index_middle_j3_delta_rad 0.1 \
  --enable_cameras

python scripts/diffusion_policy/rollout_forge_vitacdp.py \
  --task forge_nut \
  --task_name NutThread_rgb \
  --data_num 100 \
  --checkpoint_num 500 \
  --num_envs 1 \
  --max_episodes 100 \
  --max_steps 0 \
  --max_episode_steps 250 \
  --env_spacing 3.0 \
  --policy_action_mode normalized_action \
  --index_middle_j3_delta_rad 0.1 \
  --enable_cameras

python scripts/diffusion_policy/rollout_forge_dp.py \
  --task forge_insert \
  --task_name PegInsert \
  --data_num 100 \
  --checkpoint_num 1000 \
  --num_envs 1 \
  --max_episodes 100 \
  --max_steps 0 \
  --max_episode_steps 250 \
  --index_middle_j3_delta_rad 0.1 \
  --env_spacing 3.0 \
  --enable_cameras

python scripts/diffusion_policy/rollout_forge_vitacdp.py \
  --task forge_insert \
  --task_name PegInsert_rgb \
  --data_num 100 \
  --checkpoint_num 1500 \
  --num_envs 1 \
  --max_episodes 100 \
  --max_steps 0 \
  --max_episode_steps 500 \
  --policy_action_mode normalized_action \
  --env_spacing 3.0 \
  --enable_cameras