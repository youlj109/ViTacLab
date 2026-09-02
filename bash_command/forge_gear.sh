python scripts/rsl_rl/full_tra/play_full_tra_single_v5.py   --task forge_gear   --trajectory-file "scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeGearMeshEnvCfg.json"   --num_envs 1   --enable_cameras   --record-data   --record-path "~/Code/lightwheel/IsaacLab_510/ViTacLab_output/forge_dexhand/gear_mesh"   --record-max-episodes 200 
python scripts/rsl_rl/full_tra/play_full_tra_single_v5.py \
  --task forge_gear \
  --trajectory-file "scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeGearMeshEnvCfg.json" \
  --num_envs 1 \
  --enable_cameras \
  --record-data \
  --record-path "~/Code/lightwheel/IsaacLab_510/ViTacLab_output/forge_dexhand/gear_mesh/0524" \
  --record-max-episodes 200 \
  --post-arm-reached-steps 30 \
  --stable-steps 30