python scripts/rsl_rl/full_tra/play_full_tra_single_v5.py   --task forge_nut   --trajectory-file "scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeNutThreadEnvCfg.json"  --num_envs 1   --enable_cameras   --record-data   --record-path "~/Code/lightwheel/IsaacLab_510/ViTacLab_output/forge_dexhand/penut_thread"  --record-max-episodes 100 --post-arm-reached-steps 30 --stable-steps 30/penut_thread/episod
python scripts/rsl_rl/full_tra/play_full_tra_single_v5.py \
  --task forge_nut \
  --trajectory-file "scripts/rsl_rl/full_tra/pose_keyframes/UR10eShadowHandForgeEnv__UR10eShadowHandForgeNutThreadEnvCfg.json" \
  --num_envs 1 \
  --enable_cameras \
  --record-data \
  --record-path "~/Code/lightwheel/IsaacLab_510/ViTacLab_output/forge_dexhand/nut_thread/0523" \
  --record-max-episodes 200 \
  --post-arm-reached-steps 30 \
  --stable-steps 30