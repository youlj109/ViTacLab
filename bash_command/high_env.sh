python scripts/rsl_rl/full_tra/record_full_tra_simple_gripper.py \
  --task simple_forge_nut \
  --enable-high-fidelity-scene \
  --high-fidelity-scene-usd "source/ViTacLab/ViTacLab/assets/data/Scene/kitchen/kitchen_9/kitchen9.usd" \
  --high-fidelity-scene-scale 1.0 1.0 1.0 \
  --num_envs 1 --enable_cameras

python scripts/rsl_rl/full_tra/record_full_tra_single.py \
  --task Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 --enable-high-fidelity-scene \
  --high-fidelity-scene-usd "source/ViTacLab/ViTacLab/assets/data/Scene/livingroom/livingroom_13/room13.usd" \
  --high-fidelity-object-scale 1 1 1 \
  --num_envs 1 --enable_cameras 

python scripts/rsl_rl/full_tra/record_full_tra_dual.py \
  --task Isaac-UR10e-Dual-Shadow-Hand-BiBlindBinDrop-Direct-v0 --enable-high-fidelity-scene \
  --high-fidelity-scene-usd "source/ViTacLab/ViTacLab/assets/data/Scene/kitchen/kitchen_9/kitchen9.usd" \
  --high-fidelity-object-scale 1 1 1 \
  --num_envs 1 --enable_cameras


Use one of: bi_blind_grasp, bi_peg, hand_over, unscrew; a dual Gym id alias (Isaac-UR10e-Dual-Shadow-Hand-BiBlindGrasp-Direct-v0, Isaac-UR10e-Dual-Shadow-Hand-BiPeg-Direct-v0, Isaac-UR10e-Dual-Shadow-Hand-HandOver-Direct-v0, Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0