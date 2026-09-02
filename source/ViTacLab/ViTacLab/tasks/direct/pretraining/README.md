# 质量任务
python scripts/teleoperation/gui_teleop/run_gelsight_finger_pretrain.py --task mass --num_envs 1 --enable_cameras

# 摩擦任务
python scripts/teleoperation/gui_teleop/run_gelsight_finger_pretrain.py --task friction --num_envs 1 --enable_cameras

# 也可直接用注册的 Gym ID
python scripts/teleoperation/gui_teleop/run_gelsight_finger_pretrain.py \
  --task Isaac-GelsightFinger-MassPretrain-Direct-v0