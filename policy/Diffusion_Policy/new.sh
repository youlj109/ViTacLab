#!/bin/bash
python data2zarr_dp.py Isaac-Forge-PegInsert-Direct-v0_42 200
python data2zarr_dp.py Isaac-Forge-PegInsert-Direct-v0_42 100
python data2zarr_dp.py Isaac-Forge-PegInsert-Direct-v0_42 50
bash train.sh Isaac-Forge-PegInsert-Direct-v0_42 200 42 0 False
bash train.sh Isaac-Forge-PegInsert-Direct-v0_42 100 42 0 False
bash train.sh Isaac-Forge-PegInsert-Direct-v0_42 50 42 0 False
cd ../..
python scripts/rsl_rl/train.py --task=Isaac-Repose-Cube-Shadow-Direct-v0 --enable_cameras --num_envs=8196 --headless