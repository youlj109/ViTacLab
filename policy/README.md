进入对应目录后，先
```shell
python data2zarr_vitac_dex.py [task_name] [data_num]
```
然后
```shell
bash train_multi.sh [task_name] [data_num] 42 [gpu_ids] False [config_name]
```
DP的config_name是dex_task Ours的config_name是dex_tac_rgb

导包路径和data2zarr的路径可能需要调整，evaluation的时候用deploy_policy就行