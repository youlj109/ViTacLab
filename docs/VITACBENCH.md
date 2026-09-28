# ViTacBench 数据采集增量

按用户要求：**保留当前本地 ViTacSim，实现远端数据采集代码的兼容接入，
不合并远端任务环境。** 采集代码来自 `wip/advisor-vitacbench` 的 `d5c5e2d`，
本地集成前版本为 `2ca8b79`。

## 保留与引入的范围

- 整个 `source/` 与集成前版本一致：触觉渲染、力到深度纠正、标定、marker、
  任务资产、物理参数、奖励、成功判定和 actor observation 均不从远端覆盖。
- README、项目主页、已发布的 Xense 六质量对比图和数值结果保持原样。
- 引入采集/回放脚本、关键帧、IK 演示配置、一键 shell、NPZ 检查与视频导出，
  以及 `policy/prepare_dataset.py` 的数据转换支持。
- 通用 RL 播放器补齐录像参数和步数上限，并保留本地触觉可视化实现。
- 通用 RL/Pour 采集不跨 reset 拼接轨迹。RL 保存 step 前的独立快照，
  不伪造自动 reset 后无法获取的 terminal 画面；未成功尝试的缓存会丢弃。
- Pour 演示 YAML 仅配置脚本控制器，不关闭本地失败条件或重置随机化。
- 双臂录像所需的 TacSL 适配位于采集脚本侧；不替换任务物体、物理参数或场景内容。
  Handover 仍使用本地香蕉，**不是**远端训练时的黄色球。

最初较宽的集成曾包含任务环境修正；这些源码变更已撤回。
相应的 donor observation/reward 单测不再适用于本次数采范围。
不要把之前 `(1, 7620)` 的运行日志当作本版的 actor observation 验证结果。

## 运行方式

从仓库根目录，在配置好的 Isaac Lab 环境内运行。启动器支持
`PYTHON_SIM=/path/to/python.sh`、`ISAACLAB_SH=/path/to/isaaclab.sh` 和
`PYTHON_OFFLINE=/path/to/python`，不绑定特定 conda 环境名。
在现有 Isaac Lab Docker 内可使用 `PYTHON_SIM=/isaac-sim/python.sh`。
离线校验/视频解释器需要 NumPy/OpenCV；Zarr 导出还需要对应依赖。

| 任务 | 一键入口 |
| --- | --- |
| Pickup | `bash bash_command/demo_data_collection_replay.sh` |
| Pour | `bash bash_command/demo_pour_collection_replay.sh` |
| In-hand | `bash bash_command/demo_inhand_collection_replay.sh` |
| Handover | `bash bash_command/demo_handover_collection_replay.sh` |
| Forge DexHand | `bash bash_command/demo_forge_all_collection_replay.sh` |
| Forge Franka RL | `bash bash_command/demo_forge_franka_rl_all_collection_replay.sh` |
| Unscrew / bottle | `bash bash_command/demo_bottle_collection_replay.sh` |
| BiStab | `bash bash_command/demo_bistab_collection_replay.sh` |
| Blind Classification | `bash bash_command/demo_blind_classification_all_shapes.sh` |
| Blind Grasp | `bash bash_command/demo_blind_grasp_all_phases.sh` |
| Blind Retrieval | `bash bash_command/demo_blind_retrieval_collection_replay.sh` |
| Blind Bin Drop | `bash bash_command/demo_bi_blind_bin_drop_all_phases.sh` |

通用选项：`MODE=collect|visualize|full`、`DEVICE=cuda:0`。
需要 policy 的入口必须提供与**本地**环境匹配的 checkpoint；本次没有附带或重新训练模型。
NPZ、视频和报告写入 `logs/demo_<task>/` 下的新运行子目录，不预先删除旧文件。
回放已有数据时，`MODE=visualize RECORD_DIR=...` 应指向包含 NPZ 的具体运行目录。
各脚本输出实际路径，完整参数以 `--help` 为准。

`SAVE_OUTCOME=completed` 只代表轨迹完成，不代表任务成功。
Forge 的可选几何成功阈值不是经过标定的 benchmark 设置；放宽或关闭判据的输出
只能标为演示数据。源环境的奖励和成功函数不随这些采集选项改写。

## 数据、actor 与演示的区别

- `policy_record_profile=tactile_only` / `--profile tactile_only` 限制**归档或转换后的
  数据字段**，不更改环境 actor observation；原环境可能仍包含物体状态等特权信息。
- `play_blind_retrieval.py` 和 privileged grasp collector 使用物体几何辅助，
  不能作为自主盲操作策略评测，也未验证成功举升率。
- Classification 在本地环境不支持固定形状参数时，通过常规 reset 抽样获取指定类别，
  不修改源环境。记录的类别必须与实际 episode 一致。
- Blind Grasp 使用本地固定初始位置，采集默认 `--object-xy-noise 0`；
  不静默导入远端初始位置随机化。远端关键帧在本地物理场景中的成功性需要重新验证。
- NPZ 原始演示归档可以包含用于诊断的相机画面/位姿。
  `scripts/debug/validate_play_record_npz.py --profile tactile_only` 校验受限记录，
  转为 policy 输入时用 `policy/prepare_dataset.py --profile tactile_only`。
- ViTacDP 的 `single_arm_tactile_only_{force,rgb}.yaml` 仅为数据形状模板。
  现有训练 loader 仍依赖相机/位姿，**不能直接视为已适配的训练配置**。

## 验证与限制

验证分为：脚本语法/CLI、离线数据与启动器回归、真实 Isaac 短时采集。
离线单测不代表完整十二任务成功率测试；未验证远端 checkpoint 对本地场景的兼容性。
原始论文文件和本地三质量论文图不包含在此次提交中。

2026-09-28 已完成：

- `scripts/audit_project.py --repro-only`：0 error / 0 warning；不是完整风格审计结果。
- 启动器测试：12 个入口的 24 个 full/visualize 场景，旧 NPZ/视频/报告保留。
- 数据测试：13 项，由现有 RMBench（Zarr）和 embodiedgen（PyTorch3D）环境
  互补覆盖，包含实际 Zarr 往返和 pose 转换。
- 采集兼容测试、Retrieval helper 测试、RL episode 边界测试通过。
- 双臂适配离线测试通过；实际 Isaac Kit 配置加载、复制、序列化及传感器构造通过，
  香蕉/橙子/盘子、机器人和物理配置保持原样。
- Xense measured-marker CPU/CUDA 六质量图像回归逐像素一致，已发布 MAE/RMSE
  从提交的 RGB 图像重新计算通过。

上述双臂检查是配置/传感器构造验证，不是完整策略 rollout。

无 checkpoint 的实际采集检查（在 Isaac Lab 容器内）：

```bash
/isaac-sim/python.sh scripts/debug/smoke_advisor_integration.py \
  --headless --steps 2 --output logs/collection_check/record.npz
python scripts/debug/validate_play_record_npz.py logs/collection_check/record.npz
```

输出路径必须尚不存在。该检查使用未修改的本地任务，验证有限步仿真和 joint、
触觉力/RGB、相机及位姿记录，并可实际写入 NPZ。它会单独报告 actor observation
与声明空间不匹配的问题，**不会为了通过检查而修补源环境**；因此采集通过不等于
policy 可加载，也不代表成功操作。Kit 关闭前显式设置退出码，避免异常被掩盖。

本地 Pickup 派生任务存在触觉 observation 缓冲初始化/维度问题；这属于原环境，
本次数采兼容不改动它。记录器可直接读取现有 sensor record 接口，与 actor 缓冲
不是同一路径。需要修正 actor 或训练策略时应另开任务处理。
