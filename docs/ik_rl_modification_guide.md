# IK-RL 项目修改与扩增规范（团队）

本文面向基于 **Isaac Lab + ViTacLab** 使用 `scripts/rsl_rl/ik_rl/train_ik_rl_single.py` 训练、`play_ik_rl_single.py` 回放与录制的开发流程，约定**改场景 / 改奖励 / 改保存**等常见修改应动哪些代码、如何验证、如何避免踩坑。

**前置阅读（仓库已有文档）**

- 环境与命令速查：`scripts/rsl_rl/QUICKSTART.md`、`scripts/rsl_rl/README.md`
- IK YAML 语义：`scripts/rsl_rl/ik_rl/configs/README.md`
- Headless / 相机门控：`docs/enable_cameras_headless_rl.md`

---

## 1. 标准工作流（团队对齐）

| 阶段 | 入口 | 说明 |
|------|------|------|
| 环境 | Isaac Lab 官方安装 + `pip install -e source/ViTacLab` | 使用与团队一致的 Isaac / Isaac Lab 版本；见根目录 `README.md`。 |
| 训练 | `scripts/rsl_rl/ik_rl/train_ik_rl_single.py` | **仅手部**策略 + **差分 IK 臂** + 脚本化掌心轨迹；与 `full_rl/train.py` **不共用 checkpoint**。 |
| 回放 / 评测 / 录数据 | `scripts/rsl_rl/ik_rl/play_ik_rl_single.py` | `--checkpoint` 或 `--resume --load_run ...`；可选 `--record_data`。 |

**参数顺序约定**：脚本先用 `argparse` 解析**前半段**（`--task`、`--num_envs`、`--trajectory`、`--ik-config` 等），剩余参数交给 **Hydra**。因此：**所有脚本自有参数必须写在最前面**，Hydra 的 `key=value` 写在后面。见 `scripts/rsl_rl/README.md`「Hydra 与参数顺序」。

**训练 vs 相机**

- 默认训练：**`--headless`**，且**不要**传 `--enable_cameras`（省显存与传感器开销），除非要调试触觉/第三视角。
- 需要触觉画面或 `--show_rgb` / `--show_ff`：**传 `--enable_cameras`**，一般**不要** `--headless`。详见 `docs/enable_cameras_headless_rl.md`。

---

## 2. 修改导航：我要改什么？

### 2.1 场景 / USD / 物体与机器人模型

| 目标 | 主要改动位置 | 规范与注意 |
|------|----------------|------------|
| 换机器人 USD、换桌台、加减刚体障碍物 | 对应任务的 `*_env_cfg.py`（`SceneCfg` / `AssetBaseCfg` / `spawn` 路径） | 资产路径优先走 `ViTacLab` 包内 `data/` 或 Omniverse Nucleus；改完后确认 `gym.make(..., cfg=...)` 能加载。 |
| 换抓取物、改初始位姿/随机范围 | `*_env_cfg.py` 中 object / goal 的 `init_state`、`*_range` | 若 **IK 轨迹**里用 `object`、`cup` 等名字，需保证环境里仍有同名 `self.object` / `self.cup` 或 `*_pos`/`*_rot` 张量（见下节「轨迹锚点名」）。 |
| 可形变体（如杯子）、流体等 | 对应 `*_env.py` 的 `_setup_scene`、cfg 里 MPM / deformable 配置 | 改动物体后检查 `_compute_intermediate_values` 是否仍更新 `cup_pos`、`water_pos` 等缓存，奖励是否仍基于正确句柄。 |
| 触觉阵列布局、传感器贴附 | `UR10eShadowHandDirectBaseEnv` 及子类 scene cfg、传感器 prim 路径 | `enable_cameras=False` 时部分传感器**不创建**；训练全触觉观测需约定团队统一开 `enable_cameras` 或改 cfg 默认。 |

**检查清单**：改 USD 后至少跑一次 **短步数** `train_ik_rl_single.py`（或 `play`）确认无加载错误、无 NaN；若改观测维度，必须同步 **policy 配置**（见 2.3）。

---

### 2.2 奖励、成功条件、回合长度

| 目标 | 主要改动位置 | 规范与注意 |
|------|----------------|------------|
| 改 shaping 奖励、稀疏奖励 | 对应 `*_env.py` 的 `_get_rewards()` | 若任务把分量记在 `self.extras["log"]`，保持键名稳定便于 wandb/tensorboard 对比。 |
| 改成功条件、提前终止 | `_get_dones()`、`_reset_idx()` | 若 `play_ik_rl_single.py` 要打印成功率，环境可实现 `get_episode_success_stats()`（及可选 `get_episode_success_rate_ema`）；否则仅日志无统计。 |
| 改最大步数 | `*_env_cfg.py` 中 `episode_length_s` 或 RL 脚本 `--max_episode_length`（**环境步数**） | `train_ik_rl_single` 中 `--max_episode_length` 会换算 `episode_length_s = steps * sim.dt * decimation`；与 cfg 冲突时以**脚本覆盖行为**为准，团队内应文档化。 |

**规范**：改奖励后**必须**在 PR / 说明里写清：旧策略是否作废、是否需重新训练；**不要**假设旧 checkpoint 仍最优。

---

### 2.3 观测空间（policy 输入维度）

| 目标 | 主要改动位置 | 规范与注意 |
|------|----------------|------------|
| 增减观测项（位姿、触觉、速度等） | `*_env.py` 的 `_get_observations()` 与 cfg 里 `observation_space` 或动态维度 | 改维度后：**必须**重新训练；旧 `model_*.pt` 不兼容。 |
| 触觉从「摘要」切「全阵列」 | `hand_pickup_cfg` / `UR10eShadowHandDirectBaseEnv` 等 `use_full_tactile_obs`、`enable_cameras` | 全触觉通常依赖相机/渲染；与 `enable_cameras_headless_rl.md` 一致。 |

---

### 2.4 IK-RL 专用：EE 轨迹列表、`--ik-config` YAML

**实现参考**：`scripts/rsl_rl/ik_rl/utils/ik_rl_hand_vec_env.py`（`EeWaypoint`、`ArmIkHandActionExpander`）。

| 目标 | 改动位置 | 规范与注意 |
|------|----------|------------|
| 改分段与末端位姿 | `ik_rl_*.yaml` 的 `trajectory`（或双臂 `trajectory_right` / `trajectory_left`） | 每段：`pos` [m]、`quat` wxyz、`steps`（环境步；`-1` 表示到回合结束）。姿态为 **EE 连杆**（`ee_body`，默认 `wrist_3_link`）在 **世界系** 下的目标。 |
| 与物体随机位姿对齐 | 任务 env 在 reset 时设置 `ik_rl_trajectory_xyz_offset`（与物体/bottle 位置噪声一致） | IK 层会把该向量**加在**所有 waypoint 的 `pos` 上（见拧瓶盖 env）。 |
| IK 阻尼等 | `ee_body`、`ik_method`、`ik_lambda` 于 YAML 或 CLI | **训练与回放必须一致**。 |
| 新任务接入 IK-RL | 新增 `ik_rl_xxx.yaml`；文档见 `ik_rl/configs/README.md` | Pour 类任务常需显式 `--ik-config`。 |

---

### 2.5 回放、数据保存与命名

**实现参考**：`play_ik_rl_single.py` 中 `_save_play_episode_npz`、`record_data` 分支。

| 行为 | 约定 |
|------|------|
| 默认输出目录 | `./play_records/<task>_<时间戳>/`（可用 `--record_path` 覆盖） |
| 元数据 | 同目录 `meta.json`：`task`、`checkpoint`、`record_env_index`、`trajectory`、`num_envs` |
| 每个 episode 文件 | `episode_XXXX.npz`（**整段**完成时 `done=True` 触发） |
| 中途结束 | `episode_XXXX_partial.npz` |
| **npz 字段**（当前实现） | `policy_obs`（T×D）、`actions`（T×手动作维）、`rewards`（T,）、`dones`（T,） |

**规范**

| 项 | 建议 |
|----|------|
| 扩展保存内容（如 RGB、触觉图、关节全状态） | **不要**在无人评审时直接改 `play_*.py` 破坏现有字段；优先：新增可选 flag（如 `--record_extras`）或子类封装，并在本文档与 `README` 更新字段说明。 |
| 版本与可复现 | `meta.json` 中应能定位 **checkpoint 路径**与 **trajectory**；若改环境或奖励，在实验记录里注明 **commit hash** 与 **cfg 覆盖**。 |
| 并行环境只录一条 | `--record_env_index` 默认 `0`；多环境并行时确认索引与你要观察的 env 一致。 |

---

### 2.6 日志与 Checkpoint

| 项 | 约定 |
|----|------|
| 日志根 | `get_rsl_rl_log_root`（见 `scripts/rsl_rl/ik_rl/utils/rsl_rl_log_utils.py`），通常含 `task` 名 |
| IK-RL 与 full_rl | **禁止**混用 checkpoint：动作维度与控制栈不同 |
| 恢复训练 | `--resume --load_run <目录名> --checkpoint model_XXXX.pt` |

---

## 3. 提交前自检（建议）

- [ ] `train_ik_rl_single.py` 与 `play_ik_rl_single.py` 对同一 `--task`、同一 `--ik-config`（若存在）、同一 `--trajectory` 能跑通。
- [ ] 改观测或动作维度后，已重新训练并更新文档中的实验名 / 配置说明。
- [ ] 改 `enable_cameras` 相关行为时，已对照 `docs/enable_cameras_headless_rl.md`。
- [ ] 若改录数据格式，已更新 `meta.json` 或 npz 字段说明，并知会数据消费方。

---

## 4. 相关代码路径（便于检索）

| 主题 | 路径 |
|------|------|
| IK 训练入口 | `scripts/rsl_rl/ik_rl/train_ik_rl_single.py` |
| IK 回放与录制 | `scripts/rsl_rl/ik_rl/play_ik_rl_single.py` |
| IK + 包装器 | `scripts/rsl_rl/ik_rl/utils/ik_rl_hand_vec_env.py` |
| YAML 合并 | `scripts/rsl_rl/ik_rl/utils/ik_rl_load_config.py` |
| CLI 共用 | `scripts/rsl_rl/ik_rl/utils/cli_args.py` |
| UR10e 基类（相机/触觉门控） | `source/ViTacLab/.../ur10e_shadowhand_direct_base_single/ur10e_shadowhand_direct_base_env.py` |
| Pickup / Pour / 双机械臂倒水 等 | `source/ViTacLab/ViTacLab/tasks/direct/simple_dexhand/`（含 `pour_water/`、`dual_pour_water/`） |
| Forge dexhand / 拧瓶盖 等 | `source/ViTacLab/ViTacLab/tasks/direct/medium_dexhand/` |

---

*文档版本：与当前仓库 `play_ik_rl_single.py` 中 `record_data` 保存字段一致；若实现变更，请同步更新本文件「2.5」节。*
