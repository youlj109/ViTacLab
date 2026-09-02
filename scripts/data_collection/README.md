# ViTacLab 数据采集统一入口

本目录是 ViTacLab 唯一的数据采集脚本目录。所有可执行的任务数据采集实现、
采集配置、关键帧与手工录制配置都在这里；`scripts/rsl_rl` 只保留训练入口和
不允许传入数采参数的评测兼容入口。数采脚本不会再转发到 `scripts/rsl_rl`。

训练、推理与数采共同使用但本身不可执行的 CLI、日志路径和 IK 控制模块位于
`scripts/common/rl/`。这样可以避免复制代码，同时保持数采入口和实现完全归档。

## 目录分类

- `rl/`：使用完整 RSL-RL policy 自动采集。
  - `play_record.py`：直接包含 rollout、成功判断、缓存和 NPZ 落盘实现。
- `ik/`：IK 或 IK + policy 数采。
  - `play_ik_policy.py`：手部 RSL-RL policy + GPU 差分 IK 机械臂。
  - `play_full_ik_single.py`：分阶段手型/抓取脚本 + GPU 差分 IK；支持无 checkpoint 纯脚本数采。
  - `play_waypoint_ik.py`：Forge/Franka 纯 waypoint IK；复用 agent 配置创建环境，但不加载或执行 policy checkpoint。
  - `configs/ik_rl/`：IK-RL Pickup/Pour 唯一配置。
  - `configs/full_ik/`：Full-IK Pickup/Pour 唯一配置。
- `full_trajectory/`：完整轨迹关键帧录制、回放、随机扰动和分阶段采集。
  - `record_single.py` / `record_dual.py`：交互式关键帧与轨迹 JSON 录制。
  - `record_action.py` / `play_action.py`：任意 Gym Box/Dict/Tuple action space 的通用
    normalized-action 关键帧录制与回放，用于 Franka、纯 ShadowHand、预训练和其他
    不共享 UR10e marker-IK schema 的环境。
  - `play_single.py`：单臂完整轨迹回放和 NPZ 采集。
  - `play_single_phase.py --phase 1|2|3`：单臂业务阶段采集，共享一个实现。
  - `play_dual_phase.py --phase 1|2|3|4`：双臂业务阶段采集，共享一个实现。
- `manual/`：GUI/marker 手工录制机械臂位姿、抓取位姿、手型和原始观测。
  - `record_arm_pose.py`：交互式机械臂/手部目标调节与 YAML 导出。
  - `record_grasp_pose.py`：抓取闭合与腕部/手掌参数调节。
  - `record_handshape.py`：手型与触觉 RGB/力场可视化调节。
  - `record_observations.py`：统一单臂任务观测查看、随机/RL rollout 与 NPZ 记录；该实现已从 debug 目录迁入本目录。
- `tools/`：数据统计、上传等辅助工具，不创建仿真数据。

## 运行约定

所有命令均从 ViTacLab 仓库根目录执行。远程 Isaac Lab 环境可使用：

```bash
./isaaclab.sh -p <script.py> <arguments>
```

若已经激活包含 Isaac Sim/Isaac Lab 的 Python 环境，也可使用：

```bash
python <script.py> <arguments>
```

涉及第三视角或触觉图像时传 `--enable_cameras`。`--show_rgb` / `--show_ff`
会自动要求相机；大量并行环境数采时不要开启 GUI 可视化。

## 1. 完整 RSL-RL policy 数采

适用任务：能够加载 RSL-RL agent 配置、checkpoint，并在 observation 中提供
`record` 字段的任务。已知目标用例包括 Forge GearMesh/NutThread 等 Forge 任务，
以及实现相同 `obs['record']` 协议的 ViTacLab Direct 任务。

```bash
python scripts/data_collection/rl/play_record.py \
  --task Isaac-ViTac-Forge-GearMesh-Direct-v0 \
  --checkpoint logs/rsl_rl/forge_gear_mesh_direct/<run>/model_1800.pt \
  --num_envs 32 --enable_cameras \
  --save_data --num_episodes 100 --max_steps 100 \
  --data_path play_records/forge_gear_mesh
```

关键参数：

- `--save_data`：启用数据落盘；不传时只 rollout。
- `--data_path`：输出目录；不传则在 `play_records/` 下生成带时间戳目录。
- `--num_episodes`：所有并行环境合计需要保存的成功 episode 数。
- `--max_steps`：单次 rollout 最大步数；`0` 表示不额外限制。
- `--checkpoint`：模型文件，可为绝对路径或配合 `--resume --load_run` 解析。
- `--show_rgb`、`--show_ff`、`--fps`、`--env_index`：触觉可视化选项。

脚本会在 `--save_data` 启用时检查 `obs['record']`；任务未实现该字段会在启动时
明确报错，而不会生成不完整数据。

## 2. IK-RL policy 数采

适用任务：单 UR10e + ShadowHand、policy 只控制手部、机械臂由差分 IK 控制的任务。
当前预设：

- Pickup：`Isaac-UR10eShadowHand-Pickup-Direct-v0`
- Pour：`Isaac-UR10eShadowHand-PourDeformable-Direct-v0`

```bash
./isaaclab.sh -p scripts/data_collection/ik/play_ik_policy.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --ik-config scripts/data_collection/ik/configs/ik_rl/ik_rl_pickup.yaml \
  --checkpoint /absolute/path/model_1000.pt \
  --num_envs 1 --device cuda:0 --enable_cameras \
  --record_data --record_path play_records/pickup_ik_policy \
  --record_env_index 0 --record_max_episodes 10 --max_play_steps 20000
```

`--trajectory` 格式为 `target:env_steps:use_rotation`，多段以逗号分隔；
`env_steps=-1` 表示保持到 episode 结束，`use_rotation` 为 `0` 或 `1`。
`--ik-config none` 禁用 YAML。CLI 同名参数优先于 YAML。

## 3. Full-IK 分阶段脚本数采

适用任务：单 UR10e + ShadowHand Pickup/Pour；通过 `phase_schedule` 控制接近、
闭手、保持手型和后续 IK 轨迹。可以加载 checkpoint，也可以用 `--no-checkpoint`
进行纯脚本数采。

Full-IK 的仓库级验收只要求链路正确：配置能够加载，anchor/palm/wrist 坐标变换、
IK 求解、arm/hand action 扩展、环境 step、Camera/TacSL 与可选 NPZ 记录均无异常，
且输出 tensor 有正确 shape/dtype/finite 值。仓库内 YAML 是每个任务的唯一规范配置
和标定起点，但不承诺对所有物体、随机化范围或抓取目标都是最优轨迹。具体的
offset、orientation、phase 时长、抓取手型、成功条件和轨迹美观度由实际任务操作员
在生产数采前通过 GUI 调整。链路 smoke test 通过不等于该参数组合已经适合生成训练数据。

Pickup：

```bash
./isaaclab.sh -p scripts/data_collection/ik/play_full_ik_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --full-ik-config scripts/data_collection/ik/configs/full_ik/full_ik_pickup_fixed_hand.yaml \
  --num_envs 1 --device cuda:0 --enable_cameras --no-checkpoint \
  --record_data --record_path play_records/pickup_scripted \
  --record_env_index 0 --record_max_episodes 10 --max_play_steps 20000
```

Pour：

```bash
./isaaclab.sh -p scripts/data_collection/ik/play_full_ik_single.py \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --full-ik-config scripts/data_collection/ik/configs/full_ik/full_ik_pour.yaml \
  --num_envs 1 --device cuda:0 --enable_cameras --no-checkpoint \
  --record_data --record_path play_records/pour_scripted
```

配置内的 `task` 应与 `--task` 一致；手型 YAML 必须存在。脚本在加载配置时会
对任务不匹配、空 `phase_schedule` 和缺失手型文件给出明确错误或警告。

## 4. Forge/Franka Waypoint-IK 数采

适用任务：提供 `set_franka_ik_target`、`fixed_pos`、`_held_asset` 等 Factory/Forge IK
接口的 Franka 任务。脚本复用注册的 RSL-RL agent 配置来创建环境，但不会加载或执行
checkpoint；环境 action 为零，机械臂由生成式 waypoint IK 驱动。

```bash
python scripts/data_collection/ik/play_waypoint_ik.py \
  --task Isaac-ViTac-Forge-GearMesh-Direct-v0 \
  --num_envs 1 --enable_cameras \
  --save_data --data_path play_records/waypoint_ik \
  --num_episodes 10 --max_steps 400 --waypoint_max_steps 100
```

- `--waypoint_max_steps`：每个 waypoint 的最长停留步数；`0` 不强制跳过。
- `--num_episodes` 只统计成功且实际写入的数据；timeout 永远不能冒充成功。
- `--save-outcome all`：额外保存 `attempt_XXXX_timeout.npz` 供链路诊断。
- `--max-attempts`：限制成功/timeout 尝试总数，便于自动验收；`0` 表示不限制。

## 5. 完整轨迹采集

关键帧交互录制：

```bash
python scripts/data_collection/full_trajectory/record_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 1 --manual-reset-only --enable_cameras --show_rgb --show_ff
```

单臂轨迹回放并保存成功 NPZ：

```bash
python scripts/data_collection/full_trajectory/play_single.py \
  --task pickup --num_envs 1 --enable_cameras \
  --trajectory-file scripts/data_collection/full_trajectory/records/pickup.json \
  --record-data --record-path play_records/pickup_full_trajectory
```

双臂分阶段采集示例：

```bash
python scripts/data_collection/full_trajectory/play_dual_phase.py \
  --phase 2 --task bi_blind_bin_drop --num-envs 1 --num-episodes 150 \
  --enable_cameras --record-path play_records/dual_blind_bin_drop
```

`phase1`～`phase4` 表示同一任务流程的业务阶段，不是脚本版本号。

### Full-trajectory task-family coverage

- `record_single.py` and `play_single.py` share registry resolution for all 12
  single-UR10e/ShadowHand tasks. Built-in aliases are `pickup`, `pour`,
  `inhand`, `inhand_openai`, `inhand_tactile`, `forge_peg`,
  `forge_gear`, `forge_nut`, `blind_grasp`, `blind_grasp_replay`,
  `blind_classification`, and `blind_retrieval`.
- `record_dual.py` and `play_dual_phase.py --trajectory-json ...` share
  registry resolution for all nine dual-UR10e tasks. Built-in aliases are
  `bi_blind_grasp`, `bi_blind_bin_drop`, `bi_blind_inhand`, `bi_blind_peg`,
  `bi_peg`, `bi_stab`, `hand_over`, `dual_pour`, and `unscrew`.
- Replay success is read first from canonical environment success signals.
  Pickup lift/goal-Z and BinDrop object/trash-XY checks are task-specific
  fallbacks only. `--save-outcome success|completed|all` controls whether a
  structurally completed or failed replay is retained for chain diagnostics.
- `record_action.py` / `play_action.py` provide the shared action-space adapter
  for the six Franka Forge tasks, standalone ShadowHand tasks, and both
  GelSight pretraining tasks. The mass-pretraining action is intentionally a
  no-op, so its full-trajectory test validates action/record flow rather than
  visible robot motion.
- Both HandOver tasks remain valid RSL-RL tasks. In particular,
  `Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0` supports RSL-RL NPZ collection
  and matching Diffusion Policy/ViTacDP checkpoints.

## 从训练到成功轨迹的完整使用手册

本节给出所有数采入口的功能、前置条件和推荐执行顺序。完整 argparse 参数仍以
`python <script> --help` 和 `docs/SCRIPT_USAGE.md` 为准。

### 1. 全部可执行脚本索引

本目录当前有 15 个主要可执行数采脚本。辅助模块
`common_record_utils.py`、`task_entries_single.py` 和 `task_entries_dual.py`
由这些入口导入，不应单独执行。

| 类型 | 脚本 | 主要输入 | 主要输出 | 成功数据语义 |
|---|---|---|---|---|
| 完整 RL policy | `rl/play_record.py` | Gym task、匹配的 RSL-RL checkpoint | episode NPZ | 只保存成功 rollout |
| IK + hand policy | `ik/play_ik_policy.py` | IK YAML、匹配的 hand-policy checkpoint | success/failed/partial NPZ | 需按 outcome 筛选 |
| Full-IK | `ik/play_full_ik_single.py` | Full-IK YAML、可选 checkpoint | episode NPZ | 只保存成功 episode |
| Forge waypoint | `ik/play_waypoint_ik.py` | Forge task、waypoint 参数 | success/timeout NPZ | timeout 不计成功 |
| 单臂轨迹录制 | `full_trajectory/record_single.py` | marker、hand sliders | `vitatlab_full_tra_v1` JSON | JSON 不是最终训练数据 |
| 单臂轨迹回放 | `full_trajectory/play_single.py` | 单臂 JSON | success/completed/done NPZ | 生产使用 success |
| 双臂轨迹录制 | `full_trajectory/record_dual.py` | 左右 marker/hand sliders | `vitatlab_full_tra_dual_v1` JSON | JSON 不是最终训练数据 |
| 双臂轨迹回放 | `full_trajectory/play_dual_phase.py` | 内置 phase 或双臂 JSON | success/completed/failed NPZ | 生产使用 success |
| 通用 action 录制 | `full_trajectory/record_action.py` | Box/Dict/Tuple action space | `vitatlab_action_trajectory_v1` JSON | JSON 不是最终训练数据 |
| 通用 action 回放 | `full_trajectory/play_action.py` | normalized-action JSON | success/completed/done NPZ | 生产使用 success |
| 单臂业务 phase | `full_trajectory/play_single_phase.py` | BlindGrasp keyframe JSON | phase episode NPZ | 生产使用 success |
| 手臂/手型标定 | `manual/record_arm_pose.py` | marker、hand sliders | arm+hand/Full-IK YAML | 配置，不是 episode |
| 抓取姿态标定 | `manual/record_grasp_pose.py` | arm YAML、closure marker | grasp YAML | 配置，不是 episode |
| 触觉手型标定 | `manual/record_handshape.py` | arm/closed-hand YAML | handshape YAML | 配置，不是 episode |
| 原始观测记录 | `manual/record_observations.py` | zero/random/policy action | 每步 PT/NPZ | 不按成功 episode 聚合 |

### 2. YAML、JSON、NPZ 与 outcome

- YAML 保存 6D UR10e 姿势、24D Shadow Hand 手型、palm/wrist 变换、IK trajectory
  和 phase schedule。
- JSON 保存稀疏控制示范，例如 marker pose、arm/hand keyframe、normalized action、
  `hold_steps` 和环境入口。JSON 不包含完整逐步触觉训练数据。
- NPZ 保存逐仿真步或按间隔采样的时序数据，包括 `joint_pos`、`action`、触觉、
  相机和任务特定状态。

启用真实相机/TacSL 的记录至少应有：

```text
tactile_pos
tactile_normal_force
tactile_shear_force
tactile_rgb_image
```

常见传感器数量：GelSight pretraining=1、Franka Forge=2、单 Shadow Hand=5、
双手=10。相机字段由任务 canonical `record` 决定，常见字段为
`third_person_camera`、`third_person_camera_pos`，Forge 还可有
`twist_camera`。

Full-trajectory outcome：

- `--save-outcome success`：只保存环境 canonical success；生产数据使用此项。
- `--save-outcome completed`：额外保存走完关键帧但未满足任务成功的轨迹。
- `--save-outcome all`：再保存 done/failed/timeout；仅用于诊断。

`play_ik_policy.py` 输出 `*_success.npz`、`*_failed.npz`、
`*_partial.npz`，并写入 `episode_complete`、`outcome_success`。正式数据必须满足：

```text
episode_complete == True
outcome_success == True
```

### 3. 完整 RSL-RL：train -> plain play -> successful rollout

训练、普通评测和数据落盘是三个入口。不要把数采参数传给
`scripts/rsl_rl/full_rl/play.py`。

训练：

```bash
python scripts/rsl_rl/full_rl/train.py \
  --task Isaac-ViTac-Forge-GearMesh-Direct-v0 \
  --num_envs 128 \
  --device cuda:0 \
  --headless
```

普通 play，先验证 checkpoint 与当前环境匹配：

```bash
python scripts/rsl_rl/full_rl/play.py \
  --task Isaac-ViTac-Forge-GearMesh-Direct-v0 \
  --checkpoint /absolute/path/model_XXXX.pt \
  --num_envs 1 \
  --enable_cameras
```

确认 actor observation/action 维度、action 语义和任务成功率后，使用 collector：

```bash
python scripts/data_collection/rl/play_record.py \
  --task Isaac-ViTac-Forge-GearMesh-Direct-v0 \
  --checkpoint /absolute/path/model_XXXX.pt \
  --num_envs 1 \
  --save_data \
  --data_path play_records/forge_gear_mesh \
  --num_episodes 100 \
  --max_steps 500 \
  --show_rgb \
  --show_ff \
  --enable_cameras
```

`--num_episodes` 是所有并行环境合计要保存的成功 episode 数。达到
`--max_steps` 而未成功的 rollout 会 reset 并清空未保存缓冲区。批量生产时可增大
`--num_envs`，移除 viewer，并使用 `--headless --enable_cameras`。

checkpoint 必须匹配 policy family。旧 Pickup IK-hand 模型可能是
`133D observation -> 24D hand action`，full direct 环境可能是
`7633D observation -> 30D full action`；前者应使用 `play_ik_policy.py`，
不能用 `play_record.py` 强行加载。

### 4. IK-RL：train -> IK-policy play -> record

当前维护 Pickup/Pour：

```text
scripts/data_collection/ik/configs/ik_rl/ik_rl_pickup.yaml
scripts/data_collection/ik/configs/ik_rl/ik_rl_pour.yaml
```

训练 hand-only policy：

```bash
python scripts/rsl_rl/ik_rl/train_ik_rl_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --ik-config scripts/data_collection/ik/configs/ik_rl/ik_rl_pickup.yaml \
  --num_envs 256 \
  --device cuda:0 \
  --max_iterations 10000 \
  --headless
```

无记录可视化推理：

```bash
python scripts/data_collection/ik/play_ik_policy.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --ik-config scripts/data_collection/ik/configs/ik_rl/ik_rl_pickup.yaml \
  --checkpoint /absolute/path/model_XXXX.pt \
  --num_envs 1 \
  --policy-tactile-obs none \
  --max_play_steps 1000 \
  --show_rgb \
  --show_ff \
  --enable_cameras
```

记录 episode：

```bash
python scripts/data_collection/ik/play_ik_policy.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --ik-config scripts/data_collection/ik/configs/ik_rl/ik_rl_pickup.yaml \
  --checkpoint /absolute/path/model_XXXX.pt \
  --num_envs 1 \
  --policy-tactile-obs none \
  --record_data \
  --record_path play_records/pickup_ik_policy \
  --record_env_index 0 \
  --record_max_episodes 100 \
  --max_play_steps 100000 \
  --enable_cameras
```

`--policy-tactile-obs none` 保持 legacy policy observation，但不会关闭 NPZ 中的真实
TacSL。只有 checkpoint 训练时使用 summary/dense，推理时才选择相同模式。
`--record_max_episodes` 按保存的完整 episode 数停止，不是成功目标数；采集后必须过滤
`_success.npz` 和 outcome 字段。

### 5. Full-IK：标定 -> 可选训练 -> scripted/policy play -> record

推荐标定链：

```text
record_arm_pose.py
    -> arm+hand YAML / Pickup Full-IK YAML
record_grasp_pose.py
    -> grasp YAML
record_handshape.py（可选，使用真实触觉微调）
    -> final handshape YAML
把最终 hand YAML 写入 Full-IK phase_schedule/freeze_hand_yaml
```

可选训练：

```bash
python scripts/rsl_rl/full_ik/train_full_ik_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --full-ik-config scripts/data_collection/ik/configs/full_ik/full_ik_pickup_fixed_hand.yaml \
  --num_envs 256 \
  --device cuda:0 \
  --max_iterations 10000 \
  --headless
```

无 checkpoint 的脚本 dry run：

```bash
python scripts/data_collection/ik/play_full_ik_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --full-ik-config scripts/data_collection/ik/configs/full_ik/full_ik_pickup_fixed_hand.yaml \
  --num_envs 1 \
  --no-checkpoint \
  --max_play_steps 2000 \
  --show_rgb \
  --show_ff \
  --enable_cameras
```

成功 episode 数采：

```bash
python scripts/data_collection/ik/play_full_ik_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --full-ik-config scripts/data_collection/ik/configs/full_ik/full_ik_pickup_fixed_hand.yaml \
  --num_envs 1 \
  --no-checkpoint \
  --record_data \
  --record_path play_records/pickup_full_ik \
  --record_env_index 0 \
  --record_max_episodes 100 \
  --max_play_steps 100000 \
  --enable_cameras
```

如使用训练模型，用匹配的 `--checkpoint` 替换 `--no-checkpoint`。失败 episode 和
进程结束时的 partial buffer 会丢弃。`dropped failed episode` 表示任务未成功，
不表示记录链路损坏。

### 6. 单臂 Full-Trajectory：record JSON -> dry replay -> success NPZ

覆盖 12 个单 UR10e + Shadow Hand alias：

```text
pickup, pour, inhand, inhand_openai, inhand_tactile,
forge_peg, forge_gear, forge_nut,
blind_grasp, blind_grasp_replay, blind_classification, blind_retrieval
```

录制稀疏 JSON：

```bash
python scripts/data_collection/full_trajectory/record_single.py \
  --task pickup \
  --num_envs 1 \
  --manual-reset-only \
  --show_rgb \
  --show_ff \
  --enable_cameras
```

在 Isaac Sim 中移动 `/World/Debug/ArmIkTarget`，调整 Shadow Hand sliders，执行
`Start Recording -> Snapshot -> Stop Recording`。建议关键帧：

```text
reset -> approach -> pre-contact -> contact -> close/grasp ->
lift/manipulate -> target/release -> finish
```

只回放、不记录：

```bash
python scripts/data_collection/full_trajectory/play_single.py \
  --task pickup \
  --trajectory-file scripts/data_collection/full_trajectory/records/pickup.json \
  --num_envs 1 \
  --max-episodes 1 \
  --enable_cameras
```

保存一次 all-outcome 诊断：

```bash
python scripts/data_collection/full_trajectory/play_single.py \
  --task pickup \
  --trajectory-file scripts/data_collection/full_trajectory/records/pickup.json \
  --num_envs 1 \
  --max-episodes 1 \
  --record-data \
  --record-path play_records/pickup_diagnostic \
  --record-env-index 0 \
  --record-step-interval 1 \
  --save-outcome all \
  --enable_cameras
```

批量保存成功轨迹：

```bash
python scripts/data_collection/full_trajectory/play_single.py \
  --task pickup \
  --trajectory-file scripts/data_collection/full_trajectory/records/pickup.json \
  --num_envs 1 \
  --max-episodes 1000 \
  --record-data \
  --record-path play_records/pickup_success \
  --record-env-index 0 \
  --record-step-interval 1 \
  --record-max-episodes 100 \
  --save-outcome success \
  --enable_cameras
```

框架不只支持 Pickup，但正式生产应为每个任务分别录制 JSON。JSON 保存
`env_entry/cfg_entry`、物体初始位置和世界坐标 marker；仅传另一个 `--task`
不会自动把 Pickup 几何转换成 Pour/Forge/BlindGrasp 几何。

`play_single.py` 会记录真实触觉，但当前没有实时触觉 viewer。完成成功 NPZ 后可查看：

```bash
python scripts/debug/visualize_pickup_episode_npz.py \
  play_records/pickup_success/episode_0000_success.npz \
  --fps 20
```

### 7. 双臂 Full-Trajectory：record dual JSON -> play -> success NPZ

覆盖 9 个 alias：

```text
bi_blind_grasp, bi_blind_bin_drop, bi_blind_inhand, bi_blind_peg,
bi_peg, bi_stab, hand_over, dual_pour, unscrew
```

录制：

```bash
python scripts/data_collection/full_trajectory/record_dual.py \
  --task hand_over \
  --num-envs 1 \
  --record-dir scripts/data_collection/full_trajectory/records_dual/hand_over \
  --record-name hand_over \
  --manual-reset-only \
  --enable_cameras
```

使用 `/World/Debug/ArmIkTargetRight` 和
`/World/Debug/ArmIkTargetLeft`，分别调整左右手，执行
`Start Recording -> Snapshot -> Stop Recording`。

诊断回放；非 BinDrop 任务必须传匹配的 `--trajectory-json`，普通自录 JSON 推荐
`--phase 1`：

```bash
python scripts/data_collection/full_trajectory/play_dual_phase.py \
  --phase 1 \
  --task hand_over \
  --trajectory-json /absolute/path/hand_over_YYYYMMDD_HHMMSS.json \
  --num-envs 1 \
  --num-episodes 1 \
  --record-path play_records/hand_over_diagnostic \
  --save-outcome all \
  --enable_cameras
```

批量成功数据：

```bash
python scripts/data_collection/full_trajectory/play_dual_phase.py \
  --phase 1 \
  --task hand_over \
  --trajectory-json /absolute/path/hand_over_YYYYMMDD_HHMMSS.json \
  --num-envs 1 \
  --num-episodes 1000 \
  --record-path play_records/hand_over_success \
  --record-env-index 0 \
  --record-step-interval 1 \
  --save-outcome success \
  --enable_cameras
```

`--num-episodes` 是尝试 episode 上界，不是成功目标数。使用相同
`--record-path --resume` 可从已有成功编号继续。不传 `--trajectory-json` 时使用
BiBlindBinDrop 内置 keyframe 和 `--phase 1|2|3|4`；四个 phase 是独立业务分支，
不是一个 episode 中连续的四段。

### 8. 通用 action trajectory：record normalized action -> play -> success NPZ

推荐用于剩余 10 个不共享 UR10e marker-IK schema 的任务：

```text
6 个 Franka Forge（普通/Breakable PegInsert、GearMesh、NutThread）
2 个 GelSight Mass/Friction pretraining
Isaac-UR10eShadowHand-Repose-Cube-Vision-Direct-v0
Isaac-ViTac-Shadow-Hand-Over-Direct-v0
```

录制：

```bash
python scripts/data_collection/full_trajectory/record_action.py \
  --task Isaac-ViTac-Forge-GearMesh-Direct-v0 \
  --num-envs 1 \
  --hold-steps 30 \
  --record-dir scripts/data_collection/full_trajectory/records_action/forge_gear \
  --record-name forge_gear \
  --enable_cameras
```

GUI 为 flatten 后的每个 action 维度提供 `[-1, 1]` slider。当前 slider action 会
实时发送给环境；执行 `Start Recording -> Snapshot -> Stop + Save`。

诊断：

```bash
python scripts/data_collection/full_trajectory/play_action.py \
  --trajectory-file /absolute/path/forge_gear_YYYYMMDD_HHMMSS.json \
  --num-envs 1 \
  --num-episodes 1 \
  --record-data \
  --record-path play_records/forge_gear_diagnostic \
  --record-step-interval 1 \
  --save-outcome all \
  --enable_cameras
```

生产：

```bash
python scripts/data_collection/full_trajectory/play_action.py \
  --trajectory-file /absolute/path/forge_gear_YYYYMMDD_HHMMSS.json \
  --num-envs 1 \
  --num-episodes 1000 \
  --record-data \
  --record-path play_records/forge_gear_success \
  --record-env-index 0 \
  --record-step-interval 1 \
  --save-outcome success \
  --enable_cameras
```

JSON 保存 task 和 `action_dim`；回放会拒绝维度不匹配。即使维度相同，动作语义也
可能不同，所以生产时仍应为每个任务单独录 JSON。

两个 HandOver 不得混用：

- `Isaac-ViTac-Shadow-Hand-Over-Direct-v0`：standalone ShadowHand，走 action JSON。
- `Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0`：双 UR10e/双 ShadowHand，走 dual JSON。

### 9. 内置业务 phase

BlindGrasp 单臂 phase：

- phase 1：`blind_grasp` 第一分支；
- phase 2：`blind_grasp` 第二分支；
- phase 3：`blind_grasp_replay`。

```bash
for phase in 1 2 3; do
  python scripts/data_collection/full_trajectory/play_single_phase.py \
    --phase "$phase" \
    --num-envs 1 \
    --num-episodes 100 \
    --record-path "play_records/blind_single_phase_$phase" \
    --record-step-interval 1 \
    --save-outcome success \
    --enable_cameras
done
```

BiBlindBinDrop 双臂 phase：

```bash
for phase in 1 2 3 4; do
  python scripts/data_collection/full_trajectory/play_dual_phase.py \
    --phase "$phase" \
    --task bi_blind_bin_drop \
    --num-envs 1 \
    --num-episodes 100 \
    --record-path "play_records/bin_drop_phase_$phase" \
    --record-step-interval 1 \
    --save-outcome success \
    --enable_cameras
done
```

phase 脚本直接写 NPZ，不需要 `--record-data`。可用 `--keyframe-json` 覆盖内置
关键帧。

### 10. 手工 YAML 标定完整顺序

当前预设为 `pickup|pour|inhand`。`--env/--cfg` 可覆盖入口，但环境仍须兼容
单 UR10e + Shadow Hand joint/IK schema。

手臂和基础手型：

```bash
python scripts/data_collection/manual/record_arm_pose.py \
  --task pickup \
  --num_envs 1 \
  --save-yaml play_records/calibration/pickup_arm_hand.yaml \
  --save-full-ik-yaml play_records/calibration/pickup_full_ik.yaml \
  --show_rgb \
  --show_ff \
  --enable_cameras
```

快捷键：`p` 打印、`f` 锁定/解除、`s` 保存、`q` 退出。

抓取几何：

```bash
python scripts/data_collection/manual/record_grasp_pose.py \
  --task pickup \
  --num_envs 1 \
  --arm-yaml play_records/calibration/pickup_arm_hand.yaml \
  --closed-hand-yaml scripts/data_collection/manual/config/pickup_grasp.yaml \
  --save-yaml play_records/calibration/pickup_grasp_tuned.yaml \
  --enable_cameras
```

真实触觉下微调手型：

```bash
python scripts/data_collection/manual/record_handshape.py \
  --task pickup \
  --num_envs 1 \
  --arm-control marker \
  --pickup-ik-yaml play_records/calibration/pickup_full_ik.yaml \
  --closed-hand-yaml play_records/calibration/pickup_grasp_tuned.yaml \
  --save-yaml play_records/calibration/pickup_handshape_final.yaml \
  --show_rgb \
  --show_ff \
  --viewer matplotlib \
  --enable_cameras
```

快捷键：`f` 锁定、`p` 打印 24D、`s` 保存、`g` 跳过 approach helper、
`q` 退出。最终 hand YAML 应被 Full-IK/IK 配置引用。

### 11. 原始 observation 查看与逐步记录

`record_observations.py` 用于传感器检查和 zero/random/policy 单步采样，不组成成功
episode，也不自动过滤失败。

随机动作：

```bash
python scripts/data_collection/manual/record_observations.py \
  --task pickup \
  --num_envs 1 \
  --random_actions \
  --record_path play_records/observations/pickup/record \
  --record_format npz \
  --record_every 1 \
  --max_steps 1000 \
  --show_rgb \
  --show_ff \
  --enable_cameras
```

policy observation：

```bash
python scripts/data_collection/manual/record_observations.py \
  --task pickup \
  --num_envs 1 \
  --play \
  --resume_path /absolute/path/model_XXXX.pt \
  --record_path play_records/observations/pickup_policy/record \
  --record_format npz \
  --record_every 1 \
  --max_steps 1000 \
  --show_rgb \
  --show_ff \
  --enable_cameras
```

输出为 `record_step_XXXXXX.npz`。需要成功 trajectory 时应使用 episode collector。

### 12. Forge/Franka waypoint-IK

该入口不训练、不加载 checkpoint。它只复用 agent cfg 创建环境，然后调用 Forge/Factory
IK 接口。先用 `--save-outcome all --max-attempts 1` 检查 timeout/record 链路；调整
waypoint 直到 canonical success；生产时使用：

```bash
python scripts/data_collection/ik/play_waypoint_ik.py \
  --task Isaac-ViTac-Forge-GearMesh-Direct-v0 \
  --num_envs 1 \
  --save_data \
  --data_path play_records/waypoint_gear \
  --num_episodes 100 \
  --max_steps 400 \
  --waypoint_max_steps 100 \
  --save-outcome success \
  --enable_cameras
```

`attempt_XXXX_timeout.npz` 只能诊断，不能计入成功。

### 13. 31 个任务的 trajectory 分流

当前 31 个注册环境全部有 trajectory 入口，但不是由同一个 recorder 处理：

- 12 个单 UR10e + Shadow Hand：`record_single.py -> play_single.py`。
- 9 个双 UR10e + 双 Shadow Hand：`record_dual.py -> play_dual_phase.py --trajectory-json`。
- 6 个 Franka Forge、2 个 GelSight pretraining、Vision Repose、standalone
  ShadowHand-Over：`record_action.py -> play_action.py`。

`record_action.py` 在 action-space 层面也能打开其他 Box/Dict/Tuple 任务，但 30D/60D
UR10e 控制用 marker recorder 更容易制作可解释示范。完整逐任务 ID 见
`docs/ENVIRONMENT_MATRIX.md`。

“采集框架覆盖全部任务”不表示“一个 Pickup JSON 可直接用于全部任务”。每个任务/cfg
应有自己的 JSON、成功验证和版本号；跨任务轨迹只能作为待重新标定的模板。

### 14. 生产数采验收清单

1. `--help` 能正常退出，且不会误启动仿真。
2. 使用 `--enable_cameras` 创建真实 TacSL/GelSight。
3. observation/action 维度与 checkpoint 或 JSON 一致。
4. 可视化确认机器人、对象、触觉和第三人称相机时序合理。
5. 至少保存一个诊断 NPZ，检查所有数组 finite、首维时间长度一致。
6. 生产目录只接收 canonical success；completed/failed/timeout/partial 分目录或删除。
7. 每个 task、cfg、轨迹版本、checkpoint 和 seed 使用独立输出目录并保留 `meta.json`。
8. checkpoint 必须匹配 observation、action 语义、camera 顺序、tactile 数量和 policy family。

推荐目录：

```text
play_records/
  <task>/
    <collector_or_trajectory_version>/
      <checkpoint_or_script_version>/
        seed_<N>/
          meta.json
          episode_XXXX_success.npz
```

### 15. 辅助工具

统计 policy validation：

```bash
python scripts/data_collection/tools/count_validation_success.py \
  data/validation/<TASK>/<POLICY_FAMILY>
```

上传 `_success.npz`：

```bash
UPLOAD_SSH_HOST=user@host \
UPLOAD_SSH_PORT=22 \
REMOTE_DIR=/remote/dataset/task \
LOCAL_DIR1=/local/play_records/task \
bash scripts/data_collection/tools/upload_records.sh
```

`upload_records.sh` 只自动选择文件名以 `_success.npz` 结尾的数据。
`play_record.py` 当前输出的 `episode_N.npz` 已由 collector 保证成功，但不会被这一
glob 自动选中；上传前应使用经过校验的清单或统一命名流程。

## 完整参数与任务兼容性

- 每个脚本的全部显式 argparse 参数：`docs/SCRIPT_USAGE.md`
- 环境与数采方式对应关系：`docs/ENVIRONMENT_MATRIX.md`
- 远程逐项验证命令：`docs/VITACLAB_REMOTE_VALIDATION_CHECKLIST.md`

所有采集脚本还接受其调用的 Isaac `AppLauncher` 参数；常用参数包括
`--headless`、`--device`、`--enable_cameras`。RSL-RL collector 还接受
`scripts/common/rl/cli_args.py` 中列出的 checkpoint、resume 和日志参数。
