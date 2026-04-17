# `scripts/rsl_rl` — RSL-RL 训练与回放

本目录提供三类入口：

1. **`full_rl/`** — **全关节**策略：UR10e 臂 + Shadow Hand 同时由 RL 输出（与常见 Isaac Lab `train.py` 一致）。
2. **`ik_rl/`** — **仅手部**策略 + **差分 IK** 驱动机械臂 + **脚本化掌心轨迹**（跟物体 / 目标锚点）。适用于 Pickup、Pour；**In-hand**（仅手）时 expander 会退化为直通、不跑 IK。
3. **`full_ik/`** — 与 `ik_rl` **同一套回合节奏**：仍是「**仅手策略位 + GPU 差分 IK 臂 + `--trajectory` 分段**」（例如 Pickup：`object` 段接近物体 → `goal` 段搬向目标）。**差别只在抓握相关环节**：在需要闭合/定型手型的时刻，用 GUI 录制的 **`hand_yaml` 固定手型**，并可用 **`freeze_hand_after_script`** 在后续步数里**保持该手型不变**，由 IK 继续驱动手臂完成后半段；**不是**整回合都与 `ik_rl` 不同。Pour 任务同理，用 `phase_schedule` 描述「接近 → 抓手型 → 倒出」等阶段。**实现独立**：不 import `ik_rl`，IK 核心在 `full_ik/utils/full_ik_arm_ik_expander.py`。

**环境要求**：须在 **已安装 Isaac Lab / Isaac Sim** 的 Python 中运行。文档示例统一写为：

```bash
python scripts/rsl_rl/...
```

若你使用项目自带的 `./isaaclab.sh -p`，把 `python` 换成 `./isaaclab.sh -p` 即可。

### 训练 vs 录制：`headless` 与 `enable_cameras`（建议）

- **训练**：优先 **`--headless`**，且 **不要** 传 `--enable_cameras`（除非你要在训练里开相机/触觉做调试）。可减少渲染与传感器开销。
- **录制数据、回放并需要触觉/画面**：优先 **`--enable_cameras`**，且 **不要** 使用 `--headless`（便于观察，并确保依赖渲染的传感器按预期创建）。若仅用 `--record_data` 存关节与标量、且确认无头环境足够，可再酌情 headless。

与任务侧 `enable_cameras` 门控的说明见仓库根目录 **`docs/enable_cameras_headless_rl.md`**。

**团队修改规范（场景 USD / 奖励 / 录制与 IK 轨迹等）**：见 **`docs/ik_rl_modification_guide.md`**。

---

## 目录结构

```
scripts/rsl_rl/
├── README.md                 # 本文件
├── QUICKSTART.md             # 一键命令速查
├── full_rl/
│   ├── train.py
│   └── play.py
├── ik_rl/
│   ├── train_ik_rl_single.py
│   ├── play_ik_rl_single.py
│   ├── configs/
│   │   ├── ik_rl_pickup.yaml
│   │   ├── ik_rl_pour.yaml
│   │   └── README.md
│   └── utils/
│       ├── cli_args.py
│       ├── ik_rl_hand_vec_env.py
│       ├── ik_rl_load_config.py
│       └── rsl_rl_log_utils.py
└── full_ik/
    ├── train_full_ik_single.py
    ├── play_full_ik_single.py
    ├── configs/
    │   ├── full_ik_pour.yaml
    │   ├── full_ik_pour_cup_relative.yaml
    │   └── full_ik_pickup_fixed_hand.yaml
    └── utils/
        ├── full_ik_arm_ik_expander.py   # 与 ik_rl 中 ArmIkHandActionExpander 等价的 vendored 副本
        ├── full_ik_hand_vec_env.py    # 分阶段脚本 + Phased wrapper
        └── full_ik_load_config.py       # full_ik 专用 YAML 合并（IK 键与 ik_rl 对齐）
```

---

## Hydra 与参数顺序

脚本先用 `argparse` 解析 **前半段参数**，再把剩余参数交给 **Hydra**（任务与 agent 的 YAML）。因此：

- **脚本自有参数**（`--task`、`--num_envs`、`--trajectory` 等）须写在 **最前面**（在 Hydra 的 `key=value` 之前）。
- **环境 / Agent 的 Hydra 覆盖** 写在后面，例如：  
  `env.scene.num_envs=128` `agent.seed=42`

示例：

```bash
python scripts/rsl_rl/full_rl/train.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 \
  env.some_field=value
```

具体可覆盖字段以各任务 `EnvCfg` / agent 注册名为准（见 Isaac Lab / ViTacLab 任务包）。

---

## `full_rl/train.py` — 参数说明

| 参数 | 说明 |
|------|------|
| `--task` | 注册的 Gym 环境 ID（必填）。 |
| `--agent` | Agent 配置入口名，默认 `rsl_rl_cfg_entry_point`。 |
| `--num_envs` | 并行环境数；覆盖 Hydra 默认。 |
| `--seed` | 随机种子。 |
| `--max_iterations` | PPO 等训练迭代次数。 |
| `--video` | 训练时录屏（会打开相机相关开销）。 |
| `--video_length` / `--video_interval` | 与录屏片段长度、间隔（步数）。 |
| `--distributed` | 多 GPU / 多节点（需正确配置进程与设备）。 |
| `--export_io_descriptors` | 导出 IO 描述符（Manager 类环境等）。 |

**`cli_args` 组（与 `ik_rl` 共用）**

| 参数 | 说明 |
|------|------|
| `--experiment_name` | 日志根目录下的实验名子路径。 |
| `--run_name` | 单次 run 目录后缀。 |
| `--resume` | 从 checkpoint 继续训。 |
| `--load_run` | 日志目录下某次 run 的文件夹名。 |
| `--checkpoint` | checkpoint 文件名，如 `model_1000.pt`。 |
| `--logger` | `wandb` / `tensorboard` / `neptune`。 |
| `--log_project_name` | 外部 logger 的项目名。 |

**AppLauncher**（Isaac Sim）

常见：`--headless`、`--device cuda:0`、`--enable_cameras` 等；以 `AppLauncher.add_app_launcher_args` 为准。

---

## `full_rl/play.py` — 额外参数

在 `train` 基础上常见：

| 参数 | 说明 |
|------|------|
| `--use_pretrained_checkpoint` | 使用 Nucleus 等发布的预训练权重（按 Isaac 任务逻辑）。 |
| `--real-time` | 尽量实时仿真。 |
| `--disable_fabric` | 关闭 Fabric，走 USD I/O（视版本而定）。 |

Checkpoint：通常 `--checkpoint` 为 **绝对路径** 的 `model_*.pt`，或与 `--resume`、`--load_run` 组合解析到 `logs/rsl_rl/<task>/...`。

---

## `ik_rl/train_ik_rl_single.py` — 参数说明

在 **`full_rl/train`** 的共有参数之外，增加 **IK / 末端轨迹**；并与 **`ik_rl/configs/*.yaml`** 合并（CLI 覆盖 YAML）。

### IK 配置文件

| 参数 | 说明 |
|------|------|
| `--ik-config PATH` | 指定 YAML；`none` / `false` 表示不读文件。 |
| （默认） | 若未指定且存在 `ik_rl/configs/ik_rl_pickup.yaml`，会作为默认合并。Pour 建议 **显式** `--ik-config scripts/rsl_rl/ik_rl/configs/ik_rl_pour.yaml`。 |

YAML 内常见键：`task`、`trajectory`（**列表**），双臂另有 `trajectory_right` / `trajectory_left`；以及 `ee_body`、`ik_method`、`ik_lambda`。详见 `ik_rl/configs/README.md`。

**轨迹**：`trajectory` 为若干段，每段 `{pos: [x,y,z], quat: [w,x,y,z], steps: int}` — `quat` 为 **Isaac Lab wxyz**（标量在前）；world 系下 **EE 连杆**（默认 `wrist_3_link`）目标位姿；`steps` 为环境步数；`-1` 表示持续到回合结束。若环境在 reset 时写入 `ik_rl_trajectory_xyz_offset`（例如 bottle 位置随机），则该偏移会**加在**所有段的 `pos` 上。

### 可选 CLI（覆盖 YAML）

| CLI | 含义 |
|-----|------|
| `--ee-body` | EE link 名，默认 `wrist_3_link`。 |
| `--ik-method` | `pinv` / `svd` / `trans` / `dls`。 |
| `--ik-lambda` | `dls` 阻尼 λ；默认用控制器内置。 |

### 仅 IK 训练脚本特有

| 参数 | 说明 |
|------|------|
| `--max_episode_length` | 用 RL **步数**覆盖回合长度（内部换算 `episode_length_s = steps * sim.dt * decimation`）。 |

---

## `ik_rl/play_ik_rl_single.py` — 额外参数

继承 `train_ik_rl_single` 的 IK/trajectory/`ik-config`，并增加：

| 参数 | 说明 |
|------|------|
| `--max_play_steps` | 跑满 N 步后退出；`0` 表示不限制（直到关窗口或 Ctrl+C）。 |
| `--play_success_interval` | 打印成功率等日志的间隔（秒）。 |
| `--show_rgb` / `--show_ff` | 触觉 RGB / 力场可视化（会开相机）。 |
| `--env_index` | 可视化用的并行环境下标。 |
| `--fps` | 触觉显示目标帧率。 |
| `--record_data` | 将 `policy` 观测、手部动作、`reward`、`done` 写入 `play_records/...` 下 `episode_XXXX.npz`。 |
| `--record_path` | 输出目录；默认 `./play_records/<task>_<时间>/`。 |
| `--record_env_index` | 记录哪一路并行环境。 |
| `--record_max_episodes` | 成功保存满 N 个完整 episode 后退出。 |

Checkpoint 解析逻辑与训练一致：可直接 **`--checkpoint /绝对路径/model.pt`**，或使用 **`--resume --load_run <目录名> --checkpoint model_XXXX.pt`** 在 `logs/rsl_rl/<task>/` 下解析。

---

## `full_ik/train_full_ik_single.py` / `full_ik/play_full_ik_single.py`

与 `ik_rl` 共用 **Hydra、`cli_args`、AppLauncher** 等用法；**手掌 / IK / `--trajectory`** 的 CLI 含义与 `train_ik_rl_single` 相同。

### 回合步骤（以 Pickup 为例，与 `ik_rl` 对齐）

- **`--trajectory`** 与 `ik_rl_pickup` 类似：`object:N` 段内锚在 **物体**上，`goal` 段跟目标完成后半段。`ik_rl` 常用 `N=150`；**`full_ik_pickup_fixed_hand.yaml` 里 `N` 必须 ≥ `phase_schedule` 里「仍在物体旁抓握」的步数之和**（例如先开手接近 150 步，再 **多步**保持 `hand_yaml` 抓握且仍跟 `object`，然后才进入 `goal`）。若 `N` 与脚本阶段对齐错误，会在 **切入 `goal` 抬升的同时** 才切换手型，出现「边上升边闭合」。
- **`full_ik` 的特化**：在 **`phase_schedule`** 里「接近」段 **开手**，随后一段 **在物体锚点未结束前** 切换到 **`hand_yaml`** 并保持若干步，让抓握在 **仍在 `object` 段** 时完成；**`freeze_hand_after_script`** 则让之后 **`goal`** 段手指固定、只由 IK 动手臂。与 `ik_rl` 的 `hand_freeze_phase_target` + `hand_freeze_yaml` 同类目的，`full_ik` 用 `phase_schedule` 表达时机。
- **Pour**：逻辑类似——接近杯、再施加抓取手型、再倒出等，具体步数与模式见 `full_ik_pour.yaml` 等配置。

### `--full-ik-config`（主配置）

| 参数 | 说明 |
|------|------|
| `--full-ik-config PATH` | 默认 `full_ik/configs/full_ik_pour.yaml`。含 `phase_schedule`（脚本阶段：关节回放、`ik_trajectory` 开手/抓手型、`cup_relative` 等）、`freeze_hand_after_script`、`freeze_hand_yaml`、`cup_relative_stable_cup_rotation`、`train_env_overrides` 等。 |
| `none` / `false` / 空 | 不读文件 → **`phase_schedule` 为空会报错**（与 ik_rl 不同，full_ik 依赖分阶段脚本）。 |

可选 **`--ik-config PATH`**：仅合并 **IK 键**（`trajectory`、`palm_*`、`ee_body`、`ik_method` 等，见 `full_ik_load_config.IK_YAML_KEYS`），**不**替代 `phase_schedule`；与训练时一致即可。

**手部收敛门控（可选，YAML）**：`wait_hand_convergence_before_goal: true` 时，在 **`trajectory` 第一段结束、即将切入带 `env_steps:-1` 的段** 时，若手指关节尚未到达 `phase_schedule` 里最后一个 `hand_yaml` 的关节目标，则 **IK 仍按上一段的物体锚点** 计算，直到 `max|q-q*| ≤ hand_convergence_pos_tol_rad` 或超过 `hand_convergence_max_hold_steps`（避免死等）。与带 `TimeOffset` 的 scripted 臂阶段（`joint_targets` / `cup_relative`）**不兼容**，该情况下会自动关闭门控并打日志。

### 训练日志

单次 run 的 `params/` 下会写入 **`full_ik_hand.yaml`**（记录轨迹、palm、是否冻结手型等），**不是** `ik_rl_hand.yaml`。

### 回放 `play_full_ik_single.py`

- 必须使用与训练 **相同的 `--full-ik-config`**（及一致的 `--task`、palm/trajectory），否则动作展开与训练不一致。
- 其余与 `ik_rl/play_ik_rl_single.py` 相同：`--max_play_steps`、`--record_data`、`--show_rgb` / `--show_ff` 等。

---

## 任务 ID 参考（ViTacLab 常见）

| 任务 | Gym ID |
|------|--------|
| UR10e 抓取 | `Isaac-UR10eShadowHand-Pickup-Direct-v0` |
| UR10e 倒水（可形变杯） | `Isaac-UR10eShadowHand-PourDeformable-Direct-v0` |
| 掌中立方（仅手） | `Isaac-Repose-Cube-Shadow-Direct-v0` 等（见 `shadow_hand/__init__.py`） |

---

## 日志与 checkpoint 路径

`rsl_rl_log_utils.get_rsl_rl_log_root` 决定日志根目录（通常含任务名）。训练产生的 `model_*.pt` 位于对应时间戳子目录下；**`full_rl`、`ik_rl`、`full_ik` 之间不要混用 checkpoint**（动作维度与控制栈不同）。

---
