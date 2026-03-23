# `scripts/rsl_rl` — RSL-RL 训练与回放

本目录提供两类入口：

1. **`full_rl/`** — **全关节**策略：UR10e 臂 + Shadow Hand 同时由 RL 输出（与常见 Isaac Lab `train.py` 一致）。
2. **`ik_rl/`** — **仅手部**策略 + **差分 IK** 驱动机械臂 + **脚本化掌心轨迹**（跟物体 / 目标锚点）。适用于 Pickup、Pour；**In-hand**（仅手）时 expander 会退化为直通、不跑 IK。

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
└── ik_rl/
    ├── train_ik_rl_single.py
    ├── play_ik_rl_single.py
    ├── configs/
    │   ├── ik_rl_pickup.yaml
    │   ├── ik_rl_pour.yaml
    │   └── README.md
    └── utils/
        ├── cli_args.py
        ├── ik_rl_hand_vec_env.py
        ├── ik_rl_load_config.py
        └── rsl_rl_log_utils.py
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

在 **`full_rl/train`** 的共有参数之外，增加 **手掌 / IK / 轨迹** 相关项；并与 **`ik_rl/configs/*.yaml`** 合并（CLI 优先）。

### IK 配置文件

| 参数 | 说明 |
|------|------|
| `--ik-config PATH` | 指定 YAML；`none` / `false` 表示不读文件。 |
| （默认） | 若未指定且存在 `ik_rl/configs/ik_rl_pickup.yaml`，会作为默认合并（用于 Pickup 习惯 workflow）。Pour 建议 **显式** `--ik-config scripts/rsl_rl/ik_rl/configs/ik_rl_pour.yaml`。 |

YAML 内常见键与 CLI 同名：`task`、`trajectory`、`object_to_palm_offset`、`palm_in_wrist_pos`、`palm_in_wrist_euler`、`palm_orient`、`palm_normal_local`、`palm_yaw_offset`、`world_down`、`palm_euler`、`palm_euler_in_anchor`、`ee_body`、`ik_method`、`ik_lambda`。详见 `ik_rl/configs/README.md` 与各 yaml 注释。

### 轨迹 `--trajectory`

格式：逗号分隔多段，每段 `名称:环境步数:是否用旋转`，例如：

`object:150:0,goal:-1:0`

- **名称**：环境中锚点——刚性/可变形资产的 `env.<name>`，或 `env.<name>_pos` / `<name>_rot`，或 legacy `goal`。
- **步数**：`-1` 表示直到本回合结束。
- **use_rotation**：`0` 或 `1`；`1` 时在锚点坐标系内施加偏移并对齐掌心姿态。

### 手掌与 IK（与 YAML 键一一对应，CLI 为 `--kebab-case`）

| CLI | 含义 |
|-----|------|
| `--object-to-palm-offset OX OY OZ` | 锚点到掌心原点的偏移（米）；是否随锚点旋转由当前段 `use_rotation` 决定。 |
| `--palm-in-wrist-pos` | 掌心在 `wrist_3` 坐标系中的位置。 |
| `--palm-in-wrist-euler` | 掌心相对腕的欧拉角（弧度）。 |
| `--palm-orient` | `fixed` 或 `pickup_down`（手掌朝下抓取类姿态）。 |
| `--palm-normal-local` | `pickup_down` 时掌心法向。 |
| `--palm-yaw-offset` | 世界系绕竖直轴额外 yaw（弧度）。 |
| `--world-down` | 世界“向下”方向。 |
| `--palm-euler` | `palm-orient fixed` 时掌心世界欧拉角。 |
| `--palm-euler-in-anchor` | 段内 `use_rotation=1` 时相对锚点的欧拉角。 |
| `--ee-body` | Jacobian 末端 link，默认 `wrist_3_link`。 |
| `--ik-method` | `pinv` / `svd` / `trans` / `dls`。 |
| `--ik-lambda` | `dls` 阻尼 λ；默认用控制器内置。 |

### 仅 IK 训练脚本特有

| 参数 | 说明 |
|------|------|
| `--max_episode_length` | 用 RL **步数**覆盖回合长度（内部换算 `episode_length_s = steps * sim.dt * decimation`）。 |

---

## `ik_rl/play_ik_rl_single.py` — 额外参数

继承 `train_ik_rl_single` 的 palm/IK/trajectory/`ik-config`，并增加：

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

## 任务 ID 参考（ViTacLab 常见）

| 任务 | Gym ID |
|------|--------|
| UR10e 抓取 | `Isaac-UR10eShadowHand-Pickup-Direct-v0` |
| UR10e 倒水（可形变杯） | `Isaac-UR10eShadowHand-PourDeformable-Direct-v0` |
| 掌中立方（仅手） | `Isaac-Repose-Cube-Shadow-Direct-v0` 等（见 `shadow_hand/__init__.py`） |

---

## 日志与 checkpoint 路径

`rsl_rl_log_utils.get_rsl_rl_log_root` 决定日志根目录（通常含任务名）。训练产生的 `model_*.pt` 位于对应时间戳子目录下；**IK 与 full 的实验不要混用 checkpoint**（动作维度与控制方式不同）。

---
