# RSL-RL 脚本快速开始

在 **已激活 Isaac Lab / Isaac Sim 的 Python 环境** 下，于 **仓库根目录** 执行。以下命令均使用 `python`（不使用 `./python`）。

> 若你使用 Isaac Lab 自带的 `./isaaclab.sh -p`，只需把 `python scripts/...` 换成  
> `./isaaclab.sh -p scripts/...`（等价于用 Isaac 环境里的解释器跑同一脚本）。

---

## 训练 vs 录制：建议的 `headless` / `enable_cameras`

| 场景 | `--headless` | `--enable_cameras` | 说明 |
|------|--------------|-------------------|------|
| **训练** | ✅ **打开**（推荐） | ❌ **关闭**（不要传 `--enable_cameras`） | 省显存与渲染开销；Direct 任务下不创建第三视角 / TacSL 等依赖渲染的传感器（见 `docs/enable_cameras_headless_rl.md`）。 |
| **录制 / 需要触觉或画面** | ❌ **关闭**（不要传 `--headless`，用带界面或离屏但需渲染的启动方式） | ✅ **打开** | 需要相机、触觉图像或录屏时，应显式 `--enable_cameras`；便于观察与保证传感器已创建。 |

录制 policy 轨迹（`--record_data`）若还要 **触觉 RGB / `--show_rgb`**，脚本会自行打开相机；仍建议 **不要 `--headless`**，除非你明确在无头环境下只存关节与标量、不需要可视化。

---

## 目录约定

| 路径 | 用途 |
|------|------|
| `scripts/rsl_rl/full_rl/train.py` | 全关节 RL 训练（臂 + 手） |
| `scripts/rsl_rl/full_rl/play.py` | 全关节策略回放 |
| `scripts/rsl_rl/ik_rl/train_ik_rl_single.py` | 仅手 + 差分 IK 臂训练 |
| `scripts/rsl_rl/ik_rl/play_ik_rl_single.py` | 同上，回放 / 录数据 |
| `scripts/rsl_rl/full_ik/train_full_ik_single.py` | **Full-IK**：`phase_schedule` 脚本阶段（开手 / GUI 手型等）+ 差分 IK 臂；与 `ik_rl` 代码独立 |
| `scripts/rsl_rl/full_ik/play_full_ik_single.py` | Full-IK 回放 / 录数据（需与训练相同的 `--full-ik-config`） |

---

## 1. 全关节训练（Pickup 示例）

```bash
python scripts/rsl_rl/full_rl/train.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 --max_iterations 10000
```

## 2. 全关节回放

将 `CHECKPOINT` 换成你的 `model_*.pt` 绝对路径，或配合 `--resume --load_run <时间戳目录名> --checkpoint model_XXXX.pt`：

```bash
python scripts/rsl_rl/full_rl/play.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 64 --device cuda:0 --checkpoint CHECKPOINT
```

---

## 3. IK 栈训练 — Pickup（默认会合并 `ik_rl_pickup.yaml`，若存在）

```bash
python scripts/rsl_rl/ik_rl/train_ik_rl_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 --max_iterations 10000
```

显式指定 IK 配置：

```bash
python scripts/rsl_rl/ik_rl/train_ik_rl_single.py \
  --ik-config scripts/rsl_rl/ik_rl/configs/ik_rl_pickup.yaml \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 256 --headless --device cuda:0
```

---

## 4. IK 栈训练 — Pour

```bash
python scripts/rsl_rl/ik_rl/train_ik_rl_single.py \
  --ik-config scripts/rsl_rl/ik_rl/configs/ik_rl_pour.yaml \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 --max_iterations 100000
```

限制每回合步数（例如只练阶段 1）：

```bash
python scripts/rsl_rl/ik_rl/train_ik_rl_single.py \
  --ik-config scripts/rsl_rl/ik_rl/configs/ik_rl_pour.yaml \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 \
  --max_episode_length 390 --max_iterations 100000
```

---

## 5. IK 栈训练 — In-hand（仅手；无 IK，expander 直通）

```bash
python scripts/rsl_rl/ik_rl/train_ik_rl_single.py \
  --ik-config none \
  --task Isaac-Repose-Cube-Shadow-Direct-v0 \
  --num_envs 128 --headless --device cuda:0
```

---

## 6. Full-IK 训练 — Pour / Pickup（默认合并 `full_ik_pour.yaml`）

**和第 3、4 节 `ik_rl` 一样**，回合仍按 **`--trajectory`** 分相位走（例如 Pickup：`object` 段先移到物体附近 / 对准抓取 → `goal` 段再搬向目标）；手臂始终是 **GPU 差分 IK** 跟锚点。

**`full_ik` 只改「抓握时手指由谁决定」**：在需要**闭合/定型抓握**的环节，用 **`phase_schedule`** 从开手切到 **GUI 录制的 `hand_yaml`**；若 **`freeze_hand_after_script: true`**，则之后 **手型保持该 YAML 不变**，后半段仍只靠 IK 动手臂（PPO 对手指常为 1 维占位）。**不是**从第一步起就与 `ik_rl` 完全不同的另一条时间线。

默认 Pour 用 **`full_ik_pour.yaml`**；若要 PPO 继续学习手指，在同一 YAML 里把 **`freeze_hand_after_script`** 设为 `false`。

**Pickup** 且与 `ik_rl` 同节奏、仅在抓握段换固定手型：用 **`full_ik_pickup_fixed_hand.yaml`**（同一入口 `train_full_ik_single.py`，只换配置）。

若 reset 后杯子位置随机、但手掌相对杯子的接近几何一致，可用 **`full_ik_pour_cup_relative.yaml`**（`mode: cup_relative`，CPU ikpy，建议 **`--num_envs` 较小**）。

默认读取 `scripts/rsl_rl/full_ik/configs/full_ik_pour.yaml`。**不要**传 `--full-ik-config none`，否则 `phase_schedule` 为空会报错。

```bash
python scripts/rsl_rl/full_ik/train_full_ik_single.py \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 --max_iterations 100000
```

显式指定 Full-IK 配置（与默认等价时仅作示例）：

```bash
python scripts/rsl_rl/full_ik/train_full_ik_single.py \
  --full-ik-config scripts/rsl_rl/full_ik/configs/full_ik_pour.yaml \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --num_envs 256 --headless --device cuda:0
```

随机杯位 XY + scripted 阶段在**杯坐标系**下接近/抓取（上文 `cup_relative`；YAML 内 `train_env_overrides.reset_cup_pos_noise` 可 >0）。每步有 CPU ikpy，**降低** `--num_envs`：

```bash
python scripts/rsl_rl/full_ik/train_full_ik_single.py \
  --full-ik-config scripts/rsl_rl/full_ik/configs/full_ik_pour_cup_relative.yaml \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --num_envs 16 --headless --device cuda:0 --max_iterations 100000
```

限制每回合步数（例如只练某段 IK 相位）：

```bash
python scripts/rsl_rl/full_ik/train_full_ik_single.py \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 \
  --max_episode_length 390 --max_iterations 100000
```

可选：再用 `--ik-config scripts/rsl_rl/ik_rl/configs/某.yaml` 或任意仅含 IK 键的 YAML **额外合并** palm / trajectory（键集与 `full_ik_load_config.IK_YAML_KEYS` 一致）；日常一般只维护 `--full-ik-config` 即可。

**录制位姿**：臂关节 YAML 用 `joint_pos`（或 `arm_joint_pos`）；手用 `hand_joint_pos_shadow_order`（24 维）。见各 `full_ik/configs/*.yaml` 注释。

### Full-IK — Pickup 示例（轨迹步数与 ik_rl_pickup 一致；中段 scripted 抓握 + 可选收敛门控）

`trajectory` 与 **`ik_rl_pickup.yaml`** 相同：`object:150:0,goal:-1:0`（第 150 步起切 `goal`）。`phase_schedule` 中两段 `ik_trajectory` 步数之和须为 150（默认 125 开手 + 25 抓握 `hand_yaml`），全程仍跟踪物体锚点直至进入 `goal`。若启用 `wait_hand_convergence_before_goal`，则关节到位后才允许从 `object` 切到 `goal`，避免未合拢就抬升。手型文件需存在，例如：

```bash
python scripts/rsl_rl/full_ik/train_full_ik_single.py \
  --full-ik-config scripts/rsl_rl/full_ik/configs/full_ik_pickup_fixed_hand.yaml \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 256 --headless --device cuda:0 --max_iterations 10000
```

---

## 7. Full-IK 回放 + 可选录轨迹

与 **`ik_rl/play_ik_rl_single.py`** 类似，但使用 **`play_full_ik_single.py`**，且 **必须** 带上与训练一致的 **`--full-ik-config`**（及相同 `--task`、palm/轨迹习惯）。

```bash
python scripts/rsl_rl/full_ik/play_full_ik_single.py \
  --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \
  --num_envs 64 --device cuda:0 \
  --full-ik-config scripts/rsl_rl/full_ik/configs/full_ik_pour.yaml \
  --checkpoint /path/to/model_XXXX.pt
```

Pickup + 与训练相同的固定手型配置：

```bash
python scripts/rsl_rl/full_ik/play_full_ik_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 64 --device cuda:0 \
  --full-ik-config scripts/rsl_rl/full_ik/configs/full_ik_pickup_fixed_hand.yaml \
  --checkpoint /path/to/model_XXXX.pt
```

---

## 8. IK 栈回放 + 可选录轨迹

```bash
python scripts/rsl_rl/ik_rl/play_ik_rl_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 64 --device cuda:0 \
  --checkpoint /path/to/model_XXXX.pt
```

录制 `policy` 观测与手部动作等到 `play_records/...`（**建议开启 `--enable_cameras`，且不要加 `--headless`**）：

```bash
python scripts/rsl_rl/ik_rl/play_ik_rl_single.py \
  --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \
  --num_envs 64 --device cuda:0 \
  --enable_cameras \
  --checkpoint /path/to/model_XXXX.pt \
  --record_data --record_max_episodes 5 --max_play_steps 20000
```

---

更全的参数说明（含 `full_ik` 与 `ik_rl` 差异）见同目录 **`README.md`**。
