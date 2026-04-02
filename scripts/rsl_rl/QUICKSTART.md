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
| `scripts/rsl_rl/full_ik/train_full_ik_single.py` | **Full-IK**：脚本关节 pregrasp / grasp + 差分 IK 臂 + 仅手策略（当前面向 Pour） |

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

## 6. Full-IK 训练 — Pour（默认合并 `full_ik_pour.yaml`）

与 **第 3、4 节 IK 栈** 的区别：`full_ik` 在每回合开头按 YAML 里的 `phase_schedule` **强设**臂 + 手关节（如 GUI 导出的 pregrasp / grasp），之后由 **GPU 差分 IK** 跟踪 `--trajectory`（`cup` / `goal_cup` 等）。默认 `full_ik_pour.yaml` 里 **`freeze_hand_after_script: true`**：脚本段结束后 **手指固定为 grasp YAML**，不再由 PPO 控手（策略为 1 维占位动作）；若要练手指，在同一 YAML 中设为 `false`。若 reset 后杯子位置仍要随机、但手掌相对杯子的接近/抓握几何一致，使用 **`scripts/rsl_rl/full_ik/configs/full_ik_pour_cup_relative.yaml`**（`mode: cup_relative`，CPU ikpy，建议 `--num_envs` 较小）。

默认会读取 `scripts/rsl_rl/full_ik/configs/full_ik_pour.yaml`（含 `phase_schedule` 与 palm / IK / trajectory）；可用 `--full-ik-config` 换路径。**不要**传 `--full-ik-config none`，否则 `phase_schedule` 为空脚本会报错。

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

可选：再用 `--ik-config scripts/rsl_rl/ik_rl/configs/某.yaml` **额外合并** palm / trajectory 等字段（与 `train_ik_rl_single.py` 的 `IK_YAML_KEYS` 一致）；日常 Pour 一般只改 `full_ik_pour.yaml` 即可。

**录制位姿**：臂关节 YAML 用 `joint_pos`（或 grasp 记录里的 `arm_joint_pos`）；手用 `hand_joint_pos_shadow_order`（24 维）。各阶段 `arm_yaml` 应对齐（pregrasp 与 grasp 记录同一臂姿来源），见 `full_ik_pour.yaml` 内注释。

---

## 7. IK 栈回放 + 可选录轨迹

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

更全的参数说明见同目录 **`README.md`**。
