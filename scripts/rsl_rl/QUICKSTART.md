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

## 6. IK 栈回放 + 可选录轨迹

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
