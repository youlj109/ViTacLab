# IK + 手掌轨迹预设说明（`ik_rl_pickup.yaml` / `ik_rl_pour.yaml`）

本文档说明 `train_ik_rl_single.py` / `play_ik_rl_single.py` 通过 `--ik-config` 加载的 YAML 中各字段含义，便于从**使用者**角度理解坐标系与调参思路。

---

## 1. 两个文件分别对应什么任务？

| 文件 | 典型环境（`task`） | 用途简述 |
|------|-------------------|----------|
| `ik_rl_pickup.yaml` | `Isaac-UR10eShadowHand-Pickup-Direct-v0` | 抓取刚体物体：先跟 `object`，再跟 `goal`（放置目标）。 |
| `ik_rl_pour.yaml` | `Isaac-UR10eShadowHand-PourDeformable-Direct-v0` | 倒水：可变形杯 `cup`，目标用张量 `goal_cup`（`goal_cup_pos` / `goal_cup_rot`）。 |

- **Pickup**：若未显式传 `--ik-config`，且仓库里存在 `ik_rl_pickup.yaml`，脚本会**默认**尝试加载它（见 `default_pickup_ik_yaml_path()`）。
- **Pour**：必须显式指定，例如  
  `--ik-config scripts/rsl_rl/ik_rl/configs/ik_rl_pour.yaml`  
  （注意路径在 `ik_rl/configs/` 下，不要写成 `scripts/rsl_rl/configs/...`。）

YAML 中的键与命令行参数一一对应（下划线命名）；**命令行**传入的同名参数会覆盖 YAML。

---

## 2. `task`

- **含义**：Gym 注册的完整任务 ID，用于 Hydra 拉取环境与策略配置。
- **使用**：若未在命令行写 `--task`，合并逻辑会优先用 YAML 里的 `task`。

---

## 3. `trajectory`（分段轨迹）

格式为逗号分隔的若干段，每段：

```text
<anchor 名称>:<持续步数>:<use_rotation>
```

- **`anchor 名称`**：在环境里解析「当前段手掌要跟谁」的**世界系锚点**位姿，见下表。
- **持续步数**：该段持续的**环境步数**（`env.step` 计数）；`-1` 表示一直持续到本回合结束或重置。
- **`use_rotation`**：仅 `0` 或 `1`。
  - **`0`**：`object_to_palm_offset` 在**世界系**理解；手掌朝向由 `palm_orient`（`fixed` / `pickup_down`）等决定，**不**随锚点旋转一起转（除非用 `pickup_down` 等与 world 对齐的逻辑）。
  - **`1`**：`object_to_palm_offset` 在**锚点坐标系**里应用；手掌朝向为 **锚点姿态 × `palm_euler_in_anchor` 对应的旋转**（见 `ik_rl_hand_vec_env.py` 中 `TrajectoryPhase` 与 `palm_euler_in_anchor`）。

**锚点名称如何解析（概念上）**：

1. 若环境存在 `env.<名称>` 且为资产/物体，有 `root_pos_w` / `root_quat_w`，则用该锚点。
2. 否则若存在 `env.<名称>_pos` 与 `env.<名称>_rot`，用「环境局部位置 + 原点」与张量旋转（如 `goal_cup`）。
3. 若名为 `goal` 且上面失败，可退化到 `goal_object_pos` / `goal_object_rot`（旧接口）。

**预设示例**：

- **Pickup**：`object:150:0,goal:-1:0` — 先跟物体 150 步（不跟锚点转），再跟目标直到结束。
- **Pour**：`cup:200:0,goal_cup:200:0,goal_cup:-1:1` — 前两段 `use_rotation=0`，最后一段 `goal_cup` 且 `use_rotation=1`，适合「杯口对准目标方向」等需要跟锚点旋转的阶段。

调参时：先确认环境里**确实有** `object` / `cup` / `goal_cup` 等名称，否则轨迹解析会失败或行为不符合预期。

---

## 4. 坐标系与变换链（使用者视角）

整体链路（概念）：

1. **锚点**：世界系下的位置 + 四元数（`pos_w`, `quat_w`）。
2. **手掌原点**：在锚点旁加 `object_to_palm_offset`（世界系或锚点系，取决于当前段 `use_rotation`）。
3. **手掌姿态**：`use_rotation=0` 时用 `fixed` 或 `pickup_down`；`use_rotation=1` 时用锚点四元数 × `palm_euler_in_anchor`。
4. **手腕 `wrist_3`**：由「手掌在腕系下的固定外参」`palm_in_wrist_pos` + `palm_in_wrist_euler` 反推手腕位姿，再交给 differential IK 求臂关节。

下面按字段说明。

### 4.1 `object_to_palm_offset` `[x, y, z]`（米）

- 从**锚点**到**手掌原点**的平移。
- **当 `use_rotation=0`**：在**世界系**里加（与 `train_ik_rl_single.py` 帮助一致）。
- **当 `use_rotation=1`**：在**锚点坐标系**里加（与 `IkRlHandArmCfg` 注释一致）。

**调参**：先抓物体/抓杯时，通常希望掌心在物体上方或侧面：在仿真里看物体与手掌的相对位置，把期望偏移量（米）写进这里。Pickup 常用略偏 Z（如 `0,0,0.05`）；Pour 里可能需要在物体/杯局部系里偏一侧（如 `0, 0.05, 0`）。

### 4.2 `palm_in_wrist_pos` / `palm_in_wrist_euler`（米 / 弧度）

- **含义**：**手掌坐标系**相对于 **UR10e 末端连杆 `ee_body`（默认 `wrist_3_link`）** 的固定外参：先平移再 `euler xyz` 旋转。
- 来自机器人 URDF/USD 与 Shadow Hand 安装关系；**一般**在换末端工具或改手安装方式时才需要大改。

**调参**：若整臂姿态整体对但手掌与腕有系统性偏差，可微调这两个量；否则保持默认或与标定一致。

### 4.3 `palm_orient`：`fixed` | `pickup_down`

仅当**当前段** `use_rotation=0` 时生效。

- **`pickup_down`**：用 `palm_normal_local` 与 `world_down` 对齐（`scipy` 对齐向量），再绕世界 Z 加 `palm_yaw_offset`，得到手掌欧拉角（`pickup」类抓取姿态）。
- **`fixed`**：直接使用 `palm_euler`（世界系欧拉 `xyz`，弧度）。

### 4.4 `palm_normal_local` 与 `world_down`（单位向量）

- **`palm_normal_local`**：在**手掌坐标系**里，你希望「指向世界重力反方向」的那根轴（通常与掌心法向相关）。Pickup 默认常指向「手掌 Y」等；Pour 里可能改为 `0,0,1` 以适配杯口朝向。
- **`world_down`**：世界系下的「向下」方向，一般为 `(0,0,-1)`（Isaac 世界 Z 向上时）。

**调参**：若抓取时手心朝向反了，优先调整 `palm_normal_local` 与 `palm_yaw_offset`，而不是乱改 `palm_in_wrist_euler`。

### 4.5 `palm_yaw_offset`（弧度）

- 在 `pickup_down` 对齐之后，**绕世界 Z 轴**再转的角度。
- Pour 与 Pickup 的预设不同（例如 Pour 常为 `π`），用于把「杯柄/杯口」转到任务需要的朝向。

### 4.6 `palm_euler`（弧度）

- 在 `palm_orient: fixed` 时使用，**世界系** `xyz` 欧拉角。

### 4.7 `palm_euler_in_anchor`（弧度）

- 在 `use_rotation=1` 时用：**锚点四元数 × 由该欧拉角得到的旋转** 得到手掌姿态。
- 用于在**跟随锚点旋转**的同时，再微调手掌相对锚点的姿态（例如杯口对准目标）。

---

## 5. 微分 IK 相关

| 字段 | 含义 |
|------|------|
| `ee_body` | IK 使用的末端连杆名，默认 `wrist_3_link`，需与机器人模型一致。 |
| `ik_method` | `pinv` / `svd` / `trans` / `dls`，默认 `dls`。 |
| `ik_lambda` | DLS 阻尼；`null` 表示用控制器默认。 |

---

## 6. 实操建议（如何确定参数）

1. **先固定任务与轨迹名**：在环境里确认 `object` / `cup` / `goal_cup` 等存在，再写 `trajectory`。
2. **先调平移**：只改 `object_to_palm_offset`，让手掌大致在物体/杯口附近。
3. **再调朝向**：
   - `use_rotation=0`：用 `pickup_down` + `palm_normal_local` + `palm_yaw_offset`，或 `fixed` + `palm_euler`。
   - `use_rotation=1`：用 `palm_euler_in_anchor` 微调相对锚点的姿态。
4. **外参**：`palm_in_wrist_*` 与真实机器人安装一致时尽量少动。
5. **用 `--trajectory` 等 CLI 覆盖**做快速试验，满意后再写回 YAML。

---

## 7. 参考代码

- 轨迹解析与 `use_rotation` 语义：`scripts/rsl_rl/ik_rl/utils/ik_rl_hand_vec_env.py`（`TrajectoryPhase`、`IkRlHandArmCfg`）。
- YAML 键与 `argparse` 合并：`scripts/rsl_rl/ik_rl/utils/ik_rl_load_config.py`。
