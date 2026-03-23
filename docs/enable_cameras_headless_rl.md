# `enable_cameras` 与 Headless 强化学习

**实践约定（与 `scripts/rsl_rl/QUICKSTART.md` 一致）**

- **训练**：尽量 **`--headless`**，且 **不传** `--enable_cameras`，以省资源。
- **录制 / 需要相机或触觉**：尽量 **`--enable_cameras`**，且 **不用** `--headless`（除非确定无头即可）。

---

ViTacLab 在 `source/ViTacLab/ViTacLab/tasks/direct` 下的 Direct 任务中，**仅在 `enable_cameras` 为真时**才创建第三视角相机、触觉（TacSL / Forge 双指触觉）等依赖渲染的传感器，从而在 **`scripts/rsl_rl/full_rl/train.py` 默认 headless 训练** 时减少开销。

## 行为说明

### 1. RSL-RL 脚本（`train.py` / `play.py`）

在 `gym.make(...)` **之前** 设置：

```text
env_cfg.enable_cameras = (CLI --enable_cameras) OR (环境变量 ENABLE_CAMERAS=1)
```

与 `AppLauncher` 的相机/渲染开关保持一致。

### 2. `UR10eShadowHandDirectBaseEnv`

仅当 `cfg.enable_cameras` 为真时：

- 创建第三视角 `TiledCamera`
- 在 `UR10eShadowHandTacSLSceneCfg` 场景下创建 TacSL 触觉传感器

### 3. `ForgeEnv` / `ForgeEnvCfg`

- 配置项：`enable_cameras: bool = False`（默认关闭，适合 headless）。
- 当 `enable_cameras` 为 **False** 时：
  - 将 `cfg.scene` 替换为仅含 `InteractiveSceneCfg` 基础字段的配置，**不在** `InteractiveScene` 中生成 Forge 的触觉传感器与第三视角相机。
  - 若此时 `obs_mode == "full"`，会发出 **warning** 并 **强制改为 `"reduce"`**（完整触觉观测需要相机）。
- `_compute_intermediate_values` 在无传感器时对触觉相关张量使用 **零填充**。

### 4. UR10e 相关任务配置

以下 cfg 中增加了 `enable_cameras: bool = False` 作为默认值与文档说明；实际训练时由 `train.py` / `play.py` 按 CLI / 环境变量覆盖：

- `inhand_manipulation_env_cfg.py`（`UR10eShadowHandInHandEnvCfg` 及子类）
- `hand_pickup_env_cfg.py`
- `ur10e_shadowhand_pour_env_cfg.py`

### 5. `ShadowHandVisionEnv`（视觉 + CNN）

- `ShadowHandVisionEnvCfg.enable_cameras: bool = True`（该任务**默认需要** tiled 相机）。
- 若显式将 `enable_cameras` 设为 **False**，在环境 `__init__` 中会 **直接报错**（无法在无相机下完成视觉观测管线）。
- `_setup_scene` 仅在 `enable_cameras` 为真时注册 `tiled_camera`。

## 使用提示

| 场景 | 建议 |
|------|------|
| **默认 headless 训练** | 不传 `--enable_cameras`，不设 `ENABLE_CAMERAS` → 不创建上述相机/触觉；Forge 使用 `obs_mode="reduce"`（cfg 默认已是）。 |
| **需要触觉或第三视角调试** | 使用 `--enable_cameras` 或 `ENABLE_CAMERAS=1`；Forge 若要用完整触觉观测，需 `obs_mode="full"`。 |
| **训练过程录视频** | `--video` 仍会打开 `enable_cameras`（与原先行为一致）。 |

## 相关代码路径（便于检索）

- `scripts/rsl_rl/full_rl/train.py`、`scripts/rsl_rl/full_rl/play.py`：`env_cfg.enable_cameras` 注入
- `assets/robot/ur10e_shadowhand_direct_base_single/ur10e_shadowhand_direct_base_env.py`：UR10e 相机与 TacSL 门控
- `tasks/direct/simple_gripper/forge_env.py`、`forge_env_cfg.py`：Forge 场景剥离与触觉零填充
- `tasks/direct/simple_dexhand/shadow_hand/shadow_hand_vision_env.py`：视觉任务相机与校验

---

*文档对应实现：按 `enable_cameras` 门控 Direct 任务中的渲染类传感器，以支持 headless RL 加速训练。*
