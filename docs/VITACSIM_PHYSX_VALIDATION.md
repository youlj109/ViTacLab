# ViTacSim PhysX 触觉验证说明（Co-author / Release 用）

本文档说明 ViTacLab 中 **VisuoTactileSensorV2（ViTacSim 力场路径）** 如何与 **PhysX 接触/摩擦** 对齐验证，以及如何证明在严格配置下 **不会静默 fallback 到非 PhysX 路径**。

适用场景：对外发布（battle / demo pack / 论文补充材料）时需要向 reviewer 或合作者解释「触觉力场确实来自 PhysX，而不是纯 depth heuristic」。

---

## 1. 背景：ViTacSim V2 在做什么

ViTacSim 的 rigid-body 触觉力场（`VisuoTactileSensorV2`）采用 **两阶段** 重建：

1. **相机 depth delta**：GelSight 内参相机得到 elastomer 形变（任意物体接触都会在 depth 上产生响应）。
2. **PhysX sparse anchors**：从 PhysX `RigidContactView` 读取 **目标物体** 与 pad 之间的接触点、法向力、摩擦锚点，用于：
   - 法向力局部刚度修正（normal correction）
   - 滑移/粘滞切向重建（slip/stick reconstruction）

因此，**depth 只能证明「有东西碰到 pad」**，不能证明「力被正确归因到配置的目标物体」。验证的核心是：

- **对齐**：有 PhysX 接触时，sensor 力场与 PhysX 统计量相关（Fn/Ft、接触位置）。
- **归因**：非目标物体接触 pad 时，**不应** 在目标物体通道产生虚假力（false activation）。

实现代码：`source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/visuotactile_sensor_v2.py`

---

## 2. 「Fallback」在代码里指什么

ViTacSim V2 存在 **两层** fallback，必须区分：

### 2.1 初始化 fallback（Init-time）

当 `use_physx_sparse_anchors=True` 但无法创建有效的 PhysX rigid contact view 时：

| 配置 | 行为 |
|------|------|
| `require_physx_sparse_anchors=False`（默认） | **警告 + 静默回退** 到 dense self-anchoring |
| `require_physx_sparse_anchors=True`（严格） | **RuntimeError 直接 abort**，不允许 dense fallback |

> 论文/严格复现 run 应开启 `require_physx_sparse_anchors=True`，从机制上保证「要么 PhysX 后端可用，要么显式失败」。

### 2.2 运行时 fallback（Per-frame）

当某一帧 `_gather_sparse_anchors_rigid()` 失败或目标物体无 PhysX 接触点时：

| 配置 | 行为 |
|------|------|
| `strict_target_contact_attribution=False` | 回退到 **dense self-anchoring**（depth 驱动，可能误归因） |
| `strict_target_contact_attribution=True`（默认） | **法向/切向力置零**，不把非目标 depth 接触算到目标物体上 |

> 「没有 fallback」在归因语义上 = **`strict_target_contact_attribution=True`**：无目标 PhysX anchor 的帧不产生目标力场。

配置说明见：`visuotactile_sensor_cfg.py` 中 `use_physx_sparse_anchors` / `require_physx_sparse_anchors` / `strict_target_contact_attribution`。

---

## 3. 生产任务中的严格开关

ViTacBench 中 **Forge DexHand** 等正式数据采集任务在 TacSL policy 里同时开启：

```python
use_physx_sparse_anchors=True
require_physx_sparse_anchors=True          # 禁止 init 静默 fallback
strict_target_contact_attribution=True     # 禁止 runtime 误归因
```

见：`source/ViTacLab/ViTacLab/tasks/direct/medium_dexhand/forge_dexhand/ur10e_shadowhand_forge_env_cfg.py`

单臂 UR10e ShadowHand 默认 policy（`TacSLSensorPolicyCfg`）为：

- `use_physx_sparse_anchors=True`
- `strict_target_contact_attribution=True`
- `require_physx_sparse_anchors=False`（允许 init fallback，便于兼容旧场景）

---

## 4. 自动化验证流程（Quantitative Gate）

### 4.1 入口脚本

| 文件 | 作用 |
|------|------|
| `scripts/demo/eval_visuotactile_physx_alignment.py` | 6-case 定量 benchmark，输出 JSON |
| `bash_command/visuotactile_physx_alignment_check.sh` | 一键运行 + 解析 JSON，`overall_pass=false` 时 **exit 2** |

### 4.2 测试场景

独立最小场景：**GelSight 短指 USD + target nut + interference nut**（非完整 robot task，专门测机制）。

6 个 case（每个 case 约 220 sim steps，可调）：

| Case | 设置 | 验证目的 |
|------|------|----------|
| `no_contact` | 两物体悬空 | 空接触假阳性率（FPR） |
| `normal_press_center` | 目标 nut 压 pad | Fn 对齐、接触 recall、质心误差 |
| `normal_press_edge` | 目标 nut 偏置 XY | 边缘接触鲁棒性 |
| `shear_slide_constant` | 恒定 Z 力矩 | 切向 Ft、slip F1 |
| `stick_then_slip` |  ramp 力矩 | 粘滑转换 |
| **`interference_only`** | **目标 nut 抬起，干扰 nut 接触 pad** | **归因：depth 有接触但目标 PhysX 力应为 0** |

### 4.3 每步采集的对比量

- **PhysX 侧**：`fn_total`, `ft_total`, `contact_count`, contact points（ground truth）
- **Sensor 侧**：`tactile_normal_force`, `tactile_shear_force`, contact/slip mask
- **诊断**：`sparse_used`（该帧是否实际使用了 PhysX sparse anchor 路径）
- **Raw depth**：`raw_depth_contact_rate`（任意物体 depth 接触率，用于 interference case）

### 4.4 Pass 判据（摘要）

**一般接触 case：**

- `contact_recall ≥ 0.70`
- `fn_corr ≥ 0.30`（与 PhysX Fn 时间序列相关）
- `centroid_error_mean ≤ 0.03 m`
- 滑移 case 额外：`slip_f1 ≥ 0.50`，`ft_corr ≥ 0.20`（或 Ft 比足够大）

**`no_contact`：**

- `contact_fpr ≤ 0.05`

**`interference_only`（归因核心）：**

- `raw_depth_contact_rate ≥ 0.20` — 证明 pad **确实** 被干扰物碰到（depth 有响应）
- `target_false_activation_rate ≤ 0.10` — 在 PhysX 报告目标无接触时，sensor 目标通道 **几乎不** 产生虚假激活

全部 case 通过 → `alignment_summary.json` 中 `overall_pass: true`。

### 4.5 复现命令

```bash
cd ViTacLab   # repo root
bash bash_command/visuotactile_physx_alignment_check.sh
```

可选环境变量：

```bash
NUM_ENVS=1 STEPS_PER_CASE=220 SEED=42 \
  OUT_DIR=logs/alignment_visuotactile_v2 \
  bash bash_command/visuotactile_physx_alignment_check.sh
```

输出：

- `logs/alignment_visuotactile_v2/alignment_summary.json`
- `logs/alignment_visuotactile_v2/alignment.log`

---

## 5. 可视化验证（Attribution A/B，给 demo / battle 用）

定量 JSON 之外，提供 **STRICT=0 vs STRICT=1** 并排视频，直观展示「误归因 vs 抑制误归因」。

| 文件 | 作用 |
|------|------|
| `scripts/demo/demo_visuotactile_alignment_visual.py` | interference-only 场景录视频 + 叠加指标 |
| `bash_command/visuotactile_alignment_visual.sh` | `STRICT=0/1` 切换 |
| `scripts/demo/make_visuotactile_alignment_demo_video.py` | 合成 presentation 视频 |
| `bash_command/vitacsim_report_pack.sh` | Mentor demo pack（`INCLUDE_ATTRIBUTION=1` 时生成对比片） |

```bash
# Loose：允许 dense fallback，干扰物接触可能产生虚假目标力
STRICT=0 OUT_VIDEO=logs/alignment_visuotactile_v2/interference_loose.mp4 \
  bash bash_command/visuotactile_alignment_visual.sh

# Strict：strict_target_contact_attribution=True，虚假激活被抑制
STRICT=1 OUT_VIDEO=logs/alignment_visuotactile_v2/interference_strict.mp4 \
  bash bash_command/visuotactile_alignment_visual.sh

# 合成对比 presentation
LOOSE_VIDEO=.../interference_loose.mp4 \
STRICT_VIDEO=.../interference_strict.mp4 \
OUT_VIDEO=.../interference_demo_present.mp4 \
  bash bash_command/visuotactile_alignment_demo.sh
```

**解读要点（可直接用于 slide）：**

- 当 `raw_depth_contact_rate > 0` 且 PhysX 目标 `fn = 0` 时：
  - **STRICT=0**：sensor 仍可能输出非零 `fn` → **归因错误**
  - **STRICT=1**：sensor `fn / false_rate ≈ 0` → **误归因被抑制**
- 这不否定 depth 触觉成像；它证明 **力场通道** 在 strict 模式下 **gated by PhysX target anchors**。

---

## 6. 如何向外部说明「没有 silent fallback」

对外 release / battle 建议分 **三层表述**（由强到弱）：

### Level A — 机制层（代码保证）

在 **严格任务配置**（Forge DexHand 等）中：

1. `require_physx_sparse_anchors=True` → PhysX contact view 建不起来则 **进程失败**，不会悄悄用 dense anchor。
2. `strict_target_contact_attribution=True` → 无目标 PhysX anchor 的帧 **力场为零**，不会用 depth-only 伪造目标力。

### Level B — 定量回归（CI / pre-release gate）

运行 `visuotactile_physx_alignment_check.sh`，要求：

- `overall_pass=true`
- 特别关注 `interference_only`：`target_false_activation_rate ≤ 0.10` 且 `raw_depth_contact_rate ≥ 0.20`

同时检查 JSON 中 `sparse_used_mean`：

- 有目标接触的 case 应 **> 0**（证明 PhysX sparse 路径被使用）
- `interference_only` 中 `sparse_used_mean ≈ 0` 是 **预期**（目标无 PhysX 接触）

### Level C — 演示层（human-visible）

提供 STRICT=0/1 对比视频 + Forge 任务 success demo（`vitacsim_report_pack.sh`），回答两个问题：

1. 系统能否跑通 manipulation task？
2. interference 场景下归因 bug 是否被 strict 模式修掉？

---

## 7. 与 Pretraining（V1）的边界

`Isaac-GelsightFinger-*-Pretrain-Direct-v0` 等 **pretraining** 任务使用 **VisuoTactileSensor V1**（SDF / 非 PhysX sparse anchor 路径），**不在本文 PhysX 验证范围内**。

PhysX 验证仅针对 **VisuoTactileSensorV2** + rigid target attribution 路径。

---

## 8. 常见问题（Co-author FAQ）

**Q: 验证脚本本身有没有开 `require_physx_sparse_anchors=True`？**

A: `eval_visuotactile_physx_alignment.py` 显式设置 `use_physx_sparse_anchors=True`，依赖默认 `strict_target_contact_attribution=True`（cfg 默认）做 **runtime 归因** 验证；**未** 强制 `require_physx_sparse_anchors=True`。  
→ 定量 gate 主要验证 **对齐 + 归因**；**init 无 silent fallback** 由 **生产任务 cfg**（Forge 等）的 `require_physx_sparse_anchors=True` 保证。

**Q: 「没有 fallback」是否等于「每帧都必须有 PhysX 接触」？**

A: 否。无接触帧力场为零是正常行为。`interference_only` case 正是：**有 depth 接触、无目标 PhysX 接触** 时，strict 模式不应输出目标力。

**Q: NPZ 回放里 force 全零时 visualize 会用 RGB fallback 上色，这算 PhysX fallback 吗？**

A: **不算**。`visualize_play_record_npz.py` 的 RGB fallback 仅是 **离线可视化** 手段，与仿真中 ViTacSim 力场重建路径无关。

---

## 9. 相关文件索引

| 类型 | 路径 |
|------|------|
| 传感器 V2 实现 | `source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/visuotactile_sensor_v2.py` |
| 配置与 flag 文档 | `source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/visuotactile_sensor_cfg.py` |
| 定量验证 | `scripts/demo/eval_visuotactile_physx_alignment.py` |
| 验证 shell | `bash_command/visuotactile_physx_alignment_check.sh` |
| 可视化 A/B | `scripts/demo/demo_visuotactile_alignment_visual.py` |
| Demo pack | `bash_command/vitacsim_report_pack.sh` |
| 对外打包说明 | `README_send.md` §3 Critical Mechanism Flags |
| V2 机制 ablation | `scripts/demo/demo_visuotactile_sensor_v2_ablation.py` |

---

## 10. 一句话 Summary（可贴进 release note）

> ViTacSim rigid tactile force fields are reconstructed from **PhysX sparse contact/friction anchors** on the configured target object, not from camera depth alone. We validate **PhysX–sensor alignment** on six scripted cases and **target attribution under interference contact** (`interference_only`: depth contact present, target PhysX force absent → false activation rate ≤ 10%). Production configs set **`require_physx_sparse_anchors=True`** and **`strict_target_contact_attribution=True`** to prevent silent fallback to dense depth-only anchoring.

---

*Document version: 2026-08-05 · ViTacLab repo internal*
