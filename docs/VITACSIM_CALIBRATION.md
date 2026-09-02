# ViTacSim 触觉 RGB + Marker 联合标定

本文档对应《共谋大业》① **ViTacSim — Task 2**（了解 TacSL 校准方式，完成 ViTacSim 与真机 RGB 校准，并与 marker 联合标定），并为 Task 3 物理一致性验证预留接口。

> **相关文档**：marker 仿真原理见 [`VITACSIM_MARKER_SIMULATION.md`](VITACSIM_MARKER_SIMULATION.md)（Task 1，已完成）。

---

## 1. 文档用途（写完之后用来干什么）

| 用途 | 说明 |
|------|------|
| **对内工作说明书** | 标定 case 协议、目录结构、命令顺序固定下来，真机数据到了按同一流程跑，不重搭 pipeline。 |
| **给导师/实验室的采集规范** | `data/calibration/tactile/sim_reference/` 为仿真参考图；`real/` 为真机落盘位置；本文 §4 为采集 SOP。 |
| **参数与优化基线** | §6 列出当前仿真默认值；拟合完成后在 §8 填入最终数值、损失与收敛结论。 |
| **Task 3 报表模板** | §9–§10 预留 Normal / Tangential 验证表结构；真机 GT 到了只填「real」列，不改脚本约定。 |
| **组会/阶段汇报** | 可直接引用：已完成 sim sweep、待填 real 的清单（§11 TODO 表）。 |

**本文不修改任何代码默认值**；拟合结果写入 `data/calibration/tactile/fitted_params.json`，由脚本显式加载，不影响导师后续补材料。

---

## 2. 标定目标（Task 2）

在 **同一套多重量压痕协议** 下，使仿真输出与真机对齐：

1. **Taxim RGB**：背景差分后的压痕外观（依赖 `bg.jpg` + `polycalib.npz`）。
2. **FOTS marker 位移**：各 case 上 marker 最大位移曲线随载荷变化（依赖 `marker_displacement_gain` 等）。
3. **（可选扩展）** ViTacSim 力场参数：`normal_correction_k_ref`、`sticking_interp_sigma` 等——《共谋大业》要求参数清单 + 敏感性 + 网格→蚁群/退火；**当前脚本仅拟合 2 个标量**，见 §7。

联合损失（现有实现）：

```
L = α · L_rgb + β · L_marker
```

- `L_rgb`：各 case 的 bg 差分幅值图 L1（sim vs real）。
- `L_marker`：NF + lateral 各 case 的 `max ||Δmarker||` 曲线 MSE（sim gain 缩放后 vs real）。

默认 `α=β=1`（`fit_vitacsim_rgb_marker.py --alpha-rgb / --beta-marker`）。

---

## 3. TacSL 校准在做什么（原理摘要）

TacSL / Taxim 路径（ViTacLab 沿用）：

| 组件 | 文件 | 作用 |
|------|------|------|
| 无接触背景 | `gelsight_r15_data/bg.jpg` | RGB 差分基准 |
| Taxim 多项式标定 | `gelsight_r15_data/polycalib.npz` | 高度图 → RGB 映射 |
| 渲染分辨率 | 240×320（GelSight R15） | 与 `GELSIGHT_R15_CFG` 一致 |

代码入口：`gelsight_calibrated_cfg.py` → `calibrated_gelsight_r15_cfg()` / `validation_gelsight_render_cfg()`。

Marker 层（TacEx/FOTS）在 Taxim RGB 之上叠加，见 Task 1 文档。

---

## 4. 标定 Case 协议（11 cases）

与仿真 sweep **一一对应**（`bash_command/run_vitacsim_calibration_sweep.sh`）。

### 4.1 法向力 NF（无夹爪）

水平 GelSight pad；砝码 **竖直** 放置；**大圆柱底面中心** 对准传感器图像中心；静稳后采 **单帧** RGB。

| case_id | 质量 | 约 Fn (N) |
|---------|------|-----------|
| `no_contact` | — | 0 |
| `W200` | 200 g | ~1.96 |
| `W100` | 100 g | ~0.98 |
| `W050` | 50 g | ~0.49 |
| `W020` | 20 g | ~0.20 |
| `W010` | 10 g | ~0.10 |

砝码几何：见 `tasks/direct/vitacsim_validation/weight_spec.py`（大圆柱 Ø25×25 mm + 顶小圆柱 + 球）。

### 4.2 切向力 lateral（W100 + Fx）

底载 W100；对砝码施加 **+x 方向** 恒定力（仿真侧 `LATERAL_W100_FX`）：

| Fx (N) | 目录 tag |
|--------|----------|
| 0.00 | `Fx0_Fy0` |
| 0.05 | `Fx0.05_Fy0` |
| 0.10 | `Fx0.1_Fy0` |
| 0.15 | `Fx0.15_Fy0` |
| 0.20 | `Fx0.2_Fy0` |

> **待导师确认 `[TBD-PROTOCOL]`**：真机 lateral 是悬挂配重、平推、还是夹爪施力？若与上表不一致，只改 §4.2 文字与 case 名，**不改** `calibration_io.py` 除非双方对齐后统一改。

---

## 5. 目录结构与文件约定

### 5.1 仿真 sweep（已完成）

```
logs/vitacsim_calibration/sweep/
├── normal_force/
│   ├── no_contact/vitacsim/   # tactile_rgb.png, tactile_marker_displacement.npy, summary.json
│   └── W200|W100|.../vitacsim/
└── shear_force/lateral/Fx*/W100/vitacsim/
```

生成：`bash bash_command/run_vitacsim_calibration_sweep.sh`  
报告：`logs/vitacsim_calibration/report/SIM_CALIBRATION_REPORT.md`

### 5.2 真机数据（导师填入）

```
data/calibration/tactile/real/
├── manifest.json
├── normal_force/
│   ├── no_contact/rgb.png
│   └── W200|rgb.png, marker_displacement.npy (可选，可由 track 脚本生成)
└── lateral_force/W100/Fx*/rgb.png
```

模板：`python3 scripts/calibration/prepare_real_calibration_template.py`

### 5.3 仿真参考（给导师对照采图）

```
data/calibration/tactile/sim_reference/
```

生成：`python3 scripts/calibration/export_sim_reference.py`

### 5.4 拟合输出

```
data/calibration/tactile/fitted_params.json   # 真机就绪后生成
```

---

## 6. 仿真参数清单（当前默认值）

来源：`visuotactile_sensor_cfg.py`（`GelSightRenderCfg` + `VisuoTactileSensorV2` 相关项）。  
**拟合状态**：见「拟合变量」列；真机标定完成后更新 §8，**不**在此文档中改代码默认。

### 6.0 Advisor / 实验室 Xense（当前主线，`profile=advisor`）

| 参数 | 当前值 | 拟合 / 状态 |
|------|--------|-------------|
| 分辨率 | **400×700** | 固定（mp4 原生） |
| `marker_pattern` | **xense** | 固定 |
| `mm_per_pixel` | **≈0.052** | 固定（GelSight 外推，待实验室确认） |
| `bg_clean.jpg` | 实验室 file-000 | ✅ |
| `polycalib.npz` | **GelSight R15 拷贝** | ❌ 等导师 50 次球压 |
| `marker_displacement_gain` | 默认 0.35 → **拟合 0.15** | ✅ `fitted_params.json` |
| `rgb_diff_scale` → `k_ref` | **0.5**（sweep 用 `k_ref /= scale`） | ✅ |
| 接触物 | M2 螺母 + G010–G210 | ✅ |
| `finger_root_z` | 0.441 | 待 Fn 对齐 sweep |
| TacSL | `enable_corrected_force_render=False` | depth→Taxim |
| ViTacSim | `enable_corrected_force_render=True` | force-corrected Taxim |

sweep 加载拟合：`FITTED_PARAMS=data/calibration/tactile/fitted_params.json bash bash_command/run_vitacsim_calibration_sweep.sh`

### 6.1 Taxim / 渲染（GelSightRenderCfg，legacy cylinder 为 240×320）

| 参数 | 默认值 | 拟合变量 | 说明 |
|------|--------|----------|------|
| `image_height` × `width` | 240 × 320 | 固定 | R15 |
| `mm_per_pixel` | （R15 cfg） | 固定 | Taxim 尺度 |
| `num_bins` | 120 | 固定 | Taxim 梯度分箱 |
| `background` + `polycalib` | `bg.jpg` + `polycalib.npz` | **`[TBD-REAL]`** | 需为**实验室传感器**上采的标定包 |
| `enable_marker_simulation` | validation 默认 True | 固定 | NF/SF demo |
| `marker_pattern` | `gelsight` | **`[TBD-REAL]`** | 若真机为 Xense → `xense` |
| `marker_lambda_d` | 0.0025 | 待扩展 | FOTS dilate |
| `marker_displacement_gain` | 0.35 | **是（grid）** | 联合标定主变量 |
| `marker_shear_gain` | 8.0 | 待扩展 | shear proxy |
| `marker_deadband_mm` | 0.02 | 待扩展 | |
| `marker_blend_alpha` | 0.92 | 待扩展 | |

### 6.2 ViTacSim 力场（VisuoTactileSensorV2）

| 参数 | 默认值 | 拟合变量 | 说明 |
|------|--------|----------|------|
| `normal_correction_k_ref` | 1e4 | 待扩展 | 法向校正参考刚度 |
| `normal_correction_knn` | 8 | 固定 | |
| `normal_correction_trim_ratio` | 0.2 | 固定 | RobustMean |
| `sticking_interp_sigma` | 0.02 | **`[TBD-ACO]`** | 共谋大业要求纳入联合优化 |
| `slip_speed_threshold` | 1e-3 m/s | 固定 | |
| `tangential_gamma_clip` | (0.2, 5.0) | 待扩展 | |
| `normal_contact_stiffness` | 1e4 | 待扩展 | baseline TacSL 路径 |
| `depth_penetration_deadband` | 0.002 m | 固定 | |
| `enable_corrected_force_render` | False（validation） | 固定 | 标定 sweep 用 depth→Taxim |

### 6.3 当前联合拟合脚本实际优化的变量

| 变量 | 搜索范围（grid） | 输出字段 |
|------|------------------|----------|
| `marker_displacement_gain` | 0.15 … 0.75 | `fitted_params.json` |
| `rgb_diff_scale` | 0.6 … 1.6 | `recommended_force_render_k_ref_scale` |

**`[TBD-ACO]`**：《共谋大业》要求网格搜索后再蚁群/模拟退火——**尚未实现**；实现时不得改变 `real/` 目录约定与 `rgb.png` 文件名。

---

## 7. 操作流程

### Phase A — 仿真（已完成，可重复）

```bash
cd ViTacLab
bash bash_command/run_vitacsim_calibration_sweep.sh
python3 scripts/calibration/collect_sim_calibration_index.py
python3 scripts/calibration/report_sim_calibration_sweep.py
python3 scripts/calibration/export_sim_reference.py
```

### Phase B — 真机采集（导师 / 实验室）

1. 按 §4 与 `sim_reference/` 对照采图。
2. 放入 `data/calibration/tactile/real/`（§5.2）。
3. **可选**：若实验室另有 Taxim 标定包，替换  
   `source/.../gelsight_r15_data/bg.jpg` 与 `polycalib.npz`（**替换前备份**）。

### Phase C — 联合拟合（真机 rgb 就绪后）

```bash
python3 scripts/calibration/track_real_markers.py --real-root data/calibration/tactile/real
bash bash_command/run_vitacsim_calibration_fit.sh
```

或指定实验室 bg：

```bash
python3 scripts/calibration/fit_vitacsim_rgb_marker.py \
  --bg-path path/to/lab/bg.jpg \
  --out data/calibration/tactile/fitted_params.json
```

**管道测试（无需真机）**：

```bash
python3 scripts/calibration/fit_vitacsim_rgb_marker.py --sim-only
```

### Phase D — 应用拟合结果

```python
from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg

render_cfg = validation_gelsight_render_cfg(
    fitted_params_path="data/calibration/tactile/fitted_params.json",
)
```

然后重跑 sweep，对比拟合前后 `SIM_CALIBRATION_REPORT.md` 与 loss。

---

## 8. 拟合结果与收敛（Task 2 — advisor 数据，2026-08-22）

| 项 | 值 |
|----|-----|
| 拟合日期 | 2026-08-22 |
| 真机数据源 | `file-000.mp4` + `correct.zip`（M2 螺母 + 10–210 g） |
| bg 帧 | file-000 `f0029`（151 帧中 center_std 最小，跳过前 10 帧 warmup） |
| `marker_displacement_gain` | **0.75** |
| `rgb_diff_scale` | **0.5** |
| `loss_total` | **63.72**（default 75.93 → 改善 **16%**） |
| `loss_rgb_mean` | **0.329** |
| 是否收敛 | grid 上相对 default 有改善；**marker 项仍大**（sim 圆柱 vs 真机 M2） |
| 蚁群/退火 | 否（当前 grid search） |

一键重跑：`bash bash_command/run_task2_advisor_calibration.sh`  
报告：`logs/vitacsim_calibration/task2/TASK2_CALIBRATION_REPORT.md`  
旧 `bg.jpg` 备份：`gelsight_r15_data/bg.jpg.bak_before_advisor`

---

## 9. Normal 物理验证（Task 3，表结构预留）

《共谋大业》：同形状砝码、不同质量 → **图像质量差异 + 法向力差异**；输出类似 ACQ2 的表/图。

| 质量 | MSE ↓ | SSIM ↑ | PSNR ↑ | Fn error (N) | 备注 |
|------|-------|--------|--------|--------------|------|
| W010 | _sim TBD_ | | | | real 列待填 |
| W020 | | | | | |
| W050 | | | | | |
| W100 | | | | | |
| W200 | | | | | |

- **Sim 列**：可用 `summarize_vitacsim_normal_force_validation.py` + 现有 NF sweep 先填。
- **Real 列 `[TBD-REAL]`**：需真机 RGB + 实测 Fn（力台/秤）。
- **对比对象**：TacSL baseline vs ViTacSim full vs Real。

---

## 10. Tangential 物理验证（Task 3，表结构预留）

《共谋大业》：砝码 + 多切向方向/多力 → 真机 vs ViTacSim **切向力场 cosine similarity**；1k+ case 分布。

| 状态 | 说明 |
|------|------|
| Sim 批量统计 | _待实现脚本；GT 可用 PhysX per-cell_ |
| Real dense 力场 GT | **`[TBD-REAL]`** 需实验室 mesh+力感知器或等价方案 |
| 切向方向集合 | **`[TBD-PROTOCOL]`** 当前仿真仅 +x Fx |

---

## 11. TODO / 待填入 / 待修改清单

| ID | 位置 | 内容 | 谁填 | 是否阻塞 Task 2 |
|----|------|------|------|-----------------|
| `TBD-REAL` | §5.2, §8 | 真机 `rgb.png`（11 cases） | 导师/实验室 | **是** |
| `TBD-REAL` | §6.1 | 实验室 `bg.jpg` / `polycalib.npz` | 导师 | **是**（若与现文件不一致） |
| `TBD-REAL` | §6.1 | 传感器型号 → `marker_pattern` gelsight/xense | 导师 | **是**（Xense 时） |
| `TBD-PROTOCOL` | §4.2, §10 | 真机 lateral 施力方式、切向方向集合 | 导师 | 部分 |
| `TBD-RESULT` | §8 | 拟合数值、loss、收敛结论 | 你（真机后） | **是**（最终交付） |
| `TBD-ACO` | §6, §7 | 蚁群/退火 + 多参数敏感性 | 你（可选，真机前 sim-only） | 否 |
| `TBD-REAL` | §9–§10 | Normal/Tangential 表 real 列 | 导师 + 你 | Task 3 |

**本文档本身**：导师确认协议后，只需改 §4.2 / §11 中 `[TBD-PROTOCOL]` 文字；**无需**为等导师而改代码。

---

## 12. 命令速查

```bash
# 模板 + 仿真参考（可随时重跑，不覆盖 real/ 里已有 png）
python3 scripts/calibration/prepare_real_calibration_template.py
python3 scripts/calibration/export_sim_reference.py

# 真机就绪后
python3 scripts/calibration/track_real_markers.py
bash bash_command/run_vitacsim_calibration_fit.sh

# 进度（sweep 期间）
python3 scripts/calibration/watch_calibration_sweep_progress.py
```

---

## 13. 与《共谋大业》① 交付物对应

| 共谋大业要求 | 本文档 / 产物 |
|-------------|---------------|
| 了解 TacSL 校准 | §3 |
| RGB + marker 联合标定 | §2, §4–§7 |
| 校准原理 / 过程文档 | 全文 |
| 参数、值说明 | §6 |
| 优化损失、是否收敛 | §8（待填） |
| 网格→蚁群、敏感性 | §6.3, §8, §11 `TBD-ACO` |
| 类似 ACQ2 表图 | §9–§10 模板 + sim sweep panel |
| Sim2real（三任务×300） | **不在本文**；属共谋大业 ① 另一条，需与导师确认分工 |
