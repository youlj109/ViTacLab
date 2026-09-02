#!/usr/bin/env python3
"""Organize vitacsim_validation outputs into v1/v2 and build advisor zip."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "logs" / "vitacsim_validation"
V1 = ROOT / "v1"
V2 = ROOT / "v2"
ADVISOR = V2 / "advisor_delivery"

NF_V1_SCHEMA = "nf_v2_chamfer_rgb"
NF_V2_SCHEMA = "nf_v3_beta"
SF_V1_SCHEMA = "sf_lateral_v1"
SF_V2_SCHEMA = "sf_lateral_v2"

WEIGHTS = ("W200", "W100", "W050", "W020", "W010")

V1_NF_AUX = (
    "_zprobe",
    "_zprobe_f0.438",
    "_zprobe_f0.440",
    "_zprobe_f0.442",
    "_zprobe_f0.444",
    "_zprobe_f0.446",
    "_zprobe_f0.448",
    "_smoke_v2",
    "_exit_test",
    "_exit_verify",
)

V1_ROOT_FILES = (
    "curve_sf_ft_vs_fx.png",  # legacy plot (pre rebattle)
)

V2_ROOT_PLOTS = (
    "curve_nf_fn_vs_mass.png",
    "curve_sf_ft_vs_fx_all_valid.png",
    "rebattle_sf_main.png",
    "rebattle_sf_panel.png",
    "bar_sf_ft_at_fx0.15.png",
    "bar_sf_ft_at_fx0.2.png",
    "panel_sweep_pen_heatmap.png",
    "panel_sf_lateral_rgb.png",
)

V2_DOCS = (
    "EXPERIMENT_PLAN.md",
    "CALIBRATION.md",
)


def _rm_tree(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p)
    elif p.is_file():
        p.unlink()


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        _rm_tree(dst)
    shutil.copytree(src, dst)


def _read_schema(summary: Path) -> str:
    if not summary.is_file():
        return ""
    return json.loads(summary.read_text(encoding="utf-8")).get("output_schema", "")


def _copy_trial_dir(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    if dst.exists():
        _rm_tree(dst)
    shutil.copytree(src, dst)


def _collect_sf_trials(lat_root: Path, schema: str) -> list[Path]:
    out: list[Path] = []
    for summary in lat_root.glob("**/summary.json"):
        if _read_schema(summary) != schema:
            continue
        out.append(summary.parent)
    return sorted(out)


def _write_v1_readme() -> None:
    text = """# ViTacSim 验证 Beta — v1 归档

本目录为 **2026-08-12 工程改进前** 的第一版 sim beta 交付物归档。

## 内容

- `normal_force/`：z-probe / smoke 等开发试跑；**完整 NF 10-trial 原始目录已被 v3 覆盖**，数值见 `_full_sweep_v2.log`
- `shear_force/lateral/`：schema `sf_lateral_v1` 的试跑（含旧 Fx sweep 0/0.1/0.2/0.3/0.5）
- 根目录旧版曲线 `curve_sf_ft_vs_fx.png`、首版 `BETA_DELIVERY.md`

## 新版

请查看同级目录 `../v2/` 与 `../v2/advisor_delivery/`（发给导师）。
"""
    (V1 / "README.md").write_text(text, encoding="utf-8")


def _write_v2_readme() -> None:
    text = """# ViTacSim 验证 Beta — v2（当前）

工程改进后重跑：**NF nf_v3_beta** + **SF sf_lateral_v2**（分重量 Fx 上限、contact_valid、rebattle 主图）。

## 发给导师

直接使用 **`advisor_delivery/`** 文件夹，或根部的 **`vitacsim_validation_advisor_v2.zip`**。

## 复现

```bash
SKIP_EXISTING=0 FINGER_ROOT_Z=0.444 bash bash_command/demo_vitacsim_normal_force_validation.sh
SKIP_EXISTING=0 FINGER_ROOT_Z=0.444 bash bash_command/demo_vitacsim_lateral_force_validation.sh
python3 scripts/demo/plot_vitacsim_validation_beta.py
```
"""
    (V2 / "README.md").write_text(text, encoding="utf-8")


def _write_v2_beta_delivery() -> None:
    text = """# ViTacSim 砝码验证 Beta v2（交付导师）

## 实验范围

| 阶段 | 方法 | 状态 |
|------|------|------|
| **法向 NF** | 水平 GelSight + 5 砝码自重，无夹爪 | ✅ nf_v3_beta 10/10 |
| **切向 SF（退路）** | 5 砝码 + **分重量 Fx 上限** + XY 虚拟墙 | ✅ sf_lateral_v2 50/50 |
| **切向 SF 夹爪** | Franka 平行夹爪 | ⏸ 未交付 |

## 主图（rebattle 推荐）

1. **`rebattle_sf_main.png`** — W100+W200，PhysX Ft vs Fx（仅 valid 点）
2. **`curve_nf_fn_vs_mass.png`** — PhysX Fn vs 质量 + ratio 柱状图
3. **`rebattle_sf_panel.png`** — SF diff-bg @ Fx≈0.15N
4. **`bar_sf_ft_at_fx0.15.png`** — 五重量 @ 固定 Fx

## 关键结论

**NF**：PhysX Fn 随质量单调；`physx_fn/nominal ≈ 0.82`（sim 系统偏低 ~18%）。vitacsim 自适应 k_ref 后 **fn_peak 随砝码单调**，diff-bg 面板可见「越重越亮」。

**SF**：W100/W200 在 valid Fx 内摩擦响应合理；轻砝码高 Fx 点 `contact_valid=false` 已从曲线剔除。

## 已知限制

- 无真机 weighted RGB 对比
- PhysX 总量未校准到 1.0×mg（finger z 可继续微调）
- 夹爪 SF 下一阶段

## 归档

v1 旧版见 `../v1/`。
"""
    (V2 / "BETA_DELIVERY.md").write_text(text, encoding="utf-8")


def _write_advisor_readme() -> None:
    text = """# ViTacSim Sim Beta — 发给导师（v2）

**日期**：2026-08-12  
**内容**：GelSight R15 五砝码仿真验证（tacsl vs vitacsim），仅 sim、无法向真机图。

## 建议阅读顺序

1. `01_说明/BETA_DELIVERY.md` — 一页摘要  
2. `03_主图/` — rebattle 四张主图  
3. `02_汇总表/` — NF/SF 数值表  
4. `04_补充图/` — RGB diff、全重量 valid 曲线  

## 口径

- **NF**：水平传感器 + 砝码自重 → 法向力 Fn（PhysX 总量 vs mg）  
- **SF 退路**：砝码在垫上 + 世界系 +X 恒力 → 切向摩擦 Ft  
- 轻砝码过大 Fx 会失接触，表中/曲线中已标注并剔除 invalid 点  

## 复现（仓库 ViTacLab）

见 `v2/README.md`。
"""
    (ADVISOR / "00_README_发给导师.md").write_text(text, encoding="utf-8")


def organize() -> None:
    nf_root = ROOT / "normal_force"
    lat_root = ROOT / "shear_force" / "lateral"

    if V1.exists():
        _rm_tree(V1)
    if V2.exists():
        _rm_tree(V2)
    V1.mkdir(parents=True)
    V2.mkdir(parents=True)
    ADVISOR.mkdir(parents=True)

    v1_nf = V1 / "normal_force"
    v2_nf = V2 / "normal_force"
    v1_lat = V1 / "shear_force" / "lateral"
    v2_lat = V2 / "shear_force" / "lateral"
    v1_nf.mkdir(parents=True)
    v2_nf.mkdir(parents=True)
    v1_lat.mkdir(parents=True)
    v2_lat.mkdir(parents=True)

    # --- NF v2: main weight trials ---
    for wid in WEIGHTS:
        for sub in ("tacsl", "vitacsim"):
            src = nf_root / wid / sub
            if _read_schema(src / "summary.json") == NF_V2_SCHEMA:
                _copy_tree(src, v2_nf / wid / sub)

    for name in (
        "SWEEP_REPORT.md",
        "panel_sweep_rgb.png",
        "panel_sweep_rgb_depth.png",
        "panel_sweep_rgb_diff_bg.png",
        "STATUS.md",
        "weight_preview.png",
        "_full_sweep_v3.log",
    ):
        _copy_file(nf_root / name, v2_nf / name)

    # --- NF v1: aux + logs ---
    for aux in V1_NF_AUX:
        _copy_tree(nf_root / aux, v1_nf / aux)
    for name in (
        "_full_sweep.log",
        "_full_sweep_v2.log",
        "_smoke_w100_tacsl.log",
        "_debug_center_diff_x5.png",
        "_sweep_progress.json",
    ):
        _copy_file(nf_root / name, v1_nf / name)

    # Archived v1 NF report snippet from log if present
    v1_report = nf_root / "SWEEP_REPORT.md"
    if v1_report.is_file():
        txt = v1_report.read_text(encoding="utf-8")
        if NF_V1_SCHEMA in txt or "nf_v2" in txt:
            (v1_nf / "SWEEP_REPORT_nf_v2_archived.md").write_text(txt, encoding="utf-8")

    # --- SF v2 trials ---
    for trial in _collect_sf_trials(lat_root, SF_V2_SCHEMA):
        rel = trial.relative_to(lat_root)
        _copy_trial_dir(trial, v2_lat / rel)

    for name in ("SWEEP_REPORT.md", "panel_sweep_rgb.png", "panel_sweep_rgb_diff_bg.png", "_full_sweep_v2.log"):
        _copy_file(lat_root / name, v2_lat / name)

    # --- SF v1 trials (v1-only paths) ---
    v2_paths = {t.relative_to(lat_root) for t in _collect_sf_trials(lat_root, SF_V2_SCHEMA)}
    for trial in _collect_sf_trials(lat_root, SF_V1_SCHEMA):
        rel = trial.relative_to(lat_root)
        if rel in v2_paths:
            continue
        _copy_trial_dir(trial, v1_lat / rel)

    for name in ("_full_sweep.log", "_full_sweep_5weights.log", "panel_sweep_rgb.png"):
        src = lat_root / name
        if src.is_file():
            _copy_file(src, v1_lat / name)

    # --- Root plots & docs ---
    _copy_file(ROOT / "BETA_DELIVERY.md", V1 / "BETA_DELIVERY.md")
    for f in V1_ROOT_FILES:
        _copy_file(ROOT / f, V1 / f)

    for f in V2_ROOT_PLOTS:
        _copy_file(ROOT / f, V2 / f)
    for f in V2_DOCS:
        _copy_file(ROOT / f, V2 / f)

    _write_v1_readme()
    _write_v2_readme()
    _write_v2_beta_delivery()

    # --- Advisor package ---
    for sub, src in (
        ("01_说明/BETA_DELIVERY.md", V2 / "BETA_DELIVERY.md"),
        ("01_说明/EXPERIMENT_PLAN.md", V2 / "EXPERIMENT_PLAN.md"),
        ("01_说明/CALIBRATION.md", V2 / "CALIBRATION.md"),
        ("02_汇总表/NF_SWEEP_REPORT.md", v2_nf / "SWEEP_REPORT.md"),
        ("02_汇总表/SF_SWEEP_REPORT.md", v2_lat / "SWEEP_REPORT.md"),
        ("03_主图/rebattle_sf_main.png", V2 / "rebattle_sf_main.png"),
        ("03_主图/curve_nf_fn_vs_mass.png", V2 / "curve_nf_fn_vs_mass.png"),
        ("03_主图/rebattle_sf_panel.png", V2 / "rebattle_sf_panel.png"),
        ("03_主图/bar_sf_ft_at_fx0.15.png", V2 / "bar_sf_ft_at_fx0.15.png"),
        ("04_补充图/curve_sf_ft_vs_fx_all_valid.png", V2 / "curve_sf_ft_vs_fx_all_valid.png"),
        ("04_补充图/NF_panel_diff_bg.png", v2_nf / "panel_sweep_rgb_diff_bg.png"),
        ("04_补充图/NF_pen_heatmap.png", V2 / "panel_sweep_pen_heatmap.png"),
        ("04_补充图/SF_panel_diff_bg.png", v2_lat / "panel_sweep_rgb_diff_bg.png"),
    ):
        _copy_file(src, ADVISOR / sub)

    _write_advisor_readme()

    zip_path = ROOT / "vitacsim_validation_advisor_v2.zip"
    if zip_path.is_file():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(ADVISOR.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(V2).as_posix())

    # Root: only README + v1 + v2 + zip (remove legacy flat outputs if re-run)
    for name in (
        "BETA_DELIVERY.md",
        "CALIBRATION.md",
        "EXPERIMENT_PLAN.md",
        "curve_nf_fn_vs_mass.png",
        "curve_sf_ft_vs_fx.png",
        "curve_sf_ft_vs_fx_all_valid.png",
        "rebattle_sf_main.png",
        "rebattle_sf_panel.png",
        "panel_sf_lateral_rgb.png",
        "panel_sweep_pen_heatmap.png",
        "bar_sf_ft_at_fx0.15.png",
        "bar_sf_ft_at_fx0.2.png",
    ):
        p = ROOT / name
        if p.is_file():
            p.unlink()
    for dname in ("normal_force", "shear_force"):
        p = ROOT / dname
        if p.is_dir():
            shutil.rmtree(p)

    index = f"""# vitacsim_validation 目录说明

| 目录 | 内容 |
|------|------|
| **v1/** | 第一版 beta 归档（nf_v2 / sf_lateral_v1 时代） |
| **v2/** | 工程改进后重跑（nf_v3_beta / sf_lateral_v2） |
| **v2/advisor_delivery/** | 发给导师的精选包 |
| **vitacsim_validation_advisor_v2.zip** | 同上，zip 便于传输 |

整理时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
    (ROOT / "README.md").write_text(index, encoding="utf-8")
    print(f"[INFO] v1 -> {V1}")
    print(f"[INFO] v2 -> {V2}")
    print(f"[INFO] advisor -> {ADVISOR}")
    print(f"[INFO] zip -> {zip_path} ({zip_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    organize()
