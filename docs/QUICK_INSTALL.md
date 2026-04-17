# ViTacLab 快速安装

：**已装 Isaac Sim / Isaac Lab**，只想尽快把本仓库跑起来。

## 前置条件

1. **Isaac Sim 5.1.0** 与 **Isaac Lab** 按官方文档装好，且终端里能调用带 
Isaac 的 Python。  
   官方安装索引：<https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

2. 本仓库 **单独 clone**，不要放在上游 `IsaacLab` 源码目录里面。

## 三步安装

### 1. 进入「有 Isaac Lab 的」Python
```bash
conda activate env_isaaclab_510
```

下面统一记为 **`python`**。

### 2. 安装本扩展（可编辑模式）

在 **ViTacLab 仓库根目录**：

```bash
cd /path/to/ViTacLab
python -m pip install -e source/ViTacLab
```

## 验证


```bash
python scripts/rsl_rl/full_tra/record_full_tra_single.py --task Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 --num_envs 1 --enable_cameras  --show_rgb --show_ff
```

