# ViTacLab 快速安装

适用于：**已经安装 Isaac Sim / Isaac Lab**，只想尽快运行本仓库。

完整的资产下载、配置和验收流程见
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

## 前置条件

1. **Isaac Sim 5.1.0** 与 **Isaac Lab 2.3.2** 按官方文档装好，且终端里能调用带 
Isaac 的 Python。  
   官方安装索引：<https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

2. 本仓库 **单独 clone**，不要放在上游 `IsaacLab` 源码目录里面。

## 三步安装

### 1. 进入安装了 Isaac Lab 的 Python 环境
```bash
conda activate <isaac-lab-environment>
```

下面统一记为 **`python`**。

### 2. 安装本扩展（可编辑模式）

在 **ViTacLab 仓库根目录**：

```bash
cd /path/to/ViTacLab
python -m pip install -e source/ViTacLab
python -m pip install -r requirements/repro.txt
python -m venv .venv_hf
.venv_hf/bin/python -m pip install -U huggingface_hub
bash bash_command/download_hf_assets.sh Yanlj/ViTacLab-assets
```

## 验证

```bash
bash bash_command/verify_repro_env.sh

python scripts/zero_agent.py \
  --task Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 \
  --num_envs 1 --max-steps 20 --enable_cameras --headless
```

