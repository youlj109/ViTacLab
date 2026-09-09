# ViTacLab 快速安装

适用于：**已经安装 Isaac Sim / Isaac Lab**，只想尽快运行本仓库。

完整的资产下载、配置和验收流程见
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

## 前置条件

1. 按[官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 **Isaac Sim 5.1.0** 与 **Isaac Lab 2.3.2**，并确认当前 Python 能导入 Isaac Lab。
2. 将 ViTacLab 单独 clone，不要放入上游 `IsaacLab` 源码目录。

## 安装与验证

### 1. 激活 Isaac Lab 的 Python 环境

以下名称仅为示例，请换成你的实际环境名：

### 1. 进入安装了 Isaac Lab 的 Python 环境
```bash
conda activate <isaac-lab-environment>
```

### 2. 安装 ViTacLab 扩展

在 ViTacLab 仓库根目录执行：

```bash
python -m pip install -e source/ViTacLab
python -m pip install -r requirements/repro.txt
python -m venv .venv_hf
.venv_hf/bin/python -m pip install -U huggingface_hub
bash bash_command/download_hf_assets.sh Yanlj/ViTacLab-assets
```

### 3. 验证注册与基础运行

先列出当前 checkout 注册的环境：

```bash
python scripts/list_envs.py
```

再运行不依赖论文外部资产的模板环境。该命令会持续运行；看到环境与 action/observation space 正常创建后，按 `Ctrl+C` 退出。

```bash
python scripts/zero_agent.py \
  --task Template-Vitaclab-Direct-v0 \
  --num_envs 1 --headless
```

这一步只验证 Isaac Lab 与 ViTacLab 的安装和环境注册，不验证论文中的触觉仿真或操作任务。

## 运行触觉操作任务

```bash
bash bash_command/verify_repro_env.sh

python scripts/zero_agent.py \
  --task Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 \
  --num_envs 1 --max-steps 20 --enable_cameras --headless
```

需要触觉 RGB 或相机观测的任务通常必须添加 `--enable_cameras`。无显示器运行、相机渲染和常见排错请参阅[相机与 headless 指南](enable_cameras_headless_rl.md)。
