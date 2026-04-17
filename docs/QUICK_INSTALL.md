# ViTacLab 快速安装

面向：**已准备装 Isaac Sim / Isaac Lab**，只想尽快把本仓库跑起来。

## 前置条件

1. **Isaac Sim** 与 **Isaac Lab** 按官方文档装好，且终端里能调用带 Isaac 的 Python。  
   官方安装索引：<https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

2. 本仓库 **单独 clone**，不要放在上游 `IsaacLab` 源码目录里面。

## 三步安装

### 1. 进入「有 Isaac Lab 的」Python

任选其一：

- 若用 Isaac Lab 提供的脚本（常见）：

  ```bash
  cd /path/to/IsaacLab
  ./isaaclab.sh -p -c "import isaaclab; print('ok')"
  ```

- 若用 Isaac Sim 自带解释器（示例路径按你本机修改）：

  ```bash
  /path/to/isaac-sim/python.sh -c "import isaaclab; print('ok')"
  ```

下面统一记为 **`python`**；若你习惯 `./isaaclab.sh -p` 或 `./python.sh`，把命令里的 `python` 换成即可。

### 2. 安装本扩展（可编辑模式）

在 **ViTacLab 仓库根目录**：

```bash
cd /path/to/ViTacLab
python -m pip install -e source/ViTacLab
```

### 3.（可选）视频遥操作依赖

仅当你要跑 `source/video_teleop` 或相关脚本时：

```bash
python -m pip install -r source/video_teleop/requirements.txt
```

## 验证

```bash
cd /path/to/ViTacLab
python scripts/list_envs.py
```

能列出含 `ViTacLab` 的任务 ID 即说明扩展已注册。

再试一个轻量检查（零动作智能体）：

```bash
python scripts/zero_agent.py --task=<某个 ViTacLab 注册的 Gym ID>
```

训练入口见 [`scripts/rsl_rl/README.md`](../scripts/rsl_rl/README.md)、[`scripts/rsl_rl/QUICKSTART.md`](../scripts/rsl_rl/QUICKSTART.md)。

## 关于 `requirements-lock.txt`

仓库根目录若存在 **`requirements-lock.txt`**，一般是某次环境的 **`pip freeze` 快照**，体积大、且与 Isaac 内置包强绑定，**不适合**作为通用「最小安装列表」。日常以：

- `pip install -e source/ViTacLab`
- 按需 `pip install -r source/video_teleop/requirements.txt`

为主即可。

## 可选：Omniverse 里启用扩展

见根目录 [`README.md`](../README.md) 中 **Omniverse extension (optional)**：在 Extension Search Paths 中加入本仓库的 **`source`** 目录。

## 可选：IDE / 补全

见根目录 [`README.md`](../README.md) 的 **IDE setup** 与 **Troubleshooting**（`extraPaths`、Pylance 等）。
