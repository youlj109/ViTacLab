# Video teleop — Quick start

在 ViTacLab 仓库根目录下执行下列命令（将路径中的 conda 环境名换成你本机实际名称）。

## 1. 环境

视频采集与 MediaPipe 依赖 **一个** Conda 环境；Isaac Sim / Isaac Lab 使用 **另一个** 环境。发送端与接收端分别运行，因此需要两个终端、两套环境。

| 角色 | 典型环境名 | 用途 |
|------|------------|------|
| 发送端（摄像头 + 手势） | `video_teleoperator` | OpenCV、MediaPipe、pyzmq、msgpack、PyYAML 等 |
| 接收端（仿真或仅可视化） | `env_isaaclab_510` 等 | Isaac Lab、`./python.sh` |

```bash
conda env list
```

若缺少依赖，在发送端环境中安装，例如：`pip install opencv-python pyyaml pyzmq msgpack mediapipe`（以你环境实际为准）。

## 2. 标定

### 2.1 查看摄像头序号

```bash
conda activate video_teleoperator
python scripts/teleoperation/video_teleop/list_cameras.py --max-index 15
```

记下可用的 `--camera` 索引（例如 `6`）。

### 2.2 相机内参（棋盘格）

首次使用或更换相机后建议标定，供 AprilTag / 投影使用。工具仅支持 **真实摄像头** 采集（无屏幕 ROI 流程）。

```bash
python scripts/teleoperation/video_teleop/camera_calibration.py \
    --camera 6 \
    --rows 5 \
    --cols 7 \
    --square-size 0.025 \
    --output scripts/teleoperation/video_teleop/config/camera_calibration.yaml
```

操作要点：

1. 打印棋盘格：**5×7 内角点**（与 `--rows` / `--cols` 一致），方格边长 **25 mm**（与 `--square-size 0.025` 一致）。
2. 手持棋盘格在画面中多角度、多距离出现；窗口里检测到角点后会显示叠加线。
3. **空格**：保存当前帧；至少采集 **5 张** 有效图。
4. **q**：结束采集并写出 YAML；**ESC**：放弃退出。

也可从仓库根目录直接调用实现模块：

```bash
PYTHONPATH=source python source/video_teleop/tools/camera_calibration.py --help
```

**验证标定（去畸变预览）：**

```bash
python scripts/teleoperation/video_teleop/camera_calibration.py \
    --camera 6 \
    --verify scripts/teleoperation/video_teleop/config/camera_calibration.yaml
```

### 2.3 手部关节映射（人手 → Shadow Hand）

按工具提示完成「张开」「握拳」等采样，生成关节范围映射：

```bash
python source/video_teleop/tools/calibrate_hand_ranges.py --camera 6 --side left
```

默认写入 `scripts/teleoperation/video_teleop/config/hand_calibration.yaml`；可用 `--output` 指定路径。

---

## 3. 完整遥操作（UR10e + Shadow Hand，单任务）

默认通过 **ZeroMQ IPC** 通信：`ipc:///tmp/shadowhand_teleop_video.ipc`。发送端须先启动（PUB），再启动接收端（SUB）。

### 终端 1 — 发送端

```bash
conda activate video_teleoperator
python scripts/teleoperation/video_teleop/run_video_teleop_sender.py \
    --camera 6 \
    --hand-mode left
```

默认读取：

- 相机标定：`scripts/teleoperation/video_teleop/config/camera_calibration.yaml`（`--calibration-file`）
- 手部标定：`scripts/teleoperation/video_teleop/config/hand_calibration.yaml`（`--hand-calibration`）

常用参数：`--zmq-address`、`--send-rate`、`--disable-visualization`、`--enable-landmarks`。

### 终端 2 — 仿真里控机器人

在 **Isaac Lab** 环境中使用项目提供的 `python.sh`（或你环境中等价入口）：

```bash
conda activate env_isaaclab_510
./python.sh scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py \
    --task pour \
    --hand-mode left \
    --zmq-address ipc:///tmp/shadowhand_teleop_video.ipc
```

任务预设：`pour`、`pickup`、`inhand`。可加 `--enable-visualization` 显示手腕处 VisualCuboid；`--debug` 打印 IK 关节解。

---

## 4. 可选：仅接收 + Isaac 可视化（不控机器人）

用于确认网络与坐标系，不进行 IK / 环境 `step`：

```bash
./python.sh scripts/teleoperation/video_teleop/run_video_teleop_receiver.py \
    --zmq-address ipc:///tmp/shadowhand_teleop_video.ipc \
    --enable-visualization \
    --hand-mode left
```

`--tag0-world-*` / `--tag1-hand-*` / `--flip-*` 与任务脚本一致，用于对齐世界系与手腕标记。

---

## 5. 故障排查

- **收不到数据**：先起发送端再起接收端；IPC 路径双方必须一致；PUB/SUB 有 “slow joiner”，接收端启动后等约 0.1～1 s。
- **发送端与 Isaac 同机 GPU/CPU 争用**：发送端已限制 OpenMP/MediaPipe 线程数；仍异常可试 `--disable-visualization` 或降低 `--send-rate`。
- **更多架构与协议**：见 `source/video_teleop/docs/README.md`。
