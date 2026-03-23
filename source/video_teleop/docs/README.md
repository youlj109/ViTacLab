# `video_teleop` package documentation

Python 包路径：`source/video_teleop`（在仓库根目录下将 `source` 加入 `PYTHONPATH` 后 `import video_teleop`）。

## 作用

将 **单目 RGB 摄像头** 与 **MediaPipe Hands** 结合，估计双手关节与（在 AprilTag 可用时）手腕位姿，通过 **ZeroMQ** 将数据发给 **Isaac Sim** 侧进程；仿真侧可只做可视化，或通过 **IK** 驱动 **UR10e + Shadow Hand** 与任务环境交互。

## 目录结构

| 路径 | 说明 |
|------|------|
| `config_paths.py` | 标定 YAML 的默认路径（指向 `scripts/teleoperation/video_teleop/config/`） |
| `core/video_listener.py` | 采集帧、MediaPipe、手腕/AprilTag 融合、可视化 |
| `core/mediapipe_shadowhand.py` | 手部关键点 → Shadow Hand 24 维关节 |
| `core/arm_pose_estimator.py` | AprilTag 手腕位姿（依赖相机内参） |
| `core/video_teleop_sender.py` | 打包观测 → ZMQ PUB（msgpack） |
| `core/video_teleop_receiver.py` | ZMQ SUB → 解析最新一帧消息 |
| `core/video_teleop_control.py` | 手腕位姿 + Shadow 关节 → UR10e IK → 臂+手目标 |
| `core/shadowhand_joints.py` | Shadow Hand 关节命名约定 |
| `tools/list_cameras.py` | 枚举摄像头索引 |
| `tools/camera_calibration.py` | 棋盘格标定，写 YAML |
| `tools/calibrate_hand_ranges.py` | 人手开合采样 → 关节范围 YAML |
| `tools/get_realsense_rgbd.py` | RealSense 调试采集（非主流程） |

命令行封装在 **`scripts/teleoperation/video_teleop/`**（见该目录 [README.md](../../../scripts/teleoperation/video_teleop/README.md)）。

## 延伸阅读

- [ENGINEERING_SUMMARY.md](./ENGINEERING_SUMMARY.md)：消息格式、坐标系、与 Isaac 侧对接要点。
