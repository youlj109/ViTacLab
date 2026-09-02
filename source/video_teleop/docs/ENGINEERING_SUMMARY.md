# Video teleoperation — engineering summary

本文描述 `source/video_teleop` 与 `scripts/teleoperation/video_teleop` 的职责划分、运行时数据流和主要工程约定，便于扩展与联调。

## 1. 进程划分

| 进程 | 典型 Conda 环境 | Isaac |
|------|------------------|--------|
| Sender | `video_teleoperator`（OpenCV + MediaPipe + pyzmq） | 否 |
| Receiver / Task | Isaac Lab 环境 | 是 |

发送端 **不** 导入 Isaac；接收端与任务脚本通过 `AppLauncher` 启动 Omniverse。

## 2. 端到端数据流

1. **VideoListener**（`core/video_listener.py`）从 `cv2.VideoCapture` 读帧；用 **MediaPipe** 得到手部关键点并映射为 Shadow Hand 24 维关节；若启用 AprilTag，**ArmPoseEstimator** 在已知相机内参下估计手腕位姿。
2. **VideoTeleopSender** 以固定频率将观测打包，经 **ZeroMQ PUB** 发出。使用 **msgpack** 序列化。
3. **VideoTeleopReceiver** 在独立线程中 **SUB** 订阅，保留 **最新一条** 消息供主线程读取（适合实时控制、允许丢帧）。
4. **任务脚本**（`run_video_teleop_ur10e_shadowhand_single.py`）读取最新消息，将 `robot_frame` 下手腕位姿经 **tag0/tag1 与世界系** 变换后，交给 **VideoTeleopControl** 做 UR10e IK，再映射为环境 `action`。

默认 ZMQ 地址：`ipc:///tmp/shadowhand_teleop_video.ipc`。IPC 仅本机；跨机可改为 `tcp://`。

## 3. 标定文件

默认路径由 `config_paths.video_teleop_config_dir()` 指向：

`scripts/teleoperation/video_teleop/config/`

- **camera_calibration.yaml**  
  - `camera_matrix`（3×3）、`distortion_coefficients`、可选 `image_width` / `image_height`、棋盘参数与 `reprojection_error`。  
  - 由 `tools/camera_calibration.py` 生成，供 AprilTag 与几何一致投影。

- **hand_calibration.yaml**  
  - `joint_ranges`：每个关节 `human_min`/`human_max` → `robot_min`/`robot_max` 分段线性映射。  
  - 由 `tools/calibrate_hand_ranges.py` 根据采样生成。

发送端 CLI 参数命名：**`--calibration-file`** 对应 **相机** YAML；**`--hand-calibration`** 对应 **手部** YAML（与 `VideoListener` 构造参数一致：`calibration_file`=手，`camera_calibration_file`=相机）。

## 4. 坐标系与变换（仿真侧）

发送端主要输出 **相机系 / 算法内部系** 下的观测；消息里包含 **`robot_frame`** 字段，供接收端与任务脚本与 **世界系、机械臂基座** 对齐。

任务脚本与 `run_video_teleop_receiver.py` 使用同一类变换：

- `T_tag0_world`：世界到 AprilTag「基」标记（默认欧拉可由 CLI 覆盖）。
- `T_tag1_hand`：手腕标记到「手」刚体。
- 从消息中取 `robot_frame.wrist_position` / `wrist_orientation`（欧拉 xyz）构造 `T_tag0_tag1`，再链式乘到世界系下的手腕位姿。

可选 **`--flip-axis`** / **`--flip-where`** 用于镜像或对齐不同安装方式。

**VideoTeleopControl**（`core/video_teleop_control.py`）将世界系下手腕位姿转到 **臂基座**，对 UR10e 子链做 IK（6 维臂 + 24 维手目标来自遥操作）。

任务脚本中还有 **`--pos-scale`**（默认 `(4,1,2)`）对平移进行分量缩放，用于匹配仿真尺度；调参时与 `--tag1-hand-*`、`--arm-base-*` 一并考虑。

## 5. 消息内容（概念）

序列化对象为字典（具体键以 `video_teleop_sender` 打包逻辑为准），通常包含：

- 头部：`sequence`、时间戳等。
- `left_hand` / `right_hand`：`detected`、关节角、`robot_frame`（若可用）、可选 landmarks。

接收端可打印或仅更新可视化；任务脚本只关心检测到的手、IK 是否成功，失败时可保持上一帧目标。

## 6. 稳定性与线程

- ZMQ **PUB/SUB** 存在 “slow joiner”，发布端绑定后建议短暂延时再连订阅端。
- MediaPipe / OpenCV / BLAS 多线程在 **与 Isaac 同机** 时可能触发调度问题；发送端入口已限制若干 `*_NUM_THREADS` 与 OpenCV 线程数，必要时关闭可视化（`--disable-visualization`）。
- **VideoListener** 内对 `get_pose` 使用锁，避免发送线程与主线程并发进入同一推理路径。

## 7. 扩展点

- 新任务：在 `run_video_teleop_ur10e_shadowhand_single.py` 的 `_TASK_PRESETS` 中注册 env/cfg，或直接用 `--env` / `--cfg` 传入 `module:Class`。
- 新相机：换索引并重新跑相机标定；分辨率变化时建议重标定。
- 仅调试链路：先跑 sender + `run_video_teleop_receiver.py --enable-visualization`，确认位姿再开任务脚本。
