import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1️⃣ 创建 pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 2️⃣ L515 推荐分辨率（稳定）
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 3️⃣ 启动
    profile = pipeline.start(config)

    # 4️⃣ 深度对齐到 RGB
    align = rs.align(rs.stream.color)

    # 5️⃣ 深度 scale（很重要）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale:", depth_scale)  # L515 一般是 0.001

    try:
        while True:
            # 6️⃣ 获取帧
            frames = pipeline.wait_for_frames()

            # 7️⃣ 对齐
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 8️⃣ 转 numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 9️⃣ 深度可视化（伪彩色）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # ������ 拼接显示（resize depth 以匹配 RGB）
            depth_colormap_resized = cv2.resize(
                depth_colormap,
                (color_image.shape[1], color_image.shape[0])
            )

            combined = np.hstack((color_image, depth_colormap_resized))

            cv2.imshow("RGB | Depth (Aligned)", combined)

            # 1️⃣1️⃣ 读取中心点距离（单位：米）
            h, w = depth_image.shape
            distance = depth_frame.get_distance(w // 2, h // 2)
            print(f"Center distance: {distance:.3f} m")

            # ESC 退出
            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()