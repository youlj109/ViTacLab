"""从 play.py 等保存的 .npz 生成视频：相机 RGB、触觉 RGB、触觉力场（法向+切向，与 TacSL demo 一致）。"""

from __future__ import annotations

import argparse
import os

import cv2
import imageio
import numpy as np


def compute_tactile_shear_image(
    tactile_normal_force: np.ndarray,
    tactile_shear_force: np.ndarray,
    normal_force_threshold: float = 0.00008,
    shear_force_threshold: float = 0.0005,
    resolution: int = 30,
) -> np.ndarray:
    """与 IsaacLab ``tacsl_sensor.py`` / ``visuotactile_render.compute_tactile_shear_image`` 相同逻辑。"""
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]
    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)
    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            sx = float(tactile_shear_force[row, col, 0])
            sy = float(tactile_shear_force[row, col, 1])
            nf_v = float(tactile_normal_force[row, col])
            loc1_x = float(loc0_x) + sx / shear_force_threshold * resolution
            loc1_y = float(loc0_y) + sy / shear_force_threshold * resolution
            # OpenCV 4.x：color 须为 Python 标量，不能是 numpy 标量
            color = (
                0.0,
                float(max(0.0, 1.0 - nf_v / normal_force_threshold)),
                float(min(1.0, nf_v / normal_force_threshold)),
            )
            cv2.arrowedLine(
                imgs_tactile,
                (int(loc0_y), int(loc0_x)),
                (int(loc1_y), int(loc1_x)),
                color,
                6,
                tipLength=0.4,
            )
    return imgs_tactile


def concat_tactile_rgb_image(tactile_rgb_image):
    """将 (5, H, W, 3) 或长度为 5 的数组序列沿宽度拼接。"""
    x = np.asarray(tactile_rgb_image)
    if x.ndim == 4 and x.shape[0] > 1:
        return np.concatenate([x[i] for i in range(x.shape[0])], axis=1)
    return np.concatenate(tactile_rgb_image, axis=1)


def _squeeze_time_env_normal(nf: np.ndarray) -> np.ndarray:
    """得到 (5, H, W)。"""
    nf = np.asarray(nf, dtype=np.float32)
    if nf.ndim == 5 and nf.shape[-1] == 1:
        nf = nf[..., 0]
    return nf


def frame_tactile_force_field(
    tactile_normal_force_t: np.ndarray,
    tactile_shear_force_t: np.ndarray,
    normal_thr: float,
    shear_thr: float,
    resolution: int,
) -> np.ndarray:
    """单帧：5 个指尖力场图横向拼接（与触觉 RGB 拼接方向一致）。"""
    nf = _squeeze_time_env_normal(tactile_normal_force_t)
    sf = np.asarray(tactile_shear_force_t, dtype=np.float32)
    if nf.shape[0] != sf.shape[0]:
        raise ValueError(f"指尖数不一致: normal {nf.shape}, shear {sf.shape}")
    pieces = []
    for f in range(nf.shape[0]):
        img = compute_tactile_shear_image(
            nf[f],
            sf[f],
            normal_force_threshold=normal_thr,
            shear_force_threshold=shear_thr,
            resolution=resolution,
        )
        pieces.append((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.concatenate(pieces, axis=1)


def main():
    parser = argparse.ArgumentParser(description="将记录目录中的 .npz 转为 mp4（含触觉力场）。")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/rsl_rl/Isaac-Repose-Cube-Shadow-Tactile-Direct-v0_42",
        help="含子目录（按时间戳）的根路径；默认取最新子目录",
    )
    parser.add_argument("--show_dir", type=str, default="tmp", help="输出 mp4 目录")
    parser.add_argument("--fps", type=float, default=5.0, help="每集视频帧率")
    parser.add_argument("--fps_all", type=float, default=200.0, help="合并长视频的帧率")
    parser.add_argument("--normal_thr", type=float, default=0.00008, help="法向力可视化阈值（与 TacSL 默认一致）")
    parser.add_argument("--shear_thr", type=float, default=0.0005, help="切向力可视化阈值（与 TacSL 默认一致）")
    parser.add_argument("--ff_resolution", type=int, default=30, help="力场栅格放大倍数（每格边长像素）")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    subdirs = sorted(os.listdir(data_dir))
    latest_data_dir = os.path.join(data_dir, subdirs[-1])

    show_dir = os.path.abspath(os.path.expanduser(args.show_dir))
    os.makedirs(show_dir, exist_ok=True)

    data_files = sorted(os.listdir(latest_data_dir))

    video_writer_camera_all = imageio.get_writer(os.path.join(show_dir, "all_camera.mp4"), fps=args.fps_all)
    video_writer_tactile_all = None
    video_writer_tactile_ff_all = None

    for file in data_files:
        print(file)
        if not file.endswith(".npz"):
            continue
        path = os.path.join(latest_data_dir, file)
        data = np.load(path, allow_pickle=True)
        print(data["joint_pos"].shape)

        third_person_camera = data["camera_rgb"]
        tactile_rgb_image = data["tactile_rgb_image"] if "tactile_rgb_image" in data.files else None
        has_ff = "tactile_normal_force" in data.files and "tactile_shear_force" in data.files
        tactile_normal_force = data["tactile_normal_force"] if has_ff else None
        tactile_shear_force = data["tactile_shear_force"] if has_ff else None

        stem = file[:-4]
        video_writer_camera = imageio.get_writer(os.path.join(show_dir, f"{stem}_camera.mp4"), fps=args.fps)
        video_writer_tactile = None
        if tactile_rgb_image is not None:
            video_writer_tactile = imageio.get_writer(os.path.join(show_dir, f"{stem}_tactile.mp4"), fps=args.fps)
        video_writer_tactile_ff = None
        if has_ff:
            video_writer_tactile_ff = imageio.get_writer(
                os.path.join(show_dir, f"{stem}_tactile_force_field.mp4"), fps=args.fps
            )

        for i in range(third_person_camera.shape[0]):
            video_writer_camera.append_data(third_person_camera[i])
            video_writer_camera_all.append_data(third_person_camera[i])

            if tactile_rgb_image is not None and video_writer_tactile is not None:
                tr = concat_tactile_rgb_image(tactile_rgb_image[i])
                print(tr.shape)
                video_writer_tactile.append_data(tr)
                if video_writer_tactile_all is None:
                    video_writer_tactile_all = imageio.get_writer(
                        os.path.join(show_dir, "all_tactile.mp4"), fps=args.fps_all
                    )
                video_writer_tactile_all.append_data(tr)

            if has_ff and video_writer_tactile_ff is not None:
                assert tactile_normal_force is not None and tactile_shear_force is not None
                ff_frame = frame_tactile_force_field(
                    tactile_normal_force[i],
                    tactile_shear_force[i],
                    normal_thr=args.normal_thr,
                    shear_thr=args.shear_thr,
                    resolution=args.ff_resolution,
                )
                video_writer_tactile_ff.append_data(ff_frame)
                if video_writer_tactile_ff_all is None:
                    video_writer_tactile_ff_all = imageio.get_writer(
                        os.path.join(show_dir, "all_tactile_force_field.mp4"), fps=args.fps_all
                    )
                video_writer_tactile_ff_all.append_data(ff_frame)

        video_writer_camera.close()
        if video_writer_tactile is not None:
            video_writer_tactile.close()
        if video_writer_tactile_ff is not None:
            video_writer_tactile_ff.close()

    video_writer_camera_all.close()
    if video_writer_tactile_all is not None:
        video_writer_tactile_all.close()
    if video_writer_tactile_ff_all is not None:
        video_writer_tactile_ff_all.close()


if __name__ == "__main__":
    main()
