"""从 play.py 等保存的 .npz 生成视频：所有键名以 ``camera`` 结尾的 RGB 序列、触觉 RGB、触觉力场（法向+切向，与 TacSL demo 一致）。"""

from __future__ import annotations

import argparse
import os
from typing import Any

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


def load_camera_arrays_by_suffix(data: Any) -> dict[str, np.ndarray]:
    """加载 ``.npz`` 中所有键名以 ``camera`` 结尾的数组（键名排序，便于输出顺序稳定）。"""
    return {k: np.asarray(data[k]) for k in sorted(data.files) if k.endswith("camera")}


def episode_frame_count(
    npz_path: str,
    cameras: dict[str, np.ndarray],
    joint_pos: np.ndarray | None,
) -> int:
    """确定本集时间步 T：优先与各路 camera 对齐；若无 camera 键则使用 ``joint_pos``。"""
    if cameras:
        lengths = {k: int(v.shape[0]) for k, v in cameras.items()}
        t0 = next(iter(lengths.values()))
        if any(t != t0 for t in lengths.values()):
            raise ValueError(f"{npz_path}: 以 camera 结尾的键时间步不一致: {lengths}")
        if joint_pos is not None and int(joint_pos.shape[0]) != t0:
            print(f"警告: {npz_path} joint_pos T={joint_pos.shape[0]} 与 camera T={t0} 不一致，按 camera 帧数导出")
        return t0
    if joint_pos is not None:
        return int(joint_pos.shape[0])
    raise ValueError(f"{npz_path}: 无键名以 camera 结尾且无 joint_pos，无法确定帧数")


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

    all_camera_writers: dict[str, Any] = {}
    video_writer_tactile_all = None
    video_writer_tactile_ff_all = None

    for file in data_files:
        print(file)
        if not file.endswith(".npz"):
            continue
        path = os.path.join(latest_data_dir, file)
        data = np.load(path, allow_pickle=True)
        joint_pos = np.asarray(data["joint_pos"]) if "joint_pos" in data.files else None
        if joint_pos is not None:
            print(joint_pos.shape)

        cameras = load_camera_arrays_by_suffix(data)
        n_frames = episode_frame_count(path, cameras, joint_pos)

        tactile_rgb_image = data["tactile_rgb_image"] if "tactile_rgb_image" in data.files else None
        print("tactile_rgb_image", tactile_rgb_image.shape, tactile_rgb_image.min(), tactile_rgb_image.max())
        has_ff = "tactile_normal_force" in data.files and "tactile_shear_force" in data.files
        tactile_normal_force = data["tactile_normal_force"] if has_ff else None
        print("tactile_normal_force", tactile_normal_force.shape, tactile_normal_force.min(), tactile_normal_force.max())
        tactile_shear_force = data["tactile_shear_force"] if has_ff else None
        print("tactile_shear_force", tactile_shear_force.shape, tactile_shear_force.min(), tactile_shear_force.max())

        if tactile_rgb_image is not None and tactile_rgb_image.shape[0] != n_frames:
            raise ValueError(
                f"{file}: tactile_rgb_image T={tactile_rgb_image.shape[0]}, 期望与主时间步 {n_frames} 一致"
            )
        if has_ff:
            assert tactile_normal_force is not None and tactile_shear_force is not None
            if tactile_normal_force.shape[0] != n_frames or tactile_shear_force.shape[0] != n_frames:
                raise ValueError(
                    f"{file}: 力场 T normal={tactile_normal_force.shape[0]} "
                    f"shear={tactile_shear_force.shape[0]}, 期望 {n_frames}"
                )

        stem = file[:-4]
        camera_writers = {
            k: imageio.get_writer(os.path.join(show_dir, f"{stem}_{k}.mp4"), fps=args.fps)
            for k in cameras
        }
        video_writer_tactile = None
        if tactile_rgb_image is not None:
            video_writer_tactile = imageio.get_writer(os.path.join(show_dir, f"{stem}_tactile.mp4"), fps=args.fps)
        video_writer_tactile_ff = None
        if has_ff:
            video_writer_tactile_ff = imageio.get_writer(
                os.path.join(show_dir, f"{stem}_tactile_force_field.mp4"), fps=args.fps
            )

        for i in range(n_frames):
            for key, frames in cameras.items():
                frame = frames[i]
                camera_writers[key].append_data(frame)
                if key not in all_camera_writers:
                    all_camera_writers[key] = imageio.get_writer(
                        os.path.join(show_dir, f"all_{key}.mp4"), fps=args.fps_all
                    )
                all_camera_writers[key].append_data(frame)

            if tactile_rgb_image is not None and video_writer_tactile is not None:
                tr = concat_tactile_rgb_image(tactile_rgb_image[i])
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

        for w in camera_writers.values():
            w.close()
        if video_writer_tactile is not None:
            video_writer_tactile.close()
        if video_writer_tactile_ff is not None:
            video_writer_tactile_ff.close()

    for w in all_camera_writers.values():
        w.close()
    if video_writer_tactile_all is not None:
        video_writer_tactile_all.close()
    if video_writer_tactile_ff_all is not None:
        video_writer_tactile_ff_all.close()


if __name__ == "__main__":
    main()