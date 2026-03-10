import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    # 加载数据 (仅支持 .pt)
    data = torch.load(args.path, map_location="cpu", weights_only=False)
    if "viewport_rgb" not in data:
        print(f"错误: 数据中未发现 viewport_rgb 键。现有键: {list(data.keys())}")
        return

    img_data = data["viewport_rgb"].cpu().numpy()
    num_frames = img_data.shape[0]

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(bottom=0.15)

    # 处理数据格式 [T, H, W, 3] 或 [T, Env, H, W, 3]
    display_data = img_data[:, 0] if img_data.ndim == 5 else img_data

    # 归一化到 0-1 (如果是 float)
    if display_data.dtype != np.uint8:
        display_data = np.clip(display_data, 0, 1)

    im_display = ax.imshow(display_data[0])
    ax.set_title(f"Viewport Large View - Frame 0")
    ax.axis("off")

    # 添加滑块
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        im_display.set_data(display_data[idx])
        ax.set_title(f"Viewport Large View - Frame {idx}/{num_frames-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # 键盘支持
    def on_key(event):
        if event.key == "right":
            slider.set_val(min(num_frames-1, slider.val + 1))
        elif event.key == "left":
            slider.set_val(max(0, slider.val - 1))

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

if __name__ == "__main__":
    main()