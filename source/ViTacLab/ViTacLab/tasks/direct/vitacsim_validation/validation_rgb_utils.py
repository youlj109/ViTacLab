"""Background-subtracted RGB visualization for validation (sim − bg.jpg)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class BgDiffCfg:
    mode: str = "enhanced"
    gain: float = 3.8
    gamma: float = 0.50
    pct_lo: float = 0.5
    pct_hi: float = 99.2
    neutral: int = 128
    min_mag: float = 0.25


DEFAULT_BG_DIFF = BgDiffCfg()


def load_bg_rgb(render_cfg) -> np.ndarray | None:
    bg_path = Path(render_cfg.base_data_path) / render_cfg.sensor_data_dir_name / "bg.jpg"
    if not bg_path.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    return np.asarray(Image.open(bg_path).convert("RGB"), dtype=np.float32)


def load_bg_from_path(bg_path: Path) -> np.ndarray | None:
    if not bg_path.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    return np.asarray(Image.open(bg_path).convert("RGB"), dtype=np.float32)


def _align_bg(bg: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if bg.shape[:2] == (h, w):
        return bg
    try:
        from PIL import Image
    except ImportError:
        return bg
    bg_img = Image.fromarray(bg.astype(np.uint8)).resize((w, h))
    return np.asarray(bg_img, dtype=np.float32)


def rgb_diff_vis(rgb_u8: np.ndarray, bg: np.ndarray | None, *, cfg: BgDiffCfg = DEFAULT_BG_DIFF) -> np.ndarray | None:
    if bg is None:
        return None
    sim = rgb_u8.astype(np.float32)
    bg_a = _align_bg(bg, sim.shape[:2])
    diff = sim - bg_a

    if cfg.mode == "raw":
        return np.clip(diff * cfg.gain + cfg.neutral, 0, 255).astype(np.uint8)

    if cfg.mode == "heat":
        mag = np.linalg.norm(diff, axis=-1)
        sign = np.sign(diff.mean(axis=-1))
        lo, hi = _percentile_span(mag, cfg)
        norm = np.clip((mag - lo) / (hi - lo + 1e-6), 0.0, 1.0) ** cfg.gamma
        try:
            import matplotlib.cm as cm

            base = cm.get_cmap("inferno")(norm)[..., :3]
            sign_mask = sign[..., None] < 0
            base = np.where(sign_mask, base * np.array([0.55, 0.75, 1.0]), base)
            return (base * 255).astype(np.uint8)
        except ImportError:
            gray = (norm * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

    mag = np.linalg.norm(np.abs(diff), axis=-1)
    lo, hi = _percentile_span(mag, cfg)
    norm = np.clip((mag - lo) / (hi - lo + 1e-6), 0.0, 1.0) ** cfg.gamma
    norm = np.clip(norm * cfg.gain, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _percentile_span(mag: np.ndarray, cfg: BgDiffCfg) -> tuple[float, float]:
    active = mag[mag > cfg.min_mag]
    if active.size < 16:
        return 0.0, max(float(mag.max()), 1.0)
    lo = float(np.percentile(active, cfg.pct_lo))
    hi = float(np.percentile(active, cfg.pct_hi))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def save_rgb_diff_bg(path: Path, rgb_u8: torch.Tensor | np.ndarray, bg: np.ndarray | None, *, cfg: BgDiffCfg = DEFAULT_BG_DIFF) -> None:
    if bg is None:
        return
    if isinstance(rgb_u8, torch.Tensor):
        arr = rgb_u8.detach().cpu().numpy()
    else:
        arr = np.asarray(rgb_u8)
    vis = rgb_diff_vis(arr, bg, cfg=cfg)
    if vis is None:
        return
    try:
        from PIL import Image
    except ImportError:
        np.save(path.with_suffix(".npy"), vis)
        return
    Image.fromarray(vis).save(path)


def bg_diff_cfg_dict(cfg: BgDiffCfg = DEFAULT_BG_DIFF) -> dict[str, float | str | int]:
    return {
        "bg_diff_mode": cfg.mode,
        "bg_diff_gain": cfg.gain,
        "bg_diff_gamma": cfg.gamma,
        "bg_diff_pct_lo": cfg.pct_lo,
        "bg_diff_pct_hi": cfg.pct_hi,
    }
