# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""GelSight R15 / lab Xense render cfg (Taxim bg + polycalib + optional FOTS markers)."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.sensors import GELSIGHT_R15_CFG

from .visuotactile_sensor_cfg import GelSightRenderCfg

# Native advisor mp4 resolution (width, height).
XENSE_LAB_IMAGE_WIDTH = 400
XENSE_LAB_IMAGE_HEIGHT = 700
# Qianjue Xense datasheet: sensing area 17.5 mm (width) x 29.5 mm (height) @ 400x700.
XENSE_LAB_SENSING_MM = (17.5, 29.5)
XENSE_LAB_MM_PER_PIXEL = XENSE_LAB_SENSING_MM[0] / float(XENSE_LAB_IMAGE_WIDTH)


def local_gelsight_r15_data_dir() -> Path:
    return Path(__file__).resolve().parent / "gelsight_r15_data"


def local_xense_lab_data_dir() -> Path:
    return Path(__file__).resolve().parent / "xense_lab_data"


def _local_gelsight_render_cfg(**overrides) -> GelSightRenderCfg:
    """Build ViTacLab :class:`GelSightRenderCfg` (supports marker fields) from R15 defaults."""

    base = GELSIGHT_R15_CFG
    fields = {
        "base_data_path": f"{ISAACLAB_NUCLEUS_DIR}/TacSL",
        "sensor_data_dir_name": base.sensor_data_dir_name,
        "background_path": base.background_path,
        "calib_path": base.calib_path,
        "real_background": base.real_background,
        "image_height": base.image_height,
        "image_width": base.image_width,
        "num_bins": base.num_bins,
        "mm_per_pixel": base.mm_per_pixel,
        "enable_marker_simulation": False,
        "marker_pattern": "gelsight",
        "marker_rest_path": "",
    }
    fields.update(overrides)
    return GelSightRenderCfg(**fields)


def calibrated_gelsight_r15_cfg(
    *,
    prefer_local: bool = True,
    enable_marker_simulation: bool = False,
    marker_pattern: str = "gelsight",
):
    """Return GelSight R15 Taxim cfg; default uses on-disk real calibration when present."""

    marker_kwargs = {
        "enable_marker_simulation": bool(enable_marker_simulation),
        "marker_pattern": str(marker_pattern),
    }
    if prefer_local:
        local_dir = local_gelsight_r15_data_dir()
        if (local_dir / "bg.jpg").is_file() and (local_dir / "polycalib.npz").is_file():
            return _local_gelsight_render_cfg(
                base_data_path=str(local_dir.parent),
                sensor_data_dir_name=local_dir.name,
                **marker_kwargs,
            )
    return _local_gelsight_render_cfg(
        base_data_path=f"{ISAACLAB_NUCLEUS_DIR}/TacSL",
        sensor_data_dir_name="gelsight_r15_data",
        **marker_kwargs,
    )


def advisor_xense_render_cfg(
    *,
    enable_marker_simulation: bool = True,
    marker_pattern: str = "xense",
    fitted_params_path: str | Path | None = None,
):
    """Lab Xense advisor cfg: native 400x700, clean bg, lab marker rest coordinates."""

    extra: dict = {}
    if fitted_params_path is not None:
        path = Path(fitted_params_path).expanduser()
        if path.is_file():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            rec = data.get("recommended_gelsight_render_cfg", {})
            gain = rec.get("marker_displacement_gain")
            if gain is not None:
                extra["marker_displacement_gain"] = float(gain)
            ths = rec.get("taxim_height_scale")
            if ths is None:
                ths = data.get("recommended_force_render_k_ref_scale")
            if ths is not None:
                extra["taxim_height_scale"] = float(ths)
            pattern = rec.get("marker_pattern")
            if pattern:
                marker_pattern = str(pattern)

    extra.setdefault("taxim_height_scale", 0.45)

    local_dir = local_xense_lab_data_dir()
    base = GELSIGHT_R15_CFG
    bg_name = "bg_clean.jpg" if (local_dir / "bg_clean.jpg").is_file() else "bg.jpg"
    marker_rest = "marker_rest.npy" if (local_dir / "marker_rest.npy").is_file() else ""

    cfg = _local_gelsight_render_cfg(
        base_data_path=str(local_dir.parent),
        sensor_data_dir_name=local_dir.name,
        background_path=bg_name,
        image_width=XENSE_LAB_IMAGE_WIDTH,
        image_height=XENSE_LAB_IMAGE_HEIGHT,
        num_bins=base.num_bins,
        mm_per_pixel=float(XENSE_LAB_MM_PER_PIXEL),
        enable_marker_simulation=bool(enable_marker_simulation),
        marker_pattern=str(marker_pattern) if enable_marker_simulation else "none",
        marker_rest_path=marker_rest,
        marker_max_displacement_px=3.0,
        marker_shear_gain=0.0,
        marker_deadband_mm=0.008,
        marker_height_taxim_mm_max=80.0,
        marker_height_scale=0.0033,
        **extra,
    )
    if not (local_dir / bg_name).is_file() or not (local_dir / "polycalib.npz").is_file():
        # Fallback until import --install-bg has been run.
        return calibrated_gelsight_r15_cfg(
            prefer_local=True,
            enable_marker_simulation=enable_marker_simulation,
            marker_pattern=marker_pattern,
        )
    return cfg


def validation_gelsight_render_cfg(
    *,
    enable_marker: bool = True,
    marker_pattern: str = "gelsight",
    fitted_params_path: str | Path | None = None,
    profile: str = "cylinder",
):
    """Render cfg for ViTacSim NF/Shear validation demos (Taxim + optional FOTS markers)."""

    if profile == "advisor":
        return advisor_xense_render_cfg(
            enable_marker_simulation=enable_marker,
            marker_pattern=marker_pattern,
            fitted_params_path=fitted_params_path,
        )

    extra: dict = {}
    if fitted_params_path is not None:
        path = Path(fitted_params_path).expanduser()
        if path.is_file():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            rec = data.get("recommended_gelsight_render_cfg", {})
            gain = rec.get("marker_displacement_gain")
            if gain is not None:
                extra["marker_displacement_gain"] = float(gain)
            pattern = rec.get("marker_pattern")
            if pattern:
                marker_pattern = str(pattern)

    cfg = calibrated_gelsight_r15_cfg(
        prefer_local=True,
        enable_marker_simulation=bool(enable_marker),
        marker_pattern=str(marker_pattern) if enable_marker else "none",
    )
    if extra:
        cfg = cfg.replace(**extra)
    return cfg
