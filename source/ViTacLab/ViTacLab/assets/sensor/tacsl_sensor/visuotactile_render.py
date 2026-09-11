# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy
import torch

from isaaclab.utils.assets import retrieve_file_path

from .visuotactile_marker import MarkerSimulator

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .visuotactile_sensor_cfg import GelSightRenderCfg


def compute_tactile_shear_image(
    tactile_normal_force: np.ndarray,
    tactile_shear_force: np.ndarray,
    normal_force_threshold: float = 0.00008,
    shear_force_threshold: float = 0.0005,
    resolution: int = 30,
) -> np.ndarray:
    """Visualize the tactile shear field.

    This function creates a visualization of tactile forces using arrows to represent shear forces
    and color coding to represent normal forces. The thresholds are used to normalize forces for
    visualization, chosen empirically to provide clear visual representation.

    Args:
        tactile_normal_force: Array of tactile normal forces. Shape: (H, W).
        tactile_shear_force: Array of tactile shear forces. Shape: (H, W, 2).
        normal_force_threshold: Threshold for normal force visualization. Defaults to 0.00008.
        shear_force_threshold: Threshold for shear force visualization. Defaults to 0.0005.
        resolution: Resolution for the visualization. Defaults to 30.

    Returns:
        Image visualizing the tactile shear forces. Shape: (H * resolution, W * resolution, 3).
    """
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)

    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_shear_force[row, col][0] / shear_force_threshold * resolution
            loc1_y = loc0_y + tactile_shear_force[row, col][1] / shear_force_threshold * resolution
            color = (
                0.0,
                max(0.0, 1.0 - tactile_normal_force[row][col] / normal_force_threshold),
                min(1.0, tactile_normal_force[row][col] / normal_force_threshold),
            )

            cv2.arrowedLine(
                imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), color, 6, tipLength=0.4
            )

    return imgs_tactile


def compute_penetration_depth(
    penetration_depth_img: np.ndarray, resolution: int = 5, depth_multiplier: float = 300.0
) -> np.ndarray:
    """Visualize the penetration depth.

    Args:
        penetration_depth_img: Image of penetration depth. Shape: (H, W).
        resolution: Resolution for the upsampling; each pixel expands to a (res x res) block. Defaults to 5.
        depth_multiplier: Multiplier for the depth values. Defaults to 300.0 (scales ~3.3mm to 1.0).
            (e.g. typical Gelsight sensors have maximum penetration depths < 2.5mm,
            see https://dspace.mit.edu/handle/1721.1/114627).

    Returns:
        Upsampled image visualizing the penetration depth. Shape: (H * resolution, W * resolution).
    """
    # penetration_depth_img_upsampled = penetration_depth.repeat(resolution, 0).repeat(resolution, 1)
    penetration_depth_img_upsampled = np.kron(penetration_depth_img, np.ones((resolution, resolution)))
    penetration_depth_img_upsampled = np.clip(penetration_depth_img_upsampled, 0.0, 1.0) * depth_multiplier
    return penetration_depth_img_upsampled


class GelsightRender:
    """Class to handle GelSight rendering using the Taxim example-based approach from :cite:t:`si2022taxim`.

    Reference:
        Si, Z., & Yuan, W. (2022). Taxim: An example-based simulation model for GelSight
        tactile sensors. IEEE Robotics and Automation Letters, 7(2), 2361-2368.
        https://arxiv.org/abs/2109.04027
    """

    def __init__(self, cfg: GelSightRenderCfg, device: str | torch.device):
        """Initialize the GelSight renderer.

        Args:
            cfg: Configuration object for the GelSight sensor.
            device: Device to use ('cpu' or 'cuda').

        Raises:
            ValueError: If :attr:`GelSightRenderCfg.mm_per_pixel` is zero or negative.
            FileNotFoundError: If render data files cannot be retrieved.
        """
        self.cfg = cfg
        self.device = device

        # Validate configuration parameters
        eps = 1e-9
        if self.cfg.mm_per_pixel < eps:
            raise ValueError(f"Input 'mm_per_pixel' must be positive (>= {eps}), got {self.cfg.mm_per_pixel}")

        # Retrieve render data files using the configured base path
        bg_path = self._get_render_data(self.cfg.sensor_data_dir_name, self.cfg.background_path)
        calib_path = self._get_render_data(self.cfg.sensor_data_dir_name, self.cfg.calib_path)

        if bg_path is None or calib_path is None:
            raise FileNotFoundError(
                "Failed to retrieve GelSight render data files. "
                f"Base path: {self.cfg.base_data_path or 'default (Isaac Lab Nucleus)'}, "
                f"Data dir: {self.cfg.sensor_data_dir_name}"
            )

        image_height = self.cfg.image_height
        image_width = self.cfg.image_width

        self.background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
        if self.background.shape[0] != image_height or self.background.shape[1] != image_width:
            self.background = cv2.resize(
                self.background,
                (image_width, image_height),
                interpolation=cv2.INTER_AREA,
            )

        # Load calibration data directly
        calib_data = np.load(calib_path)
        calib_grad_r = calib_data["grad_r"]
        calib_grad_g = calib_data["grad_g"]
        calib_grad_b = calib_data["grad_b"]

        num_bins = self.cfg.num_bins
        [xx, yy] = np.meshgrid(range(image_width), range(image_height))
        xf = xx.flatten()
        yf = yy.flatten()
        self.A = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(image_height * image_width)]).T

        binm = num_bins - 1
        self.x_binr = 0.5 * np.pi / binm  # x [0,pi/2]
        self.y_binr = 2 * np.pi / binm  # y [-pi, pi]

        kernel_size = int(getattr(self.cfg, "taxim_smoothing_kernel_size", 5))
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = self._get_filtering_kernel(kernel_size=kernel_size)
        self.kernel = torch.tensor(kernel, dtype=torch.float, device=self.device)
        normal_kernel_size = self._normalize_kernel_size(
            int(getattr(self.cfg, "taxim_normal_smoothing_kernel_size", 1))
        )
        self._normal_smoothing_kernel = None
        if normal_kernel_size > 1:
            normal_kernel = self._get_filtering_kernel(kernel_size=normal_kernel_size)
            self._normal_smoothing_kernel = torch.tensor(
                normal_kernel, dtype=torch.float, device=self.device
            )
        edge_kernel_size = self._normalize_kernel_size(int(getattr(self.cfg, "taxim_contact_edge_denoise_kernel_size", 9)))
        edge_kernel = self._get_filtering_kernel(kernel_size=edge_kernel_size)
        self._edge_denoise_kernel = torch.tensor(edge_kernel, dtype=torch.float, device=self.device)

        self.calib_data_grad_r = torch.tensor(calib_grad_r, device=self.device)
        self.calib_data_grad_g = torch.tensor(calib_grad_g, device=self.device)
        self.calib_data_grad_b = torch.tensor(calib_grad_b, device=self.device)

        self.A_tensor = torch.tensor(self.A.reshape(image_height, image_width, 6), device=self.device).unsqueeze(0)
        self.background_tensor = torch.tensor(self.background, device=self.device)
        self._illum_gain_tensor: torch.Tensor | None = None
        self._illum_bias_tensor: torch.Tensor | None = None

        illum_ref = str(getattr(self.cfg, "taxim_illumination_reference_path", "") or "").strip()
        if illum_ref:
            illum_ref_path = os.path.expanduser(illum_ref)
            if os.path.isfile(illum_ref_path):
                ref_rgb = cv2.cvtColor(cv2.imread(illum_ref_path), cv2.COLOR_BGR2RGB)
                if ref_rgb.shape[0] != image_height or ref_rgb.shape[1] != image_width:
                    ref_rgb = cv2.resize(ref_rgb, (image_width, image_height), interpolation=cv2.INTER_AREA)
                illum_kernel_size = self._normalize_kernel_size(
                    int(getattr(self.cfg, "taxim_illumination_kernel_size", 81))
                )
                illum_kernel_np = self._get_filtering_kernel(kernel_size=illum_kernel_size)
                illum_kernel = torch.tensor(illum_kernel_np, dtype=torch.float, device=self.device)
                ref_tensor = torch.tensor(ref_rgb, dtype=torch.float, device=self.device).unsqueeze(0)
                bg_tensor = self.background_tensor.to(torch.float).unsqueeze(0)
                ref_lf = self._gaussian_filtering_rgb(ref_tensor, illum_kernel)
                bg_lf = self._gaussian_filtering_rgb(bg_tensor, illum_kernel)
                self._illum_gain_tensor = torch.clamp(ref_lf / (bg_lf + 1.0e-6), min=0.6, max=1.6)
                self._illum_bias_tensor = torch.clamp(ref_lf - bg_lf, min=-45.0, max=45.0)
            else:
                logger.warning("illumination reference image not found: %s", illum_ref_path)

        # Pre-allocate buffer for RGB output (will be resized if needed)
        self._sim_img_rgb_buffer = torch.empty((1, image_height, image_width, 3), device=self.device)

        self._marker_sim: MarkerSimulator | None = None
        self._last_marker_displacements: torch.Tensor | None = None
        enable_markers = bool(getattr(self.cfg, "enable_marker_simulation", False))
        marker_pattern = str(getattr(self.cfg, "marker_pattern", "none"))
        if enable_markers and marker_pattern != "none":
            pattern = marker_pattern
            rest_override = None
            rest_file = str(getattr(self.cfg, "marker_rest_path", "") or "").strip()
            if rest_file:
                rest_path = self._get_render_data(self.cfg.sensor_data_dir_name, rest_file)
                if rest_path and os.path.isfile(rest_path):
                    rest_override = np.load(rest_path).astype(np.float32)
            self._marker_sim = MarkerSimulator(
                pattern=pattern,  # type: ignore[arg-type]
                image_height=image_height,
                image_width=image_width,
                device=self.device,
                lambda_d=float(getattr(self.cfg, "marker_lambda_d", 0.0025)),
                displacement_gain=float(getattr(self.cfg, "marker_displacement_gain", 0.35)),
                shear_gain=float(getattr(self.cfg, "marker_shear_gain", 8.0)),
                deadband_mm=float(getattr(self.cfg, "marker_deadband_mm", 0.02)),
                blend_alpha=float(getattr(self.cfg, "marker_blend_alpha", 0.85)),
                marker_shape=str(getattr(self.cfg, "marker_shape", "disk")),
                gaussian_sigma_x_px=float(getattr(self.cfg, "marker_gaussian_sigma_x_px", 2.0)),
                gaussian_sigma_y_px=float(getattr(self.cfg, "marker_gaussian_sigma_y_px", 2.0)),
                gaussian_truncate=float(getattr(self.cfg, "marker_gaussian_truncate", 3.0)),
                max_displacement_px=float(getattr(self.cfg, "marker_max_displacement_px", 25.0)),
                rest_xy_override=rest_override,
            )
            logger.info("Gelsight marker simulation enabled (pattern=%s, M=%d).", pattern, self._marker_sim.num_markers)

        logger.info("Gelsight renderer initialization done!")

    @property
    def num_markers(self) -> int:
        """Number of simulated markers (0 if disabled)."""
        if self._marker_sim is None:
            return 0
        return self._marker_sim.num_markers

    @property
    def last_marker_displacements(self) -> torch.Tensor | None:
        """Last batch marker displacements from :meth:`render`, shape (N, M, 2) in pixels."""
        return self._last_marker_displacements

    def render(
        self,
        height_map: torch.Tensor,
        *,
        marker_height_map: torch.Tensor | None = None,
        marker_shear_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render the height map using the GelSight sensor.

        Args:
            height_map: Input height map tensor (m). Shape is (N, H, W).
            marker_height_map: Optional separate height (m) for FOTS markers; defaults to ``height_map``.
            marker_shear_map: Optional ViT shear displacement field in pixels, shape (N, H, W, 2).

        Returns:
            Rendered image tensor. Shape is (N, H, W, 3).
        """
        physical_peak_depth_mm = torch.amax(
            torch.abs(height_map), dim=(1, 2), keepdim=True
        ) * 1000.0
        taxim_scale = float(getattr(self.cfg, "taxim_height_scale", 1.0))
        if abs(taxim_scale - 1.0) > 1e-9:
            height_map = height_map * taxim_scale
        edge_denoise_blend = float(getattr(self.cfg, "taxim_contact_edge_denoise_blend", 0.0))
        if edge_denoise_blend > 1.0e-6:
            # Source-level smoothing: denoise only around the contact boundary before Taxim gradient lookup.
            depth_abs = torch.abs(height_map)
            depth_max = torch.amax(depth_abs, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            depth_n = torch.clamp(depth_abs / depth_max, min=0.0, max=1.0)
            band_center = float(getattr(self.cfg, "taxim_contact_edge_denoise_center", 0.10))
            band_width = max(float(getattr(self.cfg, "taxim_contact_edge_denoise_bandwidth", 0.10)), 1.0e-4)
            band = torch.exp(-((depth_n - band_center) ** 2) / (2.0 * band_width * band_width))
            h_blur = self._gaussian_filtering(height_map.unsqueeze(-1), self._edge_denoise_kernel).squeeze(-1)
            alpha = torch.clamp(edge_denoise_blend * band, min=0.0, max=1.0)
            height_map = (1.0 - alpha) * height_map + alpha * h_blur
        height_map = self._height_m_to_taxim_mm(height_map)
        # Gradient zero is shared by the center of a real indentation and the
        # entire no-contact background. The fitted zero-gradient bin may contain
        # a genuine center response, so gate lookup-table RGB by actual contact
        # support instead of tinting every zero-height pixel in the image.
        contact_support = torch.abs(height_map) > 1.0e-8

        grad_mag, grad_dir = self._generate_normals(height_map)

        idx_x = torch.floor(grad_mag / self.x_binr).long()
        idx_y = torch.floor((grad_dir + np.pi) / self.y_binr).long()

        # Clamp indices to valid range to prevent out-of-bounds errors
        max_idx = self.cfg.num_bins - 1
        idx_x = torch.clamp(idx_x, 0, max_idx)
        idx_y = torch.clamp(idx_y, 0, max_idx)

        params_r = self.calib_data_grad_r[idx_x, idx_y, :]
        params_g = self.calib_data_grad_g[idx_x, idx_y, :]
        params_b = self.calib_data_grad_b[idx_x, idx_y, :]

        # Reuse pre-allocated buffer, resize if batch size changed
        target_shape = (*idx_x.shape, 3)
        if self._sim_img_rgb_buffer.shape != target_shape:
            self._sim_img_rgb_buffer = torch.empty(target_shape, device=self.device)
        sim_img_rgb = self._sim_img_rgb_buffer

        sim_img_rgb[..., 0] = torch.sum(self.A_tensor * params_r, dim=-1)  # R
        sim_img_rgb[..., 1] = torch.sum(self.A_tensor * params_g, dim=-1)  # G
        sim_img_rgb[..., 2] = torch.sum(self.A_tensor * params_b, dim=-1)  # B
        sim_img_rgb *= contact_support.unsqueeze(-1)
        rgb_gain = float(getattr(self.cfg, "taxim_rgb_response_gain", 1.0))
        if abs(rgb_gain - 1.0) > 1e-9:
            sim_img_rgb = sim_img_rgb * rgb_gain
        load_gain_min = float(getattr(self.cfg, "taxim_response_load_gain_min", 1.0))
        load_gain_max = float(getattr(self.cfg, "taxim_response_load_gain_max", 1.0))
        if abs(load_gain_max - load_gain_min) > 1.0e-9 or abs(load_gain_min - 1.0) > 1.0e-9:
            reference_depth_mm = max(
                float(getattr(self.cfg, "taxim_response_load_reference_depth_mm", 0.42)),
                1.0e-9,
            )
            load_exponent = max(
                float(getattr(self.cfg, "taxim_response_load_exponent", 1.0)),
                1.0e-6,
            )
            load_weight = torch.clamp(
                physical_peak_depth_mm / reference_depth_mm, min=0.0, max=1.0
            ) ** load_exponent
            response_gain = load_gain_min + (load_gain_max - load_gain_min) * load_weight
            sim_img_rgb = sim_img_rgb * response_gain.unsqueeze(-1)

        mesh_scale = max(int(getattr(self.cfg, "taxim_response_mesh_scale", 1)), 1)
        mesh_iterations = max(
            int(getattr(self.cfg, "taxim_response_mesh_smooth_iterations", 0)), 0
        )
        mesh_blend = float(getattr(self.cfg, "taxim_response_mesh_blend", 1.0))
        if mesh_scale > 1 and mesh_iterations > 0 and mesh_blend > 1.0e-6:
            # The official FEM renderer shades a coarse gel mesh and then
            # interpolates it to camera resolution. Reproduce that optical
            # low-pass on RGB response only: corrected height and the FOTS
            # marker-driving height remain untouched.
            response_chw = sim_img_rgb.permute(0, 3, 1, 2)
            low_h = max(response_chw.shape[-2] // mesh_scale, 1)
            low_w = max(response_chw.shape[-1] // mesh_scale, 1)
            response_low = torch.nn.functional.interpolate(
                response_chw,
                size=(low_h, low_w),
                mode="area",
            )
            for _ in range(mesh_iterations):
                response_low = torch.nn.functional.avg_pool2d(
                    response_low, kernel_size=3, stride=1, padding=1
                )
            response_mesh = torch.nn.functional.interpolate(
                response_low,
                size=response_chw.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            alpha_mesh = min(max(mesh_blend, 0.0), 1.0)
            sim_img_rgb = (1.0 - alpha_mesh) * sim_img_rgb + alpha_mesh * response_mesh

        # write tactile image
        sim_img = sim_img_rgb + self.background_tensor  # /255.0
        illum_blend = float(getattr(self.cfg, "taxim_illumination_blend", 0.0))
        illum_bias_blend = float(getattr(self.cfg, "taxim_illumination_bias_blend", 0.0))
        if self._illum_gain_tensor is not None and illum_blend > 1.0e-6:
            gain = 1.0 + (self._illum_gain_tensor - 1.0) * illum_blend
            sim_img = sim_img * gain
        if self._illum_bias_tensor is not None and illum_bias_blend > 1.0e-6:
            sim_img = sim_img + self._illum_bias_tensor * illum_bias_blend
        chroma_gain = float(getattr(self.cfg, "taxim_contact_chroma_gain", 1.0))
        if abs(chroma_gain - 1.0) > 1e-9:
            # Emphasize deep-contact chroma without changing no-contact background.
            depth_mm = torch.abs(height_map)
            depth_max = torch.amax(depth_mm, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            depth_w = torch.clamp(depth_mm / depth_max, min=0.0, max=1.0) ** 0.8
            local_gain = 1.0 + (chroma_gain - 1.0) * depth_w.unsqueeze(-1)
            sim_img = self.background_tensor + (sim_img - self.background_tensor) * local_gain
        soften_blend = float(getattr(self.cfg, "taxim_contact_soften_blend", 0.0))
        if soften_blend > 1.0e-6:
            depth_mm = torch.abs(height_map)
            depth_max = torch.amax(depth_mm, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            depth_w = torch.clamp(depth_mm / depth_max, min=0.0, max=1.0)
            blur = torch.nn.functional.avg_pool2d(
                sim_img.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1
            ).permute(0, 2, 3, 1)
            blur_wide = torch.nn.functional.avg_pool2d(
                sim_img.permute(0, 3, 1, 2), kernel_size=7, stride=1, padding=3
            ).permute(0, 2, 3, 1)
            gy_s, gx_s = torch.gradient(depth_mm, dim=(1, 2))
            gmag_s = torch.sqrt(gx_s * gx_s + gy_s * gy_s)
            gmax_s = torch.amax(gmag_s, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            edge_s = torch.clamp(gmag_s / gmax_s, min=0.0, max=1.0).unsqueeze(-1)
            jelly = (1.0 - edge_s) * blur + edge_s * blur_wide
            alpha = (soften_blend * (depth_w**0.65)).unsqueeze(-1)
            sim_img = (1.0 - alpha) * sim_img + alpha * jelly
        edge_soften = float(getattr(self.cfg, "taxim_edge_soften_strength", 0.0))
        if edge_soften > 1.0e-6:
            depth_mm = torch.abs(height_map)
            gy, gx = torch.gradient(depth_mm, dim=(1, 2))
            gmag = torch.sqrt(gx * gx + gy * gy)
            gmax = torch.amax(gmag, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            edge_w = torch.clamp(gmag / gmax, min=0.0, max=1.0).unsqueeze(-1)
            blur_edge = torch.nn.functional.avg_pool2d(
                sim_img.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1
            ).permute(0, 2, 3, 1)
            blur_edge_wide = torch.nn.functional.avg_pool2d(
                sim_img.permute(0, 3, 1, 2), kernel_size=5, stride=1, padding=2
            ).permute(0, 2, 3, 1)
            blur_edge_mix = 0.45 * blur_edge + 0.55 * blur_edge_wide
            alpha_e = edge_soften * (edge_w**0.78)
            sim_img = (1.0 - alpha_e) * sim_img + alpha_e * blur_edge_mix
        psf_blend = float(getattr(self.cfg, "taxim_contact_psf_blend", 0.0))
        if psf_blend > 1.0e-6:
            depth_mm = torch.abs(height_map)
            depth_max = torch.amax(depth_mm, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            contact_w = torch.clamp(depth_mm / depth_max, min=0.0, max=1.0).unsqueeze(-1)
            psf_kernel_size = self._normalize_kernel_size(int(getattr(self.cfg, "taxim_contact_psf_kernel_size", 5)))
            psf_kernel = torch.tensor(self._get_filtering_kernel(psf_kernel_size), dtype=torch.float, device=self.device)
            psf_img = self._gaussian_filtering_rgb(sim_img, psf_kernel)
            # Additional broad PSF pass to suppress synthetic polygonal edge hardness.
            wide_kernel = torch.tensor(
                self._get_filtering_kernel(self._normalize_kernel_size(psf_kernel_size + 6)),
                dtype=torch.float,
                device=self.device,
            )
            psf_img_wide = self._gaussian_filtering_rgb(sim_img, wide_kernel)
            psf_mix = 0.45 * psf_img + 0.55 * psf_img_wide
            alpha_psf = psf_blend * (0.35 + 0.65 * (contact_w**0.45))
            sim_img = (1.0 - alpha_psf) * sim_img + alpha_psf * psf_mix
        red_tilt = float(getattr(self.cfg, "taxim_contact_red_tilt_strength", 0.0))
        if abs(red_tilt) > 1.0e-6:
            depth_mm = torch.abs(height_map)
            depth_max = torch.amax(depth_mm, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            red_tilt_power = float(getattr(self.cfg, "taxim_contact_red_tilt_power", 1.0))
            if bool(getattr(self.cfg, "taxim_contact_tint_gradient_weight", False)):
                gy_tint, gx_tint = torch.gradient(depth_mm, dim=(1, 2))
                grad_tint = torch.sqrt(gx_tint * gx_tint + gy_tint * gy_tint)
                grad_tint = self._gaussian_filtering(
                    grad_tint.unsqueeze(-1), self._edge_denoise_kernel
                ).squeeze(-1)
                grad_max = torch.amax(grad_tint, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
                contact_base = torch.clamp(grad_tint / grad_max, min=0.0, max=1.0)
            else:
                contact_base = torch.clamp(depth_mm / depth_max, min=0.0, max=1.0)
            contact_w = contact_base.unsqueeze(-1) ** max(red_tilt_power, 1.0e-6)
            w = sim_img.shape[2]
            x = torch.linspace(-1.0, 1.0, w, device=sim_img.device, dtype=sim_img.dtype).view(1, 1, w, 1)
            right = torch.clamp((x + 1.0) * 0.5, min=0.0, max=1.0) ** 1.25
            left = torch.clamp((1.0 - x) * 0.5, min=0.0, max=1.0) ** 1.25
            depth_factor = 0.35 + 0.65 * torch.clamp(
                depth_mm / depth_max, min=0.0, max=1.0
            ).unsqueeze(-1)
            tint = contact_w * right * depth_factor
            tint_left = contact_w * left * depth_factor
            # Tint the Taxim contact response, not the absolute background RGB.
            # Multiplying a bright background creates a filled pink polygon for
            # flat indenters even though their signal should live on gradients.
            contact_delta = sim_img - self.background_tensor
            delta_r = contact_delta[..., 0:1] * (1.0 + 1.10 * red_tilt * tint)
            delta_g = contact_delta[..., 1:2] * (1.0 - 0.28 * red_tilt * tint)
            delta_b = contact_delta[..., 2:3] * (1.0 - 0.14 * red_tilt * tint)
            sim_img = self.background_tensor + torch.cat((delta_r, delta_g, delta_b), dim=-1)
            red_add = float(getattr(self.cfg, "taxim_contact_red_tilt_additive", 0.0))
            if abs(red_add) > 1.0e-6:
                # Xense's opposed illumination produces a red right edge and a
                # weaker cyan left edge on the real M2-nut indentation.
                sim_img[..., 0:1] = sim_img[..., 0:1] + red_add * (tint - 0.25 * tint_left)
                sim_img[..., 1:2] = sim_img[..., 1:2] + 0.25 * red_add * tint_left
                sim_img[..., 2:3] = sim_img[..., 2:3] + 0.55 * red_add * tint_left
        final_psf_blend = float(getattr(self.cfg, "taxim_final_response_psf_blend", 0.0))
        if final_psf_blend > 1.0e-6:
            # Apply the final optical spread after chroma and directional
            # lighting. Unlike the earlier contact-masked PSF, this operates on
            # RGB response relative to the clean background, allowing light to
            # diffuse beyond the exact geometric contact boundary without
            # blurring the background or the later marker overlay.
            final_kernel_size = self._normalize_kernel_size(
                int(getattr(self.cfg, "taxim_final_response_psf_kernel_size", 15))
            )
            final_kernel = torch.tensor(
                self._get_filtering_kernel(final_kernel_size),
                dtype=torch.float,
                device=self.device,
            )
            contact_response = sim_img - self.background_tensor
            response_blurred = self._gaussian_filtering_rgb(contact_response, final_kernel)
            alpha_final = min(max(final_psf_blend, 0.0), 1.0)
            sim_img = self.background_tensor + (
                (1.0 - alpha_final) * contact_response + alpha_final * response_blurred
            )
        sim_img = torch.clip(sim_img, 0, 255, out=sim_img).to(torch.uint8)

        if self._marker_sim is not None and self._marker_sim.enabled:
            raw_markers = marker_height_map if marker_height_map is not None else height_map
            marker_scale = float(getattr(self.cfg, "marker_height_scale", 1.0))
            taxim_max = float(getattr(self.cfg, "marker_height_taxim_mm_max", 100.0))
            height_for_markers = self._height_m_to_taxim_mm(raw_markers * marker_scale).abs().clamp(
                min=0.0, max=taxim_max
            )
            sim_img, self._last_marker_displacements = self._marker_sim.composite_batch(
                sim_img,
                height_for_markers,
                shear_disp_px_batch=marker_shear_map,
            )
        else:
            self._last_marker_displacements = None

        return sim_img

    def _height_m_to_taxim_mm(self, height_m: torch.Tensor) -> torch.Tensor:
        """Convert penetration depth (m, >=0) to Taxim height-map mm units."""
        h = height_m.clone()
        h[torch.abs(h) < 1e-6] = 0
        h = h * -1000.0
        h /= self.cfg.mm_per_pixel
        return self._gaussian_filtering(h.unsqueeze(-1), self.kernel).squeeze(-1)

    """
    Internal Helpers.
    """

    def _get_render_data(self, data_dir: str, file_name: str) -> str:
        """Gets the path for the GelSight render data file.

        Args:
            data_dir: The data directory name containing the render data.
            file_name: The specific file name to retrieve.

        Returns:
            The local path to the file.

        Raises:
            FileNotFoundError: If the file is not found locally or on Nucleus.
        """
        # Construct path using the configured base path
        file_path = os.path.join(self.cfg.base_data_path, data_dir, file_name)

        # Cache directory for downloads
        cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_dir)

        # Use retrieve_file_path to handle local/Nucleus paths and caching
        return retrieve_file_path(file_path, download_dir=cache_dir, force_download=False)

    def _generate_normals(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the gradient magnitude and direction of the height map.

        Args:
            img: Input height map tensor. Shape: (N, H, W).

        Returns:
            Tuple containing gradient magnitude tensor and gradient direction tensor. Shape: (N, H, W).
        """
        img_grad = torch.gradient(img, dim=(1, 2))
        dzdx, dzdy = img_grad
        if self._normal_smoothing_kernel is not None:
            # Match mesh-renderer normal interpolation without changing the
            # force-corrected height map or the height map used by FOTS.
            dzdx = self._gaussian_filtering(
                dzdx.unsqueeze(-1), self._normal_smoothing_kernel
            ).squeeze(-1)
            dzdy = self._gaussian_filtering(
                dzdy.unsqueeze(-1), self._normal_smoothing_kernel
            ).squeeze(-1)

        grad_mag_orig = torch.sqrt(dzdx**2 + dzdy**2)
        grad_suppress = float(getattr(self.cfg, "taxim_gradient_edge_suppress", 0.0))
        if grad_suppress > 1.0e-6:
            depth_abs = torch.abs(img)
            depth_max = torch.amax(depth_abs, dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
            depth_n = torch.clamp(depth_abs / depth_max, min=0.0, max=1.0)
            band_center = float(getattr(self.cfg, "taxim_contact_edge_denoise_center", 0.10))
            band_width = max(float(getattr(self.cfg, "taxim_contact_edge_denoise_bandwidth", 0.10)), 1.0e-4)
            band = torch.exp(-((depth_n - band_center) ** 2) / (2.0 * band_width * band_width))
            grad_blur = self._gaussian_filtering(grad_mag_orig.unsqueeze(-1), self._edge_denoise_kernel).squeeze(-1)
            alpha = torch.clamp(grad_suppress * band, min=0.0, max=1.0)
            grad_mag_orig = (1.0 - alpha) * grad_mag_orig + alpha * grad_blur
        grad_mag = torch.arctan(grad_mag_orig)  # seems that arctan is used as a squashing function
        grad_dir = torch.arctan2(dzdx, dzdy)
        grad_dir[grad_mag_orig == 0] = 0

        # handle edges
        grad_mag = torch.nn.functional.pad(grad_mag[:, 1:-1, 1:-1], pad=(1, 1, 1, 1))
        grad_dir = torch.nn.functional.pad(grad_dir[:, 1:-1, 1:-1], pad=(1, 1, 1, 1))

        return grad_mag, grad_dir

    def _get_filtering_kernel(self, kernel_size: int = 5) -> np.ndarray:
        """Create a Gaussian filtering kernel.

        For kernel derivation, see https://cecas.clemson.edu/~stb/ece847/internal/cvbook/ch03_filtering.pdf

        Args:
            kernel_size: Size of the kernel. Defaults to 5.

        Returns:
            Filtering kernel. Shape is (kernel_size, kernel_size).
        """
        filter_1D = scipy.special.binom(kernel_size - 1, np.arange(kernel_size))
        filter_1D /= filter_1D.sum()
        filter_1D = filter_1D[..., None]

        kernel = filter_1D @ filter_1D.T
        return kernel

    def _normalize_kernel_size(self, kernel_size: int) -> int:
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _gaussian_filtering(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian filtering to the input image tensor.

        Args:
            img: Input image tensor. Shape is (N, H, W, 1).
            kernel: Filtering kernel tensor. Shape is (K, K).

        Returns:
            Filtered image tensor. Shape is (N, H, W, 1).
        """
        img_output = torch.nn.functional.conv2d(
            img.permute(0, 3, 1, 2), kernel.unsqueeze(0).unsqueeze(0), stride=1, padding="same"
        ).permute(0, 2, 3, 1)
        return img_output

    def _gaussian_filtering_rgb(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply the separable binomial/Gaussian kernel to three RGB channels."""
        # Every kernel produced by _get_filtering_kernel is v @ v.T. Recovering
        # v from its diagonal makes large final PSFs far cheaper than three
        # dense KxK convolutions while remaining numerically equivalent.
        kernel_1d = torch.sqrt(torch.clamp(torch.diagonal(kernel), min=0.0))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp(min=1.0e-12)
        size = int(kernel_1d.numel())
        pad = size // 2
        image_chw = img.permute(0, 3, 1, 2)
        horizontal = kernel_1d.view(1, 1, 1, size).repeat(3, 1, 1, 1)
        vertical = kernel_1d.view(1, 1, size, 1).repeat(3, 1, 1, 1)
        filtered = torch.nn.functional.conv2d(
            image_chw, horizontal, padding=(0, pad), groups=3
        )
        filtered = torch.nn.functional.conv2d(
            filtered, vertical, padding=(pad, 0), groups=3
        )
        return filtered.permute(0, 2, 3, 1)
