# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

from dataclasses import MISSING
from typing import cast

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import VISUO_TACTILE_SENSOR_MARKER_CFG
from isaaclab.sensors import SensorBaseCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .visuotactile_sensor import VisuoTactileSensor

##
# GelSight Render Configuration
##


@configclass
class GelSightRenderCfg:
    """Configuration for GelSight sensor rendering parameters.

    This configuration defines the rendering parameters for example-based tactile image synthesis
    using the Taxim approach.

    Reference:
        Si, Z., & Yuan, W. (2022). Taxim: An example-based simulation model for GelSight
        tactile sensors. IEEE Robotics and Automation Letters, 7(2), 2361-2368.
        https://arxiv.org/abs/2109.04027

    Data Directory Structure:
        The sensor data should be organized in the following structure::

            base_data_path/
            └── sensor_data_dir_name/
                ├── bg.jpg              # Background image (required)
                ├── polycalib.npz       # Polynomial calibration data (required)
                └── real_bg.npy         # Real background data (optional)

    Example:
        Using predefined sensor configuration::

            from ViTacLab.assets.sensor.tacsl_sensor import VisuoTactileSensorCfg
            # from ViTacLab.assets.sensor.tacsl_sensor import VisuoTactileSensorCfg

            from isaaclab_assets.sensors import GELSIGHT_R15_CFG

            sensor_cfg = VisuoTactileSensorCfg(render_cfg=GELSIGHT_R15_CFG)

        Using custom sensor data::

            custom_cfg = GelSightRenderCfg(
                base_data_path="/path/to/my/sensors",
                sensor_data_dir_name="my_custom_sensor",
                image_height=480,
                image_width=640,
                mm_per_pixel=0.05,
            )
    """

    base_data_path: str = f"{ISAACLAB_NUCLEUS_DIR}/TacSL"
    """Base path to the directory containing sensor calibration data. Defaults to
    Isaac Lab Nucleus directory at ``{ISAACLAB_NUCLEUS_DIR}/TacSL``.
    """

    sensor_data_dir_name: str = cast(str, MISSING)
    """Directory name containing the sensor calibration and background data.

    This should be a relative path (directory name) inside the :attr:`base_data_path`.
    """

    background_path: str = "bg.jpg"
    """Filename of the background image within the data directory."""

    calib_path: str = "polycalib.npz"
    """Filename of the polynomial calibration data within the data directory."""

    real_background: str = "real_bg.npy"
    """Filename of the real background data within the data directory."""

    image_height: int = cast(int, MISSING)
    """Height of the tactile image in pixels."""

    image_width: int = cast(int, MISSING)
    """Width of the tactile image in pixels."""

    num_bins: int = 120
    """Number of bins for gradient magnitude and direction quantization."""

    mm_per_pixel: float = cast(float, MISSING)
    """Millimeters per pixel conversion factor for reconstructing 2D tactile image from the height map."""

    taxim_height_scale: float = 1.0
    """Scale penetration depth (m) before Taxim RGB synthesis (Task2 ``rgb_diff_scale``)."""

    enable_marker_simulation: bool = False
    """Whether to overlay FOTS-style marker motion on Taxim RGB (TacEx-compatible)."""

    marker_pattern: str = "gelsight"
    """Marker layout: ``gelsight`` (sparse black dots), ``xense`` (denser staggered black dots), or ``none``."""

    marker_lambda_d: float = 0.0025
    """Gaussian falloff for FOTS dilate displacement (1/pixel^2 scale)."""

    marker_displacement_gain: float = 0.35
    """Scale for normal-indentation marker displacement."""

    marker_shear_gain: float = 8.0
    """Scale for shear proxy from height-map gradient."""

    marker_deadband_mm: float = 0.02
    """Ignore height-map cells below this penetration (Taxim mm units) for marker contacts."""

    marker_blend_alpha: float = 0.92
    """Marker color blend weight when compositing onto Taxim RGB."""

    marker_max_displacement_px: float = 25.0
    """Clamp FOTS marker displacement magnitude (px) to avoid gradient blow-up at high resolution."""

    marker_height_taxim_mm_max: float = 100.0
    """Upper clamp on Taxim-mm height fed to FOTS markers (prevents silent saturation)."""

    marker_height_scale: float = 1.0
    """Scale penetration (m) before Taxim-mm conversion for FOTS markers only (RGB unchanged)."""

    marker_rest_path: str = ""
    """Optional filename (within sensor data dir) of lab-measured marker rest (M,2) pixel coords."""


##
# Visuo-Tactile Sensor Configuration
##


@configclass
class VisuoTactileSensorCfg(SensorBaseCfg):
    """Configuration for the visuo-tactile sensor.

    This sensor provides both camera-based tactile sensing and force field tactile sensing.
    It can capture tactile RGB/depth images and compute penalty-based contact forces.
    """

    class_type: type = VisuoTactileSensor

    # Sensor type and capabilities
    render_cfg: GelSightRenderCfg = cast(GelSightRenderCfg, MISSING)
    """Configuration for GelSight sensor rendering.

    This defines the rendering parameters for converting depth maps to realistic tactile images.

    For simplicity, you can use the predefined configs for standard sensor models:

    - :attr:`isaaclab_assets.sensors.GELSIGHT_R15_CFG`
    - :attr:`isaaclab_assets.sensors.GELSIGHT_MINI_CFG`

    """

    enable_camera_tactile: bool = True
    """Whether to enable camera-based tactile sensing."""

    enable_force_field: bool = True
    """Whether to enable force field tactile sensing."""

    # Force field configuration
    tactile_array_size: tuple[int, int] = cast(tuple[int, int], MISSING)
    """Number of tactile points for force field sensing in (rows, cols) format."""

    tactile_margin: float = cast(float, MISSING)
    """Margin for tactile point generation (in meters).

    This parameter defines the exclusion margin from the edges of the elastomer mesh when generating
    the tactile point grid. It ensures that force field points are not generated on the very edges
    of the sensor surface where geometry might be unstable or less relevant for contact.
    """

    contact_object_prim_path_expr: str | None = None
    """Prim path expression to find the contact object for force field computation.

    This specifies the object that will make contact with the tactile sensor. The sensor will automatically
    find the SDF collision mesh within this object for optimal force field computation.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/ContactObject`` will be replaced with ``/World/envs/env_.*/ContactObject``.

    .. attention::
        For force field computation to work properly, the contact object must have an SDF collision mesh.
        The sensor will search for the first SDF mesh within the specified prim hierarchy.
    """

    contact_object_is_deformable: bool = False
    """If True, the contact object is treated as a PhysX soft body (deformable).

    Used by :class:`VisuoTactileSensorV2` for velocity at the nearest simulation vertex.
    Ignored by the legacy :class:`VisuoTactileSensor` (SDF-based).
    """

    depth_penetration_deadband: float = 0.002
    """Deadband on depth-difference penetration, used only by :class:`VisuoTactileSensorV2`.

    Effective penetration is ``max(0, (z_ref - z_cur) - depth_penetration_deadband)`` at each sample.
    Same units as camera depth (typically meters). A small positive value (e.g. ``1e-4``) reduces spurious
    forces from depth noise when there is no contact. Ignored by the legacy SDF-based sensor.
    """

    enable_normal_correction: bool = True
    """Enable local normal correction for :class:`VisuoTactileSensorV2`.

    When enabled, V2 estimates a local effective stiffness from neighboring samples and maps raw
    depth penetration to a corrected normal-force field around ``normal_correction_k_ref``.
    """

    normal_correction_knn: int = 8
    """Neighborhood size for local stiffness estimation in V2 normal correction."""

    normal_correction_eps: float = 1e-6
    """Small epsilon used in normal-correction ratios/divisions to avoid zero-division."""

    normal_correction_trim_ratio: float = 0.2
    """Trim ratio for robust local mean in V2 normal correction (0 disables trimming)."""

    normal_correction_k_ref: float = 1e4
    """Reference stiffness ``k_ref`` for V2 normal correction.

    If set to ``<= 0``, V2 falls back to :attr:`normal_contact_stiffness`.
    """

    enable_slip_stick_reconstruction: bool = True
    """Enable slip/stick tangential reconstruction in :class:`VisuoTactileSensorV2`."""

    slip_speed_threshold: float = 1e-3
    """Tangential-speed threshold ``tau_v`` (m/s) for sliding vs sticking partition in V2."""

    sticking_interp_sigma: float = 0.02
    """Gaussian width ``sigma`` (m, pad local frame) for sticking-anchor interpolation in V2."""

    tangential_anchor_knn: int = 8
    """Neighborhood size for sliding-anchor gamma estimation in V2."""

    tangential_anchor_eps: float = 1e-6
    """Small epsilon used in tangential reconstruction ratios/divisions."""

    tangential_gamma_clip: tuple[float, float] = (0.2, 5.0)
    """Clip range ``(min,max)`` for sliding scale ``gamma`` in V2."""

    use_physx_sparse_anchors: bool = True
    """Use PhysX sparse contact/friction anchors in :class:`VisuoTactileSensorV2`.

    When disabled (or if PhysX contact view is unavailable), V2 falls back to dense self-anchoring
    for normal correction and slip/stick reconstruction.
    """

    sparse_anchor_max_contact_data_count_per_env: int = 1024
    """Max PhysX contact/friction buffer entries reserved per environment for sparse anchors in V2."""

    require_physx_sparse_anchors: bool = False
    """Require PhysX sparse-anchor backend in :class:`VisuoTactileSensorV2`.

    When True together with ``use_physx_sparse_anchors=True`` (rigid-object path), V2 raises an error during
    initialization if a valid PhysX rigid-contact view cannot be created. This disables silent fallback to dense
    anchors and is useful for strict paper-reproduction / mechanism-validation runs.
    """

    strict_target_contact_attribution: bool = True
    """Strictly gate V2 force reconstruction by target-object PhysX sparse anchors.

    When True with ``use_physx_sparse_anchors=True`` (rigid-object path), V2 zeros normal/tangential
    forces for frames where no sparse anchors are reported for the configured target object. This prevents
    non-target contacts (that only appear in depth delta) from being mis-attributed to the target object's
    relative velocity.
    """

    enable_corrected_force_render: bool = False
    """Enable Stage-C render correction in :class:`VisuoTactileSensorV2`.

    When enabled, V2 blends the camera depth-difference height map with a force-derived
    corrected penetration map reconstructed from corrected normal force.
    """

    corrected_force_render_blend: float = 1.0
    """Blend factor for Stage-C render correction in V2.

    The final render height is ``(1-blend)*depth_delta + blend*force_delta`` and clamped to ``[0, +inf)``.
    Recommended range is ``[0, 1]``.
    """

    force_height_max_m: float = 0.006
    """Upper bound (m) on force-derived penetration used for Taxim RGB and marker height maps."""

    marker_load_ref_fn_n: float = 0.72
    """Reference PhysX normal force (N) for ViTacSim marker load scaling (advisor G110 ~0.72 N)."""

    marker_load_scale_exponent: float = 0.5
    """Exponent on ``(sparse_fn / marker_load_ref_fn_n)`` when scaling depth for markers."""

    marker_depth_gamma: float = 1.0
    """Superlinear exponent on depth (m) for FOTS markers at high PhysX load; >1 compresses saturation."""

    marker_depth_gamma_low_load: float = 1.0
    """Depth exponent at low load (keeps G010 markers above FOTS deadband when paired with ``marker_depth_gamma``)."""

    marker_depth_gamma_load_t0: float = 0.35
    """Load-scale pivot: below this use ``marker_depth_gamma_low_load``, above blend toward ``marker_depth_gamma``."""

    marker_shear_from_force_field: bool = False
    """When True (ViTacSim), add marker shear from ``tactile_shear_force`` scattered to the camera grid."""

    marker_shear_force_gain: float = 3.0
    """Pixel gain applied to ``tactile_shear_force`` when driving marker displacement."""

    marker_shear_force_ref_n: float = 0.05
    """Reference shear magnitude (N) mapped to ``marker_shear_force_gain`` pixels."""

    tactile_uv_shift_px: tuple[float, float] = (0.0, 0.0)
    """In-plane shift (du, dv) in pixels applied to height maps before Taxim render.

    Lab advisor alignment: positive du samples from the right, positive dv from below,
    moving the contact imprint toward the lab Xense image coordinates.
    """

    normal_correction_max_stiffness_ratio: float = 0.0
    """Cap on ``F_n/(d+eps)`` in local stiffness estimation. ``<=0`` disables (uses ``50 * k_ref``)."""

    # Force field physics parameters
    normal_contact_stiffness: float = 1e4
    """Normal contact stiffness for penalty-based force computation."""

    friction_coefficient: float = 0.5
    """Friction coefficient for shear forces."""

    tangential_stiffness: float = 100
    """Tangential stiffness for shear forces."""

    camera_cfg: TiledCameraCfg | None = None
    """Camera configuration for tactile RGB/depth sensing.

    If None, camera-based sensing will be disabled even if :attr:`enable_camera_tactile` is True.
    """

    # Visualization
    visualizer_cfg: VisualizationMarkersCfg = VISUO_TACTILE_SENSOR_MARKER_CFG.replace(
        prim_path="/Visuals/TactileSensor"
    )
    """The configuration object for the visualization markers.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """

    trimesh_vis_tactile_points: bool = False
    """Whether to visualize tactile points for debugging using trimesh. Defaults to False."""

    visualize_sdf_closest_pts: bool = False
    """Whether to visualize SDF closest points for debugging. Defaults to False."""
