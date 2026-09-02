# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuo-tactile sensor V2: depth-based force field (no object SDF queries).

Public API matches :class:`VisuoTactileSensor` (same cfg/data methods) except the force pipeline uses
camera depth (nominal vs current) instead of ``create_sdf_shape_view`` / ``get_sdf_and_gradients``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from pxr import PhysxSchema, UsdPhysics

from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.configclass import configclass

from .visuotactile_sensor import VisuoTactileSensor
from .visuotactile_sensor_cfg import VisuoTactileSensorCfg

logger = logging.getLogger(__name__)


class VisuoTactileSensorV2(VisuoTactileSensor):
    r"""Same external interface as :class:`VisuoTactileSensor`, but force field uses depth-camera sampling.

    **Force pipeline (rigid contact object):**
        1. Fixed UV grid on the tactile depth image (``tactile_array_size``).
        2. Penetration :math:`d = z_{\mathrm{ref}} - z` from nominal vs current depth.
        3. Back-project samples with camera intrinsics + pose to world; normals from finite differences on depth.
        4. Rigid object velocity at the contact patch: :math:`v = v_{\mathrm{lin}} + \omega \times r` (same COM correction
           as the SDF implementation).
        5. Same penalty normal force and Coulomb / tangential stiffness model, mapped to tactile frame.

    **Deformable contact object:** ``contact_object_is_deformable=True`` uses nearest simulation vertex velocity
    from ``SoftBodyView`` nodal data.

    **Requirements:**
        * ``enable_camera_tactile=True`` and ``get_initial_render()`` before sim (nominal depth baseline).
        * ``contact_object_prim_path_expr`` set when ``enable_force_field=True``.
    """

    cfg: VisuoTactileSensorCfg

    def __init__(self, cfg: VisuoTactileSensorCfg):
        self._sample_u: torch.Tensor | None = None
        self._sample_v: torch.Tensor | None = None
        self._sample_u_idx: torch.Tensor | None = None
        self._sample_v_idx: torch.Tensor | None = None
        self._sample_flat_idx: torch.Tensor | None = None
        self._force_corrected_height_map: torch.Tensor | None = None
        self._contact_soft_body_view: Any | None = None
        self._contact_physx_view = None
        self._sparse_anchor_warned: bool = False
        self._sparse_anchor_sensor_pattern: str | None = None
        self._sparse_anchor_backend_ok: bool = False
        self._last_sparse_anchor_count: int = 0
        self._last_sparse_contact_count: int = 0
        self._last_sparse_friction_count: int = 0
        self._last_sparse_used: bool = False
        self._deformable_max_vertices: int = 0
        self._cached_sparse_anchors: dict[str, torch.Tensor] | None = None
        self._cached_p_pad_dense: torch.Tensor | None = None
        super().__init__(cfg)

    def _initialize_force_field(self):
        if not self.cfg.enable_camera_tactile:
            raise RuntimeError(
                "VisuoTactileSensorV2 requires enable_camera_tactile=True for depth-based force field."
            )
        self._build_uv_sample_grid()
        self._tactile_pos_local = torch.zeros((self.num_tactile_points, 3), dtype=torch.float32, device=self._device)
        self._tactile_quat_local = math_utils.quat_from_euler_xyz(
            torch.tensor(0.0, device=self._device),
            torch.tensor(0.0, device=self._device),
            torch.tensor(-torch.pi, device=self._device),
        ).unsqueeze(0).repeat(self.num_tactile_points, 1)

        self._create_physx_views_v2()
        self._initialize_force_field_buffers()
        logger.info("VisuoTactileSensorV2: depth-based force field initialized.")

    def _initialize_force_field_buffers(self):
        """Same as base, but per-env storage for expanded pose (base uses ``expand``, which breaks V2 slice writes)."""

        super()._initialize_force_field_buffers()
        self._tactile_pos_expanded = self._tactile_pos_local.unsqueeze(0).repeat(self._num_envs, 1, 1)
        self._tactile_quat_expanded = self._tactile_quat_local.unsqueeze(0).repeat(self._num_envs, 1, 1)
        self._force_corrected_height_map = torch.zeros(
            (self._num_envs, int(self.cfg.camera_cfg.height), int(self.cfg.camera_cfg.width)),
            dtype=torch.float32,
            device=self._device,
        )
        self._sparse_fn_total = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Stage-C order for V2: force update first, then camera render update."""
        if len(env_ids) == self._num_envs:
            internal_env_ids: Sequence[int] | slice = slice(None)
        else:
            internal_env_ids = env_ids

        if self.cfg.enable_force_field:
            self._update_force_field(internal_env_ids)
        if self.cfg.enable_camera_tactile:
            self._update_camera_tactile(internal_env_ids)
        self._update_tri_modal_output()

    def _marker_load_scale(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        """Per-env scale from sparse PhysX normal force (ViTacSim marker load consistency)."""
        ref = float(getattr(self.cfg, "marker_load_ref_fn_n", 0.72))
        exp = float(getattr(self.cfg, "marker_load_scale_exponent", 0.5))
        fn = self._sparse_fn_total[env_ids].clamp(min=1e-9)
        return (fn / max(ref, 1e-9)).pow(exp)

    def _update_sparse_fn_total(
        self,
        env_idx: Sequence[int] | slice,
        sparse_anchors: dict[str, torch.Tensor] | None,
    ) -> None:
        if isinstance(env_idx, slice):
            env_list = list(range(self._num_envs))
        else:
            env_list = list(env_idx)
        for e_local, e_global in enumerate(env_list):
            if sparse_anchors is None or sparse_anchors["normal_fn"].numel() == 0:
                self._sparse_fn_total[e_global] = 0.0
                continue
            mask = sparse_anchors["normal_env"] == e_local
            self._sparse_fn_total[e_global] = (
                sparse_anchors["normal_fn"][mask].sum() if bool(mask.any()) else 0.0
            )

    def _height_map_for_markers(self, depth_delta: torch.Tensor, env_ids: Sequence[int] | slice) -> torch.Tensor:
        """Dense height (m) driving FOTS markers.

        TacSL baseline: camera depth delta only (no PhysX load scaling).
        ViTacSim: load-scaled depth with load-adaptive exponent γ (low load γ≈1, high load γ>1).
        """
        gamma_hi = float(getattr(self.cfg, "marker_depth_gamma", 1.0))
        gamma_lo = float(getattr(self.cfg, "marker_depth_gamma_low_load", 1.0))
        load_t0 = float(getattr(self.cfg, "marker_depth_gamma_load_t0", 0.35))
        base = depth_delta.clamp(min=0.0)

        if not bool(self.cfg.enable_corrected_force_render):
            if abs(gamma_hi - 1.0) > 1e-6:
                base = base.pow(gamma_hi)
            return base

        scale = self._marker_load_scale(env_ids).view(-1, 1, 1)
        t = ((scale - load_t0) / max(1.0 - load_t0, 1e-6)).clamp(0.0, 1.0)
        eff_gamma = gamma_lo * (1.0 - t) + gamma_hi * t
        base = torch.pow(base, eff_gamma)
        return base * scale

    def _shift_vector_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """Apply configured (du, dv) pixel shift to batch maps (N, H, W, C)."""
        shift = getattr(self.cfg, "tactile_uv_shift_px", (0.0, 0.0))
        du = float(shift[0]) if shift else 0.0
        dv = float(shift[1]) if len(shift) > 1 else 0.0
        if abs(du) < 1e-6 and abs(dv) < 1e-6:
            return maps
        n, ht, wd, ch = maps.shape
        yy, xx = torch.meshgrid(
            torch.arange(ht, device=maps.device, dtype=torch.float32),
            torch.arange(wd, device=maps.device, dtype=torch.float32),
            indexing="ij",
        )
        src_x = xx.unsqueeze(0).expand(n, -1, -1) + du
        src_y = yy.unsqueeze(0).expand(n, -1, -1) + dv
        gx = 2.0 * src_x / max(float(wd - 1), 1.0) - 1.0
        gy = 2.0 * src_y / max(float(ht - 1), 1.0) - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        maps_chw = maps.permute(0, 3, 1, 2).float()
        out = F.grid_sample(
            maps_chw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out.permute(0, 2, 3, 1).to(dtype=maps.dtype)

    def _shear_map_for_markers(self, env_ids: Sequence[int] | slice) -> torch.Tensor | None:
        """Build (N, H, W, 2) marker shear displacement in pixels from the tactile force field."""
        if not bool(getattr(self.cfg, "marker_shear_from_force_field", False)):
            return None
        gain = float(getattr(self.cfg, "marker_shear_force_gain", 0.0))
        if gain <= 0.0:
            return None
        sf = self._data.tactile_shear_force
        if sf is None or self._sample_u_idx is None or self._sample_v_idx is None:
            return None

        sf_batch = sf[env_ids]
        rows, cols = self.cfg.tactile_array_size
        n = sf_batch.shape[0]
        cam_h = int(self.cfg.camera_cfg.height)
        cam_w = int(self.cfg.camera_cfg.width)
        flat_sf = sf_batch.view(n, rows * cols, 2)
        out = torch.zeros(n, cam_h, cam_w, 2, device=sf.device, dtype=sf.dtype)
        u_idx = self._sample_u_idx
        v_idx = self._sample_v_idx
        out[:, v_idx, u_idx, :] = flat_sf

        ref = float(getattr(self.cfg, "marker_shear_force_ref_n", 0.05))
        out = out * (gain / max(ref, 1e-9))
        return self._shift_vector_maps(out)

    def _shift_height_maps(self, height_m: torch.Tensor) -> torch.Tensor:
        """Apply configured (du, dv) pixel shift to batch height maps (N, H, W)."""
        shift = getattr(self.cfg, "tactile_uv_shift_px", (0.0, 0.0))
        du = float(shift[0]) if shift else 0.0
        dv = float(shift[1]) if len(shift) > 1 else 0.0
        if abs(du) < 1e-6 and abs(dv) < 1e-6:
            return height_m
        n, ht, wd = height_m.shape
        yy, xx = torch.meshgrid(
            torch.arange(ht, device=height_m.device, dtype=torch.float32),
            torch.arange(wd, device=height_m.device, dtype=torch.float32),
            indexing="ij",
        )
        src_x = xx.unsqueeze(0).expand(n, -1, -1) + du
        src_y = yy.unsqueeze(0).expand(n, -1, -1) + dv
        gx = 2.0 * src_x / max(float(wd - 1), 1.0) - 1.0
        gy = 2.0 * src_y / max(float(ht - 1), 1.0) - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        return torch.nn.functional.grid_sample(
            height_m.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

    def _tactile_uv_shift_du_dv(self) -> tuple[float, float]:
        shift = getattr(self.cfg, "tactile_uv_shift_px", (0.0, 0.0))
        if not shift:
            return 0.0, 0.0
        du = float(shift[0])
        dv = float(shift[1]) if len(shift) > 1 else 0.0
        return du, dv

    def _shift_rgb_batch(self, rgb: torch.Tensor) -> torch.Tensor:
        """Shift rendered tactile RGB (N,H,W,3) to align sim contact with lab Xense."""
        du, dv = self._tactile_uv_shift_du_dv()
        if abs(du) < 1e-6 and abs(dv) < 1e-6:
            return rgb
        n, ht, wd, _ = rgb.shape
        yy, xx = torch.meshgrid(
            torch.arange(ht, device=rgb.device, dtype=torch.float32),
            torch.arange(wd, device=rgb.device, dtype=torch.float32),
            indexing="ij",
        )
        src_x = xx.unsqueeze(0).expand(n, -1, -1) + du
        src_y = yy.unsqueeze(0).expand(n, -1, -1) + dv
        gx = 2.0 * src_x / max(float(wd - 1), 1.0) - 1.0
        gy = 2.0 * src_y / max(float(ht - 1), 1.0) - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        rgb_chw = rgb.permute(0, 3, 1, 2).float()
        out = torch.nn.functional.grid_sample(
            rgb_chw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out.permute(0, 2, 3, 1).to(dtype=rgb.dtype)

    def _update_camera_tactile(self, env_ids: Sequence[int] | slice):
        """Update camera tactile images, with optional Stage-C force-corrected rendering."""
        if self._nominal_tactile is None:
            raise RuntimeError("Nominal tactile is not set. Please call get_initial_render() first.")
        self._camera_sensor.update(self._sim_physics_dt)
        camera_data = self._camera_sensor.data

        depth_key = None
        if "distance_to_image_plane" in camera_data.output:
            depth_key = "distance_to_image_plane"
        elif "depth" in camera_data.output:
            depth_key = "depth"
        if depth_key is None:
            return

        self._data.tactile_depth_image[env_ids] = camera_data.output[depth_key][env_ids].clone()
        diff = self._nominal_tactile[depth_key][env_ids] - self._data.tactile_depth_image[env_ids]
        depth_delta = torch.clamp(diff.squeeze(-1), min=0.0)
        marker_h = self._height_map_for_markers(depth_delta, env_ids)
        shear_map = self._shear_map_for_markers(env_ids)
        rgb_base = self._tactile_rgb_render.render(
            depth_delta,
            marker_height_map=marker_h,
            marker_shear_map=shear_map,
        )
        rgb_base = self._shift_rgb_batch(rgb_base)
        self._data.tactile_rgb_image[env_ids] = rgb_base
        marker_disp = self._tactile_rgb_render.last_marker_displacements

        if bool(self.cfg.enable_corrected_force_render) and self._force_corrected_height_map is not None:
            alpha = float(self.cfg.corrected_force_render_blend)
            force_delta = self._force_corrected_height_map[env_ids]
            blended = torch.clamp((1.0 - alpha) * depth_delta + alpha * force_delta, min=0.0)
            self._data.tactile_height_map_corrected[env_ids] = blended
            rgb_corr = self._tactile_rgb_render.render(
                blended,
                marker_height_map=marker_h,
                marker_shear_map=shear_map,
            )
            rgb_corr = self._shift_rgb_batch(rgb_corr)
            self._data.tactile_rgb_image_corrected[env_ids] = rgb_corr
            marker_disp = self._tactile_rgb_render.last_marker_displacements
        else:
            self._data.tactile_height_map_corrected[env_ids] = depth_delta
            self._data.tactile_rgb_image_corrected[env_ids] = rgb_base

        if self._data.tactile_marker_displacement is not None and marker_disp is not None:
            self._data.tactile_marker_displacement[env_ids] = marker_disp

    def _update_force_corrected_height_map(self, env_ids: Sequence[int] | slice) -> None:
        """Reconstruct dense image-space height map from corrected normal force."""
        if (
            self._force_corrected_height_map is None
            or self._sample_flat_idx is None
            or self._sample_u_idx is None
            or self._sample_v_idx is None
        ):
            return
        nf = self._data.tactile_normal_force[env_ids]
        if nf is None:
            self._force_corrected_height_map[env_ids].zero_()
            return

        eps = float(max(self.cfg.normal_correction_eps, 1e-12))
        k_ref_cfg = float(self.cfg.normal_correction_k_ref)
        k_ref = float(self.cfg.normal_contact_stiffness if k_ref_cfg <= 0.0 else k_ref_cfg)
        delta = torch.clamp(nf / (k_ref + eps), min=0.0)
        max_h_base = float(getattr(self.cfg, "force_height_max_m", 0.006))
        if max_h_base > 0.0:
            load_scale = self._marker_load_scale(env_ids).view(-1, 1).clamp(min=0.05, max=3.0)
            max_h = max_h_base * load_scale
            delta = torch.minimum(delta, max_h)

        target = self._force_corrected_height_map[env_ids]
        target.zero_()
        # Vectorized scatter-max: UV indices are fixed at init (see _build_uv_sample_grid).
        flat_idx = self._sample_flat_idx.unsqueeze(0).expand(delta.shape[0], -1)
        target.view(delta.shape[0], -1).scatter_reduce_(1, flat_idx, delta, reduce="amax", include_self=True)

    def _build_uv_sample_grid(self):
        rows, cols = self.cfg.tactile_array_size
        h = float(self.cfg.camera_cfg.height)
        w = float(self.cfg.camera_cfg.width)
        margin_u = 0.5
        margin_v = 0.5
        u_idx = torch.linspace(margin_u, w - 1.0 - margin_u, cols, device=self._device, dtype=torch.float32)
        v_idx = torch.linspace(margin_v, h - 1.0 - margin_v, rows, device=self._device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v_idx, u_idx, indexing="ij")
        self._sample_u = uu.reshape(-1)
        self._sample_v = vv.reshape(-1)
        self.num_tactile_points = rows * cols
        cam_h = int(self.cfg.camera_cfg.height)
        cam_w = int(self.cfg.camera_cfg.width)
        self._sample_u_idx = torch.clamp(torch.round(self._sample_u).long(), min=0, max=cam_w - 1)
        self._sample_v_idx = torch.clamp(torch.round(self._sample_v).long(), min=0, max=cam_h - 1)
        self._sample_flat_idx = self._sample_v_idx * cam_w + self._sample_u_idx

    def _create_physx_views(self) -> None:
        """Legacy SDF path is unused; see :meth:`_create_physx_views_v2`."""
        pass

    def _create_physx_views_v2(self) -> None:
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        elastomer_pattern = self._parent_prims[0].GetPath().pathString.replace("env_0", "env_*")
        self._elastomer_body_view = self._physics_sim_view.create_rigid_body_view([elastomer_pattern])
        self._elastomer_tip_view = self._physics_sim_view.create_rigid_body_view([elastomer_pattern.replace("elastomer", "elastomer_tip")])
        self._elastomer_com_b = self._elastomer_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]

        self._contact_object_body_view = None
        self._contact_object_com_b = None
        self._contact_soft_body_view = None
        self._contact_physx_view = None
        self._sparse_anchor_warned = False
        self._sparse_anchor_sensor_pattern = None
        self._sparse_anchor_backend_ok = False
        self._last_sparse_anchor_count = 0
        self._last_sparse_contact_count = 0
        self._last_sparse_friction_count = 0
        self._last_sparse_used = False
        self._deformable_max_vertices = 0

        if self.cfg.contact_object_prim_path_expr is None:
            return

        contact_prim = sim_utils.find_first_matching_prim(self.cfg.contact_object_prim_path_expr)
        if contact_prim is None:
            raise RuntimeError(
                f"No contact object prim found matching pattern: {self.cfg.contact_object_prim_path_expr}"
            )

        if self.cfg.contact_object_is_deformable:
            root_prims = sim_utils.get_all_matching_child_prims(
                contact_prim.GetPath().pathString,
                predicate=lambda prim: prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI),
                traverse_instance_prims=False,
            )
            if len(root_prims) == 0:
                raise RuntimeError(
                    f"No PhysxDeformableBodyAPI under contact object at {contact_prim.GetPath().pathString}"
                )
            if len(root_prims) > 1:
                logger.warning(
                    "Multiple deformable roots under contact object; using the first: %s",
                    root_prims[0].GetPath().pathString,
                )
            root_path = root_prims[0].GetPath().pathString.replace("env_0", "env_*")
            root_path = root_path.replace(".*", "*")
            self._contact_soft_body_view = self._physics_sim_view.create_soft_body_view(root_path)
            if self._contact_soft_body_view._backend is None:
                raise RuntimeError("Failed to create soft body view for contact object.")
            self._deformable_max_vertices = self._contact_soft_body_view.max_sim_vertices_per_body
            return

        rigid = self._find_parent_rigid_body(contact_prim)
        if rigid is None:
            rigid = sim_utils.get_first_matching_child_prim(
                contact_prim.GetPath(),
                predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI),
            )
        if rigid is None:
            raise RuntimeError(f"No rigid body found for contact object at {contact_prim.GetPath().pathString}")

        body_path_pattern = rigid.GetPath().pathString.replace("env_0", "env_*")
        self._contact_object_body_view = self._physics_sim_view.create_rigid_body_view([body_path_pattern])
        self._contact_object_com_b = self._contact_object_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]
        if bool(self.cfg.use_physx_sparse_anchors):
            max_count = max(
                int(self.cfg.sparse_anchor_max_contact_data_count_per_env) * int(self._num_envs),
                int(self._num_envs),
            )
            sensor_candidates = [elastomer_pattern, elastomer_pattern.replace("elastomer", "elastomer_tip")]
            sparse_fail_reasons: list[str] = []
            # Pre-check in env_0 to avoid noisy plugin errors when contact reporter API is absent.
            filter_local = rigid.GetPath().pathString
            filter_has_contact_api = self._ensure_contact_report_api(rigid)
            if not filter_has_contact_api:
                sparse_fail_reasons.append(f"filter_no_contact_report_api:{filter_local}")
                logger.warning(
                    "VisuoTactileSensorV2: filter rigid body has no PhysxContactReportAPI: %s. "
                    "Sparse anchors disabled for this sensor.",
                    filter_local,
                )
                sensor_candidates = []
            for sensor_pattern in sensor_candidates:
                sensor_local = sensor_pattern.replace("env_*", "env_0")
                sensor_prim = sim_utils.find_first_matching_prim(sensor_local)
                if sensor_prim is None:
                    sparse_fail_reasons.append(f"sensor_not_found:{sensor_local}")
                    continue
                if not self._ensure_contact_report_api(sensor_prim):
                    sparse_fail_reasons.append(f"sensor_no_contact_report_api:{sensor_local}")
                    continue
                # Keep threshold at zero so small forces still produce contact report entries.
                try:
                    PhysxSchema.PhysxContactReportAPI(sensor_prim).CreateThresholdAttr().Set(0.0)
                except Exception:
                    pass
                try:
                    PhysxSchema.PhysxContactReportAPI(rigid).CreateThresholdAttr().Set(0.0)
                except Exception:
                    pass
                try:
                    view = self._physics_sim_view.create_rigid_contact_view(
                        sensor_pattern,
                        filter_patterns=[body_path_pattern],
                        max_contact_data_count=max_count,
                    )
                    backend = getattr(view, "_backend", None)
                    if backend is None:
                        sparse_fail_reasons.append(f"view_backend_none:{sensor_pattern}->{body_path_pattern}")
                        continue
                    fcnt = int(view.filter_count)
                    if fcnt <= 0:
                        sparse_fail_reasons.append(f"view_filter_count_zero:{sensor_pattern}->{body_path_pattern}")
                        continue
                    self._contact_physx_view = view
                    self._sparse_anchor_sensor_pattern = sensor_pattern
                    self._sparse_anchor_backend_ok = True
                    logger.info(
                        "VisuoTactileSensorV2: sparse anchors enabled with sensor pattern=%s, filters=%d",
                        sensor_pattern,
                        fcnt,
                    )
                    break
                except Exception as e:
                    sparse_fail_reasons.append(f"view_create_exception:{sensor_pattern}:{e}")
                    continue
            if self._contact_physx_view is None:
                require_sparse = bool(getattr(self.cfg, "require_physx_sparse_anchors", False))
                msg = (
                    "VisuoTactileSensorV2: no valid rigid contact view for sparse anchors "
                    f"(candidates={sensor_candidates}, filter={body_path_pattern}, reasons={sparse_fail_reasons})"
                )
                if require_sparse:
                    raise RuntimeError(msg + ". require_physx_sparse_anchors=True, aborting instead of dense fallback.")
                logger.warning(
                    msg + ", fallback to dense anchors."
                )
                self._sparse_anchor_backend_ok = False

    @staticmethod
    def _ensure_contact_report_api(prim) -> bool:
        if prim is None:
            return False
        try:
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api = PhysxSchema.PhysxContactReportAPI(prim)
            try:
                api.CreateThresholdAttr().Set(0.0)
            except Exception:
                pass
            return prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
        except Exception:
            return False

    @staticmethod
    def _find_parent_rigid_body(prim) -> Any | None:
        current_prim = prim
        while current_prim and current_prim.IsValid():
            if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return current_prim
            current_prim = current_prim.GetParent()
            if current_prim.GetPath() == "/":
                break
        return None

    def _update_force_field(self, env_ids: Sequence[int] | slice):
        if self._nominal_tactile is None:
            raise RuntimeError("VisuoTactileSensorV2: call get_initial_render() before simulation for nominal depth.")

        env_idx = env_ids if isinstance(env_ids, slice) else env_ids
        if not isinstance(env_idx, slice) and len(env_idx) == 0:
            return
        if self._force_corrected_height_map is not None:
            self._force_corrected_height_map[env_idx].zero_()

        elastomer_pos_w, elastomer_quat_w = self._elastomer_body_view.get_transforms().split([3, 4], dim=-1)
        elastomer_quat_w = math_utils.convert_quat(elastomer_quat_w, to="wxyz")
        elastomer_pos_w = elastomer_pos_w[env_idx]
        elastomer_quat_w = elastomer_quat_w[env_idx]

        depth_key = None
        if "distance_to_image_plane" in self._nominal_tactile:
            depth_key = "distance_to_image_plane"
        elif "depth" in self._nominal_tactile:
            depth_key = "depth"
        if depth_key is None:
            raise RuntimeError("Nominal tactile dict has no depth key.")
        self._camera_sensor.update(self._sim_physics_dt)

        cam_data = self._camera_sensor.data
        z_cur = cam_data.output[depth_key][env_idx]
        z_ref = self._nominal_tactile[depth_key][env_idx]
        k_mat = cam_data.intrinsic_matrices[env_idx]
        elastomer_tip_pos_w, elastomer_tip_quat_w = self._elastomer_tip_view.get_transforms().split([3, 4], dim=-1)
        elastomer_tip_quat_w = math_utils.convert_quat(elastomer_tip_quat_w, to="wxyz")
        #cam相对于elastomer_tip的位姿 = wxyz 0100 xyz-0.01859
        cam_quat_rel = torch.tensor([0, 1, 0, 0], device=self._device)   # wxyz
        cam_pos_rel  = torch.tensor([-0.01859, 0.0, 0.0], device=self._device)

        # expand到batch
        cam_quat_rel = cam_quat_rel.unsqueeze(0).expand_as(elastomer_tip_quat_w)
        cam_pos_rel  = cam_pos_rel.unsqueeze(0).expand_as(elastomer_tip_pos_w)

        # world pose
        cam_quat_w = math_utils.quat_mul(elastomer_tip_quat_w, cam_quat_rel)

        cam_pos_w = elastomer_tip_pos_w + math_utils.quat_apply(
            elastomer_tip_quat_w, cam_pos_rel
        )


        penetration, normals_world, points_world = self._depth_samples_to_penetration_and_geometry(
            z_cur, z_ref, k_mat, cam_pos_w, cam_quat_w
        )

        self._tactile_pos_expanded[env_idx] = math_utils.quat_apply_inverse(
            elastomer_quat_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1),
            points_world - elastomer_pos_w.unsqueeze(1),
        )

        num_pts = self.num_tactile_points
        quat_expanded = elastomer_quat_w.unsqueeze(1).expand(-1, num_pts, -1)
        self._data.tactile_points_pos_w[env_idx] = points_world
        self._data.tactile_points_quat_w[env_idx] = math_utils.quat_mul(
            quat_expanded, self._tactile_quat_expanded[env_idx]
        )

        if self.cfg.visualize_sdf_closest_pts:
            dbg = points_world + normals_world * penetration.unsqueeze(-1) * 0.01
            if isinstance(env_idx, slice):
                self.debug_closest_points_wolrd = dbg
            else:
                if not hasattr(self, "debug_closest_points_wolrd") or self.debug_closest_points_wolrd.shape != (
                    self._num_envs,
                    num_pts,
                    3,
                ):
                    self.debug_closest_points_wolrd = torch.zeros(
                        (self._num_envs, num_pts, 3), device=self._device, dtype=torch.float32
                    )
                self.debug_closest_points_wolrd[env_idx] = dbg

        if self._contact_object_body_view is None and self._contact_soft_body_view is None:
            self._data.penetration_depth[env_idx].zero_()
            if self._force_corrected_height_map is not None:
                self._force_corrected_height_map[env_idx].zero_()
            return

        if self._contact_object_body_view is not None:
            self._compute_forces_rigid(
                env_idx,
                penetration,
                normals_world,
                points_world,
                elastomer_pos_w,
                elastomer_quat_w,
            )
        else:
            self._compute_forces_deformable(
                env_idx,
                penetration,
                normals_world,
                points_world,
                elastomer_pos_w,
                elastomer_quat_w,
            )
        self._update_force_corrected_height_map(env_idx)

    def _grid_sample_depth(self, z_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sample depth (N,H,W) at float pixel coords (P,) per env -> (N,P)."""
        n, h, w = z_hw.shape
        p = u.shape[0]
        # grid_sample normalized coords; avoid div by zero when H or W is 1
        denom_w = max(float(w - 1), 1.0)
        denom_h = max(float(h - 1), 1.0)
        grid_x = 2.0 * u / denom_w - 1.0
        grid_y = 2.0 * v / denom_h - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, p, 1, 2).expand(n, -1, -1, -1)
        zin = z_hw.unsqueeze(1)
        out = F.grid_sample(zin, grid, align_corners=True, padding_mode="border")
        return out.squeeze(1).squeeze(-1)

    def _unproject_cam(
        self, z_px: torch.Tensor, u: torch.Tensor, v: torch.Tensor, invk: torch.Tensor
    ) -> torch.Tensor:
        """(N,P) depth, shared (P,) u,v, (N,3,3) invk -> (N,P,3) camera-frame points."""
        hom = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        ray_dir = torch.einsum("nij,pj->npi", invk, hom)
        return ray_dir * z_px.unsqueeze(-1)

    def _depth_samples_to_penetration_and_geometry(
        self,
        z_cur: torch.Tensor,
        z_ref: torch.Tensor,
        k_mat: torch.Tensor,
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self._sample_u
        v = self._sample_v
        p = self.num_tactile_points
        zc = z_cur.squeeze(-1)
        zr = z_ref.squeeze(-1)
        
        invk = torch.linalg.inv(k_mat)
        z_s = self._grid_sample_depth(zc, u, v)

        z_ref_s = self._grid_sample_depth(zr, u, v)

        delta = z_ref_s - z_s
        db = float(self.cfg.depth_penetration_deadband)
        penetration = torch.clamp(delta - db, min=0.0)
        u_r = torch.clamp(u + 1.0, max=float(zc.shape[2]) - 1.0)
        u_l = torch.clamp(u - 1.0, min=0.0)
        v_d = torch.clamp(v + 1.0, max=float(zc.shape[1]) - 1.0)
        v_u = torch.clamp(v - 1.0, min=0.0)
        z_ur = self._grid_sample_depth(zc, u_r, v)
        z_ul = self._grid_sample_depth(zc, u_l, v)
        z_vd = self._grid_sample_depth(zc, u, v_d)
        z_vu = self._grid_sample_depth(zc, u, v_u)

        p_c = self._unproject_cam(z_s, u, v, invk)

        p_ur = self._unproject_cam(z_ur, u_r, v, invk)
        p_ul = self._unproject_cam(z_ul, u_l, v, invk)
        p_vd = self._unproject_cam(z_vd, u, v_d, invk)
        p_vu = self._unproject_cam(z_vu, u, v_u, invk)
        # Central differences on the depth surface: ∂p/∂u ≈ (p(u+1) − p(u−1)) / 2, same for v.
        t_u = 0.5 * (p_ur - p_ul)
        t_v = 0.5 * (p_vd - p_vu)
        n_cam = torch.linalg.cross(t_u, t_v, dim=-1)
        n_cam = torch.nn.functional.normalize(n_cam, dim=-1, eps=1e-9)

        quat_exp = quat_w.unsqueeze(1).expand(-1, p, -1)
        #normals_world = math_utils.quat_apply(quat_exp, n_cam)
        points_world = math_utils.quat_apply(quat_exp, p_c) + pos_w.unsqueeze(1).expand(-1, p, -1)

        center = points_world.mean(dim=1, keepdim=True)   # (N,1,3)
        # Tactile pad outward normal in world frame (per env, broadcast to points); used as contact normal n_w.
        normals_world = center - pos_w.unsqueeze(1)

        return penetration, normals_world, points_world

    def _compute_forces_rigid(
        self,
        env_idx: Sequence[int] | slice,
        penetration: torch.Tensor,
        normals_world: torch.Tensor,
        points_world: torch.Tensor,
        elastomer_pos_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
    ) -> None:
        tactile_nf = self._data.tactile_normal_force[env_idx]
        tactile_sf = self._data.tactile_shear_force[env_idx]
        tangential_speed = self._data.tactile_tangential_speed[env_idx]
        depth_buf = self._data.penetration_depth[env_idx]
        tactile_nf.zero_()
        tactile_sf.zero_()
        tangential_speed.zero_()
        depth_buf[:] = penetration

        num_pts = self.num_tactile_points
        contact_object_pos_w, contact_object_quat_w = self._contact_object_body_view.get_transforms().split(
            [3, 4], dim=-1
        )
        contact_object_quat_w = math_utils.convert_quat(contact_object_quat_w, to="wxyz")
        contact_object_pos_w = contact_object_pos_w[env_idx]
        contact_object_quat_w = contact_object_quat_w[env_idx]

        co_vel = self._contact_object_body_view.get_velocities()[env_idx]
        em_vel = self._elastomer_body_view.get_velocities()[env_idx]

        co_lin_com = co_vel[:, :3]
        co_ang = co_vel[:, 3:]
        em_lin_com = em_vel[:, :3]
        em_ang = em_vel[:, 3:]

        co_com_off = math_utils.quat_apply(contact_object_quat_w, self._contact_object_com_b[env_idx])
        co_lin = co_lin_com - torch.cross(co_ang, co_com_off, dim=-1)
        em_com_off = math_utils.quat_apply(elastomer_quat_w, self._elastomer_com_b[env_idx])
        em_lin = em_lin_com - torch.cross(em_ang, em_com_off, dim=-1)

        co_quat_e = contact_object_quat_w.unsqueeze(1).expand(-1, num_pts, -1)

        p_obj, _ = self._transform_points_to_contact_object_local(
            points_world, contact_object_pos_w, contact_object_quat_w
        )
        p_pad_dense = math_utils.quat_apply_inverse(
            elastomer_quat_w.unsqueeze(1).expand(-1, num_pts, -1),
            points_world - elastomer_pos_w.unsqueeze(1),
        )
        tactile_vel = em_lin.unsqueeze(1) + torch.cross(
            em_ang.unsqueeze(1),
            points_world - elastomer_pos_w.unsqueeze(1),
            dim=-1,
        )

        closest_vel = co_lin.unsqueeze(1) + torch.cross(
            co_ang.unsqueeze(1),
            math_utils.quat_apply(co_quat_e, p_obj),
            dim=-1,
        )


        n_w = torch.nn.functional.normalize(normals_world, dim=-1, eps=1e-9)
        sparse_anchors = self._gather_sparse_anchors_rigid(
            env_ids=env_idx,
            elastomer_pos_w=elastomer_pos_w,
            elastomer_quat_w=elastomer_quat_w,
            p_pad_dense=p_pad_dense,
            penetration=penetration,
        )
        self._cached_sparse_anchors = sparse_anchors
        self._cached_p_pad_dense = p_pad_dense.detach()
        self._update_sparse_fn_total(env_idx, sparse_anchors)
        if (
            sparse_anchors is None
            and bool(self.cfg.use_physx_sparse_anchors)
            and bool(getattr(self.cfg, "strict_target_contact_attribution", False))
            and (not bool(self.cfg.contact_object_is_deformable))
        ):
            # No target-object PhysX anchors this frame: suppress target tactile outputs
            # to avoid attributing depth-only contacts from non-target objects.
            depth_buf.zero_()
            tangential_speed.zero_()
            tactile_nf.zero_()
            tactile_sf.zero_()
            return
        fc_norm = self._compute_corrected_normal_force(
            penetration=penetration,
            p_pad_dense=p_pad_dense,
            sparse_anchors=sparse_anchors,
        )
        rel_v = tactile_vel - closest_vel
        vt = rel_v - n_w * torch.sum(n_w * rel_v, dim=-1, keepdim=True)
        vt_n = torch.norm(vt, dim=-1)
        tangential_speed[:] = vt_n

        ft_s = self.cfg.tangential_stiffness * vt_n
        ft_d = self.cfg.friction_coefficient * fc_norm
        mask = (penetration > 0).unsqueeze(-1)
        ft_world_base = -torch.minimum(ft_s, ft_d).unsqueeze(-1) * vt / (vt_n.unsqueeze(-1).clamp(min=1e-9))
        ft_world_base = ft_world_base * mask
        if bool(self.cfg.enable_slip_stick_reconstruction):
            ft_world = self._reconstruct_tangential_force_slip_stick(
                p_pad_dense=p_pad_dense,
                vt_world=vt,
                vt_norm=vt_n,
                fc_norm=fc_norm,
                ft_world_dense=ft_world_base,
                contact_mask=mask,
                sparse_anchors=sparse_anchors,
            )
        else:
            ft_world = ft_world_base

        tactile_nf[:] = fc_norm
        tactile_sf[:] = math_utils.quat_apply_inverse(self._data.tactile_points_quat_w[env_idx], ft_world)[..., :2]

    def get_physx_shear_gt_tactile(self, env_ids: Sequence[int] | slice | None = None) -> torch.Tensor | None:
        """Dense PhysX friction GT on the tactile grid via IDW (rebuttal Table 1 proxy).

        Returns:
            Tensor of shape (N, rows, cols, 2) in the tactile frame, or ``None`` when anchors are unavailable.
        """
        anchors = self._cached_sparse_anchors
        p_pad = self._cached_p_pad_dense
        if anchors is None or p_pad is None:
            return None

        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        p_pad = p_pad[env_ids]
        quat_w = self._data.tactile_points_quat_w[env_ids]

        rows, cols = self.cfg.tactile_array_size
        n_env = p_pad.shape[0]
        n_pts = p_pad.shape[1]
        sigma = float(max(self.cfg.sticking_interp_sigma, 1e-6))
        out = torch.zeros(n_env, n_pts, 2, device=p_pad.device, dtype=p_pad.dtype)

        if isinstance(env_ids, slice):
            env_local = list(range(*env_ids.indices(self._num_envs)))
        elif torch.is_tensor(env_ids):
            env_local = [int(x) for x in env_ids.detach().cpu().tolist()]
        else:
            env_local = [int(x) for x in env_ids]

        for e_local, e_global in enumerate(env_local):
            mask_e = anchors["friction_env"] == e_local
            if not bool(mask_e.any()):
                continue
            pos_e = anchors["friction_pos_pad"][mask_e]
            ft_w = anchors["friction_ft_world"][mask_e]
            dist_ap = torch.cdist(pos_e.unsqueeze(0), p_pad[e_local].unsqueeze(0)).squeeze(0)
            nn = torch.argmin(dist_ap, dim=-1)
            q_nn = quat_w[e_local, nn]
            ft_t = math_utils.quat_apply_inverse(q_nn, ft_w)[..., :2]
            anchor_mag = torch.norm(ft_t, dim=-1)
            keep = anchor_mag > 1e-9
            if not bool(keep.any()):
                continue
            pos_e = pos_e[keep]
            ft_t = ft_t[keep]
            dist = torch.cdist(p_pad[e_local].unsqueeze(0), pos_e.unsqueeze(0)).squeeze(0)
            w = torch.exp(-(dist * dist) / (sigma * sigma))
            denom = w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            out[e_local] = torch.einsum("pa,ak->pk", w, ft_t) / denom

        return out.view(n_env, rows, cols, 2)

    def _compute_forces_deformable(
        self,
        env_idx: Sequence[int] | slice,
        penetration: torch.Tensor,
        normals_world: torch.Tensor,
        points_world: torch.Tensor,
        elastomer_pos_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
    ) -> None:
        tactile_nf = self._data.tactile_normal_force[env_idx]
        tactile_sf = self._data.tactile_shear_force[env_idx]
        tangential_speed = self._data.tactile_tangential_speed[env_idx]
        depth_buf = self._data.penetration_depth[env_idx]
        tactile_nf.zero_()
        tactile_sf.zero_()
        tangential_speed.zero_()
        depth_buf[:] = penetration

        num_pts = self.num_tactile_points
        nodal_pos = self._contact_soft_body_view.get_sim_nodal_positions()[env_idx, : self._deformable_max_vertices, :]
        nodal_vel = self._contact_soft_body_view.get_sim_nodal_velocities()[env_idx, : self._deformable_max_vertices, :]

        dist = torch.cdist(points_world, nodal_pos)
        idx = torch.argmin(dist, dim=-1)

        n_env = points_world.shape[0]
        bi = torch.arange(n_env, device=self._device).unsqueeze(1).expand(-1, num_pts)
        closest_vel = nodal_vel[bi, idx]

        em_vel = self._elastomer_body_view.get_velocities()[env_idx]
        em_lin_com = em_vel[:, :3]
        em_ang = em_vel[:, 3:]
        em_com_off = math_utils.quat_apply(elastomer_quat_w, self._elastomer_com_b[env_idx])
        em_lin = em_lin_com - torch.cross(em_ang, em_com_off, dim=-1)

        tactile_vel = em_lin.unsqueeze(1) + torch.cross(
            em_ang.unsqueeze(1),
            points_world - elastomer_pos_w.unsqueeze(1),
            dim=-1,
        )
        p_pad_dense = math_utils.quat_apply_inverse(
            elastomer_quat_w.unsqueeze(1).expand(-1, num_pts, -1),
            points_world - elastomer_pos_w.unsqueeze(1),
        )

        n_w = torch.nn.functional.normalize(normals_world, dim=-1, eps=1e-9)

        fc_norm = self._compute_corrected_normal_force(
            penetration=penetration,
            p_pad_dense=p_pad_dense,
            sparse_anchors=None,
        )

        rel_v = tactile_vel - closest_vel
        vt = rel_v - n_w * torch.sum(n_w * rel_v, dim=-1, keepdim=True)
        vt_n = torch.norm(vt, dim=-1)
        tangential_speed[:] = vt_n
        ft_s = self.cfg.tangential_stiffness * vt_n
        ft_d = self.cfg.friction_coefficient * fc_norm
        mask = (penetration > 0).unsqueeze(-1)
        ft_world_base = -torch.minimum(ft_s, ft_d).unsqueeze(-1) * vt / (vt_n.unsqueeze(-1).clamp(min=1e-9))
        ft_world_base = ft_world_base * mask
        if bool(self.cfg.enable_slip_stick_reconstruction):
            ft_world = self._reconstruct_tangential_force_slip_stick(
                p_pad_dense=p_pad_dense,
                vt_world=vt,
                vt_norm=vt_n,
                fc_norm=fc_norm,
                ft_world_dense=ft_world_base,
                contact_mask=mask,
                sparse_anchors=None,
            )
        else:
            ft_world = ft_world_base

        tactile_nf[:] = fc_norm
        tactile_sf[:] = math_utils.quat_apply_inverse(self._data.tactile_points_quat_w[env_idx], ft_world)[..., :2]

    def _compute_corrected_normal_force(
        self,
        *,
        penetration: torch.Tensor,
        p_pad_dense: torch.Tensor,
        sparse_anchors: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Compute normal force with optional local stiffness correction.

        Baseline (disabled correction): ``phi_n = k_ref * d_raw``.
        Corrected (enabled): local robust mean of ``F_n/(d+eps)`` over KNN neighbors, then
        ``delta_corr = d_raw * k_local/(k_ref+eps)``, ``phi_n = k_ref * delta_corr``.
        """
        eps = float(max(self.cfg.normal_correction_eps, 1e-12))
        k_ref_cfg = float(self.cfg.normal_correction_k_ref)
        k_ref = float(self.cfg.normal_contact_stiffness if k_ref_cfg <= 0.0 else k_ref_cfg)
        d_raw = torch.clamp(penetration, min=0.0)
        # Keep no-contact cells at zero in all modes.
        contact_mask = d_raw > 0.0
        if (not bool(self.cfg.enable_normal_correction)) or int(self.cfg.normal_correction_knn) <= 1:
            return (k_ref * d_raw) * contact_mask

        # Fallback to dense self-anchoring if sparse anchors are unavailable.
        if sparse_anchors is None:
            if (
                bool(self.cfg.use_physx_sparse_anchors)
                and bool(getattr(self.cfg, "strict_target_contact_attribution", False))
                and (not bool(self.cfg.contact_object_is_deformable))
            ):
                # No target-object anchors this frame: avoid mis-attributing depth-only contacts
                # (possibly from non-target objects) to the configured target object.
                return torch.zeros_like(d_raw)
            return (k_ref * d_raw) * contact_mask

        a_pos = sparse_anchors["normal_pos_pad"]
        a_fn = sparse_anchors["normal_fn"]
        a_env = sparse_anchors["normal_env"]
        a_depth = sparse_anchors["normal_depth"]

        n_env, n_pts, _ = p_pad_dense.shape
        k_local = torch.full((n_env, n_pts), k_ref, device=self._device, dtype=torch.float32)
        ratio_trim = float(max(0.0, min(0.49, self.cfg.normal_correction_trim_ratio)))
        k_knn = int(max(1, self.cfg.normal_correction_knn))

        for e in range(n_env):
            mask_e = a_env == e
            if int(mask_e.sum().item()) == 0:
                continue
            pos_e = a_pos[mask_e]  # (A,3)
            fn_e = torch.clamp(a_fn[mask_e], min=0.0)
            d_e = torch.clamp(a_depth[mask_e], min=0.0)
            ratio_cap = float(getattr(self.cfg, "normal_correction_max_stiffness_ratio", 0.0))
            if ratio_cap <= 0.0:
                ratio_cap = k_ref * 50.0
            ratio_e = torch.clamp(fn_e / (d_e + eps), max=ratio_cap)
            # dist: (P, A)
            dist = torch.cdist(p_pad_dense[e].unsqueeze(0), pos_e.unsqueeze(0)).squeeze(0)
            k_use = int(min(k_knn, pos_e.shape[0]))
            nn_idx = torch.topk(dist, k=k_use, dim=-1, largest=False).indices
            ratios = ratio_e[nn_idx]  # (P, K)
            if k_use >= 3 and ratio_trim > 0.0:
                trim_n = int(ratio_trim * k_use)
                if trim_n > 0 and (k_use - 2 * trim_n) >= 1:
                    ratios = torch.sort(ratios, dim=-1).values[..., trim_n : (k_use - trim_n)]
            k_local[e] = torch.mean(ratios, dim=-1)

        delta_corr = d_raw * (k_local / (k_ref + eps))
        phi_n = k_ref * delta_corr
        return phi_n * contact_mask

    @staticmethod
    def _project_to_friction_disk(ft_vec: torch.Tensor, radius: torch.Tensor, eps: float) -> torch.Tensor:
        """Project 3D tangential vectors onto per-point Coulomb disk in their tangent plane."""
        mag = torch.norm(ft_vec, dim=-1)
        scale = torch.minimum(torch.ones_like(mag), radius / (mag + eps))
        return ft_vec * scale.unsqueeze(-1)

    def _reconstruct_tangential_force_slip_stick(
        self,
        *,
        p_pad_dense: torch.Tensor,
        vt_world: torch.Tensor,
        vt_norm: torch.Tensor,
        fc_norm: torch.Tensor,
        ft_world_dense: torch.Tensor,
        contact_mask: torch.Tensor,
        sparse_anchors: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Slip/sticking tangential reconstruction with Coulomb projection."""
        eps = float(max(self.cfg.tangential_anchor_eps, 1e-12))
        tau_v = float(max(self.cfg.slip_speed_threshold, 0.0))
        mu = float(self.cfg.friction_coefficient)
        radius = torch.clamp(mu * fc_norm, min=0.0)
        contact = contact_mask.squeeze(-1)
        if sparse_anchors is None:
            if (
                bool(self.cfg.use_physx_sparse_anchors)
                and bool(getattr(self.cfg, "strict_target_contact_attribution", False))
                and (not bool(self.cfg.contact_object_is_deformable))
            ):
                return torch.zeros_like(ft_world_dense)
            return self._project_to_friction_disk(ft_world_dense, radius, eps) * contact_mask

        # partition
        is_sl = (vt_norm > tau_v) & contact
        is_st = contact & (~is_sl)

        a_pos = sparse_anchors["friction_pos_pad"]
        a_ft = sparse_anchors["friction_ft_world"]
        a_env = sparse_anchors["friction_env"]

        n_env, n_pts, _ = p_pad_dense.shape
        out = self._project_to_friction_disk(ft_world_dense, radius, eps)
        k_knn = int(max(1, self.cfg.tangential_anchor_knn))
        gmin, gmax = float(self.cfg.tangential_gamma_clip[0]), float(self.cfg.tangential_gamma_clip[1])
        if gmax < gmin:
            gmin, gmax = gmax, gmin
        sigma = float(max(self.cfg.sticking_interp_sigma, 1e-6))

        ft_dense_mag = torch.norm(ft_world_dense, dim=-1)

        for e in range(n_env):
            mask_e = a_env == e
            if int(mask_e.sum().item()) == 0:
                continue
            pos_e = a_pos[mask_e]
            ft_e = a_ft[mask_e]

            # Anchor-to-dense nearest mapping for anchor-type partition and gamma source ratio.
            dist_ad = torch.cdist(pos_e.unsqueeze(0), p_pad_dense[e].unsqueeze(0)).squeeze(0)  # (A,P)
            nn_d = torch.argmin(dist_ad, dim=-1)  # (A,)
            vt_anchor = vt_norm[e, nn_d]
            is_sl_anchor = vt_anchor > tau_v
            is_st_anchor = ~is_sl_anchor

            # Sliding region: local gamma from sliding anchors.
            dist = torch.cdist(p_pad_dense[e].unsqueeze(0), pos_e.unsqueeze(0)).squeeze(0)  # (P,A)
            k_use = int(min(k_knn, pos_e.shape[0]))
            nn_idx = torch.topk(dist, k=k_use, dim=-1, largest=False).indices
            anchor_mag = torch.norm(ft_e, dim=-1)
            ratio_anchor = anchor_mag / (ft_dense_mag[e, nn_d] + eps)
            ratio_anchor = torch.clamp(ratio_anchor, min=gmin, max=gmax)
            ratio_nn = ratio_anchor[nn_idx]  # (P,K)
            sl_anchor_nn = is_sl_anchor[nn_idx].float()
            gamma = torch.sum(ratio_nn * sl_anchor_nn, dim=-1) / torch.clamp(torch.sum(sl_anchor_nn, dim=-1), min=1.0)
            gamma = torch.clamp(gamma, min=gmin, max=gmax)
            ft_sl = self._project_to_friction_disk(gamma.unsqueeze(-1) * ft_world_dense[e], radius[e], eps)

            # Sticking region: Gaussian interpolation from sticking anchors.
            if bool(is_st_anchor.any().item()):
                pos_st = pos_e[is_st_anchor]
                ft_st_anchor = ft_e[is_st_anchor]
                dist_st = torch.cdist(p_pad_dense[e].unsqueeze(0), pos_st.unsqueeze(0)).squeeze(0)  # (P,Ast)
                w = torch.exp(-(dist_st * dist_st) / (sigma * sigma))
                denom = torch.sum(w, dim=-1, keepdim=True).clamp(min=eps)
                ft_st_interp = torch.einsum("pa,ak->pk", w, ft_st_anchor) / denom
                ft_st = self._project_to_friction_disk(ft_st_interp, radius[e], eps)
            else:
                ft_st = self._project_to_friction_disk(ft_world_dense[e], radius[e], eps)

            out[e] = torch.where(is_sl[e].unsqueeze(-1), ft_sl, ft_st)

        out = out * contact_mask
        return out

    def _gather_sparse_anchors_rigid(
        self,
        *,
        env_ids: Sequence[int] | slice,
        elastomer_pos_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
        p_pad_dense: torch.Tensor,
        penetration: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        """Gather PhysX sparse contact/friction anchors for the elastomer-vs-object pair.

        Returns local-batch-indexed anchor tensors. Returns ``None`` when unavailable.
        """
        if (self._contact_physx_view is None) or (not bool(self.cfg.use_physx_sparse_anchors)):
            self._sparse_anchor_backend_ok = False
            self._last_sparse_anchor_count = 0
            self._last_sparse_contact_count = 0
            self._last_sparse_friction_count = 0
            self._last_sparse_used = False
            return None
        # Some scenes can produce a RigidContactView object whose backend is invalid.
        backend = getattr(self._contact_physx_view, "_backend", None)
        if backend is None:
            self._sparse_anchor_backend_ok = False
            self._last_sparse_anchor_count = 0
            self._last_sparse_contact_count = 0
            self._last_sparse_friction_count = 0
            self._last_sparse_used = False
            if not self._sparse_anchor_warned:
                logger.warning(
                    "VisuoTactileSensorV2: sparse PhysX contact view backend is None; "
                    "falling back to dense anchors."
                )
                self._sparse_anchor_warned = True
            return None
        try:
            num_filters = int(self._contact_physx_view.filter_count)
        except Exception as exc:
            self._sparse_anchor_backend_ok = False
            self._last_sparse_anchor_count = 0
            self._last_sparse_contact_count = 0
            self._last_sparse_friction_count = 0
            self._last_sparse_used = False
            if not self._sparse_anchor_warned:
                logger.warning(
                    "VisuoTactileSensorV2: sparse PhysX contact view unavailable (%s); "
                    "falling back to dense anchors.",
                    exc,
                )
                self._sparse_anchor_warned = True
            return None
        if num_filters <= 0:
            self._sparse_anchor_backend_ok = False
            self._last_sparse_anchor_count = 0
            self._last_sparse_contact_count = 0
            self._last_sparse_friction_count = 0
            self._last_sparse_used = False
            if not self._sparse_anchor_warned:
                logger.warning(
                    "VisuoTactileSensorV2: sparse PhysX contact view has zero filters; "
                    "falling back to dense anchors."
                )
                self._sparse_anchor_warned = True
            return None
        try:
            forces, points, _normals, _seps, buffer_count, buffer_start_indices = self._contact_physx_view.get_contact_data(
                dt=self._sim_physics_dt
            )
            friction_forces, friction_points, buffer_count_f, buffer_start_indices_f = self._contact_physx_view.get_friction_data(
                dt=self._sim_physics_dt
            )
        except Exception as exc:
            self._sparse_anchor_backend_ok = False
            self._last_sparse_anchor_count = 0
            self._last_sparse_contact_count = 0
            self._last_sparse_friction_count = 0
            self._last_sparse_used = False
            if not self._sparse_anchor_warned:
                logger.warning(
                    "VisuoTactileSensorV2: sparse PhysX contact query failed (%s); fallback to dense anchors.",
                    exc,
                )
                self._sparse_anchor_warned = True
            return None
        # shape: (num_envs * num_bodies(=1), num_filters)
        counts = buffer_count.view(self._num_envs, num_filters)
        starts = buffer_start_indices.view(self._num_envs, num_filters)
        counts_f = buffer_count_f.view(self._num_envs, num_filters)
        starts_f = buffer_start_indices_f.view(self._num_envs, num_filters)

        if isinstance(env_ids, slice):
            env_global = list(range(*env_ids.indices(self._num_envs)))
        elif torch.is_tensor(env_ids):
            env_global = [int(x) for x in env_ids.detach().cpu().tolist()]
        else:
            env_global = [int(x) for x in env_ids]

        n_list_pos: list[torch.Tensor] = []
        n_list_fn: list[torch.Tensor] = []
        n_list_depth: list[torch.Tensor] = []
        n_list_env: list[torch.Tensor] = []
        t_list_pos: list[torch.Tensor] = []
        t_list_ft: list[torch.Tensor] = []
        t_list_env: list[torch.Tensor] = []
        eps = float(max(self.cfg.normal_correction_eps, 1e-12))
        total_contact_count = 0
        total_friction_count = 0

        for e_local, e_global in enumerate(env_global):
            q = elastomer_quat_w[e_local]
            p0 = elastomer_pos_w[e_local]
            for f_idx in range(num_filters):
                cnt = int(counts[e_global, f_idx].item())
                total_contact_count += cnt
                if cnt > 0:
                    st = int(starts[e_global, f_idx].item())
                    sl = slice(st, st + cnt)
                    p_w = points[sl]
                    fn_vals = torch.clamp(forces[sl].reshape(-1), min=0.0)
                    p_pad = math_utils.quat_apply_inverse(q.unsqueeze(0).expand(cnt, -1), p_w - p0.unsqueeze(0))
                    # Anchor depth proxy: nearest dense penetration.
                    dist_ap = torch.cdist(p_pad.unsqueeze(0), p_pad_dense[e_local].unsqueeze(0)).squeeze(0)  # (A,P)
                    nn_d = torch.argmin(dist_ap, dim=-1)
                    d_anchor = torch.clamp(penetration[e_local, nn_d], min=0.0) + eps
                    n_list_pos.append(p_pad)
                    n_list_fn.append(fn_vals)
                    n_list_depth.append(d_anchor)
                    n_list_env.append(torch.full((cnt,), e_local, device=self._device, dtype=torch.long))

                cnt_f = int(counts_f[e_global, f_idx].item())
                total_friction_count += cnt_f
                if cnt_f > 0:
                    stf = int(starts_f[e_global, f_idx].item())
                    slf = slice(stf, stf + cnt_f)
                    fp_w = friction_points[slf]
                    ff_w = friction_forces[slf]
                    fp_pad = math_utils.quat_apply_inverse(q.unsqueeze(0).expand(cnt_f, -1), fp_w - p0.unsqueeze(0))
                    t_list_pos.append(fp_pad)
                    t_list_ft.append(ff_w)
                    t_list_env.append(torch.full((cnt_f,), e_local, device=self._device, dtype=torch.long))

        if len(n_list_pos) == 0 and len(t_list_pos) == 0:
            self._sparse_anchor_backend_ok = True
            self._last_sparse_anchor_count = 0
            self._last_sparse_contact_count = int(total_contact_count)
            self._last_sparse_friction_count = int(total_friction_count)
            self._last_sparse_used = False
            return None

        def _cat_or_empty(lst: list[torch.Tensor], shape_tail: tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
            if len(lst) == 0:
                return torch.zeros((0, *shape_tail), device=self._device, dtype=dtype)
            return torch.cat(lst, dim=0)

        out = {
            "normal_pos_pad": _cat_or_empty(n_list_pos, (3,)),
            "normal_fn": _cat_or_empty(n_list_fn, (), dtype=torch.float32).reshape(-1),
            "normal_depth": _cat_or_empty(n_list_depth, (), dtype=torch.float32).reshape(-1),
            "normal_env": _cat_or_empty(n_list_env, (), dtype=torch.long).reshape(-1),
            "friction_pos_pad": _cat_or_empty(t_list_pos, (3,)),
            "friction_ft_world": _cat_or_empty(t_list_ft, (3,)),
            "friction_env": _cat_or_empty(t_list_env, (), dtype=torch.long).reshape(-1),
        }
        self._sparse_anchor_backend_ok = True
        self._last_sparse_anchor_count = int(out["normal_env"].numel())
        self._last_sparse_contact_count = int(total_contact_count)
        self._last_sparse_friction_count = int(total_friction_count)
        self._last_sparse_used = True
        return out


@configclass
class VisuoTactileSensorV2Cfg(VisuoTactileSensorCfg):
    """Same as :class:`VisuoTactileSensorCfg` with ``class_type`` = :class:`VisuoTactileSensorV2`."""

    class_type: type = VisuoTactileSensorV2
