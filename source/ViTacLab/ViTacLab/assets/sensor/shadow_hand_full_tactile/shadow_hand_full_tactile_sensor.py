# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand: all hand contacts binned into one palm-frame 3D voxel grid (fn + 2 friction comps)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.sensors.contact_sensor import ContactSensor
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_inverse

from .shadow_hand_full_tactile_data import ShadowHandFullTactileData

if TYPE_CHECKING:
    from .shadow_hand_full_tactile_sensor_cfg import ShadowHandFullTactileSensorCfg


class ShadowHandFullTactileSensor(ContactSensor):
    """Unifies every contact link in ``prim_path`` into a single voxel grid in the **palm** frame.

    * Normal channel: signed scalar from ``get_contact_data`` ``forces`` (same convention as PhysX docs).
    * Tangential channels: friction 3-vector from ``get_friction_data``, dotted with palm +X and +Y
      axes in world (palm body +Z is outward palm normal–oriented; tangents follow +X/+Y in the palm prim).
    * Contact positions for ``fn`` use contact ``points``; friction uses ``friction_points``, both mapped
      to palm frame via the **same** palm pose for each environment.
    """

    cfg: ShadowHandFullTactileSensorCfg

    def __init__(self, cfg: ShadowHandFullTactileSensorCfg):
        super().__init__(cfg)
        self._data = ShadowHandFullTactileData()
        self._palm_body_idx: int | None = None
        self._e_tx: torch.Tensor | None = None
        self._e_ty: torch.Tensor | None = None
        self._sum_contact_normal_palm: torch.Tensor | None = None
        self._cnt_contact_normal: torch.Tensor | None = None

    @property
    def data(self) -> ShadowHandFullTactileData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if self._data.voxel_grid is not None:
            self._data.voxel_grid[env_ids] = 0.0
        if self._data.contact_normal_points_mean_palm is not None:
            self._data.contact_normal_points_mean_palm[env_ids] = float("nan")
        if self._data.contact_normal_point_count is not None:
            self._data.contact_normal_point_count[env_ids] = 0

    def _initialize_impl(self):
        if len(self.cfg.filter_prim_paths_expr) == 0:
            raise ValueError(
                "ShadowHandFullTactileSensor requires non-empty 'filter_prim_paths_expr' "
                "(PhysX get_contact_data / get_friction_data need filter prims)."
            )
        super()._initialize_impl()

        for i, name in enumerate(self.body_names):
            if self.cfg.palm_link_name_substr in name:
                self._palm_body_idx = i
                break
        if self._palm_body_idx is None:
            raise RuntimeError(
                f"No palm body found matching substr {self.cfg.palm_link_name_substr!r} in {self.body_names}"
            )

        self._e_tx = torch.tensor([1.0, 0.0, 0.0], device=self._device, dtype=torch.float32)
        self._e_ty = torch.tensor([0.0, 1.0, 0.0], device=self._device, dtype=torch.float32)

        nx, ny, nz = self.cfg.voxel_resolution
        num_filters = self.contact_physx_view.filter_count
        self._data.voxel_grid = torch.zeros(
            self._num_envs, num_filters, nx, ny, nz, 3, device=self._device
        )

        self._sum_contact_normal_palm = torch.zeros(
            self._num_envs, num_filters, 3, device=self._device, dtype=torch.float32
        )
        self._cnt_contact_normal = torch.zeros(
            self._num_envs, num_filters, device=self._device, dtype=torch.long
        )
        self._data.contact_normal_points_mean_palm = torch.full(
            (self._num_envs, num_filters, 3), float("nan"), device=self._device, dtype=torch.float32
        )
        self._data.contact_normal_point_count = torch.zeros(
            self._num_envs, num_filters, device=self._device, dtype=torch.long
        )

        bmin = torch.tensor(self.cfg.voxel_min_bounds_palm, device=self._device, dtype=torch.float32)
        bmax = torch.tensor(self.cfg.voxel_max_bounds_palm, device=self._device, dtype=torch.float32)
        if torch.any(bmax <= bmin):
            raise ValueError("voxel_max_bounds_palm must be strictly greater than voxel_min_bounds_palm.")

    def _points_to_voxel_indices(
        self,
        p_palm: torch.Tensor,
        nx: int,
        ny: int,
        nz: int,
        bmin: torch.Tensor,
        bmax: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map palm-frame points (N,3) to integer voxel indices."""
        extent = bmax - bmin
        t = (p_palm - bmin.unsqueeze(0)) / extent.unsqueeze(0)
        ix = torch.floor(t[:, 0] * nx).long().clamp(0, nx - 1)
        iy = torch.floor(t[:, 1] * ny).long().clamp(0, ny - 1)
        iz = torch.floor(t[:, 2] * nz).long().clamp(0, nz - 1)
        return ix, iy, iz

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        if self._data.voxel_grid is None or self._palm_body_idx is None:
            return
        if self._e_tx is None or self._e_ty is None:
            return

        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        self._data.voxel_grid[env_ids] = 0.0

        assert self._sum_contact_normal_palm is not None and self._cnt_contact_normal is not None
        self._sum_contact_normal_palm.zero_()
        self._cnt_contact_normal.zero_()

        forces, points, _normals, _sep, buffer_count, buffer_start_indices = (
            self.contact_physx_view.get_contact_data(dt=self._sim_physics_dt)
        )
        f_scalars = forces.reshape(-1)

        if self.cfg.track_friction:
            friction_forces, friction_points, buffer_count_f, buffer_start_indices_f = (
                self.contact_physx_view.get_friction_data(dt=self._sim_physics_dt)
            )
        else:
            friction_forces = friction_points = None
            buffer_count_f = buffer_start_indices_f = None

        pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)
        pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")

        nx, ny, nz = self.cfg.voxel_resolution
        bmin = torch.tensor(self.cfg.voxel_min_bounds_palm, device=self._device, dtype=torch.float32)
        bmax = torch.tensor(self.cfg.voxel_max_bounds_palm, device=self._device, dtype=torch.float32)

        num_sensor_rows = self._num_envs * self._num_bodies
        num_filters = self.contact_physx_view.filter_count
        counts = buffer_count.view(num_sensor_rows, num_filters)
        starts = buffer_start_indices.view(num_sensor_rows, num_filters)
        if self.cfg.track_friction and buffer_count_f is not None:
            counts_f = buffer_count_f.view(num_sensor_rows, num_filters)
            starts_f = buffer_start_indices_f.view(num_sensor_rows, num_filters)
        else:
            counts_f = None
            starts_f = None

        if env_ids == slice(None):
            env_iter = list(range(self._num_envs))
        elif torch.is_tensor(env_ids):
            env_iter = env_ids.tolist()
        else:
            env_iter = list(env_ids)

        palm_idx = self._palm_body_idx

        for env_idx in env_iter:
            pos_palm = pose[env_idx, palm_idx, :3]
            quat_palm = pose[env_idx, palm_idx, 3:7]
            t_x_w = quat_apply(quat_palm.unsqueeze(0), self._e_tx.unsqueeze(0)).squeeze(0)
            t_y_w = quat_apply(quat_palm.unsqueeze(0), self._e_ty.unsqueeze(0)).squeeze(0)

            for body_idx in range(self._num_bodies):
                sensor_row = env_idx * self._num_bodies + body_idx

                for filt_idx in range(num_filters):
                    cnt = int(counts[sensor_row, filt_idx].item())
                    if cnt == 0:
                        continue
                    start = int(starts[sensor_row, filt_idx].item())
                    sl = slice(start, start + cnt)

                    pts_w = points[sl]
                    fn_vals = f_scalars[sl].reshape(-1)

                    p_palm_c = quat_apply_inverse(
                        quat_palm.unsqueeze(0).expand(cnt, -1),
                        pts_w - pos_palm.unsqueeze(0),
                    )
                    self._sum_contact_normal_palm[env_idx, filt_idx] += p_palm_c.sum(dim=0)
                    self._cnt_contact_normal[env_idx, filt_idx] += cnt

                    ix, iy, iz = self._points_to_voxel_indices(p_palm_c, nx, ny, nz, bmin, bmax)

                    vox = self._data.voxel_grid[env_idx, filt_idx]
                    vox[..., 0].index_put_((ix, iy, iz), fn_vals, accumulate=True)

                    if (
                        self.cfg.track_friction
                        and friction_forces is not None
                        and friction_points is not None
                        and counts_f is not None
                        and starts_f is not None
                    ):
                        cnt_f = int(counts_f[sensor_row, filt_idx].item())
                        if cnt_f > 0:
                            stf = int(starts_f[sensor_row, filt_idx].item())
                            slf = slice(stf, stf + cnt_f)
                            ff = friction_forces[slf]
                            fp_w = friction_points[slf]
                            nf = ff.shape[0]
                            n_use = min(cnt, nf, cnt_f)
                            ff = ff[:n_use]
                            fp_w = fp_w[:n_use]

                            ft1 = (ff * t_x_w.unsqueeze(0)).sum(dim=-1)
                            ft2 = (ff * t_y_w.unsqueeze(0)).sum(dim=-1)

                            p_palm_f = quat_apply_inverse(
                                quat_palm.unsqueeze(0).expand(n_use, -1),
                                fp_w - pos_palm.unsqueeze(0),
                            )
                            ixf, iyf, izf = self._points_to_voxel_indices(
                                p_palm_f, nx, ny, nz, bmin, bmax
                            )
                            vox[..., 1].index_put_((ixf, iyf, izf), ft1, accumulate=True)
                            vox[..., 2].index_put_((ixf, iyf, izf), ft2, accumulate=True)

        cnt = self._cnt_contact_normal
        s = self._sum_contact_normal_palm
        assert cnt is not None and s is not None
        valid = (cnt > 0).unsqueeze(-1)
        mean = torch.where(
            valid,
            s / cnt.unsqueeze(-1).float().clamp(min=1),
            torch.full_like(s, float("nan")),
        )
        self._data.contact_normal_points_mean_palm.copy_(mean)  # type: ignore[union-attr]
        self._data.contact_normal_point_count.copy_(cnt)  # type: ignore[union-attr]

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._palm_body_idx = None
        self._e_tx = None
        self._e_ty = None
        self._sum_contact_normal_palm = None
        self._cnt_contact_normal = None
