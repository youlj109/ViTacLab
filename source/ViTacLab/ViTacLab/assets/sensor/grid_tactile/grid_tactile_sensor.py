# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grid tactile sensor: per-contact data from PhysX, binned to a 2D pressure map."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.sensors.contact_sensor import ContactSensor
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_inverse

from .grid_tactile_sensor_data import GridTactileSensorData

if TYPE_CHECKING:
    from .grid_tactile_sensor_cfg import GridTactileSensorCfg


def _tangent_axes_from_normal(normal_axis: int) -> tuple[int, int]:
    """Return the two body axes spanning the tangent plane for a given outward normal axis."""
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    return axes[0], axes[1]


class GridTactileSensor(ContactSensor):
    """Bins per-contact data into 2D grids on the sensor rigid body.

    * **Normal**: from :meth:`RigidContactView.get_contact_data` — scalar×normal, then signed
      :math:`f \\cdot n_{pad}` per cell (contact point positions).
    * **Tangential (friction)**: from :meth:`RigidContactView.get_friction_data` — 3D friction force
      projected onto patch tangent axes ``u``, ``v``, binned using friction point positions.

    Requires non-empty ``filter_prim_paths_expr``.
    """

    cfg: GridTactileSensorCfg

    def __init__(self, cfg: GridTactileSensorCfg):
        super().__init__(cfg)
        self._data = GridTactileSensorData()
        self._pad_normal_body: torch.Tensor | None = None
        self._e_tangent_u: torch.Tensor | None = None
        self._e_tangent_v: torch.Tensor | None = None
        self._tangent_u, self._tangent_v = _tangent_axes_from_normal(self.cfg.pad_normal_axis)

    @property
    def data(self) -> GridTactileSensorData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if self._data.force_grid is not None:
            self._data.force_grid[env_ids] = 0.0
        if self._data.friction_grid_uv is not None:
            self._data.friction_grid_uv[env_ids] = 0.0

    def _initialize_impl(self):
        if len(self.cfg.filter_prim_paths_expr) == 0:
            raise ValueError(
                "GridTactileSensor requires non-empty 'filter_prim_paths_expr' "
                "(RigidContactView.get_contact_data needs filter prims)."
            )
        super()._initialize_impl()
        self._pad_normal_body = torch.zeros(3, device=self._device)
        self._pad_normal_body[self.cfg.pad_normal_axis] = 1.0
        self._e_tangent_u = torch.zeros(3, device=self._device)
        self._e_tangent_u[self._tangent_u] = 1.0
        self._e_tangent_v = torch.zeros(3, device=self._device)
        self._e_tangent_v[self._tangent_v] = 1.0
        h, w = self.cfg.grid_resolution
        num_filters = self.contact_physx_view.filter_count
        self._data.force_grid = torch.zeros(
            self._num_envs, self._num_bodies, num_filters, h, w, device=self._device
        )
        if self.cfg.track_friction:
            self._data.friction_grid_uv = torch.zeros(
                self._num_envs, self._num_bodies, num_filters, 2, h, w, device=self._device
            )
        else:
            self._data.friction_grid_uv = None

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        if self._data.force_grid is None or self._pad_normal_body is None:
            return
        if self._e_tangent_u is None or self._e_tangent_v is None:
            return

        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        self._data.force_grid[env_ids] = 0.0
        if self.cfg.track_friction and self._data.friction_grid_uv is not None:
            self._data.friction_grid_uv[env_ids] = 0.0

        forces, points, normals, _separations, buffer_count, buffer_start_indices = (
            self.contact_physx_view.get_contact_data(dt=self._sim_physics_dt)
        )

        f_scalars = forces.reshape(-1, 1)
        f_vec = f_scalars * normals

        if self.cfg.track_friction:
            friction_forces, friction_points, buffer_count_f, buffer_start_indices_f = (
                self.contact_physx_view.get_friction_data(dt=self._sim_physics_dt)
            )
        else:
            friction_forces = friction_points = None
            buffer_count_f = buffer_start_indices_f = None

        pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)
        pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")

        h, w = self.cfg.grid_resolution
        ext_u, ext_v = self.cfg.patch_extent

        num_sensor_rows = self._num_envs * self._num_bodies
        num_filters = self.contact_physx_view.filter_count

        counts = buffer_count.view(num_sensor_rows, num_filters)
        starts = buffer_start_indices.view(num_sensor_rows, num_filters)
        if self.cfg.track_friction and buffer_count_f is not None:
            counts_f = buffer_count_f.view(num_sensor_rows, num_filters)
            starts_f = buffer_start_indices_f.view(num_sensor_rows, num_filters)
        else:
            counts_f = starts_f = None

        pad_n = self._pad_normal_body
        e_u = self._e_tangent_u
        e_v = self._e_tangent_v

        if env_ids == slice(None):
            env_iter = list(range(self._num_envs))
        elif torch.is_tensor(env_ids):
            env_iter = env_ids.tolist()
        else:
            env_iter = list(env_ids)

        for env_idx in env_iter:
            for body_idx in range(self._num_bodies):
                sensor_row = env_idx * self._num_bodies + body_idx
                pos_w = pose[env_idx, body_idx, :3]
                quat_w = pose[env_idx, body_idx, 3:7]
                n_pad_w = quat_apply(quat_w.unsqueeze(0), pad_n.unsqueeze(0)).squeeze(0)
                t_u_w = quat_apply(quat_w.unsqueeze(0), e_u.unsqueeze(0)).squeeze(0)
                t_v_w = quat_apply(quat_w.unsqueeze(0), e_v.unsqueeze(0)).squeeze(0)

                for filt_idx in range(num_filters):
                    cnt = int(counts[sensor_row, filt_idx].item())
                    if cnt == 0:
                        continue
                    start = int(starts[sensor_row, filt_idx].item())
                    sl = slice(start, start + cnt)

                    pts_w = points[sl]
                    fv = f_vec[sl]
                    pressure = (fv * n_pad_w.unsqueeze(0)).sum(dim=-1)

                    rel_w = pts_w - pos_w.unsqueeze(0)
                    p_b = quat_apply_inverse(quat_w.unsqueeze(0).expand(cnt, -1), rel_w)

                    u = p_b[:, self._tangent_u]
                    v = p_b[:, self._tangent_v]

                    iu = torch.floor((u + 0.5 * ext_u) / ext_u * h).long().clamp(0, h - 1)
                    iv = torch.floor((v + 0.5 * ext_v) / ext_v * w).long().clamp(0, w - 1)

                    grid = self._data.force_grid[env_idx, body_idx, filt_idx]
                    grid.index_put_((iu, iv), pressure, accumulate=True)

                    if (
                        self.cfg.track_friction
                        and self._data.friction_grid_uv is not None
                        and friction_forces is not None
                        and friction_points is not None
                        and counts_f is not None
                        and starts_f is not None
                    ):
                        cnt_f = int(counts_f[sensor_row, filt_idx].item())
                        if cnt_f > 0:
                            start_f = int(starts_f[sensor_row, filt_idx].item())
                            sl_f = slice(start_f, start_f + cnt_f)
                            ff = friction_forces[sl_f]
                            fp_w = friction_points[sl_f]
                            n_ff = ff.shape[0]
                            if n_ff != cnt:
                                n_use = min(cnt, n_ff, cnt_f)
                                ff = ff[:n_use]
                                fp_w = fp_w[:n_use]
                            else:
                                n_use = n_ff

                            fu = (ff * t_u_w.unsqueeze(0)).sum(dim=-1)
                            fv = (ff * t_v_w.unsqueeze(0)).sum(dim=-1)

                            rel_fp = fp_w - pos_w.unsqueeze(0)
                            p_b_fp = quat_apply_inverse(quat_w.unsqueeze(0).expand(n_use, -1), rel_fp)
                            u_fp = p_b_fp[:, self._tangent_u]
                            v_fp = p_b_fp[:, self._tangent_v]
                            iu_f = torch.floor((u_fp + 0.5 * ext_u) / ext_u * h).long().clamp(0, h - 1)
                            iv_f = torch.floor((v_fp + 0.5 * ext_v) / ext_v * w).long().clamp(0, w - 1)

                            grid_f = self._data.friction_grid_uv[env_idx, body_idx, filt_idx]
                            grid_f[0].index_put_((iu_f, iv_f), fu, accumulate=True)
                            grid_f[1].index_put_((iu_f, iv_f), fv, accumulate=True)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._pad_normal_body = None
        self._e_tangent_u = None
        self._e_tangent_v = None
