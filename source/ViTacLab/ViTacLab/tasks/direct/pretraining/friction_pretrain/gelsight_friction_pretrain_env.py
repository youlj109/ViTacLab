# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""切向剪切触觉与滑动监督对齐：固定法向载荷，随机 μ，可选平面切向推力。"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import sample_uniform

from ..gelsight_finger_pretrain_base_env import (
    GelsightFingerPretrainBaseEnv,
    TACTILE_NORMAL_DIM,
    TACTILE_SHEAR_DIM,
)
from ..gelsight_pretrain_obs import pretrain_obs_dim
from .gelsight_friction_pretrain_env_cfg import GelsightFingerFrictionPretrainEnvCfg


class GelsightFingerFrictionPretrainEnv(GelsightFingerPretrainBaseEnv):
    """监督重点：触觉剪切（切向）与 ``gt_is_sliding``；法向载荷 F_n = m g 固定。"""

    cfg: GelsightFingerFrictionPretrainEnvCfg

    def __init__(self, cfg: GelsightFingerFrictionPretrainEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = pretrain_obs_dim(cfg.use_full_tactile_obs)
        super().__init__(cfg, render_mode, **kwargs)

        self._gt_mu_s = torch.zeros(self.num_envs, device=self.device)
        self._gt_mu_d = torch.zeros(self.num_envs, device=self.device)
        self._object_mass_kg = float(cfg.object_mass_kg)
        self._object_nominal_z_env = float(cfg.contact_object_cfg.init_state.pos[2])
        # 世界系 XY 侧向恒力符号；碰虚拟墙时与速度反射一起翻转。
        self._lateral_push_sign_x = torch.ones(self.num_envs, device=self.device)
        self._lateral_push_sign_y = torch.ones(self.num_envs, device=self.device)
        self._applied_force_world = torch.zeros(self.num_envs, 3, device=self.device)

        self._plot_xyz_force_live_disabled = False
        self._force_plot_fig = None
        self._force_plot_ax = None
        self._force_plot_lines: list | None = None
        self._force_plot_t: deque | None = None
        self._force_plot_fx: deque | None = None
        self._force_plot_fy: deque | None = None
        self._force_plot_fz: deque | None = None
        if cfg.plot_xyz_force_live:
            mp = max(32, int(cfg.plot_xyz_force_live_max_points))
            self._force_plot_t = deque(maxlen=mp)
            self._force_plot_fx = deque(maxlen=mp)
            self._force_plot_fy = deque(maxlen=mp)
            self._force_plot_fz = deque(maxlen=mp)

    def _flip_lateral_push_signs_from_position_only(self) -> None:
        """无虚拟墙时：按位置阈值翻转侧向力符号（不修改位姿/速度）。"""
        half = float(self.cfg.object_xy_bounds_half_extent)
        margin = float(self.cfg.boundary_flip_margin_m)
        hi = half - margin
        lo = -half + margin
        pos_env = self.contact_object.data.root_pos_w - self.scene.env_origins
        px = pos_env[:, 0]
        py = pos_env[:, 1]
        sx = self._lateral_push_sign_x
        sx = torch.where((px >= hi) & (sx > 0), -torch.ones_like(sx), sx)
        sx = torch.where((px <= lo) & (sx < 0), torch.ones_like(sx), sx)
        self._lateral_push_sign_x = sx
        sy = self._lateral_push_sign_y
        sy = torch.where((py >= hi) & (sy > 0), -torch.ones_like(sy), sy)
        sy = torch.where((py <= lo) & (sy < 0), torch.ones_like(sy), sy)
        self._lateral_push_sign_y = sy

    def _resolve_xy_walls(self) -> None:
        """env 系矩形虚拟墙：夹紧 xy、反射法向速度；可选同时翻转侧向恒力符号。"""
        half = float(self.cfg.object_xy_bounds_half_extent)
        margin = float(self.cfg.boundary_flip_margin_m)
        wall_half = max(half - margin, 1e-6)
        r = float(self.cfg.boundary_wall_restitution)

        pos_w = self.contact_object.data.root_pos_w
        quat_w = self.contact_object.data.root_quat_w
        d = self.contact_object.data
        lin_w = getattr(d, "root_lin_vel_w", d.root_link_lin_vel_w).clone()
        ang_w = d.root_ang_vel_w
        pos_env = pos_w - self.scene.env_origins
        px = pos_env[:, 0]
        py = pos_env[:, 1]
        pz = pos_env[:, 2]
        vx = lin_w[:, 0]
        vy = lin_w[:, 1]

        hit_x_hi = px > wall_half
        hit_x_lo = px < -wall_half
        hit_y_hi = py > wall_half
        hit_y_lo = py < -wall_half
        hit_any = hit_x_hi | hit_x_lo | hit_y_hi | hit_y_lo
        if not hit_any.any():
            return

        px_new = torch.clamp(px, -wall_half, wall_half)
        py_new = torch.clamp(py, -wall_half, wall_half)

        vx_new = torch.where(hit_x_hi & (vx > 0), -vx * r, vx)
        vx_new = torch.where(hit_x_lo & (vx < 0), -vx * r, vx_new)
        vy_new = torch.where(hit_y_hi & (vy > 0), -vy * r, vy)
        vy_new = torch.where(hit_y_lo & (vy < 0), -vy * r, vy_new)

        lin_w[:, 0] = vx_new
        lin_w[:, 1] = vy_new

        if self.cfg.flip_lateral_push_at_xy_bounds:
            sx = self._lateral_push_sign_x
            sx = torch.where(hit_x_hi, -torch.ones_like(sx), sx)
            sx = torch.where(hit_x_lo, torch.ones_like(sx), sx)
            self._lateral_push_sign_x = sx
            sy = self._lateral_push_sign_y
            sy = torch.where(hit_y_hi, -torch.ones_like(sy), sy)
            sy = torch.where(hit_y_lo, torch.ones_like(sy), sy)
            self._lateral_push_sign_y = sy

        pos_env_new = torch.stack([px_new, py_new, pz], dim=-1)
        pos_w_new = pos_env_new + self.scene.env_origins
        root = torch.cat([pos_w_new, quat_w], dim=-1)
        self.contact_object.write_root_pose_to_sim(root, env_ids=None)
        root_vel = torch.cat([lin_w, ang_w], dim=-1)
        self.contact_object.write_root_velocity_to_sim(root_vel, env_ids=None)

    def _setup_task_rigid_object(self) -> None:
        self.contact_object = RigidObject(self.cfg.contact_object_cfg)
        self.scene.rigid_objects["contact_object"] = self.contact_object

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.to(self.device), -1.0, 1.0)

    def _apply_action(self) -> None:
        # 每步物理后已 scene.update：先解析虚拟墙（夹紧 + 速度反射 + 力符号），再施恒力。
        if self.cfg.enforce_object_xy_walls:
            self._resolve_xy_walls()
        elif self.cfg.flip_lateral_push_at_xy_bounds:
            self._flip_lateral_push_signs_from_position_only()

        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        if self.cfg.enable_lateral_push:
            mask = self.episode_length_buf >= int(self.cfg.push_start_env_steps)
            fx0 = float(self.cfg.lateral_push_force_n)
            fy0 = float(self.cfg.lateral_push_force_y_n)
            fx = self._lateral_push_sign_x * fx0
            fy = self._lateral_push_sign_y * fy0
            forces[mask, 0, 0] = fx[mask]
            forces[mask, 0, 1] = fy[mask]
        self.contact_object.set_external_force_and_torque(
            forces,
            torch.zeros_like(forces),
            env_ids=None,
            is_global=True,
        )
        self._applied_force_world.copy_(forces.squeeze(1))

    def _get_observations(self) -> dict:
        self._update_tactile_data()

        pos_w = self.contact_object.data.root_pos_w
        object_pos_env = pos_w - self.scene.env_origins

        obs_list = [object_pos_env]
        if self._tactile_normal_force is not None:
            if self.cfg.use_full_tactile_obs:
                obs_list.append(self._tactile_normal_force)
                obs_list.append(self._tactile_shear_force)
            else:
                obs_list.append(self._tactile_normal_mean)
                obs_list.append(self._tactile_shear_mean)
        else:
            if self.cfg.use_full_tactile_obs:
                obs_list.append(torch.zeros(self.num_envs, TACTILE_NORMAL_DIM, device=self.device))
                obs_list.append(torch.zeros(self.num_envs, TACTILE_SHEAR_DIM, device=self.device))
            else:
                obs_list.append(torch.zeros(self.num_envs, 3, device=self.device))

        obs = torch.cat(obs_list, dim=-1)

        d = self.contact_object.data
        lin_w = getattr(d, "root_lin_vel_w", d.root_link_lin_vel_w)
        speed_xy = torch.norm(lin_w[:, :2], dim=-1)
        n_mean = (
            self._tactile_normal_mean.squeeze(-1)
            if self._tactile_normal_mean is not None
            else torch.zeros(self.num_envs, device=self.device)
        )
        in_contact = n_mean > float(self.cfg.contact_normal_mean_threshold)
        is_sliding = (speed_xy > float(self.cfg.slip_speed_threshold_m_s)) & in_contact

        g = float(self.cfg.gravity_m_s2)
        n_load = float(self._object_mass_kg) * g

        if "log" not in self.extras:
            self.extras["log"] = {}
        log: dict = {
            "gt_is_sliding": is_sliding.float(),
            "gt_mu_static": self._gt_mu_s,
            "gt_mu_dynamic": self._gt_mu_d,
            "gt_normal_load_n": torch.full((self.num_envs,), n_load, device=self.device),
        }
        if self._tactile_shear_mean is not None:
            sh = self._tactile_shear_mean
            log["tactile_shear_mean"] = sh
            log["tactile_shear_mean_mag"] = torch.linalg.norm(sh, dim=-1)
        if self._tactile_normal_mean is not None:
            log["tactile_normal_mean"] = self._tactile_normal_mean.squeeze(-1)
        if self.cfg.enable_lateral_push and (
            self.cfg.flip_lateral_push_at_xy_bounds or self.cfg.enforce_object_xy_walls
        ):
            log["lateral_push_sign_x"] = self._lateral_push_sign_x
            log["lateral_push_sign_y"] = self._lateral_push_sign_y

        if self.cfg.terminate_on_object_drop:
            pz = object_pos_env[:, 2]
            z_thr = self._object_nominal_z_env - float(self.cfg.object_drop_height_m)
            log["gt_object_dropped"] = (pz < z_thr).float()

        n_env = self.num_envs
        push_start = int(self.cfg.push_start_env_steps)
        if self.cfg.enable_lateral_push:
            push_active = self.episode_length_buf >= push_start
            if push_active.any():
                app_mean = self._applied_force_world[push_active].mean(dim=0)
            else:
                app_mean = torch.zeros(3, device=self.device)
        else:
            app_mean = self._applied_force_world.mean(dim=0)
        app_mean_all = self._applied_force_world.mean(dim=0)
        log["applied_force_mean_x"] = app_mean[0].expand(n_env)
        log["applied_force_mean_y"] = app_mean[1].expand(n_env)
        log["applied_force_mean_z"] = app_mean[2].expand(n_env)
        log["applied_force_mean_all_envs_x"] = app_mean_all[0].expand(n_env)
        log["applied_force_mean_all_envs_y"] = app_mean_all[1].expand(n_env)
        log["applied_force_mean_all_envs_z"] = app_mean_all[2].expand(n_env)

        if self._tactile_normal_mean is not None and self._tactile_shear_mean is not None:
            tn = self._tactile_normal_mean.squeeze(-1)
            sx = self._tactile_shear_mean[:, 0]
            sy = self._tactile_shear_mean[:, 1]
            tactile_mean = torch.stack([tn.mean(), sx.mean(), sy.mean()], dim=0)
            log["tactile_patch_mean_normal"] = tactile_mean[0].expand(n_env)
            log["tactile_patch_mean_shear_x"] = tactile_mean[1].expand(n_env)
            log["tactile_patch_mean_shear_y"] = tactile_mean[2].expand(n_env)
        else:
            tactile_mean = torch.zeros(3, device=self.device)

        self.extras["log"].update(log)

        iv = int(self.cfg.print_xyz_force_mean_interval)
        if iv > 0 and (int(self.common_step_counter) % iv == 0):
            t = float(self.common_step_counter) * float(self.step_dt)
            ax, ay, az = [float(x) for x in app_mean.tolist()]
            aax, aay, aaz = [float(x) for x in app_mean_all.tolist()]
            tx, ty, tz = [float(x) for x in tactile_mean.tolist()]
            print(
                "[friction_pretrain] "
                f"env_step={int(self.common_step_counter):6d} t={t:8.4f}s | "
                f"applied(push_active_mean)[N]=({ax:9.5f},{ay:9.5f},{az:9.5f}) | "
                f"applied(all_env_mean)[N]=({aax:9.5f},{aay:9.5f},{aaz:9.5f}) | "
                f"tacsl_patch_mean(n,sx,sy)=({tx:9.5f},{ty:9.5f},{tz:9.5f})",
                flush=True,
            )

        self._update_xyz_force_live_plot(app_mean, tactile_mean)

        return {"policy": obs}

    def _update_xyz_force_live_plot(
        self,
        app_mean: torch.Tensor,
        tactile_mean: torch.Tensor,
    ) -> None:
        if self._plot_xyz_force_live_disabled or not self.cfg.plot_xyz_force_live:
            return
        if self._force_plot_t is None:
            return
        uiv = max(1, int(self.cfg.plot_xyz_force_live_update_interval))
        if int(self.common_step_counter) % uiv != 0:
            return

        src = str(self.cfg.plot_xyz_force_live_source).lower().strip()
        if src == "applied":
            fvec = app_mean
            title_src = "applied external (push-active env mean, N)"
            ylabel = "Force (N)"
            labels = ("Fx", "Fy", "Fz")
        else:
            # default tactile; legacy "contact" 映射为触觉
            fvec = tactile_mean
            title_src = "TacSL patch mean (same tensors as FF; N, Sx, Sy)"
            ylabel = "TacSL (a.u.)"
            labels = ("patch_normal", "patch_shear_x", "patch_shear_y")

        t = float(self.common_step_counter) * float(self.step_dt)
        fx, fy, fz = (float(x) for x in fvec.detach().cpu().flatten()[:3].tolist())

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._plot_xyz_force_live_disabled = True
            return

        if self._force_plot_fig is None:
            try:
                plt.ion()
                self._force_plot_fig, self._force_plot_ax = plt.subplots(figsize=(8.0, 4.0), num="friction_pretrain F_xyz(t)")
                axp = self._force_plot_ax
                (ln0,) = axp.plot([], [], "b-", lw=1.2, label=labels[0])
                (ln1,) = axp.plot([], [], "g-", lw=1.2, label=labels[1])
                (ln2,) = axp.plot([], [], "r-", lw=1.2, label=labels[2])
                self._force_plot_lines = [ln0, ln1, ln2]
                axp.set_xlabel("t (s)")
                axp.set_ylabel(ylabel)
                axp.set_title(f"Mean force xyz vs time ({title_src})")
                axp.legend(loc="upper right", fontsize=9)
                axp.grid(True, alpha=0.35)
                self._force_plot_fig.tight_layout()
                self._force_plot_fig.show()
            except Exception:
                self._plot_xyz_force_live_disabled = True
                return

        self._force_plot_t.append(t)
        self._force_plot_fx.append(fx)
        self._force_plot_fy.append(fy)
        self._force_plot_fz.append(fz)

        tt = list(self._force_plot_t)
        self._force_plot_lines[0].set_data(tt, list(self._force_plot_fx))
        self._force_plot_lines[1].set_data(tt, list(self._force_plot_fy))
        self._force_plot_lines[2].set_data(tt, list(self._force_plot_fz))
        assert self._force_plot_ax is not None
        self._force_plot_ax.relim()
        self._force_plot_ax.autoscale_view()
        try:
            self._force_plot_fig.canvas.draw_idle()
            self._force_plot_fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            self._plot_xyz_force_live_disabled = True

    def close(self) -> None:
        if self._force_plot_fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._force_plot_fig)
            except Exception:
                pass
            self._force_plot_fig = None
            self._force_plot_ax = None
            self._force_plot_lines = None
        super().close()

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.terminate_on_object_drop:
            pos_env = self.contact_object.data.root_pos_w - self.scene.env_origins
            z_thr = self._object_nominal_z_env - float(self.cfg.object_drop_height_m)
            terminated = pos_env[:, 2] < z_thr
        return terminated, time_out

    def _set_object_mass_uniform(self, env_ids: torch.Tensor, idx_cpu: torch.Tensor, env_ids_cpu: torch.Tensor) -> None:
        m = float(self._object_mass_kg)
        m_t = torch.full((env_ids.numel(), 1), m, device=self.device, dtype=torch.float32)
        m_cpu = m_t.cpu()

        masses = self.contact_object.root_physx_view.get_masses().clone()
        masses[idx_cpu] = m_cpu
        self.contact_object.root_physx_view.set_masses(masses, env_ids_cpu)

        dm = self.contact_object.data.default_mass[idx_cpu].clamp(min=1e-9)
        ratio_cpu = m_cpu / dm
        inertias = self.contact_object.root_physx_view.get_inertias().clone()
        di = self.contact_object.data.default_inertia[idx_cpu].cpu()
        inertias[idx_cpu] = di * ratio_cpu
        self.contact_object.root_physx_view.set_inertias(inertias, env_ids_cpu)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.contact_object._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        if env_ids.numel() == 0:
            return

        env_ids_cpu = env_ids.to(torch.int32).cpu()
        idx_cpu = env_ids.cpu()

        self._set_object_mass_uniform(env_ids, idx_cpu, env_ids_cpu)

        mu_s = sample_uniform(
            float(self.cfg.static_friction_range[0]),
            float(self.cfg.static_friction_range[1]),
            (env_ids.numel(), 1),
            device=self.device,
        ).squeeze(-1)
        scale = float(self.cfg.dynamic_friction_scale)
        mu_d = torch.clamp(mu_s * scale, max=mu_s)

        self._gt_mu_s[env_ids] = mu_s
        self._gt_mu_d[env_ids] = mu_d
        self._lateral_push_sign_x[env_ids] = 1.0
        self._lateral_push_sign_y[env_ids] = 1.0

        mat = self.contact_object.root_physx_view.get_material_properties().clone()
        n = env_ids.numel()
        mu_s_cpu = mu_s.cpu()
        mu_d_cpu = mu_d.cpu()
        if mat.dim() == 2:
            mat[idx_cpu, 0] = mu_s_cpu
            mat[idx_cpu, 1] = mu_d_cpu
            mat[idx_cpu, 2] = 0.0
        else:
            n_shapes = mat.shape[1]
            mat[idx_cpu, :, 0] = mu_s_cpu.unsqueeze(-1).expand(n, n_shapes)
            mat[idx_cpu, :, 1] = mu_d_cpu.unsqueeze(-1).expand(n, n_shapes)
            mat[idx_cpu, :, 2] = 0.0
        self.contact_object.root_physx_view.set_material_properties(mat, env_ids_cpu)

        ox = sample_uniform(
            float(self.cfg.object_reset_xy_range[0]),
            float(self.cfg.object_reset_xy_range[1]),
            (env_ids.numel(), 1),
            device=self.device,
        )
        oy = sample_uniform(
            float(self.cfg.object_reset_xy_range[0]),
            float(self.cfg.object_reset_xy_range[1]),
            (env_ids.numel(), 1),
            device=self.device,
        )
        z0 = float(self.cfg.contact_object_cfg.init_state.pos[2])
        pos_env = torch.cat([ox, oy, torch.full_like(ox, z0)], dim=1)
        pos_w = pos_env + self.scene.env_origins[env_ids]

        root = torch.zeros(env_ids.numel(), 7, device=self.device)
        root[:, :3] = pos_w
        root[:, 3] = 1.0
        self.contact_object.write_root_pose_to_sim(root, env_ids)
        self.contact_object.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=self.device), env_ids)

        self.contact_object.set_external_force_and_torque(
            torch.zeros(self.num_envs, 1, 3, device=self.device),
            torch.zeros(self.num_envs, 1, 3, device=self.device),
            env_ids=None,
            is_global=True,
        )
