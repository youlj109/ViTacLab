# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mass-from-pressure pretraining environment (GelSight short finger + knob-shaped weight)."""

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
from .gelsight_mass_pretrain_env_cfg import GelsightFingerMassPretrainEnvCfg


class GelsightFingerMassPretrainEnv(GelsightFingerPretrainBaseEnv):
    """Randomize manipuland mass; observations are object pose + TacSL (normal/shear)."""

    cfg: GelsightFingerMassPretrainEnvCfg

    def __init__(self, cfg: GelsightFingerMassPretrainEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = pretrain_obs_dim(cfg.use_full_tactile_obs)
        super().__init__(cfg, render_mode, **kwargs)

        self._gt_mass_kg = torch.zeros(self.num_envs, device=self.device)
        self._mass_min = float(cfg.mass_range_kg[0])
        self._mass_max = float(cfg.mass_range_kg[1])

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

    def _setup_task_rigid_object(self) -> None:
        self.contact_object = RigidObject(self.cfg.contact_object_cfg)
        self.scene.rigid_objects["contact_object"] = self.contact_object

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.to(self.device), -1.0, 1.0)

    def _apply_action(self) -> None:
        return

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

        m = self._gt_mass_kg
        denom = max(self._mass_max - self._mass_min, 1e-9)
        mass_norm = (m - self._mass_min) / denom
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(
            {
                "gt_mass_kg": m,
                "gt_mass_normalized": mass_norm,
                "gt_normal_force_weight_n": m * float(self.cfg.gravity_m_s2),
            }
        )

        n_env = self.num_envs
        if self._tactile_normal_mean is not None and self._tactile_shear_mean is not None:
            tn = self._tactile_normal_mean.squeeze(-1)
            sx = self._tactile_shear_mean[:, 0]
            sy = self._tactile_shear_mean[:, 1]
            tactile_mean = torch.stack([tn.mean(), sx.mean(), sy.mean()], dim=0)
            self.extras["log"]["tactile_patch_mean_normal"] = tactile_mean[0].expand(n_env)
            self.extras["log"]["tactile_patch_mean_shear_x"] = tactile_mean[1].expand(n_env)
            self.extras["log"]["tactile_patch_mean_shear_y"] = tactile_mean[2].expand(n_env)
        else:
            tactile_mean = torch.zeros(3, device=self.device)

        iv = int(self.cfg.print_xyz_force_mean_interval)
        if iv > 0 and (int(self.common_step_counter) % iv == 0):
            t = float(self.common_step_counter) * float(self.step_dt)
            tx, ty, tz = [float(x) for x in tactile_mean.tolist()]
            mg = float(m.mean().item())
            print(
                "[mass_pretrain] "
                f"env_step={int(self.common_step_counter):6d} t={t:8.4f}s | "
                f"gt_mass_mean_kg={mg:8.5f} | "
                f"tacsl_patch_mean(n,sx,sy)=({tx:9.5f},{ty:9.5f},{tz:9.5f})",
                flush=True,
            )

        self._update_mass_tactile_live_plot(tactile_mean)

        return {"policy": obs}

    def _update_mass_tactile_live_plot(self, tactile_mean: torch.Tensor) -> None:
        if self._plot_xyz_force_live_disabled or not self.cfg.plot_xyz_force_live:
            return
        if self._force_plot_t is None:
            return
        uiv = max(1, int(self.cfg.plot_xyz_force_live_update_interval))
        if int(self.common_step_counter) % uiv != 0:
            return

        title_src = "TacSL patch mean (same tensors as FF; N, Sx, Sy)"
        ylabel = "TacSL (a.u.)"
        labels = ("patch_normal", "patch_shear_x", "patch_shear_y")

        t = float(self.common_step_counter) * float(self.step_dt)
        fx, fy, fz = (float(x) for x in tactile_mean.detach().cpu().flatten()[:3].tolist())

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._plot_xyz_force_live_disabled = True
            return

        if self._force_plot_fig is None:
            try:
                plt.ion()
                self._force_plot_fig, self._force_plot_ax = plt.subplots(
                    figsize=(8.0, 4.0), num="mass_pretrain TacSL patch(t)"
                )
                axp = self._force_plot_ax
                (ln0,) = axp.plot([], [], "b-", lw=1.2, label=labels[0])
                (ln1,) = axp.plot([], [], "g-", lw=1.2, label=labels[1])
                (ln2,) = axp.plot([], [], "r-", lw=1.2, label=labels[2])
                self._force_plot_lines = [ln0, ln1, ln2]
                axp.set_xlabel("t (s)")
                axp.set_ylabel(ylabel)
                axp.set_title(f"Mean vs time ({title_src})")
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
        assert self._force_plot_lines is not None
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
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.contact_object._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        if env_ids.numel() == 0:
            return

        m = sample_uniform(
            float(self.cfg.mass_range_kg[0]),
            float(self.cfg.mass_range_kg[1]),
            (env_ids.numel(), 1),
            device=self.device,
        )
        self._gt_mass_kg[env_ids] = m.squeeze(-1)

        # PhysX mass/inertia buffers from ``root_physx_view`` are CPU tensors; keep sampled mass aligned.
        env_ids_cpu = env_ids.to(torch.int32).cpu()
        idx_cpu = env_ids.cpu()
        m_cpu = m.cpu()

        masses = self.contact_object.root_physx_view.get_masses().clone()
        masses[idx_cpu] = m_cpu
        self.contact_object.root_physx_view.set_masses(masses, env_ids_cpu)

        # ``default_mass`` / ``default_inertia`` live on CPU; index with CPU long indices only.
        dm = self.contact_object.data.default_mass[idx_cpu].clamp(min=1e-9)
        ratio_cpu = m_cpu / dm
        inertias = self.contact_object.root_physx_view.get_inertias().clone()
        di = self.contact_object.data.default_inertia[idx_cpu].cpu()
        inertias[idx_cpu] = di * ratio_cpu
        self.contact_object.root_physx_view.set_inertias(inertias, env_ids_cpu)

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
