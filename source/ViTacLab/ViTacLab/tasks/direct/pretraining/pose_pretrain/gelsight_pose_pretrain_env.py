# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pose-from-touch pretraining: fixed knob, random orientation; TacSL + optional object position."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from ..gelsight_finger_pretrain_base_env import (
    GelsightFingerPretrainBaseEnv,
    TACTILE_NORMAL_DIM,
    TACTILE_SHEAR_DIM,
)
from ..gelsight_pretrain_obs import pose_pretrain_obs_dim
from .gelsight_pose_pretrain_env_cfg import GelsightFingerPosePretrainEnvCfg


class GelsightFingerPosePretrainEnv(GelsightFingerPretrainBaseEnv):
    """Same object/mass as spawner default; randomize orientation (roll/pitch/yaw) at reset.

    Observations stack optional env-frame object position and TacSL (compact or full patch).
    Ground-truth orientation is exposed in ``extras['log']`` for supervised learning.
    """

    cfg: GelsightFingerPosePretrainEnvCfg

    def __init__(self, cfg: GelsightFingerPosePretrainEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = pose_pretrain_obs_dim(cfg.use_full_tactile_obs, cfg.include_object_position_in_obs)
        super().__init__(cfg, render_mode, **kwargs)

        self._gt_euler_xyz = torch.zeros(self.num_envs, 3, device=self.device)
        self._gt_quat_wxyz = torch.zeros(self.num_envs, 4, device=self.device)

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

        obs_list = []
        if self.cfg.include_object_position_in_obs:
            obs_list.append(object_pos_env)

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

        if "log" not in self.extras:
            self.extras["log"] = {}
        e = self._gt_euler_xyz
        q = self._gt_quat_wxyz
        pi = 3.141592653589793
        self.extras["log"].update(
            {
                "gt_euler_xyz": e,
                "gt_euler_xyz_sin": torch.sin(e),
                "gt_euler_xyz_cos": torch.cos(e),
                "gt_euler_xyz_normalized": e / pi,
                "gt_object_quat_w": q,
            }
        )

        iv = int(self.cfg.print_pose_mean_interval)
        if iv > 0 and (int(self.common_step_counter) % iv == 0):
            er, ep, ey = [float(x) for x in e.mean(dim=0).tolist()]
            print(
                "[pose_pretrain] "
                f"env_step={int(self.common_step_counter):6d} | "
                f"mean_gt_euler_xyz(rad)=({er:7.4f},{ep:7.4f},{ey:7.4f})",
                flush=True,
            )

        return {"policy": obs}

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

        n = env_ids.numel()
        roll = sample_uniform(
            float(self.cfg.euler_roll_range_rad[0]),
            float(self.cfg.euler_roll_range_rad[1]),
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        pitch = sample_uniform(
            float(self.cfg.euler_pitch_range_rad[0]),
            float(self.cfg.euler_pitch_range_rad[1]),
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        yaw = sample_uniform(
            float(self.cfg.euler_yaw_range_rad[0]),
            float(self.cfg.euler_yaw_range_rad[1]),
            (n, 1),
            device=self.device,
        ).squeeze(-1)

        self._gt_euler_xyz[env_ids, 0] = roll
        self._gt_euler_xyz[env_ids, 1] = pitch
        self._gt_euler_xyz[env_ids, 2] = yaw

        quat = quat_from_euler_xyz(roll, pitch, yaw)
        self._gt_quat_wxyz[env_ids] = quat

        ox = sample_uniform(
            float(self.cfg.object_reset_xy_range[0]),
            float(self.cfg.object_reset_xy_range[1]),
            (n, 1),
            device=self.device,
        )
        oy = sample_uniform(
            float(self.cfg.object_reset_xy_range[0]),
            float(self.cfg.object_reset_xy_range[1]),
            (n, 1),
            device=self.device,
        )
        z0 = float(self.cfg.contact_object_cfg.init_state.pos[2])
        pos_env = torch.cat([ox, oy, torch.full_like(ox, z0)], dim=1)
        pos_w = pos_env + self.scene.env_origins[env_ids]

        root = torch.zeros(n, 7, device=self.device)
        root[:, :3] = pos_w
        root[:, 3:7] = quat
        self.contact_object.write_root_pose_to_sim(root, env_ids)
        self.contact_object.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)
