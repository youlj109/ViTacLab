from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_from_angle_axis, sample_uniform, saturate

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    UR10eShadowHandDirectBaseEnv,
    _tacsl_to_batched_flat,
    _unscale,
)

from .hand_pickup_env_cfg_v1 import UR10eShadowHandPickupEnvCfgV1

if TYPE_CHECKING:
    from isaaclab.envs.ui import ViewerCfg


TACTILE_POINTS_PER_SENSOR = 20 * 25
TACTILE_NORMAL_DIM = 5 * TACTILE_POINTS_PER_SENSOR
TACTILE_SHEAR_DIM = 5 * TACTILE_POINTS_PER_SENSOR * 2


class UR10eShadowHandPickupEnvV1(UR10eShadowHandDirectBaseEnv):
    """Pickup v1: task logic is pickup-specific; tactile/camera/record follow example ``inhand_manipulation_env``."""

    cfg: UR10eShadowHandPickupEnvCfgV1

    def __init__(self, cfg: UR10eShadowHandPickupEnvCfgV1, render_mode: str | None = None, **kwargs):
        base_obs_dim = 30 + 30 + 3 + 4 + 3 + 3 + 30 + 30
        tactile_dim = (TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM) if cfg.use_full_tactile_obs else (5 * (1 + 2))
        cfg.observation_space = base_obs_dim + tactile_dim
        super().__init__(cfg, render_mode, **kwargs)

        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self._object_init_z = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.goal_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_object_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

        self._goal_time_left_s = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        self._prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._action_rate = torch.zeros_like(self._prev_actions)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self._episode_success_count = 0
        self._episode_total_count = 0
        self._episode_success_rate_ema: float = 0.0

    def _setup_task_scene(self) -> None:
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions[:] = self.actions
        super()._pre_physics_step(actions)
        self._action_rate[:] = self.actions - self._prev_actions

    def _apply_action(self) -> None:
        """Match Forge behavior: skip RL action mapping in non-RL control mode."""
        if not self._use_rl_control:
            return
        super()._apply_action()

    def apply_joint_targets(self, joint_pos: torch.Tensor):
        """
        Set robot joint position targets directly (for IL/DP policy play).
        joint_pos: (num_envs, num_robot_dofs).
        """
        n_dofs = self.robot.data.joint_pos.shape[1]
        n_cmd = joint_pos.shape[1]
        print(f"joint_pos: {joint_pos.shape}")
        joint_pos = joint_pos.to(device=self.device, dtype=self.robot.data.joint_pos.dtype)
        assert n_cmd == n_dofs, "Joint position command dimension mismatch"
        self.robot.set_joint_position_target(joint_pos)

    def _get_observations(self) -> dict:
        # ``_compute_intermediate_values`` ends with ``_sync_tacsl_tactile_and_third_person_visual`` (inhand pattern).
        self._compute_intermediate_values()
        obs_elems = [
            _unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[
                :, self.actuated_dof_indices
            ],
            self.cfg.vel_obs_scale * self.robot_dof_vel[:, self.actuated_dof_indices],
            self.object_pos,
            self.object_rot,
            self.goal_object_pos,
            (self.goal_object_pos - self.object_pos),
            self.actions,
            self._action_rate,
        ]
        if self._ur10e_stacked_n > 0:
            n = self._ur10e_stacked_n
            p = self._ur10e_stacked_array_total
            tn = _tacsl_to_batched_flat(self.tactile_normal_force, self.num_envs)
            ts = _tacsl_to_batched_flat(self.tactile_shear_force, self.num_envs)
            if self.cfg.use_full_tactile_obs:
                obs_elems.append(tn)
                obs_elems.append(ts)
            else:
                nf = tn.view(self.num_envs, n, p).mean(dim=-1)
                sf = ts.view(self.num_envs, n, p, 2).mean(dim=2).reshape(self.num_envs, n * 2)
                obs_elems.append(nf)
                obs_elems.append(sf)
        obs = torch.cat(obs_elems, dim=-1)
        record_dict = self._build_pickup_style_record_dict(
            joint_pos=self.robot.data.joint_pos,
            tactile_sensor_names=tuple(getattr(self, "_ur10e_stacked_tacsl_names", ())),
            tactile_sensor_count=int(self._ur10e_stacked_n),
            tactile_normal_force=self.tactile_normal_force if self._ur10e_stacked_n > 0 else None,
            tactile_shear_force=self.tactile_shear_force if self._ur10e_stacked_n > 0 else None,
            tactile_rgb_image=self.tactile_rgb_image if self._ur10e_stacked_n > 0 else None,
            tactile_array_size=tuple(int(v) for v in self.tactile_array_size) if self._ur10e_stacked_n > 0 else None,
            tactile_image_hw=(int(self.tactile_image_height), int(self.tactile_image_width))
            if self._ur10e_stacked_n > 0
            else None,
        )
        return {"policy": obs, "record": record_dict}

    def _get_rewards(self) -> torch.Tensor:
        pos_err = self.goal_object_pos - self.object_pos
        pos_dist = torch.norm(pos_err, p=2, dim=-1)
        pos_rew = (1.0 - torch.tanh(pos_dist / (self.cfg.pos_tracking_std + 1e-6))) * self.cfg.pos_tracking_weight
        action_l2 = torch.sum(self.actions**2, dim=-1) * self.cfg.action_l2_weight
        action_rate_l2 = torch.sum(self._action_rate**2, dim=-1) * self.cfg.action_rate_l2_weight
        reward = pos_rew + action_l2 + action_rate_l2
        success_mode = str(getattr(self.cfg, "success_mode", "lift_and_goal_z")).lower()
        if success_mode == "lift_and_goal_z":
            z_obj = self.object_pos[:, 2]
            z_goal = self.goal_object_pos[:, 2]
            is_lifted = (z_obj - self._object_init_z) >= float(self.cfg.grasp_lift_min_dz)
            reached_goal_z = torch.abs(z_obj - z_goal) <= float(self.cfg.goal_z_tol)
            success_mask = is_lifted & reached_goal_z
        else:
            # legacy xyz tolerance mode
            height_err = torch.abs(pos_err[:, 2])
            xy_dist = torch.norm(pos_err[:, 0:2], p=2, dim=-1)
            success_mask = (height_err <= self.cfg.success_height_tol) & (xy_dist <= self.cfg.success_pos_tol)
        reward = torch.where(success_mask, reward + self.cfg.success_weight, reward)
        self.successes = torch.where(success_mask, torch.ones_like(self.successes), self.successes)
        self._success_streak = torch.where(success_mask, self._success_streak + 1, torch.zeros_like(self._success_streak))
        alpha = float(self.cfg.success_ema_alpha)
        self.consecutive_successes.mul_(1.0 - alpha).add_(self._success_streak.float().mean() * alpha)
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["curr_success_per_env"] = success_mask
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fall_mask = self.object_pos[:, 2] < self.cfg.fall_height
        oob = (
            (self.object_pos[:, 0] < self.cfg.out_of_bound_x[0])
            | (self.object_pos[:, 0] > self.cfg.out_of_bound_x[1])
            | (self.object_pos[:, 1] < self.cfg.out_of_bound_y[0])
            | (self.object_pos[:, 1] > self.cfg.out_of_bound_y[1])
            | (self.object_pos[:, 2] < self.cfg.out_of_bound_z[0])
            | (self.object_pos[:, 2] > self.cfg.out_of_bound_z[1])
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = fall_mask | oob | time_out
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            super()._reset_idx(slice(None))
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super()._reset_idx(env_ids)
        goal_x = sample_uniform(self.cfg.goal_pos_x_range[0], self.cfg.goal_pos_x_range[1], (len(env_ids), 1), device=self.device)
        goal_y = sample_uniform(self.cfg.goal_pos_y_range[0], self.cfg.goal_pos_y_range[1], (len(env_ids), 1), device=self.device)
        goal_z = sample_uniform(self.cfg.goal_pos_z_range[0], self.cfg.goal_pos_z_range[1], (len(env_ids), 1), device=self.device)
        self.goal_object_pos[env_ids] = torch.cat([goal_x, goal_y, goal_z], dim=1)
        tmin, tmax = self.cfg.goal_resample_time_range_s
        self._goal_time_left_s[env_ids] = sample_uniform(float(tmin), float(tmax), (len(env_ids),), device=self.device)

        obj_state = self.object.data.default_root_state.clone()[env_ids]
        base_pos = torch.tensor(self.cfg.object_cfg.init_state.pos, device=self.device, dtype=torch.float).view(1, 3)
        dx = sample_uniform(self.cfg.object_reset_pos_x_range[0], self.cfg.object_reset_pos_x_range[1], (len(env_ids), 1), device=self.device)
        dy = sample_uniform(self.cfg.object_reset_pos_y_range[0], self.cfg.object_reset_pos_y_range[1], (len(env_ids), 1), device=self.device)
        dz = sample_uniform(self.cfg.object_reset_pos_z_range[0], self.cfg.object_reset_pos_z_range[1], (len(env_ids), 1), device=self.device)
        obj_state[:, 0:3] = base_pos + torch.cat([dx, dy, dz], dim=1) + self.scene.env_origins[env_ids]
        self._object_init_z[env_ids] = (base_pos[:, 2] + dz).squeeze(-1)
        yaw = sample_uniform(self.cfg.object_reset_yaw_range[0], self.cfg.object_reset_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(-1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        obj_state[:, 3:7] = quat_from_angle_axis(yaw, z_axis)
        obj_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

        delta_max = self.robot_dof_upper_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]
        dof_pos_noise = sample_uniform(
            self.cfg.robot_reset_dof_pos_offset_range[0],
            self.cfg.robot_reset_dof_pos_offset_range[1],
            (len(env_ids), self.num_robot_dofs),
            device=self.device,
        )
        rand_delta = saturate(dof_pos_noise, delta_min, delta_max)
        dof_pos = self.robot.data.default_joint_pos[env_ids] + rand_delta
        dof_vel = sample_uniform(
            self.cfg.robot_reset_dof_vel_range[0],
            self.cfg.robot_reset_dof_vel_range[1],
            (len(env_ids), self.num_robot_dofs),
            device=self.device,
        )
        self._reset_robot_joints(env_ids, dof_pos=dof_pos, dof_vel=dof_vel)
        self.successes[env_ids] = 0.0
        self._success_streak[env_ids] = 0
        # Inhand: after partial reset, refresh TacSL so ``.data`` stays consistent across envs (uint8 indexing).
        if getattr(self.cfg, "enable_cameras", False):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def _compute_intermediate_values(self):
        self.robot_dof_pos = self.robot.data.joint_pos
        self.robot_dof_vel = self.robot.data.joint_vel
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self._goal_time_left_s -= float(self.step_dt)
        resample_ids = torch.nonzero(self._goal_time_left_s <= 0.0, as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            goal_x = sample_uniform(self.cfg.goal_pos_x_range[0], self.cfg.goal_pos_x_range[1], (len(resample_ids), 1), device=self.device)
            goal_y = sample_uniform(self.cfg.goal_pos_y_range[0], self.cfg.goal_pos_y_range[1], (len(resample_ids), 1), device=self.device)
            goal_z = sample_uniform(self.cfg.goal_pos_z_range[0], self.cfg.goal_pos_z_range[1], (len(resample_ids), 1), device=self.device)
            self.goal_object_pos[resample_ids] = torch.cat([goal_x, goal_y, goal_z], dim=1)
            tmin, tmax = self.cfg.goal_resample_time_range_s
            self._goal_time_left_s[resample_ids] = sample_uniform(float(tmin), float(tmax), (len(resample_ids),), device=self.device)
        goal_pos = self.goal_object_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_object_rot)
        self._sync_tacsl_tactile_and_third_person_visual()