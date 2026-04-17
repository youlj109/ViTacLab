from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_from_angle_axis, sample_uniform, saturate

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env import (
    TACTILE_SENSOR_NAMES,
    TACTILE_NORMAL_DIM,
    TACTILE_SHEAR_DIM,
    UR10eShadowHandPickupEnv,
    _unscale,
)

from .blind_classification_env_cfg import UR10eShadowHandBlindClassificationEnvCfg


@torch.jit.script
def _scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


class UR10eShadowHandBlindClassificationEnv(UR10eShadowHandPickupEnv):
    """Blind bin + random manipuland shape; reward when policy logits (last K actions) match the label."""

    cfg: UR10eShadowHandBlindClassificationEnvCfg

    def __init__(self, cfg: UR10eShadowHandBlindClassificationEnvCfg, render_mode: str | None = None, **kwargs):
        nc = int(cfg.num_shape_classes)
        base_obs_dim = 30 + 30 + 3 + 4 + 30 + 30
        tactile_dim = (TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM) if cfg.use_full_tactile_obs else (5 * (1 + 2))
        cfg.observation_space = base_obs_dim + tactile_dim
        cfg.action_space = 30 + nc

        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.observation_space = int(base_obs_dim + tactile_dim)
        self._robot_action_dim = 30
        self._class_action_dim = nc
        self.num_actions = 30 + nc
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._action_rate = torch.zeros((self.num_envs, self._robot_action_dim), device=self.device)

        self.shape_labels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _setup_task_scene(self) -> None:
        spawn = self.cfg.trash_can_cfg.spawn.replace(scale=self.cfg.trash_can_scale)
        tc_cfg = self.cfg.trash_can_cfg.replace(spawn=spawn)
        self.trash_can = RigidObject(tc_cfg)
        self.scene.rigid_objects["trash_can"] = self.trash_can

        self.object_cube = RigidObject(self.cfg.object_cube_cfg)
        self.object_sphere = RigidObject(self.cfg.object_sphere_cfg)
        self.object_cone = RigidObject(self.cfg.object_cone_cfg)
        self._shape_objects = [self.object_cube, self.object_sphere, self.object_cone]
        self.scene.rigid_objects["object_cube"] = self.object_cube
        self.scene.rigid_objects["object_sphere"] = self.object_sphere
        self.scene.rigid_objects["object_cone"] = self.object_cone
        self.object = self.object_cube

        if TACTILE_SENSOR_NAMES[0] in self.scene.sensors and VisuoTactileSensorData is not None:
            first = self.scene[TACTILE_SENSOR_NAMES[0]]
            sz = first.cfg.tactile_array_size
            self._tactile_array_total = sz[0] * sz[1]
            self._num_tactile_sensors = len(TACTILE_SENSOR_NAMES)
            self._tactile_normal_force = torch.zeros(
                (self.num_envs, self._num_tactile_sensors * self._tactile_array_total),
                device=self.device,
            )
            self._tactile_shear_force = torch.zeros(
                (self.num_envs, self._num_tactile_sensors * self._tactile_array_total * 2),
                device=self.device,
            )
            self._tactile_normal_mean = torch.zeros((self.num_envs, self._num_tactile_sensors), device=self.device)
            self._tactile_shear_mean = torch.zeros((self.num_envs, self._num_tactile_sensors * 2), device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions[:, : self._robot_action_dim] = self.actions[:, : self._robot_action_dim]
        a = actions.to(device=self.device)
        # Teleop/debug scripts may still provide only the 30 robot-control dims.
        # For compatibility, pad missing class-logit dims with zeros.
        if a.shape[1] < self.num_actions:
            pad = torch.zeros((a.shape[0], self.num_actions - a.shape[1]), device=self.device, dtype=a.dtype)
            a = torch.cat([a, pad], dim=1)
        elif a.shape[1] > self.num_actions:
            a = a[:, : self.num_actions]
        self.actions = torch.clamp(a, -1.0, 1.0)
        self._action_rate[:] = (
            self.actions[:, : self._robot_action_dim] - self._prev_actions[:, : self._robot_action_dim]
        )

    def _apply_action(self) -> None:
        joint_ids = self.actuated_dof_indices
        a = self.actions[:, : self._robot_action_dim]
        self.cur_targets[:, joint_ids] = _scale(
            a,
            self.robot_dof_lower_limits[:, joint_ids],
            self.robot_dof_upper_limits[:, joint_ids],
        )
        self.cur_targets[:, joint_ids] = (
            self.cfg.act_moving_average * self.cur_targets[:, joint_ids]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, joint_ids]
        )
        self.cur_targets[:, joint_ids] = saturate(
            self.cur_targets[:, joint_ids],
            self.robot_dof_lower_limits[:, joint_ids],
            self.robot_dof_upper_limits[:, joint_ids],
        )
        self.prev_targets[:, joint_ids] = self.cur_targets[:, joint_ids]
        self.robot.set_joint_position_target(self.cur_targets[:, joint_ids], joint_ids=joint_ids)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        obs_elems = [
            _unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[
                :, self.actuated_dof_indices
            ],
            self.cfg.vel_obs_scale * self.robot_dof_vel[:, self.actuated_dof_indices],
            self.object_pos,
            self.object_rot,
            self.actions[:, : self._robot_action_dim],
            self._action_rate,
        ]

        if self._tactile_normal_force is not None:
            self._update_tactile_data()
            if self.cfg.use_full_tactile_obs:
                obs_elems.append(self._tactile_normal_force)
                obs_elems.append(self._tactile_shear_force)
            else:
                obs_elems.append(self._tactile_normal_mean)
                obs_elems.append(self._tactile_shear_mean)

        obs = torch.cat(obs_elems, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        logits = self.actions[:, self._robot_action_dim : self._robot_action_dim + self._class_action_dim]
        pred = torch.argmax(logits, dim=-1)
        match = pred == self.shape_labels
        cls_rew = match.float() * float(self.cfg.classification_reward_weight)

        action_l2 = torch.sum(self.actions[:, : self._robot_action_dim] ** 2, dim=-1) * self.cfg.action_l2_weight
        action_rate_l2 = torch.sum(self._action_rate**2, dim=-1) * self.cfg.action_rate_l2_weight
        reward = cls_rew + action_l2 + action_rate_l2

        success_mask = match
        self.successes = torch.where(success_mask, torch.ones_like(self.successes), self.successes)
        self._success_streak = torch.where(success_mask, self._success_streak + 1, torch.zeros_like(self._success_streak))
        sm = self._success_streak.float().mean()
        alpha = float(self.cfg.success_ema_alpha)
        self.consecutive_successes.mul_(1.0 - alpha).add_(sm * alpha)

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["classification_accuracy"] = match.float().mean()
        self.extras["log"]["episode_success_rate"] = float(self._episode_success_rate_ema)
        self.extras["log"]["episode_success_rate_all_time"] = self.get_episode_success_rate()

        return reward

    def _compute_intermediate_values(self) -> None:
        self.robot_dof_pos = self.robot.data.joint_pos
        self.robot_dof_vel = self.robot.data.joint_vel

        self.object_pos.zero_()
        self.object_rot.zero_()
        for s in range(len(self._shape_objects)):
            mask = self.shape_labels == s
            if not mask.any():
                continue
            obj = self._shape_objects[s]
            self.object_pos[mask] = obj.data.root_pos_w[mask] - self.scene.env_origins[mask]
            self.object_rot[mask] = obj.data.root_quat_w[mask]

        self._goal_time_left_s -= float(self.step_dt)
        resample_ids = torch.nonzero(self._goal_time_left_s <= 0.0, as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._resample_goals(resample_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            e = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            e = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if e.numel() > 0:
            self._accumulate_episode_success_stats(e)

        if env_ids is None:
            super(UR10eShadowHandPickupEnv, self)._reset_idx(slice(None))
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super(UR10eShadowHandPickupEnv, self)._reset_idx(env_ids)

        nc = self._class_action_dim
        self.shape_labels[env_ids] = torch.randint(0, nc, (len(env_ids),), device=self.device, dtype=torch.long)

        goal_x = sample_uniform(
            self.cfg.goal_pos_x_range[0], self.cfg.goal_pos_x_range[1], (len(env_ids), 1), device=self.device
        )
        goal_y = sample_uniform(
            self.cfg.goal_pos_y_range[0], self.cfg.goal_pos_y_range[1], (len(env_ids), 1), device=self.device
        )
        goal_z = sample_uniform(
            self.cfg.goal_pos_z_range[0], self.cfg.goal_pos_z_range[1], (len(env_ids), 1), device=self.device
        )
        self.goal_object_pos[env_ids] = torch.cat([goal_x, goal_y, goal_z], dim=1)
        tmin, tmax = self.cfg.goal_resample_time_range_s
        self._goal_time_left_s[env_ids] = sample_uniform(float(tmin), float(tmax), (len(env_ids),), device=self.device)

        base_pos = torch.tensor(self.cfg.object_cube_cfg.init_state.pos, device=self.device, dtype=torch.float).view(1, 3)
        dx = sample_uniform(
            self.cfg.object_reset_pos_x_range[0],
            self.cfg.object_reset_pos_x_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        dy = sample_uniform(
            self.cfg.object_reset_pos_y_range[0],
            self.cfg.object_reset_pos_y_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        dz = sample_uniform(
            self.cfg.object_reset_pos_z_range[0],
            self.cfg.object_reset_pos_z_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        active_pos = base_pos + torch.cat([dx, dy, dz], dim=1) + self.scene.env_origins[env_ids]
        yaw = sample_uniform(
            self.cfg.object_reset_yaw_range[0],
            self.cfg.object_reset_yaw_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(-1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        quat_w = quat_from_angle_axis(yaw, z_axis)

        park = torch.tensor(self.cfg.park_object_pos, device=self.device, dtype=torch.float).view(1, 3)
        park_pos = park + self.scene.env_origins[env_ids]
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        n = len(env_ids)
        for s in range(len(self._shape_objects)):
            obj = self._shape_objects[s]
            st_all = obj.data.default_root_state.clone()[env_ids]
            for i in range(n):
                eid = int(env_ids[i].item())
                if self.shape_labels[eid] == s:
                    st_all[i, 0:3] = active_pos[i]
                    st_all[i, 3:7] = quat_w[i]
                else:
                    st_all[i, 0:3] = park_pos[i]
                    st_all[i, 3:7] = ident
            st_all[:, 7:] = 0.0
            obj.write_root_pose_to_sim(st_all[:, :7], env_ids)
            obj.write_root_velocity_to_sim(st_all[:, 7:], env_ids)

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
        self._compute_intermediate_values()

    def _get_record(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor | dict]:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._compute_intermediate_values()

        robot = self.robot
        robot_state = {
            "root_pos_w": robot.data.root_pos_w[ids].clone(),
            "root_quat_w": robot.data.root_quat_w[ids].clone(),
            "root_lin_vel_w": robot.data.root_lin_vel_w[ids].clone(),
            "root_ang_vel_w": robot.data.root_ang_vel_w[ids].clone(),
            "joint_pos": robot.data.joint_pos[ids].clone(),
            "joint_vel": robot.data.joint_vel[ids].clone(),
            "joint_pos_target": self.cur_targets[ids].clone(),
            "joint_pos_target_prev": self.prev_targets[ids].clone(),
            "actuated_dof_indices": torch.tensor(self.actuated_dof_indices, device=self.device, dtype=torch.long),
            "actions": self.actions[ids].clone(),
            "action_rate": self._action_rate[ids].clone(),
        }

        n = ids.numel()
        root_pos_w = torch.zeros((n, 3), device=self.device)
        root_quat_w = torch.zeros((n, 4), device=self.device)
        root_lin_vel_w = torch.zeros((n, 3), device=self.device)
        root_ang_vel_w = torch.zeros((n, 3), device=self.device)
        root_pos_env = torch.zeros((n, 3), device=self.device)
        for i in range(n):
            eid = int(ids[i].item())
            s = int(self.shape_labels[eid].item())
            o = self._shape_objects[s]
            root_pos_w[i] = o.data.root_pos_w[eid]
            root_quat_w[i] = o.data.root_quat_w[eid]
            root_lin_vel_w[i] = o.data.root_lin_vel_w[eid]
            root_ang_vel_w[i] = o.data.root_ang_vel_w[eid]
            root_pos_env[i] = o.data.root_pos_w[eid] - self.scene.env_origins[eid]

        object_state = {
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "root_lin_vel_w": root_lin_vel_w,
            "root_ang_vel_w": root_ang_vel_w,
            "root_pos_env": root_pos_env,
        }

        task_state = {
            "goal_pos_env": self.goal_object_pos[ids].clone(),
            "goal_pos_w": (self.goal_object_pos[ids] + self.scene.env_origins[ids]).clone(),
            "goal_quat_w": self.goal_object_rot[ids].clone(),
            "goal_time_left_s": self._goal_time_left_s[ids].clone(),
            "shape_labels": self.shape_labels[ids].clone(),
            "successes": self.successes[ids].clone(),
            "success_streak": self._success_streak[ids].clone(),
            "consecutive_successes": self.consecutive_successes.clone(),
            "episode_length_buf": self.episode_length_buf[ids].clone(),
        }

        sensors: dict[str, dict] = {}
        tactile: dict[str, dict[str, torch.Tensor]] = {}
        for name in TACTILE_SENSOR_NAMES:
            if name not in self.scene.sensors:
                continue
            data = self.scene[name].data
            entry: dict[str, torch.Tensor] = {}
            rgb = getattr(data, "tactile_rgb_image", None)
            if rgb is not None:
                entry["rgb"] = rgb[ids].clone()
            nf = getattr(data, "tactile_normal_force", None)
            if nf is not None:
                entry["normal_force"] = nf[ids].clone()
            sf = getattr(data, "tactile_shear_force", None)
            if sf is not None:
                entry["shear_force"] = sf[ids].clone()
            if entry:
                tactile[name] = entry
        if tactile:
            sensors["tactile"] = tactile

        if "third_person_camera" in self.scene.sensors:
            cam = self.scene["third_person_camera"]
            rgb = None
            try:
                rgb = cam.data.output.get("rgb", None)
            except Exception:
                rgb = None
            if rgb is not None:
                sensors["third_person_camera"] = {"rgb": rgb[ids].clone()}

        return {
            "env_ids": ids.clone(),
            "env_origins": self.scene.env_origins[ids].clone(),
            "robot": robot_state,
            "object": object_state,
            "task": task_state,
            "sensors": sensors,
        }
