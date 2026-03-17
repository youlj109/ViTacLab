from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    UR10eShadowHandDirectBaseEnv,
)

from .hand_pickup_env_cfg import UR10eShadowHandPickupEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs.ui import ViewerCfg


TACTILE_SENSOR_NAMES = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)
TACTILE_POINTS_PER_SENSOR = 20 * 25
TACTILE_NORMAL_DIM = 5 * TACTILE_POINTS_PER_SENSOR
TACTILE_SHEAR_DIM = 5 * TACTILE_POINTS_PER_SENSOR * 2


@torch.jit.script
def _scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class UR10eShadowHandPickupEnv(UR10eShadowHandDirectBaseEnv):
    """UR10e + ShadowHand rigid-object pickup environment with tactile sensing."""

    cfg: UR10eShadowHandPickupEnvCfg

    def __init__(self, cfg: UR10eShadowHandPickupEnvCfg, render_mode: str | None = None, **kwargs):
        # Dexsuite-style: compact obs; optional full tactile.
        # [robot_q(30) robot_dq(30) obj_pos(3) obj_quat(4) goal_pos(3) pos_err(3) actions(30) action_rate(30)]
        base_obs_dim = 30 + 30 + 3 + 4 + 3 + 3 + 30 + 30
        tactile_dim = (TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM) if cfg.use_full_tactile_obs else (5 * (1 + 2))
        cfg.observation_space = base_obs_dim + tactile_dim

        super().__init__(cfg, render_mode, **kwargs)

        # object / goal buffers
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_object_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

        # goal resampling timer (per-env)
        self._goal_time_left_s = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

        # action-rate (dexsuite action_rate_l2)
        self._prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._action_rate = torch.zeros_like(self._prev_actions)

        # tactile buffers (match pour env)
        self._tactile_normal_force: torch.Tensor | None = None
        self._tactile_shear_force: torch.Tensor | None = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0

        # tactile summary (per-sensor mean normal, mean shear xy)
        self._tactile_normal_mean: torch.Tensor | None = None
        self._tactile_shear_mean: torch.Tensor | None = None

        # success stats
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # ---------------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------------

    def _setup_task_scene(self) -> None:
        # task object
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        # Init tactile buffers (match pour env)
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
        # keep action-rate before base overwrites self.actions
        self._prev_actions[:] = self.actions
        super()._pre_physics_step(actions)
        self._action_rate[:] = self.actions - self._prev_actions

    # ---------------------------------------------------------------------
    # RL API
    # ---------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        obs_elems = [
            _unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self.actuated_dof_indices],
            self.cfg.vel_obs_scale * self.robot_dof_vel[:, self.actuated_dof_indices],
            self.object_pos,
            self.object_rot,
            self.goal_object_pos,
            (self.goal_object_pos - self.object_pos),
            self.actions,
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

    def _resample_goals(self, env_ids: torch.Tensor) -> None:
        """Sample new goal positions (command-like)."""
        if env_ids.numel() == 0:
            return
        goal_x = sample_uniform(
            self.cfg.goal_pos_x_range[0],
            self.cfg.goal_pos_x_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        goal_y = sample_uniform(
            self.cfg.goal_pos_y_range[0],
            self.cfg.goal_pos_y_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        goal_z = sample_uniform(
            self.cfg.goal_pos_z_range[0],
            self.cfg.goal_pos_z_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        self.goal_object_pos[env_ids] = torch.cat([goal_x, goal_y, goal_z], dim=1)

        # reset per-env timer
        tmin, tmax = self.cfg.goal_resample_time_range_s
        self._goal_time_left_s[env_ids] = sample_uniform(
            float(tmin), float(tmax), (len(env_ids),), device=self.device
        )

    def _update_tactile_data(self) -> None:
        if self._tactile_normal_force is None:
            return

        norm_list, shear_list = [], []
        for name in TACTILE_SENSOR_NAMES:
            if name not in self.scene.sensors:
                continue
            data = self.scene[name].data
            if getattr(data, "tactile_normal_force", None) is not None:
                norm_list.append(data.tactile_normal_force)
            if getattr(data, "tactile_shear_force", None) is not None:
                shear_list.append(data.tactile_shear_force.view(self.num_envs, -1))
        if len(norm_list) == self._num_tactile_sensors and len(shear_list) == self._num_tactile_sensors:
            self._tactile_normal_force = torch.cat(norm_list, dim=1)
            self._tactile_shear_force = torch.cat(shear_list, dim=1)
            # summaries
            if self._tactile_normal_mean is not None and self._tactile_shear_mean is not None:
                n_means = []
                s_means = []
                for name in TACTILE_SENSOR_NAMES:
                    data = self.scene[name].data
                    nf = data.tactile_normal_force  # (num_envs, P)
                    sf = data.tactile_shear_force  # (num_envs, P, 2)
                    n_means.append(nf.mean(dim=1, keepdim=True))
                    s_means.append(sf.mean(dim=1))
                self._tactile_normal_mean = torch.cat(n_means, dim=1)
                self._tactile_shear_mean = torch.cat(s_means, dim=1)

    def _get_rewards(self) -> torch.Tensor:
        # position tracking (tanh shaping like dexsuite)
        pos_err = self.goal_object_pos - self.object_pos
        pos_dist = torch.norm(pos_err, p=2, dim=-1)
        pos_rew = (1.0 - torch.tanh(pos_dist / (self.cfg.pos_tracking_std + 1e-6))) * self.cfg.pos_tracking_weight
        # shaping: distance from robot base / EE region to object
        ee_to_obj = self.object_pos - self.robot_root_pos_env
        ee_dist = torch.norm(ee_to_obj, p=2, dim=-1)
        # plateau: treat distances within saturation radius as equally good
        sat_r = self.cfg.ee_object_saturation_radius
        ee_dist_eff = torch.clamp(ee_dist, min=sat_r)
        ee_rew = (1.0 - torch.tanh(ee_dist_eff / (self.cfg.ee_object_std + 1e-6))) * self.cfg.ee_object_weight

        action_l2 = torch.sum(self.actions**2, dim=-1) * self.cfg.action_l2_weight
        action_rate_l2 = torch.sum(self._action_rate**2, dim=-1) * self.cfg.action_rate_l2_weight

        reward = pos_rew + ee_rew + action_l2 + action_rate_l2

        # success (position-only lift)
        height_err = torch.abs(pos_err[:, 2])
        xy_dist = torch.norm(pos_err[:, 0:2], p=2, dim=-1)
        success_mask = (height_err <= self.cfg.success_height_tol) & (xy_dist <= self.cfg.success_pos_tol)
        reward = torch.where(success_mask, reward + self.cfg.success_weight, reward)
        self.successes = torch.where(success_mask, torch.ones_like(self.successes), self.successes)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminate if object falls below table, out-of-bound, or timeout
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

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # IMPORTANT: Passing `None` into `scene.reset()` uses `slice(None)` inside sensors.
        # This avoids CUDA indexing asserts in case some sensors were initialized with a different
        # internal env count (e.g., when their prims are not yet present at construction time).
        if env_ids is None:
            super()._reset_idx(slice(None))
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super()._reset_idx(env_ids)

        # sample a new goal (command-like) on reset
        goal_x = sample_uniform(self.cfg.goal_pos_x_range[0], self.cfg.goal_pos_x_range[1], (len(env_ids), 1), device=self.device)
        goal_y = sample_uniform(self.cfg.goal_pos_y_range[0], self.cfg.goal_pos_y_range[1], (len(env_ids), 1), device=self.device)
        goal_z = sample_uniform(self.cfg.goal_pos_z_range[0], self.cfg.goal_pos_z_range[1], (len(env_ids), 1), device=self.device)
        self.goal_object_pos[env_ids] = torch.cat([goal_x, goal_y, goal_z], dim=1)
        # reset goal resample timer on reset
        tmin, tmax = self.cfg.goal_resample_time_range_s
        self._goal_time_left_s[env_ids] = sample_uniform(
            float(tmin), float(tmax), (len(env_ids),), device=self.device
        )

        # reset object pose around spawn with uniform randomization (dexsuite-style)
        obj_state = self.object.data.default_root_state.clone()[env_ids]
        base_pos = torch.tensor(self.cfg.object_cfg.init_state.pos, device=self.device, dtype=torch.float).view(1, 3)
        dx = sample_uniform(self.cfg.object_reset_pos_x_range[0], self.cfg.object_reset_pos_x_range[1], (len(env_ids), 1), device=self.device)
        dy = sample_uniform(self.cfg.object_reset_pos_y_range[0], self.cfg.object_reset_pos_y_range[1], (len(env_ids), 1), device=self.device)
        dz = sample_uniform(self.cfg.object_reset_pos_z_range[0], self.cfg.object_reset_pos_z_range[1], (len(env_ids), 1), device=self.device)
        obj_state[:, 0:3] = base_pos + torch.cat([dx, dy, dz], dim=1) + self.scene.env_origins[env_ids]
        yaw = sample_uniform(self.cfg.object_reset_yaw_range[0], self.cfg.object_reset_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(-1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        obj_state[:, 3:7] = quat_from_angle_axis(yaw, z_axis)
        obj_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

        # reset robot joints around default with configurable offset range
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

        # keep base-class buffers/targets in sync
        self._reset_robot_joints(env_ids, dof_pos=dof_pos, dof_vel=dof_vel)

        # reset stats
        self.successes[env_ids] = 0.0

        # update caches
        self._compute_intermediate_values()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _compute_intermediate_values(self):
        # robot state
        self.robot_dof_pos = self.robot.data.joint_pos
        self.robot_dof_vel = self.robot.data.joint_vel
        # robot base position in env frame (used as a simple proxy for end-effector region)
        self.robot_root_pos_env = self.robot.data.root_pos_w - self.scene.env_origins

        # object pose
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w

        # goal resampling (countdown in seconds)
        self._goal_time_left_s -= float(self.step_dt)
        resample_ids = torch.nonzero(self._goal_time_left_s <= 0.0, as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._resample_goals(resample_ids)

        # update goal marker (one sphere per env)
        goal_pos = self.goal_object_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_object_rot)

    def _get_record(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor | dict]:
        """Collect a full snapshot for logging / imitation learning playback.

        Returns a nested dict of torch tensors on the current device.
        """
        # normalize env_ids
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # ensure cached values are fresh (also updates goal marker)
        self._compute_intermediate_values()

        # --- robot state
        robot = self.robot
        robot_state = {
            # root (world)
            "root_pos_w": robot.data.root_pos_w[ids].clone(),
            "root_quat_w": robot.data.root_quat_w[ids].clone(),
            "root_lin_vel_w": robot.data.root_lin_vel_w[ids].clone(),
            "root_ang_vel_w": robot.data.root_ang_vel_w[ids].clone(),
            # joints (all dofs)
            "joint_pos": robot.data.joint_pos[ids].clone(),
            "joint_vel": robot.data.joint_vel[ids].clone(),
            # controller targets (all dofs)
            "joint_pos_target": self.cur_targets[ids].clone(),
            "joint_pos_target_prev": self.prev_targets[ids].clone(),
            # indexing helper
            "actuated_dof_indices": torch.tensor(self.actuated_dof_indices, device=self.device, dtype=torch.long),
            "actions": self.actions[ids].clone(),
            "action_rate": self._action_rate[ids].clone(),
        }

        # --- object state
        obj = self.object
        object_state = {
            # root (world)
            "root_pos_w": obj.data.root_pos_w[ids].clone(),
            "root_quat_w": obj.data.root_quat_w[ids].clone(),
            "root_lin_vel_w": obj.data.root_lin_vel_w[ids].clone(),
            "root_ang_vel_w": obj.data.root_ang_vel_w[ids].clone(),
            # root (env)
            "root_pos_env": (obj.data.root_pos_w[ids] - self.scene.env_origins[ids]).clone(),
        }

        # --- goal / task
        task_state = {
            "goal_pos_env": self.goal_object_pos[ids].clone(),
            "goal_pos_w": (self.goal_object_pos[ids] + self.scene.env_origins[ids]).clone(),
            "goal_quat_w": self.goal_object_rot[ids].clone(),
            "goal_time_left_s": self._goal_time_left_s[ids].clone(),
            "successes": self.successes[ids].clone(),
            "episode_length_buf": self.episode_length_buf[ids].clone(),
        }

        # --- sensors: tactile + third-person camera
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


