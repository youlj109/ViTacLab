# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Sequence
import numpy as np
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

try:
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData
except ImportError:
    VisuoTactileSensorData = None  # type: ignore

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    UR10eShadowHandDirectBaseEnv,
)

from .inhand_manipulation_env_cfg import UR10eShadowHandInHandEnvCfg, sync_inhand_rl_space_dims

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


class InHandManipulationEnv(UR10eShadowHandDirectBaseEnv):
    """In-hand cube reorientation on UR10e + Shadow Hand; policy controls hand only (arm pose fixed)."""

    cfg: UR10eShadowHandInHandEnvCfg

    def __init__(self, cfg: UR10eShadowHandInHandEnvCfg, render_mode: str | None = None, **kwargs):
        sync_inhand_rl_space_dims(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # Alias for preserved in-hand logic (expects `hand`).
        self.hand = self.robot

        # Policy actions: hand joints only (arm stays at commanded default pose).
        hand_re = re.compile(cfg.hand_joint_expr)
        self._hand_dof_indices = [i for i, name in enumerate(self.robot.joint_names) if hand_re.match(name)]
        self._hand_dof_indices.sort()
        if len(self._hand_dof_indices) != cfg.num_hand_dofs:
            raise RuntimeError(
                f"Expected {cfg.num_hand_dofs} hand DOFs from hand_joint_expr, "
                f"found {len(self._hand_dof_indices)}: {self._hand_dof_indices} / {self.robot.joint_names}"
            )
        self.actuated_dof_indices = self._hand_dof_indices
        self.num_actions = len(self._hand_dof_indices)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.num_hand_dofs = len(self._hand_dof_indices)

        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)

        self.finger_bodies = []
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self._resolve_body_index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04

        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        gp = torch.tensor(self.cfg.goal_marker_pos, device=self.device, dtype=torch.float).view(1, 3)
        self.goal_pos[:, :] = gp

        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # Cumulative episodic success for :meth:`get_episode_success_rate` (updated in :meth:`_reset_idx`).
        self._episode_success_count = 0
        self._episode_total_count = 0

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Hand-only limit tensors (for observations / unscale in preserved math).
        h = torch.tensor(self._hand_dof_indices, device=self.device, dtype=torch.long)
        self._hand_idx_t = h
        self.hand_dof_lower_limits = self.robot_dof_lower_limits[:, h]
        self.hand_dof_upper_limits = self.robot_dof_upper_limits[:, h]

        # Tactile buffers (populated in _setup_task_scene when sensors exist).
        self._tactile_normal_force: torch.Tensor | None = None
        self._tactile_shear_force: torch.Tensor | None = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0

    def orientation_success(self) -> torch.Tensor:
        """Per-environment orientation success: ``|rotation_distance(object, goal)| <= success_tolerance``.

        Same criterion as :func:`compute_rewards` / ``goal_resets`` (DexCube target pose).

        Returns:
            Boolean tensor of shape ``(num_envs,)`` on :attr:`device`.
        """
        rd = rotation_distance(self.object_rot, self.goal_rot)
        return torch.abs(rd) <= float(self.cfg.success_tolerance)

    def _accumulate_episode_success_stats(self, env_ids: torch.Tensor) -> None:
        """Update cumulative success/total episode counts when an episode ends (called from :meth:`_reset_idx`).

        Only envs that actually finished an episode **this step** (``terminated | time_out``) are counted.
        This excludes e.g. :meth:`env.reset` / full :meth:`_reset_idx` where buffers are still false, which
        would otherwise inflate the denominator and depress the reported success rate.
        """
        if env_ids.numel() == 0:
            return
        term = self.reset_terminated[env_ids]
        tout = self.reset_time_outs[env_ids]
        episode_done = term | tout
        if not episode_done.any():
            return
        not_fall = ~term
        # Episode finished without dropping: timeout / truncation (includes max-consecutive-success termination).
        completed = not_fall & tout
        max_cons = int(self.cfg.max_consecutive_success)
        if max_cons > 0:
            won = completed & (self.successes[env_ids] >= float(max_cons))
        else:
            won = completed & self.orientation_success()[env_ids]
        # Count only finished episodes (fall = failure, timeout with criteria = success/failure).
        won = won & episode_done
        self._episode_success_count += int(won.sum().item())
        self._episode_total_count += int(episode_done.sum().item())

    def get_episode_success_rate(self) -> float:
        """Fraction of **finished episodes** that ended in success (running average since env creation).

        Denominator includes only envs where ``terminated | time_out`` was true when :meth:`_reset_idx` ran
        (i.e. a real episode end from :meth:`step`), not spurious full resets such as the initial :meth:`reset`.

        **Success** (per episode):

        - If :attr:`max_consecutive_success` > 0: episode ends without a fall **and** (timeout path)
          ``successes >= max_consecutive_success`` after the last reward update (task completion), **or**
          horizon timeout with enough accumulated successes (same check).
        - If ``max_consecutive_success`` == 0: episode ends without a fall, on timeout, **and**
          :meth:`orientation_success` is true at termination (still within goal at episode end).

        **Failure**: object dropped (fall / out-of-reach termination), or horizon timeout without meeting
        the success criteria above.

        Returns:
            Scalar in ``[0, 1]``. Returns ``0.0`` if no episode has finished yet.
        """
        if self._episode_total_count <= 0:
            return 0.0
        return self._episode_success_count / float(self._episode_total_count)

    def get_episode_success_stats(self) -> tuple[int, int, float]:
        """Return ``(n_success_episodes, n_completed_episodes, success_rate)`` for :meth:`get_episode_success_rate`."""
        return (
            self._episode_success_count,
            self._episode_total_count,
            self.get_episode_success_rate(),
        )

    def reset_episode_success_statistics(self) -> None:
        """Reset running counts used by :meth:`get_episode_success_rate` (e.g. before a new eval run)."""
        self._episode_success_count = 0
        self._episode_total_count = 0

    def _resolve_body_index(self, name: str) -> int:
        names = self.robot.body_names
        if name in names:
            return names.index(name)
        for i, bn in enumerate(names):
            if bn.endswith(name) or name in bn:
                return i
        raise KeyError(f"Fingertip body {name!r} not found in robot.body_names={names!r}")

    def _setup_task_scene(self) -> None:
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        self._tactile_normal_force = None
        self._tactile_shear_force = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.to(device=self.device), -1.0, 1.0)

    def _update_tactile_data(self) -> None:
        """Read 5 GelSight sensors and fill _tactile_normal_force, _tactile_shear_force."""
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

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        use_tactile_obs = self._tactile_normal_force is not None and not getattr(self.cfg, "reduced_obs", True)
        if use_tactile_obs:
            self._update_tactile_data()
            obs = torch.cat([obs, self._tactile_normal_force, self._tactile_shear_force], dim=-1)
            if self.cfg.asymmetric_obs:
                states = torch.cat([states, self._tactile_normal_force, self._tactile_shear_force], dim=-1)

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            e = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            e = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if e.numel() > 0:
            self._accumulate_episode_success_stats(e)

        if env_ids is None:
            super()._reset_idx(slice(None))
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super()._reset_idx(env_ids)

        self._reset_target_pose(env_ids)

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        defaults = self.robot.data.default_joint_pos[env_ids]
        delta_max = self.robot_dof_upper_limits[env_ids] - defaults
        delta_min = self.robot_dof_lower_limits[env_ids] - defaults

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise

        hand_mask = torch.zeros(self.num_robot_dofs, device=self.device)
        hand_mask[self._hand_dof_indices] = 1.0
        arm_scale = float(getattr(self.cfg, "arm_reset_dof_pos_noise_scale", 0.0))
        arm_mask = 1.0 - hand_mask
        rand_delta = rand_delta * (hand_mask + arm_mask * arm_scale)

        dof_pos = defaults + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        dof_vel = self.robot.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        dof_vel = dof_vel * hand_mask.unsqueeze(0)

        self._reset_robot_joints(env_ids, dof_pos=dof_pos, dof_vel=dof_vel)
        self.hand_dof_targets[env_ids] = dof_pos

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos[:, self._hand_idx_t]
        self.hand_dof_vel = self.hand.data.joint_vel[:, self._hand_idx_t]

        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def compute_reduced_observations(self):
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                self.actions,
            ),
            dim=-1,
        )
        return states


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes


# Backward-compatible names for scripts importing unscale from this module.
__all__ = [
    "InHandManipulationEnv",
    "compute_rewards",
    "randomize_rotation",
    "rotation_distance",
    "scale",
    "unscale",
]
