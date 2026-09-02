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
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_env import (
    UR10eDualShadowHandDirectMARLBaseEnv,
)

from .bi_blind_inhand_env_cfg import (
    UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS,
    UR10eDualShadowHandBiBlindInhandEnvCfg,
)


class UR10eDualShadowHandBiBlindInhandEnv(UR10eDualShadowHandDirectMARLBaseEnv):
    """Bimanual blind in-hand: each hand aligns its own object rotation to an independent target."""

    cfg: UR10eDualShadowHandBiBlindInhandEnvCfg

    def __init__(self, cfg: UR10eDualShadowHandBiBlindInhandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.right_hand.num_joints

        self.right_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )

        hand_re = re.compile(cfg.hand_joint_expr)
        self._hand_dof_indices = [i for i, name in enumerate(self.right_hand.joint_names) if hand_re.match(name)]
        self._hand_dof_indices.sort()
        self.actuated_dof_indices = self._hand_dof_indices
        if len(self.actuated_dof_indices) != UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS:
            raise RuntimeError(
                f"Expected {UR10E_DUAL_SHADOWHAND_BI_BLIND_INHAND_NUM_HAND_DOFS} hand DOFs from hand_joint_expr, "
                f"got {len(self.actuated_dof_indices)}: {[self.right_hand.joint_names[i] for i in self.actuated_dof_indices]}"
            )
        self._hand_idx_t = torch.tensor(self._hand_dof_indices, device=self.device, dtype=torch.long)

        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.right_hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]
        h = self._hand_idx_t
        self.hand_dof_lower_limits = self.robot_dof_lower_limits[:, h]
        self.hand_dof_upper_limits = self.robot_dof_upper_limits[:, h]

        self.goal_rot_left = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot_left[:, 0] = 1.0
        self.goal_rot_right = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot_right[:, 0] = 1.0

        self.goal_markers_left = VisualizationMarkers(self.cfg.goal_left_marker_cfg)
        self.goal_markers_right = VisualizationMarkers(self.cfg.goal_right_marker_cfg)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_task_scene(self) -> None:
        self.object_left = RigidObject(self.cfg.object_left_cfg)
        self.object_right = RigidObject(self.cfg.object_right_cfg)
        self.scene.rigid_objects["object_left"] = self.object_left
        self.scene.rigid_objects["object_right"] = self.object_right

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["right_hand"],
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.right_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.right_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.right_hand_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )

        self.left_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["left_hand"],
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.left_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.left_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.left_hand_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )

        self.right_hand_prev_targets[:, self.actuated_dof_indices] = self.right_hand_curr_targets[
            :, self.actuated_dof_indices
        ]
        self.left_hand_prev_targets[:, self.actuated_dof_indices] = self.left_hand_curr_targets[
            :, self.actuated_dof_indices
        ]

        self.right_hand.set_joint_position_target(
            self.right_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self.left_hand.set_joint_position_target(
            self.left_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            "right_hand": torch.cat(
                (
                    unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                    self.cfg.vel_obs_scale * self.right_hand_dof_vel,
                    self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    self.actions["right_hand"],
                    self.object_right_pos,
                    self.object_right_rot,
                    self.object_right_linvel,
                    self.cfg.vel_obs_scale * self.object_right_angvel,
                    self.goal_rot_right,
                    quat_mul(self.object_right_rot, quat_conjugate(self.goal_rot_right)),
                ),
                dim=-1,
            ),
            "left_hand": torch.cat(
                (
                    unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                    self.cfg.vel_obs_scale * self.left_hand_dof_vel,
                    self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    self.actions["left_hand"],
                    self.object_left_pos,
                    self.object_left_rot,
                    self.object_left_linvel,
                    self.cfg.vel_obs_scale * self.object_left_angvel,
                    self.goal_rot_left,
                    quat_mul(self.object_left_rot, quat_conjugate(self.goal_rot_left)),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        return torch.cat(
            (
                unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.right_hand_dof_vel,
                self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.actions["right_hand"],
                self.object_right_pos,
                self.object_right_rot,
                self.object_right_linvel,
                self.cfg.vel_obs_scale * self.object_right_angvel,
                self.goal_rot_right,
                quat_mul(self.object_right_rot, quat_conjugate(self.goal_rot_right)),
                unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.left_hand_dof_vel,
                self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.actions["left_hand"],
                self.object_left_pos,
                self.object_left_rot,
                self.object_left_linvel,
                self.cfg.vel_obs_scale * self.object_left_angvel,
                self.goal_rot_left,
                quat_mul(self.object_left_rot, quat_conjugate(self.goal_rot_left)),
            ),
            dim=-1,
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rot_l = rotation_distance(self.object_left_rot, self.goal_rot_left)
        rot_r = rotation_distance(self.object_right_rot, self.goal_rot_right)

        rot_rew_l = 1.0 / (torch.abs(rot_l) + self.cfg.rot_eps) * self.cfg.rot_reward_scale
        rot_rew_r = 1.0 / (torch.abs(rot_r) + self.cfg.rot_eps) * self.cfg.rot_reward_scale

        action_penalty = torch.sum(self.actions["right_hand"] ** 2, dim=-1) + torch.sum(
            self.actions["left_hand"] ** 2, dim=-1
        )

        rew = rot_rew_l + rot_rew_r + self.cfg.action_penalty_scale * action_penalty

        both_ok = (torch.abs(rot_l) <= self.cfg.success_tolerance) & (torch.abs(rot_r) <= self.cfg.success_tolerance)
        rew = torch.where(both_ok, rew + self.cfg.reach_goal_bonus, rew)

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["rot_dist_left"] = rot_l.mean()
        self.extras["log"]["rot_dist_right"] = rot_r.mean()
        self.extras["log"]["rot_reward"] = (rot_rew_l + rot_rew_r).mean()

        return {"right_hand": rew, "left_hand": rew}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()

        out_l = self.object_left_pos[:, 2] <= self.cfg.fall_dist
        out_r = self.object_right_pos[:, 2] <= self.cfg.fall_dist
        out_of_reach = out_l | out_r
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: out_of_reach for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.right_hand._ALL_INDICES
        super()._reset_idx(env_ids)

        self._reset_goals(env_ids)

        for obj in (self.object_left, self.object_right):
            object_state = obj.data.default_root_state.clone()[env_ids]
            pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
            object_state[:, 0:3] = (
                object_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
            )
            rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
            object_state[:, 3:7] = randomize_rotation(
                rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
            )
            object_state[:, 7:] = torch.zeros_like(obj.data.default_root_state[env_ids, 7:])
            obj.write_root_pose_to_sim(object_state[:, :7], env_ids)
            obj.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        delta_max = self.robot_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        hand_mask = torch.zeros(self.num_hand_dofs, device=self.device)
        hand_mask[self._hand_dof_indices] = 1.0
        arm_scale = float(getattr(self.cfg, "arm_reset_dof_pos_noise_scale", 0.0))
        arm_mask = 1.0 - hand_mask
        rand_delta = rand_delta * (hand_mask + arm_mask * arm_scale)

        dof_pos = self.right_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        dof_vel = dof_vel * hand_mask.unsqueeze(0)

        self.right_hand_prev_targets[env_ids] = dof_pos
        self.right_hand_curr_targets[env_ids] = dof_pos
        self.right_hand_dof_targets[env_ids] = dof_pos

        self.right_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        delta_max = self.robot_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        rand_delta = rand_delta * (hand_mask + arm_mask * arm_scale)

        dof_pos = self.left_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        dof_vel = dof_vel * hand_mask.unsqueeze(0)

        self.left_hand_prev_targets[env_ids] = dof_pos
        self.left_hand_curr_targets[env_ids] = dof_pos
        self.left_hand_dof_targets[env_ids] = dof_pos

        self.left_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._compute_intermediate_values()
        self._update_goal_markers()

    def _reset_goals(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        rand_l = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        self.goal_rot_left[env_ids] = randomize_rotation(
            rand_l[:, 0], rand_l[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        rand_r = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        self.goal_rot_right[env_ids] = randomize_rotation(
            rand_r[:, 0], rand_r[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

    def _update_goal_markers(self) -> None:
        gl = self.object_left_pos + self.scene.env_origins
        gr = self.object_right_pos + self.scene.env_origins
        self.goal_markers_left.visualize(gl, self.goal_rot_left)
        self.goal_markers_right.visualize(gr, self.goal_rot_right)

    def _compute_intermediate_values(self):
        self.right_fingertip_pos = self.right_hand.data.body_pos_w[:, self.finger_bodies]
        self.right_fingertip_rot = self.right_hand.data.body_quat_w[:, self.finger_bodies]
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w[:, self.finger_bodies]

        self.right_hand_dof_pos = self.right_hand.data.joint_pos[:, self._hand_idx_t]
        self.right_hand_dof_vel = self.right_hand.data.joint_vel[:, self._hand_idx_t]

        self.left_fingertip_pos = self.left_hand.data.body_pos_w[:, self.finger_bodies]
        self.left_fingertip_rot = self.left_hand.data.body_quat_w[:, self.finger_bodies]
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w[:, self.finger_bodies]

        self.left_hand_dof_pos = self.left_hand.data.joint_pos[:, self._hand_idx_t]
        self.left_hand_dof_vel = self.left_hand.data.joint_vel[:, self._hand_idx_t]

        self.object_left_pos = self.object_left.data.root_pos_w - self.scene.env_origins
        self.object_left_rot = self.object_left.data.root_quat_w
        self.object_left_linvel = self.object_left.data.root_lin_vel_w
        self.object_left_angvel = self.object_left.data.root_ang_vel_w

        self.object_right_pos = self.object_right.data.root_pos_w - self.scene.env_origins
        self.object_right_rot = self.object_right.data.root_quat_w
        self.object_right_linvel = self.object_right.data.root_lin_vel_w
        self.object_right_angvel = self.object_right.data.root_ang_vel_w


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


__all__ = [
    "UR10eDualShadowHandBiBlindInhandEnv",
    "rotation_distance",
    "scale",
    "unscale",
]
