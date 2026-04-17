# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual UR10e + Shadow Hand peg-style task: rigid Cosmos hole + peg; keypoint rewards via ``factory_utils``."""

from __future__ import annotations

import re
from collections.abc import Sequence

import carb
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import saturate

from isaaclab_tasks.direct.factory import factory_utils

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import _scale
from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_env import (
    UR10eDualShadowHandDirectMARLBaseEnv,
)

from .bi_peg_env_cfg import UR10eDualShadowHandBiPegEnvCfg


@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class UR10eDualShadowHandBiPegEnv(UR10eDualShadowHandDirectMARLBaseEnv):
    """Two arms on a peg-insert-style task; hole/peg are :class:`RigidObject` (Cosmos USD), not ``ForgePegInsert``."""

    cfg: UR10eDualShadowHandBiPegEnvCfg

    def __init__(self, cfg: UR10eDualShadowHandBiPegEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_task = cfg.task
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = {
            "right_hand": torch.zeros((self.num_envs, 30), device=self.device),
            "left_hand": torch.zeros((self.num_envs, 30), device=self.device),
        }

        factory_utils.set_body_inertias(self.right_hand, self.scene.num_envs)
        factory_utils.set_body_inertias(self.left_hand, self.scene.num_envs)
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self.right_hand, self.cfg_task.robot_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self.left_hand, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

        self.num_robot_dofs = self.right_hand.num_joints
        arm_ids_r, _ = self.right_hand.find_joints(self.cfg.arm_joint_expr)
        self._arm_joint_ids = [int(i) for i in arm_ids_r]
        if len(self._arm_joint_ids) != 6:
            raise RuntimeError(f"Bi-Peg: expected 6 arm joints, got {len(self._arm_joint_ids)}")

        act_re = re.compile(
            f"(?:{self.cfg.arm_joint_expr})|(?:{self.cfg.hand_joint_expr})",
        )
        self._actuated_dof_indices = [i for i, name in enumerate(self.right_hand.joint_names) if act_re.match(name)]
        self._actuated_dof_indices.sort()
        self._act_idx_t = torch.tensor(self._actuated_dof_indices, device=self.device, dtype=torch.long)
        if len(self._actuated_dof_indices) != 30:
            raise RuntimeError(
                f"Bi-Peg: expected 30 actuated DoFs, got {len(self._actuated_dof_indices)}: "
                f"{[self.right_hand.joint_names[i] for i in self._actuated_dof_indices]}"
            )
        # Public alias (same as :class:`UR10eDualShadowHandOverEnv`) for teleop / tooling.
        self.actuated_dof_indices = self._actuated_dof_indices

        joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]

        self._ee_body_idx_r = self.right_hand.body_names.index(self.cfg.ee_body_name)
        self._ee_body_idx_l = self.left_hand.body_names.index(self.cfg.ee_body_name)

        self._identity_quat_w = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        self._prev_actions_r = torch.zeros((self.num_envs, 30), device=self.device)
        self._prev_actions_l = torch.zeros_like(self._prev_actions_r)
        self._action_rate_r = torch.zeros_like(self._prev_actions_r)
        self._action_rate_l = torch.zeros_like(self._prev_actions_r)

        self._keypoint_dist = torch.zeros((self.num_envs,), device=self.device)
        self._held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._held_base_quat = self._identity_quat_w.clone()
        self._target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_held_base_quat = self._identity_quat_w.clone()

        self.last_update_timestamp = 0.0

        self.right_hand_prev_targets = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device)
        self.right_hand_curr_targets = torch.zeros_like(self.right_hand_prev_targets)
        self.left_hand_prev_targets = torch.zeros_like(self.right_hand_prev_targets)
        self.left_hand_curr_targets = torch.zeros_like(self.right_hand_prev_targets)

    def _setup_task_scene(self) -> None:
        self._fixed_asset = RigidObject(self.cfg.hole_cfg)
        self._held_asset = RigidObject(self.cfg.peg_cfg)
        self.scene.rigid_objects["hole"] = self._fixed_asset
        self.scene.rigid_objects["peg"] = self._held_asset

    def _wrist_pose_env_right(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self.right_hand.data.body_pos_w[:, self._ee_body_idx_r]
        quat_w = self.right_hand.data.body_quat_w[:, self._ee_body_idx_r]
        pos = pos_w - self.scene.env_origins
        return pos, quat_w

    def _wrist_pose_env_left(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self.left_hand.data.body_pos_w[:, self._ee_body_idx_l]
        quat_w = self.left_hand.data.body_quat_w[:, self._ee_body_idx_l]
        pos = pos_w - self.scene.env_origins
        return pos, quat_w

    def _compute_keypoint_dist_from_bases(
        self,
        held_base_pos: torch.Tensor,
        held_base_quat: torch.Tensor,
        target_held_base_pos: torch.Tensor,
        target_held_base_quat: torch.Tensor,
    ) -> torch.Tensor:
        nk = self.cfg_task.num_keypoints
        keypoints_held = torch.zeros((self.num_envs, nk, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, nk, 3), device=self.device)
        offsets = factory_utils.get_keypoint_offsets(nk, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        identity = self._identity_quat_w
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                identity,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                identity,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        return torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

    def _compute_intermediate_values(self) -> None:
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w
        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self._held_base_pos, self._held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos,
            self.held_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        self._target_held_base_pos, self._target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        self._keypoint_dist = self._compute_keypoint_dist_from_bases(
            self._held_base_pos,
            self._held_base_quat,
            self._target_held_base_pos,
            self._target_held_base_quat,
        )
        self.last_update_timestamp = self.right_hand._data._sim_timestamp

    def _compute_keypoint_reward_terms(self, curr_successes: torch.Tensor) -> tuple[dict, dict]:
        keypoint_dist = self._keypoint_dist
        a0, b0 = tuple(self.cfg_task.keypoint_coef_baseline)
        a1, b1 = tuple(self.cfg_task.keypoint_coef_coarse)
        a2, b2 = tuple(self.cfg_task.keypoint_coef_fine)
        ar = self.actions["right_hand"]
        al = self.actions["left_hand"]
        action_penalty_ee = 0.5 * (torch.norm(ar, p=2, dim=-1) + torch.norm(al, p=2, dim=-1))
        action_grad_penalty = 0.5 * (
            torch.norm(self._action_rate_r, p=2, dim=-1) + torch.norm(self._action_rate_l, p=2, dim=-1)
        )
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    def _get_curr_successes(self, success_threshold: float, check_rot: bool) -> torch.Tensor:
        held_base_pos = self._held_base_pos
        target_held_base_pos = self._target_held_base_pos

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = xy_dist < 0.0025
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError
        is_close_or_below = z_disp < height_threshold
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.right_hand.data.body_quat_w[:, self._ee_body_idx_r])
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _log_factory_metrics(self, rew_dict: dict, curr_successes: torch.Tensor) -> None:
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        first_success = torch.logical_and(curr_successes, self.ep_succeeded == 0)
        self.ep_succeeded = torch.where(curr_successes, torch.ones_like(self.ep_succeeded), self.ep_succeeded)

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._prev_actions_r[:] = self.actions["right_hand"]
        self._prev_actions_l[:] = self.actions["left_hand"]
        self.actions = {
            "right_hand": torch.clamp(actions["right_hand"].to(self.device), -1.0, 1.0),
            "left_hand": torch.clamp(actions["left_hand"].to(self.device), -1.0, 1.0),
        }
        self._action_rate_r[:] = self.actions["right_hand"] - self._prev_actions_r
        self._action_rate_l[:] = self.actions["left_hand"] - self._prev_actions_l

    def _apply_action(self) -> None:
        jids = self._actuated_dof_indices
        for hand, prev, tgt, act in (
            (self.right_hand, self.right_hand_prev_targets, self.right_hand_curr_targets, self.actions["right_hand"]),
            (self.left_hand, self.left_hand_prev_targets, self.left_hand_curr_targets, self.actions["left_hand"]),
        ):
            tgt[:, jids] = _scale(
                act,
                self.robot_dof_lower_limits[:, self._act_idx_t],
                self.robot_dof_upper_limits[:, self._act_idx_t],
            )
            tgt[:, jids] = (
                self.cfg.act_moving_average * tgt[:, jids]
                + (1.0 - self.cfg.act_moving_average) * prev[:, jids]
            )
            tgt[:, jids] = saturate(
                tgt[:, jids],
                self.robot_dof_lower_limits[:, self._act_idx_t],
                self.robot_dof_upper_limits[:, self._act_idx_t],
            )
            prev[:, jids] = tgt[:, jids]
            hand.set_joint_position_target(tgt[:, jids], joint_ids=jids)

    def _policy_obs_single(
        self,
        robot: Articulation,
        actions: torch.Tensor,
        action_rate: torch.Tensor,
        wrist_pos: torch.Tensor,
        wrist_quat: torch.Tensor,
    ) -> torch.Tensor:
        noisy_fixed = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        held_rel = self.held_pos - noisy_fixed
        wrist_rel = wrist_pos - noisy_fixed
        obs_elems = [
            _unscale(
                robot.data.joint_pos[:, self._act_idx_t],
                self.robot_dof_lower_limits[:, self._act_idx_t],
                self.robot_dof_upper_limits[:, self._act_idx_t],
            ),
            self.cfg.vel_obs_scale * robot.data.joint_vel[:, self._act_idx_t],
            held_rel,
            self.held_quat,
            wrist_rel,
            wrist_quat,
            self._keypoint_dist.unsqueeze(-1),
            actions,
            action_rate,
        ]
        td = 5 * (1 + 2)
        obs_elems.append(torch.zeros((self.num_envs, td), device=self.device))
        return torch.cat(obs_elems, dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.last_update_timestamp < self.right_hand._data._sim_timestamp:
            self._compute_intermediate_values()

        wr_p, wr_q = self._wrist_pose_env_right()
        wl_p, wl_q = self._wrist_pose_env_left()

        obs_r = self._policy_obs_single(
            self.right_hand,
            self.actions["right_hand"],
            self._action_rate_r,
            wr_p,
            wr_q,
        )
        obs_l = self._policy_obs_single(
            self.left_hand,
            self.actions["left_hand"],
            self._action_rate_l,
            wl_p,
            wl_q,
        )
        return {"right_hand": obs_r, "left_hand": obs_l}

    def _get_states(self) -> torch.Tensor:
        if self.last_update_timestamp < self.right_hand._data._sim_timestamp:
            self._compute_intermediate_values()
        wr_p, wr_q = self._wrist_pose_env_right()
        wl_p, wl_q = self._wrist_pose_env_left()
        obs_r = self._policy_obs_single(self.right_hand, self.actions["right_hand"], self._action_rate_r, wr_p, wr_q)
        obs_l = self._policy_obs_single(self.left_hand, self.actions["left_hand"], self._action_rate_l, wl_p, wl_q)
        return torch.cat((obs_r, obs_l), dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        if self.last_update_timestamp < self.right_hand._data._sim_timestamp:
            self._compute_intermediate_values()
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold,
            check_rot=False,
        )
        rew_dict, rew_scales = self._compute_keypoint_reward_terms(curr_successes)
        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf = rew_buf + rew * rew_scales[rew_name]

        self._log_factory_metrics(rew_dict, curr_successes)
        return {"right_hand": rew_buf, "left_hand": rew_buf}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.last_update_timestamp < self.right_hand._data._sim_timestamp:
            self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        to = {a: time_out for a in self.cfg.possible_agents}
        te = {a: time_out for a in self.cfg.possible_agents}
        return to, te

    def step_sim_no_action(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def _write_rigid_root_pose(
        self,
        body: RigidObject | Articulation,
        env_ids: torch.Tensor,
        pos_env: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        n = len(env_ids)
        eo = self.scene.env_origins[env_ids]
        pos = torch.tensor(pos_env, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        quat = torch.tensor(quat_wxyz, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        st = body.data.default_root_state.clone()[env_ids]
        st[:, 0:3] = eo + pos
        st[:, 3:7] = quat
        st[:, 7:] = 0.0
        body.write_root_pose_to_sim(st[:, 0:7], env_ids=env_ids)
        body.write_root_velocity_to_sim(st[:, 7:], env_ids=env_ids)
        body.reset()

    def _set_robot_default_reset_pose(self, env_ids: torch.Tensor) -> None:
        for robot in (self.right_hand, self.left_hand):
            jp = robot.data.default_joint_pos[env_ids].clone()
            jv = torch.zeros_like(jp)
            robot.set_joint_position_target(jp, env_ids=env_ids)
            robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
            robot.reset()
        self.step_sim_no_action()

    def randomize_initial_state(self, env_ids: torch.Tensor) -> None:
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
        op = self.cfg.object_poses

        self._write_rigid_root_pose(self._fixed_asset, env_ids, op.fixed_pos, op.fixed_quat)
        self._write_rigid_root_pose(self._held_asset, env_ids, op.held_pos, op.held_quat)

        n = len(env_ids)
        ref = torch.tensor(self.cfg.fixed_obs_ref_pos, dtype=torch.float32, device=self.device)
        self.fixed_pos_obs_frame[env_ids] = ref.unsqueeze(0).expand(n, -1)
        std = torch.tensor(self.cfg.fixed_obs_noise_std, dtype=torch.float32, device=self.device)
        self.init_fixed_pos_obs_noise[env_ids] = torch.randn(n, 3, device=self.device) * std.unsqueeze(0)

        self.step_sim_no_action()

        grasp_time = 0.0
        hand_ids = [i for i in self._actuated_dof_indices if i not in self._arm_joint_ids]
        hand_ids_t = torch.tensor(hand_ids, device=self.device, dtype=torch.long)
        closed_tgt = None
        if hand_ids:
            lo = self.robot_dof_lower_limits[:, hand_ids_t]
            hi = self.robot_dof_upper_limits[:, hand_ids_t]
            t = float(self.cfg.reset_grasp_hand_joint_lerp)
            closed_tgt = lo + t * (hi - lo)
        while grasp_time < 0.25:
            if closed_tgt is not None:
                for robot in (self.right_hand, self.left_hand):
                    jp = robot.data.joint_pos.clone()
                    jp[:, hand_ids_t] = closed_tgt
                    robot.set_joint_position_target(jp)
                    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.actions["right_hand"].zero_()
        self.actions["left_hand"].zero_()
        self._prev_actions_r.zero_()
        self._prev_actions_l.zero_()
        self._action_rate_r.zero_()
        self._action_rate_l.zero_()

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            super()._reset_idx(slice(None))
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super()._reset_idx(env_ids)

        self._set_robot_default_reset_pose(env_ids_t)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids_t)

        jp_r = self.right_hand.data.joint_pos[env_ids_t]
        jp_l = self.left_hand.data.joint_pos[env_ids_t]
        self.right_hand_prev_targets[env_ids_t] = jp_r
        self.right_hand_curr_targets[env_ids_t] = jp_r
        self.left_hand_prev_targets[env_ids_t] = jp_l
        self.left_hand_curr_targets[env_ids_t] = jp_l

        self.ep_succeeded[env_ids_t] = 0
        self.ep_success_times[env_ids_t] = 0
