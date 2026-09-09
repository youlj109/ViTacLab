from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_from_angle_axis, sample_uniform, saturate

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import _scale
from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_env import (
    UR10eDualShadowHandDirectMARLBaseEnv,
)

from .bi_blind_bin_drop_env_cfg import UR10eDualShadowHandBiBlindBinDropEnvCfg


@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class UR10eDualShadowHandBiBlindBinDropEnv(UR10eDualShadowHandDirectMARLBaseEnv):
    """Dual-arm blind bin-drop: manipulate the cube from outside into the bin."""

    cfg: UR10eDualShadowHandBiBlindBinDropEnvCfg

    def __init__(self, cfg: UR10eDualShadowHandBiBlindBinDropEnvCfg, render_mode: str | None = None, **kwargs):
        choice = max(0, min(int(getattr(cfg, "object_init_choice", 0)), len(cfg.object_init_pos_candidates) - 1))
        object_pos = cfg.resolve_object_init_pos(choice)
        cfg.object_init_choice = choice
        cfg.object_cfg = cfg.object_cfg.replace(init_state=cfg.object_cfg.init_state.replace(pos=object_pos))
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = {
            "right_hand": torch.zeros((self.num_envs, 30), device=self.device),
            "left_hand": torch.zeros((self.num_envs, 30), device=self.device),
        }
        self._prev_actions_r = torch.zeros_like(self.actions["right_hand"])
        self._prev_actions_l = torch.zeros_like(self.actions["left_hand"])
        self._action_rate_r = torch.zeros_like(self.actions["right_hand"])
        self._action_rate_l = torch.zeros_like(self.actions["left_hand"])

        act_re = re.compile(f"(?:{self.cfg.arm_joint_expr})|(?:{self.cfg.hand_joint_expr})")
        self._actuated_dof_indices = [i for i, name in enumerate(self.right_hand.joint_names) if act_re.match(name)]
        self._actuated_dof_indices.sort()
        self.actuated_dof_indices = self._actuated_dof_indices
        self._act_idx_t = torch.tensor(self._actuated_dof_indices, device=self.device, dtype=torch.long)
        if len(self._actuated_dof_indices) != 30:
            raise RuntimeError(f"Expected 30 actuated DoFs, got {len(self._actuated_dof_indices)}")

        joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]

        self.right_hand_prev_targets = torch.zeros((self.num_envs, self.right_hand.num_joints), device=self.device)
        self.right_hand_curr_targets = torch.zeros_like(self.right_hand_prev_targets)
        self.left_hand_prev_targets = torch.zeros_like(self.right_hand_prev_targets)
        self.left_hand_curr_targets = torch.zeros_like(self.right_hand_prev_targets)

        self._ee_body_idx_r = self.right_hand.body_names.index(self.cfg.ee_body_name)
        self._ee_body_idx_l = self.left_hand.body_names.index(self.cfg.ee_body_name)

        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.trash_can_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.trash_can_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._success_streak = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.successes = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

    def _setup_task_scene(self) -> None:
        spawn = self.cfg.trash_can_cfg.spawn.replace(scale=self.cfg.trash_can_scale)
        trash_can_cfg = self.cfg.trash_can_cfg.replace(spawn=spawn)
        self.trash_can = RigidObject(trash_can_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["trash_can"] = self.trash_can
        self.scene.rigid_objects["object"] = self.object

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
            tgt[:, jids] = self.cfg.act_moving_average * tgt[:, jids] + (1.0 - self.cfg.act_moving_average) * prev[:, jids]
            tgt[:, jids] = saturate(
                tgt[:, jids],
                self.robot_dof_lower_limits[:, self._act_idx_t],
                self.robot_dof_upper_limits[:, self._act_idx_t],
            )
            prev[:, jids] = tgt[:, jids]
            hand.set_joint_position_target(tgt[:, jids], joint_ids=jids)

    def _compute_intermediate_values(self) -> None:
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_quat = self.object.data.root_quat_w
        self.trash_can_pos = self.trash_can.data.root_pos_w - self.scene.env_origins
        self.trash_can_quat = self.trash_can.data.root_quat_w

    def _object_in_bin(self) -> torch.Tensor:
        rel = self.object_pos - self.trash_can_pos
        xy_dist = torch.linalg.norm(rel[:, :2], dim=-1)
        z_in = (rel[:, 2] >= self.cfg.bin_success_z_min) & (rel[:, 2] <= self.cfg.bin_success_z_max)
        return (xy_dist <= self.cfg.bin_success_xy_radius) & z_in

    def _object_out_of_bounds(self) -> torch.Tensor:
        o = self.object_pos
        return (
            (o[:, 0] < self.cfg.out_of_bound_x[0])
            | (o[:, 0] > self.cfg.out_of_bound_x[1])
            | (o[:, 1] < self.cfg.out_of_bound_y[0])
            | (o[:, 1] > self.cfg.out_of_bound_y[1])
            | (o[:, 2] < self.cfg.out_of_bound_z[0])
            | (o[:, 2] > self.cfg.out_of_bound_z[1])
        )

    def _wrist_pose_env_right(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self.right_hand.data.body_pos_w[:, self._ee_body_idx_r]
        quat_w = self.right_hand.data.body_quat_w[:, self._ee_body_idx_r]
        return pos_w - self.scene.env_origins, quat_w

    def _wrist_pose_env_left(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self.left_hand.data.body_pos_w[:, self._ee_body_idx_l]
        quat_w = self.left_hand.data.body_quat_w[:, self._ee_body_idx_l]
        return pos_w - self.scene.env_origins, quat_w

    def _obs_single(self, robot, wrist_pos, wrist_quat, actions, action_rate) -> torch.Tensor:
        return torch.cat(
            (
                _unscale(
                    robot.data.joint_pos[:, self._act_idx_t],
                    self.robot_dof_lower_limits[:, self._act_idx_t],
                    self.robot_dof_upper_limits[:, self._act_idx_t],
                ),
                self.cfg.vel_obs_scale * robot.data.joint_vel[:, self._act_idx_t],
                wrist_pos,
                wrist_quat,
                self.object_pos,
                self.object_quat,
                self.trash_can_pos,
                self.trash_can_quat,
                actions,
                action_rate,
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict:
        base = super()._get_observations()
        self._compute_intermediate_values()
        wr_p, wr_q = self._wrist_pose_env_right()
        wl_p, wl_q = self._wrist_pose_env_left()
        base["right_hand"] = self._obs_single(
            self.right_hand, wr_p, wr_q, self.actions["right_hand"], self._action_rate_r
        )
        base["left_hand"] = self._obs_single(
            self.left_hand, wl_p, wl_q, self.actions["left_hand"], self._action_rate_l
        )
        return base

    def _get_states(self) -> torch.Tensor:
        obs = self._get_observations()
        return torch.cat((obs["right_hand"], obs["left_hand"]), dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        self._compute_intermediate_values()

        wr_p, _ = self._wrist_pose_env_right()
        wl_p, _ = self._wrist_pose_env_left()
        wrist_to_object = 0.5 * (
            torch.linalg.norm(wr_p - self.object_pos, dim=-1) + torch.linalg.norm(wl_p - self.object_pos, dim=-1)
        )
        object_to_bin = torch.linalg.norm((self.object_pos - self.trash_can_pos)[:, :2], dim=-1)

        reach_rew = (1.0 - torch.tanh(wrist_to_object / 0.15)) * self.cfg.wrist_to_object_reward_scale
        bin_rew = (1.0 - torch.tanh(object_to_bin / 0.12)) * self.cfg.object_to_bin_reward_scale
        act_penalty_r = torch.sum(self.actions["right_hand"] ** 2, dim=-1) * self.cfg.action_l2_weight
        act_penalty_l = torch.sum(self.actions["left_hand"] ** 2, dim=-1) * self.cfg.action_l2_weight
        rate_penalty_r = torch.sum(self._action_rate_r**2, dim=-1) * self.cfg.action_rate_l2_weight
        rate_penalty_l = torch.sum(self._action_rate_l**2, dim=-1) * self.cfg.action_rate_l2_weight

        success_mask = self._object_in_bin()
        self._success_streak = torch.where(success_mask, self._success_streak + 1, torch.zeros_like(self._success_streak))
        success_now = self._success_streak >= int(self.cfg.min_success_steps)
        self.successes = torch.where(success_now, torch.ones_like(self.successes), self.successes)

        reward_common = reach_rew + bin_rew
        reward_r = reward_common + act_penalty_r + rate_penalty_r
        reward_l = reward_common + act_penalty_l + rate_penalty_l
        reward_r = torch.where(success_now, reward_r + self.cfg.success_bonus, reward_r)
        reward_l = torch.where(success_now, reward_l + self.cfg.success_bonus, reward_l)

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["bin_drop_success_rate"] = self.successes.mean()
        self.extras["log"]["bin_drop_success_streak"] = self._success_streak.float().mean()

        return {"right_hand": reward_r, "left_hand": reward_l}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = self._success_streak >= int(self.cfg.min_success_steps)
        fall = self.object_pos[:, 2] < self.cfg.fall_height
        oob = self._object_out_of_bounds()
        terminated_flag = fall | oob | success
        terminated = {a: terminated_flag for a in self.cfg.possible_agents}
        time_outs = {a: time_out for a in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            super()._reset_idx(slice(None))
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super()._reset_idx(env_ids_t)

        n = len(env_ids_t)
        trash_can_state = self.trash_can.data.default_root_state.clone()[env_ids_t]
        object_state = self.object.data.default_root_state.clone()[env_ids_t]

        trash_x = sample_uniform(
            self.cfg.trash_can_reset_pos_x_range[0], self.cfg.trash_can_reset_pos_x_range[1], (n, 1), self.device
        )
        trash_can_state[:, 0] = trash_can_state[:, 0] + trash_x.squeeze(-1)
        trash_can_state[:, 7:] = 0.0
        self.trash_can.write_root_pose_to_sim(trash_can_state[:, :7], env_ids_t)
        self.trash_can.write_root_velocity_to_sim(trash_can_state[:, 7:], env_ids_t)

        dx = sample_uniform(self.cfg.object_reset_pos_x_range[0], self.cfg.object_reset_pos_x_range[1], (n, 1), self.device)
        dy = sample_uniform(self.cfg.object_reset_pos_y_range[0], self.cfg.object_reset_pos_y_range[1], (n, 1), self.device)
        dz = sample_uniform(self.cfg.object_reset_pos_z_range[0], self.cfg.object_reset_pos_z_range[1], (n, 1), self.device)

        object_base = torch.tensor(
            self.cfg.resolve_object_init_pos(self.cfg.object_init_choice), device=self.device, dtype=torch.float
        ).view(1, 3)
        object_state[:, 0:3] = object_base + torch.cat([dx, dy, dz], dim=1) + self.scene.env_origins[env_ids_t]

        yaw = sample_uniform(self.cfg.object_reset_yaw_range[0], self.cfg.object_reset_yaw_range[1], (n, 1), self.device).squeeze(-1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(n, 1)
        object_state[:, 3:7] = quat_from_angle_axis(yaw, z_axis)
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids_t)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids_t)

        for robot, prev, curr in (
            (self.right_hand, self.right_hand_prev_targets, self.right_hand_curr_targets),
            (self.left_hand, self.left_hand_prev_targets, self.left_hand_curr_targets),
        ):
            jp = robot.data.default_joint_pos[env_ids_t].clone()
            jv = torch.zeros_like(jp)
            robot.set_joint_position_target(jp, env_ids=env_ids_t)
            robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids_t)
            prev[env_ids_t] = jp
            curr[env_ids_t] = jp

        self.actions["right_hand"][env_ids_t] = 0.0
        self.actions["left_hand"][env_ids_t] = 0.0
        self._prev_actions_r[env_ids_t] = 0.0
        self._prev_actions_l[env_ids_t] = 0.0
        self._action_rate_r[env_ids_t] = 0.0
        self._action_rate_l[env_ids_t] = 0.0
        self._success_streak[env_ids_t] = 0
        self.successes[env_ids_t] = 0.0
        self._compute_intermediate_values()
