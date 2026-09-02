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

from .bi_blind_grasp_env_cfg import UR10eDualShadowHandBiBlindGraspEnvCfg


@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class UR10eDualShadowHandBiBlindGraspEnv(UR10eDualShadowHandDirectMARLBaseEnv):
    """Dual-arm blind grasp: left hand reaches hole, right hand reaches peg."""

    cfg: UR10eDualShadowHandBiBlindGraspEnvCfg

    def __init__(self, cfg: UR10eDualShadowHandBiBlindGraspEnvCfg, render_mode: str | None = None, **kwargs):
        choice = max(0, min(int(getattr(cfg, "object_init_choice", 0)), len(cfg.object_init_pos_candidates) - 1))
        hole_pos = type(cfg).resolve_hole_init_pos(choice)
        peg_pos = type(cfg).resolve_peg_init_pos(choice)
        cfg.object_init_choice = choice
        cfg.hole_cfg = cfg.hole_cfg.replace(init_state=cfg.hole_cfg.init_state.replace(pos=hole_pos))
        cfg.peg_cfg = cfg.peg_cfg.replace(init_state=cfg.peg_cfg.init_state.replace(pos=peg_pos))
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

        self.hole_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hole_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.peg_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.peg_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_task_scene(self) -> None:
        self.hole = RigidObject(self.cfg.hole_cfg)
        self.peg = RigidObject(self.cfg.peg_cfg)
        self.scene.rigid_objects["hole"] = self.hole
        self.scene.rigid_objects["peg"] = self.peg

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
        self.hole_pos = self.hole.data.root_pos_w - self.scene.env_origins
        self.hole_quat = self.hole.data.root_quat_w
        self.peg_pos = self.peg.data.root_pos_w - self.scene.env_origins
        self.peg_quat = self.peg.data.root_quat_w

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
                self.hole_pos,
                self.hole_quat,
                self.peg_pos,
                self.peg_quat,
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
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = 0.0
        self.extras["log"]["episode_success_rate"] = 0.0
        self.extras["log"]["episode_success_rate_all_time"] = 0.0
        z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return {"right_hand": z, "left_hand": z}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        bad = (self.hole_pos[:, 2] < self.cfg.fall_height) | (self.peg_pos[:, 2] < self.cfg.fall_height)
        terminated = {a: bad for a in self.cfg.possible_agents}
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
        hole_state = self.hole.data.default_root_state.clone()[env_ids_t]
        peg_state = self.peg.data.default_root_state.clone()[env_ids_t]

        hx = sample_uniform(self.cfg.object_reset_pos_x_range[0], self.cfg.object_reset_pos_x_range[1], (n, 1), self.device)
        hy = sample_uniform(self.cfg.object_reset_pos_y_range[0], self.cfg.object_reset_pos_y_range[1], (n, 1), self.device)
        hz = sample_uniform(self.cfg.object_reset_pos_z_range[0], self.cfg.object_reset_pos_z_range[1], (n, 1), self.device)
        px = sample_uniform(self.cfg.object_reset_pos_x_range[0], self.cfg.object_reset_pos_x_range[1], (n, 1), self.device)
        py = sample_uniform(self.cfg.object_reset_pos_y_range[0], self.cfg.object_reset_pos_y_range[1], (n, 1), self.device)
        pz = sample_uniform(self.cfg.object_reset_pos_z_range[0], self.cfg.object_reset_pos_z_range[1], (n, 1), self.device)

        hole_base = torch.tensor(
            type(self.cfg).resolve_hole_init_pos(self.cfg.object_init_choice), device=self.device, dtype=torch.float
        ).view(1, 3)
        peg_base = torch.tensor(
            type(self.cfg).resolve_peg_init_pos(self.cfg.object_init_choice), device=self.device, dtype=torch.float
        ).view(1, 3)
        hole_state[:, 0:3] = hole_base + torch.cat([hx, hy, hz], dim=1) + self.scene.env_origins[env_ids_t]
        peg_state[:, 0:3] = peg_base + torch.cat([px, py, pz], dim=1) + self.scene.env_origins[env_ids_t]

        yaw_h = sample_uniform(self.cfg.object_reset_yaw_range[0], self.cfg.object_reset_yaw_range[1], (n, 1), self.device).squeeze(-1)
        yaw_p = sample_uniform(self.cfg.object_reset_yaw_range[0], self.cfg.object_reset_yaw_range[1], (n, 1), self.device).squeeze(-1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(n, 1)
        hole_state[:, 3:7] = quat_from_angle_axis(yaw_h, z_axis)
        peg_state[:, 3:7] = quat_from_angle_axis(yaw_p, z_axis)
        hole_state[:, 7:] = 0.0
        peg_state[:, 7:] = 0.0
        self.hole.write_root_pose_to_sim(hole_state[:, :7], env_ids_t)
        self.hole.write_root_velocity_to_sim(hole_state[:, 7:], env_ids_t)
        self.peg.write_root_pose_to_sim(peg_state[:, :7], env_ids_t)
        self.peg.write_root_velocity_to_sim(peg_state[:, 7:], env_ids_t)

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
        self._compute_intermediate_values()
