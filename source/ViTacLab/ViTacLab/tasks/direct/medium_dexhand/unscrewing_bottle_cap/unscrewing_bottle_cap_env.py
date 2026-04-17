# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import re
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import sample_uniform, saturate

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_env import (
    UR10eDualShadowHandDirectMARLBaseEnv,
)

from .unscrewing_bottle_cap_env_cfg import (
    UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS,
    UR10E_DUAL_UNSCREW_NUM_HAND_DOFS,
    UR10eDualShadowHandUnscrewBottleCapEnvCfg,
    default_bottle_articulation_cfg,
    default_visual_goal_bottle_cfg,
)


def _angle_delta_shortest(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle delta in radians (avoids spikes when joint state wraps near ±π)."""

    d = curr - prev
    return torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi


class UR10eDualShadowHandUnscrewBottleCapEnv(UR10eDualShadowHandDirectMARLBaseEnv):
    """Dual-arm UR10e + Shadow Hand policy + articulated bottle (revolute + prismatic cap joints).

    Both arms are **actuated** (same DOFs as single-arm direct base: arm + hand). Prismatic DOF is
    **velocity-zeroed** until :math:`|\\theta_{\\mathrm{rot}}|` exceeds
    :attr:`UR10eDualShadowHandUnscrewBottleCapEnvCfg.cap_translation_unlock_angle_rad`
    so the cap cannot translate off the thread before enough rotation (thread-like behavior).

    **Goal:** :attr:`goal_pos` is **world** position (m) and :attr:`goal_quat_w` is world wxyz; both updated each reset. When
    :attr:`UR10eDualShadowHandUnscrewBottleCapEnvCfg.enable_visual_goal_bottle` is True, a kinematic **cylinder**
    (``visual_goal_bottle`` under each env) is posed in world frame so USD **Translate** matches the goal (not a
    PointInstancer-only marker).
    """

    cfg: UR10eDualShadowHandUnscrewBottleCapEnvCfg
    visual_goal_bottle: RigidObject | None = None

    def __init__(self, cfg: UR10eDualShadowHandUnscrewBottleCapEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.bottle_cfg is None:
            cfg = cfg.replace(
                bottle_cfg=default_bottle_articulation_cfg(
                    cap_rotation_damping=cfg.cap_rotation_damping,
                    cap_translation_damping=cfg.cap_translation_damping,
                    rotation_joint_name=cfg.cap_rotation_joint_name,
                    translation_joint_name=cfg.cap_translation_joint_name,
                    spawn_scale=cfg.bottle_spawn_scale,
                    init_pos=cfg.bottle_root_init_pos,
                ),
            )
        if cfg.enable_visual_goal_bottle and cfg.visual_goal_bottle_cfg is None:
            cfg = cfg.replace(
                visual_goal_bottle_cfg=default_visual_goal_bottle_cfg(
                    bottle_spawn_scale=cfg.bottle_spawn_scale,
                    init_pos=cfg.goal_pos_local,
                    init_rot_wxyz=cfg.goal_quat_wxyz,
                ),
            )
        super().__init__(cfg, render_mode, **kwargs)

        self.num_robot_dofs = self.right_hand.num_joints

        self.right_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_curr_targets = torch.zeros_like(self.right_hand_prev_targets)
        self.left_hand_prev_targets = torch.zeros_like(self.right_hand_prev_targets)
        self.left_hand_curr_targets = torch.zeros_like(self.right_hand_prev_targets)

        hand_re = re.compile(cfg.hand_joint_expr)
        arm_re = re.compile(cfg.arm_joint_expr)
        self._hand_dof_indices = [i for i, name in enumerate(self.right_hand.joint_names) if hand_re.match(name)]
        self._hand_dof_indices.sort()
        arm_dof_indices = [i for i, name in enumerate(self.right_hand.joint_names) if arm_re.match(name)]
        arm_dof_indices.sort()
        self.actuated_dof_indices = sorted(set(arm_dof_indices) | set(self._hand_dof_indices))
        if len(self._hand_dof_indices) != UR10E_DUAL_UNSCREW_NUM_HAND_DOFS:
            raise RuntimeError(
                f"Expected {UR10E_DUAL_UNSCREW_NUM_HAND_DOFS} hand DOFs from hand_joint_expr, "
                f"got {len(self._hand_dof_indices)}: "
                f"{[self.right_hand.joint_names[i] for i in self._hand_dof_indices]}"
            )
        if len(self.actuated_dof_indices) != UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS:
            raise RuntimeError(
                f"Expected {UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS} actuated DOFs (arm+hand), "
                f"got {len(self.actuated_dof_indices)}: "
                f"{[self.right_hand.joint_names[i] for i in self.actuated_dof_indices]}"
            )
        self._act_idx_t = torch.tensor(self.actuated_dof_indices, device=self.device, dtype=torch.long)

        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.right_hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]
        self.act_dof_lower_limits = self.robot_dof_lower_limits[:, self._act_idx_t]
        self.act_dof_upper_limits = self.robot_dof_upper_limits[:, self._act_idx_t]

        self._hand_idx_t = torch.tensor(self._hand_dof_indices, device=self.device, dtype=torch.long)
        self.hand_dof_lower_limits = self.robot_dof_lower_limits[:, self._hand_idx_t]
        self.hand_dof_upper_limits = self.robot_dof_upper_limits[:, self._hand_idx_t]

        try:
            self._cap_rot_idx = self.bottle.joint_names.index(self.cfg.cap_rotation_joint_name)
            self._cap_slide_idx = self.bottle.joint_names.index(self.cfg.cap_translation_joint_name)
        except ValueError as e:
            raise RuntimeError(
                "Bottle joint name mismatch. Available joints: "
                f"{self.bottle.joint_names}. Set cfg.cap_rotation_joint_name / cap_translation_joint_name "
                "to match mobility.usd."
            ) from e

        try:
            self._right_grasp_bottle_body_idx = self.bottle.body_names.index(self.cfg.right_grasp_contact_body_name)
        except ValueError as e:
            raise RuntimeError(
                "Right-hand grasp contact body not found on bottle articulation. Available bodies: "
                f"{self.bottle.body_names}. Set cfg.right_grasp_contact_body_name (cap link for PartNet 3517: link_1)."
            ) from e

        self._prev_cap_rot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._prev_cap_trans = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._action_rate_right = torch.zeros(
            self.num_envs, UR10E_DUAL_UNSCREW_NUM_ACTUATED_DOFS, device=self.device, dtype=torch.float32
        )
        self._action_rate_left = torch.zeros_like(self._action_rate_right)

        # IK+RL scripted EE: same env-local xyz noise as bottle root (see ``reset_position_noise`` in reset).
        self.ik_rl_trajectory_xyz_offset = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        gq = torch.tensor(self.cfg.goal_quat_wxyz, device=self.device, dtype=torch.float32)
        self.goal_quat_w = gq.unsqueeze(0).expand(self.num_envs, 4).clone()

        self._apply_bottle_solver_joint_limits()

    def _grasp_contact_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fingertip distance + per-finger opposition + bottle root linear-speed penalty; see cfg grasp_* fields.

        Right hand uses :attr:`bottle_cap_pos` (cap link COM); left hand uses :attr:`bottle_pos` (articulation root).
        Non-thumb contacts use a geometric mean so all four fingers are encouraged, not a two-finger pinch.
        """

        cfg = self.cfg
        if not cfg.enable_grasp_contact_reward:
            z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            return z, {}

        thresh = float(cfg.grasp_contact_dist_m) + 1e-8

        def _one_hand(
            finger_pos: torch.Tensor, p_obj: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # finger_pos: (N,5,3) env frame; index 4 = thumb (thdistal); p_obj: (N,3) env frame
            d = torch.norm(finger_pos - p_obj.unsqueeze(1), dim=-1)
            thumb_c = torch.clamp(1.0 - d[:, 4] / thresh, 0.0, 1.0)
            other_c = torch.clamp(1.0 - d[:, 0:4] / thresh, 0.0, 1.0)
            # Geometric mean: needs all four fingers contributing, not just two strong contacts.
            other_agg = torch.exp(torch.mean(torch.log(other_c + 1e-6), dim=-1))
            # Opposition: thumb vs each non-thumb finger (then mean), not thumb vs centroid of others.
            p_t = finger_pos[:, 4]
            v_t = p_t - p_obj
            v_t = v_t / (torch.norm(v_t, dim=-1, keepdim=True) + 1e-8)
            p_other = finger_pos[:, 0:4, :]
            v_o = p_other - p_obj.unsqueeze(1)
            v_o = v_o / (torch.norm(v_o, dim=-1, keepdim=True) + 1e-8)
            opp_each = -(v_t.unsqueeze(1) * v_o).sum(dim=-1)
            opp_each = torch.clamp(opp_each, 0.0, 1.0)
            opp = opp_each.mean(dim=-1)
            r = (
                float(cfg.grasp_thumb_w) * thumb_c
                + float(cfg.grasp_other_mean_w) * other_agg
                + float(cfg.grasp_opposition_w) * opp
            )
            return r, thumb_c, other_agg, opp, other_c.mean(dim=-1)

        r_r, t_r, og_r, op_r, om_r = _one_hand(self.right_fingertip_pos, self.bottle_cap_pos)
        r_l, t_l, og_l, op_l, om_l = _one_hand(self.left_fingertip_pos, self.bottle_pos)
        r_contact = torch.maximum(r_r, r_l)
        speed = torch.norm(self.bottle_linvel, dim=-1)
        r_stable = -speed
        w_s = float(cfg.grasp_stability_weight)
        r_total = r_contact + w_s * r_stable
        r_total = r_total * float(cfg.grasp_contact_reward_scale)

        extras = {
            "grasp_r_contact": r_contact,
            "grasp_r_stable": r_stable,
            "grasp_thumb_r": t_r,
            "grasp_thumb_l": t_l,
            "grasp_other_geom_r": og_r,
            "grasp_other_geom_l": og_l,
            "grasp_other_mean_r": om_r,
            "grasp_other_mean_l": om_l,
            "grasp_opp_r": op_r,
            "grasp_opp_l": op_l,
        }
        return r_total, extras

    def _apply_bottle_solver_joint_limits(self) -> None:
        """Push position / max-velocity limits into PhysX (see cfg ``apply_bottle_joint_limits``)."""

        if not self.cfg.apply_bottle_joint_limits:
            return

        lo_r, hi_r = self.cfg.cap_rotation_pos_limit
        # PhysX: revolute joint position limits must stay within [-2π, 2π] (reduced-coordinate articulation).
        two_pi = 2.0 * math.pi
        lo_r = max(float(lo_r), -two_pi)
        hi_r = min(float(hi_r), two_pi)
        if lo_r >= hi_r:
            lo_r, hi_r = -two_pi, two_pi
        limits_r = torch.empty(self.num_envs, 1, 2, device=self.device, dtype=torch.float32)
        limits_r[:, 0, 0] = lo_r
        limits_r[:, 0, 1] = hi_r
        self.bottle.write_joint_position_limit_to_sim(
            limits_r, joint_ids=[self._cap_rot_idx], warn_limit_violation=False
        )

        lo_t, hi_t = self.cfg.cap_translation_pos_limit
        limits_t = torch.empty(self.num_envs, 1, 2, device=self.device, dtype=torch.float32)
        limits_t[:, 0, 0] = lo_t
        limits_t[:, 0, 1] = hi_t
        self.bottle.write_joint_position_limit_to_sim(
            limits_t, joint_ids=[self._cap_slide_idx], warn_limit_violation=False
        )

        if self.cfg.cap_rotation_velocity_limit_sim is not None:
            v = torch.full(
                (self.num_envs, 1),
                float(self.cfg.cap_rotation_velocity_limit_sim),
                device=self.device,
                dtype=torch.float32,
            )
            self.bottle.write_joint_velocity_limit_to_sim(v, joint_ids=[self._cap_rot_idx])
        if self.cfg.cap_translation_velocity_limit_sim is not None:
            v = torch.full(
                (self.num_envs, 1),
                float(self.cfg.cap_translation_velocity_limit_sim),
                device=self.device,
                dtype=torch.float32,
            )
            self.bottle.write_joint_velocity_limit_to_sim(v, joint_ids=[self._cap_slide_idx])

    def _setup_task_scene(self) -> None:
        self.bottle = Articulation(self.cfg.bottle_cfg)
        self.scene.articulations["bottle"] = self.bottle
        if self.cfg.enable_visual_goal_bottle and self.cfg.visual_goal_bottle_cfg is not None:
            self.visual_goal_bottle = RigidObject(self.cfg.visual_goal_bottle_cfg)
            self.scene.rigid_objects["visual_goal_bottle"] = self.visual_goal_bottle

    def _apply_bottle_cap_translation_lock(self) -> None:
        """Zero prismatic velocity until rotation magnitude exceeds unlock threshold."""

        jp = self.bottle.data.joint_pos.clone()
        jv = self.bottle.data.joint_vel.clone()
        rot_abs = torch.abs(jp[:, self._cap_rot_idx])
        locked = rot_abs < self.cfg.cap_translation_unlock_angle_rad
        if locked.any():
            jv[locked, self._cap_slide_idx] = 0.0
            self.bottle.write_joint_state_to_sim(jp, jv)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        prev_r = self.actions["right_hand"].clone()
        prev_l = self.actions["left_hand"].clone()
        self.actions = actions
        self._action_rate_right = actions["right_hand"] - prev_r
        self._action_rate_left = actions["left_hand"] - prev_l

    def _bottle_out_of_bounds(self) -> torch.Tensor:
        """Bottle root in env frame outside configured axis-aligned bounds (N,) bool."""

        o = self.bottle_pos
        return (
            (o[:, 0] < self.cfg.out_of_bound_x[0])
            | (o[:, 0] > self.cfg.out_of_bound_x[1])
            | (o[:, 1] < self.cfg.out_of_bound_y[0])
            | (o[:, 1] > self.cfg.out_of_bound_y[1])
            | (o[:, 2] < self.cfg.out_of_bound_z[0])
            | (o[:, 2] > self.cfg.out_of_bound_z[1])
        )

    def _apply_action(self) -> None:
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["right_hand"],
            self.act_dof_lower_limits,
            self.act_dof_upper_limits,
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.right_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.right_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.right_hand_curr_targets[:, self.actuated_dof_indices],
            self.act_dof_lower_limits,
            self.act_dof_upper_limits,
        )

        self.left_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["left_hand"],
            self.act_dof_lower_limits,
            self.act_dof_upper_limits,
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.left_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.left_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.left_hand_curr_targets[:, self.actuated_dof_indices],
            self.act_dof_lower_limits,
            self.act_dof_upper_limits,
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

        self._apply_bottle_cap_translation_lock()

    def _bottle_joint_obs(self) -> torch.Tensor:
        """4 floats: rotation pos, translation pos, rotation vel, translation vel."""

        jp = self.bottle.data.joint_pos
        jv = self.bottle.data.joint_vel
        return torch.stack(
            (
                jp[:, self._cap_rot_idx],
                jp[:, self._cap_slide_idx],
                jv[:, self._cap_rot_idx],
                jv[:, self._cap_slide_idx],
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        b = self._bottle_joint_obs()
        observations = {
            "right_hand": torch.cat(
                (
                    unscale(self.right_act_dof_pos, self.act_dof_lower_limits, self.act_dof_upper_limits),
                    self.cfg.vel_obs_scale * self.right_act_dof_vel,
                    self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    self.actions["right_hand"],
                    self.bottle_pos,
                    self.bottle_rot,
                    self.bottle_linvel,
                    self.cfg.vel_obs_scale * self.bottle_angvel,
                    b,
                ),
                dim=-1,
            ),
            "left_hand": torch.cat(
                (
                    unscale(self.left_act_dof_pos, self.act_dof_lower_limits, self.act_dof_upper_limits),
                    self.cfg.vel_obs_scale * self.left_act_dof_vel,
                    self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    self.actions["left_hand"],
                    self.bottle_pos,
                    self.bottle_rot,
                    self.bottle_linvel,
                    self.cfg.vel_obs_scale * self.bottle_angvel,
                    b,
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        b = self._bottle_joint_obs()
        states = torch.cat(
            (
                unscale(self.right_act_dof_pos, self.act_dof_lower_limits, self.act_dof_upper_limits),
                self.cfg.vel_obs_scale * self.right_act_dof_vel,
                self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.actions["right_hand"],
                unscale(self.left_act_dof_pos, self.act_dof_lower_limits, self.act_dof_upper_limits),
                self.cfg.vel_obs_scale * self.left_act_dof_vel,
                self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.actions["left_hand"],
                self.bottle_pos,
                self.bottle_rot,
                self.bottle_linvel,
                self.cfg.vel_obs_scale * self.bottle_angvel,
                b,
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        goal_pos_env = self.goal_pos - self.scene.env_origins
        pos_err = goal_pos_env - self.bottle_pos
        pos_dist = torch.norm(pos_err, p=2, dim=-1)
        pos_term = 1.0 - torch.tanh(pos_dist / (cfg.staged_grasp_pos_std + 1e-6))
        # Near goal position before weighting bottle orientation (tighter scale than ``staged_grasp_pos_std``).
        rot_align_gate = 1.0 - torch.tanh(pos_dist / (cfg.grasp_rot_align_pos_std + 1e-6))

        dot_q = torch.abs(torch.sum(self.goal_quat_w * self.bottle_rot, dim=-1))
        dot_q = torch.clamp(dot_q, 0.0, 1.0)
        rot_dist = 2.0 * torch.acos(dot_q)
        rot_term = 1.0 - torch.tanh(rot_dist / (cfg.staged_grasp_rot_std + 1e-6))

        grasp_success = torch.clamp(
            cfg.grasp_success_pos_blend * pos_term
            + cfg.grasp_success_rot_blend * rot_term * rot_align_gate,
            0.0,
            1.0,
        )

        abs_cap = torch.abs(self.cap_rot)
        rotation_progress = torch.clamp(abs_cap / (cfg.rotation_progress_max_rad + 1e-6), 0.0, 1.0)
        lift_height = torch.clamp(self.cap_trans / (cfg.lift_progress_max_m + 1e-6), 0.0, 1.0)

        progress = (
            cfg.staged_progress_w1 * grasp_success
            + cfg.staged_progress_w2 * rotation_progress
            + cfg.staged_progress_w3 * lift_height
        )

        rs = (abs_cap - cfg.rotation_success_ramp_min_rad) / (
            cfg.rotation_success_ramp_max_rad - cfg.rotation_success_ramp_min_rad + 1e-6
        )
        rotation_success = torch.clamp(rs, 0.0, 1.0)

        r_grasp = pos_term * cfg.r_grasp_pos_scale + rot_term * cfg.r_grasp_rot_scale * rot_align_gate

        d_rot = _angle_delta_shortest(self._prev_cap_rot, self.cap_rot)
        r_rotate = torch.clamp(cfg.unscrew_rotation_sign * d_rot, min=0.0) * cfg.r_rotate_scale
        wrong_rot_mag = torch.clamp(-(cfg.unscrew_rotation_sign * d_rot), min=0.0)
        pen_wrong_rot = wrong_rot_mag * cfg.wrong_rotation_penalty_weight

        unlocked = abs_cap >= cfg.cap_translation_unlock_angle_rad
        d_trans = self.cap_trans - self._prev_cap_trans
        r_lift = torch.clamp(d_trans, min=0.0) * unlocked.to(torch.float) * cfg.r_lift_scale

        # No cap joint (rotation / lift) **rewards** until pose is close enough (hard mask, per env).
        cap_joint_ok = (pos_dist < cfg.cap_joint_reward_pos_thresh_m) & (
            rot_dist < cfg.cap_joint_reward_rot_thresh_rad
        )
        cap_joint_f = cap_joint_ok.to(dtype=r_rotate.dtype)
        r_rotate = r_rotate * cap_joint_f
        r_lift = r_lift * cap_joint_f

        self._prev_cap_rot = self.cap_rot.clone()
        self._prev_cap_trans = self.cap_trans.clone()

        staged = (
            r_grasp * (1.0 - rotation_progress)
            + r_rotate * grasp_success
            + r_lift * rotation_success
        )

        oob = self._bottle_out_of_bounds()
        pen_oob = oob.to(torch.float) * cfg.bottle_out_of_bounds_penalty_weight

        rew_common = staged + pen_wrong_rot + pen_oob
        r_grasp_contact, grasp_contact_log = self._grasp_contact_reward()
        rew_common = rew_common + r_grasp_contact

        pen_ar = torch.sum(self.actions["right_hand"] ** 2, dim=-1) * cfg.action_l2_penalty
        pen_al = torch.sum(self.actions["left_hand"] ** 2, dim=-1) * cfg.action_l2_penalty
        pen_rr = torch.sum(self._action_rate_right**2, dim=-1) * cfg.action_rate_l2_penalty
        pen_rl = torch.sum(self._action_rate_left**2, dim=-1) * cfg.action_rate_l2_penalty

        rew_r = rew_common - pen_ar - pen_rr
        rew_l = rew_common - pen_al - pen_rl

        if "log" not in self.extras:
            self.extras["log"] = dict()
        log = self.extras["log"]
        log["task_progress"] = progress.mean()
        log["grasp_success"] = grasp_success.mean()
        log["rotation_progress"] = rotation_progress.mean()
        log["lift_height"] = lift_height.mean()
        log["rotation_success"] = rotation_success.mean()
        log["reward_staged"] = staged.mean()
        log["reward_r_grasp_gate"] = (r_grasp * (1.0 - rotation_progress)).mean()
        log["reward_r_rotate_gate"] = (r_rotate * grasp_success).mean()
        log["reward_r_lift_gate"] = (r_lift * rotation_success).mean()
        log["pen_wrong_rot"] = pen_wrong_rot.mean()
        log["pen_oob"] = pen_oob.mean()
        log["reward_common"] = rew_common.mean()
        log["cap_rot"] = self.cap_rot.mean()
        log["cap_trans"] = self.cap_trans.mean()
        log["cap_unlocked_frac"] = unlocked.to(torch.float).mean()
        log["pos_dist_mean"] = pos_dist.mean()
        log["rot_dist_mean"] = rot_dist.mean()
        log["rot_align_gate"] = rot_align_gate.mean()
        log["cap_joint_reward_ok_frac"] = cap_joint_f.mean()
        log["reward_grasp_contact"] = r_grasp_contact.mean()
        for _k, _v in grasp_contact_log.items():
            log[_k] = _v.mean()

        return {"right_hand": rew_r, "left_hand": rew_l}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()

        fell = self.bottle_pos[:, 2] <= self.cfg.fall_height
        success = (self.cap_trans >= self.cfg.success_min_translation_m) | (
            torch.abs(self.cap_rot) >= self.cfg.success_total_rotation_rad
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: fell | success for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.right_hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # bottle root + joints
        bottle_root = self.bottle.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        dpos = self.cfg.reset_position_noise * pos_noise
        self.ik_rl_trajectory_xyz_offset[env_ids] = dpos.to(dtype=self.ik_rl_trajectory_xyz_offset.dtype)
        g = torch.tensor(self.cfg.goal_pos_local, device=self.device, dtype=torch.float32).view(1, 3).expand(
            len(env_ids), 3
        )
        origins = self.scene.env_origins[env_ids]
        if self.cfg.goal_uses_reset_position_noise:
            self.goal_pos[env_ids] = g + dpos + origins
        else:
            self.goal_pos[env_ids] = g + origins
        gq = torch.tensor(self.cfg.goal_quat_wxyz, device=self.device, dtype=torch.float32).view(1, 4).expand(
            len(env_ids), 4
        )
        self.goal_quat_w[env_ids] = gq

        bottle_root[:, 0:3] = bottle_root[:, 0:3] + dpos + self.scene.env_origins[env_ids]
        # Fixed orientation (identity from cfg); do not randomize bottle yaw/tilt.
        bottle_root[:, 7:] = 0.0
        self.bottle.write_root_pose_to_sim(bottle_root[:, :7], env_ids)
        self.bottle.write_root_velocity_to_sim(bottle_root[:, 7:], env_ids)

        njp = self.bottle.data.default_joint_pos[env_ids].clone()
        njv = self.bottle.data.default_joint_vel[env_ids].clone()
        self.bottle.write_joint_state_to_sim(njp, njv, env_ids=env_ids)

        self._prev_cap_rot[env_ids] = njp[:, self._cap_rot_idx]
        self._prev_cap_trans[env_ids] = njp[:, self._cap_slide_idx]

        delta_max = self.robot_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        hand_mask = torch.zeros(self.num_robot_dofs, device=self.device)
        hand_mask[self._hand_dof_indices] = 1.0
        arm_scale = float(self.cfg.arm_reset_dof_pos_noise_scale)
        arm_mask = 1.0 - hand_mask
        rand_delta = rand_delta * (hand_mask + arm_mask * arm_scale)

        dof_pos = self.right_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        dof_vel = dof_vel * hand_mask.unsqueeze(0)

        self.right_hand_prev_targets[env_ids] = dof_pos
        self.right_hand_curr_targets[env_ids] = dof_pos

        self.right_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        delta_max = self.robot_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        rand_delta = rand_delta * (hand_mask + arm_mask * arm_scale)

        dof_pos = self.left_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        dof_vel = dof_vel * hand_mask.unsqueeze(0)

        self.left_hand_prev_targets[env_ids] = dof_pos
        self.left_hand_curr_targets[env_ids] = dof_pos

        self.left_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        if self.visual_goal_bottle is not None:
            pos_w = self.goal_pos[env_ids]
            quat = self.goal_quat_w[env_ids]
            root = self.visual_goal_bottle.data.default_root_state.clone()[env_ids]
            root[:, 0:3] = pos_w
            root[:, 3:7] = quat
            root[:, 7:] = 0.0
            zv = torch.zeros((len(env_ids), 6), device=self.device, dtype=root.dtype)
            self.visual_goal_bottle.write_root_pose_to_sim(root[:, :7], env_ids)
            self.visual_goal_bottle.write_root_velocity_to_sim(zv, env_ids)

        self._compute_intermediate_values()

        # Avoid a large action-rate spike on the first step after reset.
        e = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if e.numel() > 0:
            self.actions["right_hand"][e] = 0.0
            self.actions["left_hand"][e] = 0.0

    def _compute_intermediate_values(self):
        self.right_fingertip_pos = self.right_hand.data.body_pos_w[:, self.finger_bodies]
        self.right_fingertip_rot = self.right_hand.data.body_quat_w[:, self.finger_bodies]
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w[:, self.finger_bodies]

        self.right_act_dof_pos = self.right_hand.data.joint_pos[:, self._act_idx_t]
        self.right_act_dof_vel = self.right_hand.data.joint_vel[:, self._act_idx_t]

        self.left_fingertip_pos = self.left_hand.data.body_pos_w[:, self.finger_bodies]
        self.left_fingertip_rot = self.left_hand.data.body_quat_w[:, self.finger_bodies]
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w[:, self.finger_bodies]

        self.left_act_dof_pos = self.left_hand.data.joint_pos[:, self._act_idx_t]
        self.left_act_dof_vel = self.left_hand.data.joint_vel[:, self._act_idx_t]

        self.bottle_pos = self.bottle.data.root_pos_w - self.scene.env_origins
        self.bottle_cap_pos = (
            self.bottle.data.body_pos_w[:, self._right_grasp_bottle_body_idx] - self.scene.env_origins
        )
        self.bottle_rot = self.bottle.data.root_quat_w
        self.bottle_linvel = self.bottle.data.root_lin_vel_w
        self.bottle_angvel = self.bottle.data.root_ang_vel_w

        self.cap_rot = self.bottle.data.joint_pos[:, self._cap_rot_idx]
        self.cap_trans = self.bottle.data.joint_pos[:, self._cap_slide_idx]


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
