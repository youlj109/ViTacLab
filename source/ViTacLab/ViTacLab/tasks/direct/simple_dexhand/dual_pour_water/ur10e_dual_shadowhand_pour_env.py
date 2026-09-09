"""Dual UR10e + ShadowHand shared deformable cup pouring (cooperative MARL).

Task dynamics mirror :mod:`ViTacLab.tasks.direct.simple_dexhand.pour_water` (two-phase pour, same rewards / terminations); each agent observes
its own joints, actions, tactile, and shared cup / goal / tracking error. Shared task reward with
per-agent action penalties (see :class:`UR10eDualShadowHandUnscrewBottleCapEnv`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    sample_uniform,
    saturate,
)

from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_env import (
    UR10eDualShadowHandDirectMARLBaseEnv,
)

from .ur10e_dual_shadowhand_pour_env_cfg import (
    UR10E_DUAL_POUR_NUM_ACTUATED_DOFS,
    UR10eDualShadowHandPourEnvCfg,
    _pour_obs_dim_per_agent,
)

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
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def _rotation_distance(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


class UR10eDualShadowHandPourEnv(UR10eDualShadowHandDirectMARLBaseEnv):
    """Dual UR10e + ShadowHand pour task (shared deformable cup + water + rigid bowl)."""

    cfg: UR10eDualShadowHandPourEnvCfg

    def __init__(self, cfg: UR10eDualShadowHandPourEnvCfg, render_mode: str | None = None, **kwargs):
        obs_dim = _pour_obs_dim_per_agent(use_full_tactile_obs=cfg.use_full_tactile_obs)
        cfg = cfg.replace(
            observation_spaces={
                "right_hand": obs_dim,
                "left_hand": obs_dim,
            },
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
        self._act_idx_t = torch.tensor(self.actuated_dof_indices, device=self.device, dtype=torch.long)

        joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]
        self.act_dof_lower_limits = self.robot_dof_lower_limits[:, self._act_idx_t]
        self.act_dof_upper_limits = self.robot_dof_upper_limits[:, self._act_idx_t]

        self._target_default_quat_w = self.target.data.default_root_state[:, 3:7].clone()

        self.cup_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.water_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cup_rot = torch.zeros((self.num_envs, 4), device=self.device)

        self.goal_cup_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_cup_rot = torch.tensor(
            cfg.goal_cup_rot_wxyz, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

        self._goal_time_left_s = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

        self._action_rate_right = torch.zeros(
            self.num_envs, UR10E_DUAL_POUR_NUM_ACTUATED_DOFS, device=self.device, dtype=torch.float32
        )
        self._action_rate_left = torch.zeros_like(self._action_rate_right)

        self._tactile_normal_force_r: torch.Tensor | None = None
        self._tactile_shear_force_r: torch.Tensor | None = None
        self._tactile_normal_force_l: torch.Tensor | None = None
        self._tactile_shear_force_l: torch.Tensor | None = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0
        self._tactile_normal_mean_r: torch.Tensor | None = None
        self._tactile_shear_mean_r: torch.Tensor | None = None
        self._tactile_normal_mean_l: torch.Tensor | None = None
        self._tactile_shear_mean_l: torch.Tensor | None = None

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.pour_water_in_bowl = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._pour_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self._episode_success_count = 0
        self._episode_total_count = 0
        self._episode_success_rate_ema: float = 0.0

    def _tactile_key(self, arm: str, name: str) -> str:
        prefix = "right_" if arm == "right" else "left_"
        return prefix + name

    def _sync_target_pose_to_goal(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ox, oy = self.cfg.target_xy_offset_from_goal
        z = self.cfg.target_z_env
        g = self.goal_cup_pos[env_ids]
        pos_env = torch.stack(
            (
                g[:, 0] + ox,
                g[:, 1] + oy,
                torch.full_like(g[:, 2], z, dtype=g.dtype),
            ),
            dim=-1,
        )
        pos_w = pos_env + self.scene.env_origins[env_ids]
        quat = self._target_default_quat_w[env_ids]
        root_pose = torch.cat([pos_w, quat], dim=-1)
        zeros = torch.zeros((len(env_ids), 6), device=self.device, dtype=root_pose.dtype)
        self.target.write_root_pose_to_sim(root_pose, env_ids)
        self.target.write_root_velocity_to_sim(zeros, env_ids)

    def _out_of_bounds(self, cup_pos_env: torch.Tensor) -> torch.Tensor:
        o = cup_pos_env
        return (
            (o[:, 0] < self.cfg.out_of_bound_x[0])
            | (o[:, 0] > self.cfg.out_of_bound_x[1])
            | (o[:, 1] < self.cfg.out_of_bound_y[0])
            | (o[:, 1] > self.cfg.out_of_bound_y[1])
            | (o[:, 2] < self.cfg.out_of_bound_z[0])
            | (o[:, 2] > self.cfg.out_of_bound_z[1])
        )

    def _water_out_of_bounds(self, water_pos_env: torch.Tensor) -> torch.Tensor:
        w = water_pos_env
        return (
            (w[:, 0] < self.cfg.water_out_of_bound_x[0])
            | (w[:, 0] > self.cfg.water_out_of_bound_x[1])
            | (w[:, 1] < self.cfg.water_out_of_bound_y[0])
            | (w[:, 1] > self.cfg.water_out_of_bound_y[1])
            | (w[:, 2] < self.cfg.water_out_of_bound_z[0])
            | (w[:, 2] > self.cfg.water_out_of_bound_z[1])
        )

    def _accumulate_episode_success_stats(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        # ``terminated_dict`` / ``time_out_dict`` are only set after the first ``step()``; skip on initial ``reset()``.
        if not hasattr(self, "terminated_dict"):
            return
        term = self.terminated_dict["right_hand"][env_ids]
        tout = self.time_out_dict["right_hand"][env_ids]
        episode_done = term | tout
        if not episode_done.any():
            return
        buf = self.episode_length_buf[env_ids]
        split = int(self.cfg.pour_phase_split_step)
        p1 = buf <= split
        water_cup = torch.norm(self.water_pos[env_ids] - self.cup_pos[env_ids], dim=-1)
        cup_goal = torch.norm(self.goal_cup_pos[env_ids] - self.cup_pos[env_ids], dim=-1)
        phase1_bad = p1 & (water_cup > self.cfg.phase1_water_cup_max_dist)
        phase2_bad = (~p1) & (cup_goal > self.cfg.phase2_cup_goal_max_dist)
        bad = (
            (self.cup_pos[env_ids][:, 2] < self.cfg.fall_height)
            | self._out_of_bounds(self.cup_pos[env_ids])
            | self._water_out_of_bounds(self.water_pos[env_ids])
            | phase1_bad
            | phase2_bad
        )
        max_cons = int(self.cfg.max_consecutive_success)
        if max_cons > 0:
            won = episode_done & ~bad & (self._pour_success_streak[env_ids].float() >= float(max_cons))
        else:
            won = episode_done & ~bad & (self.pour_water_in_bowl[env_ids] >= 1.0)
        n_done = int(episode_done.sum().item())
        n_won = int(won.sum().item())
        self._episode_success_count += n_won
        self._episode_total_count += n_done
        if n_done > 0:
            batch_rate = n_won / float(n_done)
            alpha = float(self.cfg.episode_success_ema_alpha)
            self._episode_success_rate_ema = (1.0 - alpha) * self._episode_success_rate_ema + alpha * batch_rate

    def get_episode_success_rate(self) -> float:
        if self._episode_total_count <= 0:
            return 0.0
        return self._episode_success_count / float(self._episode_total_count)

    def get_episode_success_rate_ema(self) -> float:
        return float(self._episode_success_rate_ema)

    def get_episode_success_stats(self) -> tuple[int, int, float]:
        return (
            self._episode_success_count,
            self._episode_total_count,
            self.get_episode_success_rate(),
        )

    def reset_episode_success_statistics(self) -> None:
        self._episode_success_count = 0
        self._episode_total_count = 0
        self._episode_success_rate_ema = 0.0

    def _setup_task_scene(self) -> None:
        self.cup = DeformableObject(self.cfg.cup_cfg)
        self.water = DeformableObject(self.cfg.water_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        self.scene.deformable_objects["cup"] = self.cup
        self.scene.deformable_objects["water"] = self.water
        self.scene.rigid_objects["target_cup"] = self.target

        for arm in ("right", "left"):
            if self._tactile_key(arm, TACTILE_SENSOR_NAMES[0]) in self.scene.sensors and VisuoTactileSensorData is not None:
                first = self.scene[self._tactile_key(arm, TACTILE_SENSOR_NAMES[0])]
                sz = first.cfg.tactile_array_size
                self._tactile_array_total = sz[0] * sz[1]
                self._num_tactile_sensors = len(TACTILE_SENSOR_NAMES)
                n = self._num_tactile_sensors * self._tactile_array_total
                nf = torch.zeros((self.num_envs, n), device=self.device)
                sf = torch.zeros((self.num_envs, n * 2), device=self.device)
                nm = torch.zeros((self.num_envs, self._num_tactile_sensors), device=self.device)
                sm = torch.zeros((self.num_envs, self._num_tactile_sensors * 2), device=self.device)
                if arm == "right":
                    self._tactile_normal_force_r = nf
                    self._tactile_shear_force_r = sf
                    self._tactile_normal_mean_r = nm
                    self._tactile_shear_mean_r = sm
                else:
                    self._tactile_normal_force_l = nf
                    self._tactile_shear_force_l = sf
                    self._tactile_normal_mean_l = nm
                    self._tactile_shear_mean_l = sm

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        prev_r = self.actions["right_hand"].clone()
        prev_l = self.actions["left_hand"].clone()
        self.actions = {k: torch.clamp(v.to(device=self.device), -1.0, 1.0) for k, v in actions.items()}
        self._action_rate_right = self.actions["right_hand"] - prev_r
        self._action_rate_left = self.actions["left_hand"] - prev_l

    def _update_tactile_arm(self, arm: str) -> None:
        nf_buf = self._tactile_normal_force_r if arm == "right" else self._tactile_normal_force_l
        if nf_buf is None:
            return
        norm_list, shear_list = [], []
        for name in TACTILE_SENSOR_NAMES:
            key = self._tactile_key(arm, name)
            if key not in self.scene.sensors:
                continue
            data = self.scene[key].data
            if getattr(data, "tactile_normal_force", None) is not None:
                norm_list.append(data.tactile_normal_force)
            if getattr(data, "tactile_shear_force", None) is not None:
                shear_list.append(data.tactile_shear_force.view(self.num_envs, -1))
        if len(norm_list) == self._num_tactile_sensors and len(shear_list) == self._num_tactile_sensors:
            cat_n = torch.cat(norm_list, dim=1)
            cat_s = torch.cat(shear_list, dim=1)
            if arm == "right":
                self._tactile_normal_force_r = cat_n
                self._tactile_shear_force_r = cat_s
            else:
                self._tactile_normal_force_l = cat_n
                self._tactile_shear_force_l = cat_s
            n_means, s_means = [], []
            for name in TACTILE_SENSOR_NAMES:
                key = self._tactile_key(arm, name)
                data = self.scene[key].data
                nf = data.tactile_normal_force
                sf = data.tactile_shear_force
                n_means.append(nf.mean(dim=1, keepdim=True))
                s_means.append(sf.mean(dim=1))
            nm = torch.cat(n_means, dim=1)
            sm = torch.cat(s_means, dim=1)
            if arm == "right":
                self._tactile_normal_mean_r = nm
                self._tactile_shear_mean_r = sm
            else:
                self._tactile_normal_mean_l = nm
                self._tactile_shear_mean_l = sm

    def _update_tactile_data(self) -> None:
        self._update_tactile_arm("right")
        self._update_tactile_arm("left")

    def _compute_intermediate_values(self, advance_goal_timer: bool = True):
        self.cup_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.water_pos = self.water.data.root_pos_w - self.scene.env_origins
        self.target_pos = self.target.data.root_pos_w - self.scene.env_origins
        eq = self.cup.data.sim_element_quat_w[:, 0, :]
        self.cup_rot = eq / eq.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if advance_goal_timer:
            self._goal_time_left_s -= float(self.step_dt)
            resample_ids = torch.nonzero(self._goal_time_left_s <= 0.0, as_tuple=False).squeeze(-1)
            if resample_ids.numel() > 0:
                self._resample_goals(resample_ids)

        goal_w = self.goal_cup_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_w, self.goal_cup_rot)

    def _resample_goals(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        gx = sample_uniform(
            self.cfg.goal_cup_pos_x_range[0],
            self.cfg.goal_cup_pos_x_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        gy = sample_uniform(
            self.cfg.goal_cup_pos_y_range[0],
            self.cfg.goal_cup_pos_y_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        gz = sample_uniform(
            self.cfg.goal_cup_pos_z_range[0],
            self.cfg.goal_cup_pos_z_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        self.goal_cup_pos[env_ids] = torch.cat([gx, gy, gz], dim=1)
        tmin, tmax = self.cfg.goal_resample_time_range_s
        self._goal_time_left_s[env_ids] = sample_uniform(float(tmin), float(tmax), (len(env_ids),), device=self.device)
        self._sync_target_pose_to_goal(env_ids)

    def _tactile_zeros_for_arm(self) -> list[torch.Tensor]:
        """Match live tactile tensor shapes when sensors are disabled (``enable_cameras=False``)."""

        if self.cfg.use_full_tactile_obs:
            return [
                torch.zeros((self.num_envs, TACTILE_NORMAL_DIM), device=self.device),
                torch.zeros((self.num_envs, TACTILE_SHEAR_DIM), device=self.device),
            ]
        return [
            torch.zeros((self.num_envs, 5), device=self.device),
            torch.zeros((self.num_envs, 10), device=self.device),
        ]

    def _obs_one_arm(self, arm: str) -> torch.Tensor:
        if arm == "right":
            hand = self.right_hand.data.joint_pos
            hand_vel = self.right_hand.data.joint_vel
            action_rate = self._action_rate_right
            act = self.actions["right_hand"]
        else:
            hand = self.left_hand.data.joint_pos
            hand_vel = self.left_hand.data.joint_vel
            action_rate = self._action_rate_left
            act = self.actions["left_hand"]

        pos_err = self.goal_cup_pos - self.cup_pos
        elems = [
            _unscale(hand, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self._act_idx_t],
            self.cfg.vel_obs_scale * hand_vel[:, self._act_idx_t],
            self.cup_pos,
            self.cup_rot,
            self.goal_cup_pos,
            pos_err,
            act,
            action_rate,
        ]
        if self._tactile_normal_force_r is not None:
            if self.cfg.use_full_tactile_obs:
                if arm == "right":
                    elems.append(self._tactile_normal_force_r)
                    elems.append(self._tactile_shear_force_r)
                else:
                    elems.append(self._tactile_normal_force_l)
                    elems.append(self._tactile_shear_force_l)
            else:
                if arm == "right":
                    elems.append(self._tactile_normal_mean_r)
                    elems.append(self._tactile_shear_mean_r)
                else:
                    elems.append(self._tactile_normal_mean_l)
                    elems.append(self._tactile_shear_mean_l)
        else:
            elems.extend(self._tactile_zeros_for_arm())
        return torch.cat(elems, dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._compute_intermediate_values(advance_goal_timer=False)
        if self._tactile_normal_force_r is not None:
            self._update_tactile_data()
        return {
            "right_hand": self._obs_one_arm("right"),
            "left_hand": self._obs_one_arm("left"),
        }

    def _get_states(self) -> torch.Tensor:
        obs = self._get_observations()
        return torch.cat([obs["right_hand"], obs["left_hand"]], dim=-1)

    def _shared_task_reward(self) -> torch.Tensor:
        split = int(self.cfg.pour_phase_split_step)
        p1 = self.episode_length_buf <= split

        pos_err = self.goal_cup_pos - self.cup_pos
        pos_dist = torch.norm(pos_err, p=2, dim=-1)
        pos_rew = (1.0 - torch.tanh(pos_dist / (self.cfg.pos_tracking_std + 1e-6))) * self.cfg.pos_tracking_weight

        rot_dist = _rotation_distance(self.cup_rot, self.goal_cup_rot)
        rot_rew = (1.0 - torch.tanh(rot_dist / (self.cfg.rot_tracking_std + 1e-6))) * self.cfg.rot_tracking_weight

        action_l2_r = torch.sum(self.actions["right_hand"] ** 2, dim=-1) * self.cfg.action_l2_weight
        action_l2_l = torch.sum(self.actions["left_hand"] ** 2, dim=-1) * self.cfg.action_l2_weight
        action_l2 = action_l2_r + action_l2_l

        action_rate_l2 = (
            torch.sum(self._action_rate_right**2, dim=-1) + torch.sum(self._action_rate_left**2, dim=-1)
        ) * self.cfg.action_rate_l2_weight

        reward = pos_rew + torch.where(p1, rot_rew, torch.zeros_like(rot_rew)) + action_l2 + action_rate_l2

        water_cup_dist = torch.norm(self.water_pos - self.cup_pos, dim=-1)

        wt_xy = torch.norm((self.water_pos - self.target_pos)[:, :2], dim=-1)
        wt_z = torch.abs(self.water_pos[:, 2] - self.target_pos[:, 2])
        pour_xy = (
            (1.0 - torch.tanh(wt_xy / (self.cfg.phase2_water_target_xy_std + 1e-6)))
            * self.cfg.phase2_water_target_xy_weight
        )
        pour_z = (
            (1.0 - torch.tanh(wt_z / (self.cfg.phase2_water_target_z_std + 1e-6)))
            * self.cfg.phase2_water_target_z_weight
        )
        reward = reward + torch.where(~p1, pour_xy + pour_z, torch.zeros_like(reward))

        pos_ok = pos_dist <= self.cfg.success_pos_tol
        rot_ok = torch.abs(rot_dist) <= self.cfg.success_rot_tol
        succ_track = torch.where(p1, pos_ok & rot_ok, pos_ok)
        succ_water = (wt_xy <= self.cfg.phase2_success_water_xy_tol) & (wt_z <= self.cfg.phase2_success_water_z_tol)
        pour_full = succ_track & succ_water
        success_mask = succ_track & (p1 | succ_water)
        reward = torch.where(success_mask, reward + self.cfg.success_weight, reward)
        self.successes = torch.where(success_mask, torch.ones_like(self.successes), self.successes)

        self.pour_water_in_bowl = torch.where(pour_full, torch.ones_like(self.pour_water_in_bowl), self.pour_water_in_bowl)
        self._pour_success_streak = torch.where(
            pour_full, self._pour_success_streak + 1, torch.zeros_like(self._pour_success_streak)
        )

        self._success_streak = torch.where(success_mask, self._success_streak + 1, torch.zeros_like(self._success_streak))
        sm = self._success_streak.float().mean()
        alpha = float(self.cfg.success_ema_alpha)
        self.consecutive_successes.mul_(1.0 - alpha).add_(sm * alpha)

        water_far = p1 & (water_cup_dist > self.cfg.phase1_water_far_penalty_start_dist)
        reward = torch.where(water_far, reward + self.cfg.phase1_water_far_penalty, reward)
        wscale = float(self.cfg.phase1_water_far_reward_scale)
        reward = torch.where(water_far, reward * wscale, reward)

        fall_mask = self.cup_pos[:, 2] < self.cfg.fall_height
        reward = torch.where(fall_mask, reward + self.cfg.fall_penalty, reward)

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["episode_success_rate"] = float(self._episode_success_rate_ema)
        self.extras["log"]["episode_success_rate_all_time"] = self.get_episode_success_rate()
        self.extras["log"]["pour_phase1_frac"] = p1.float().mean()
        self.extras["log"]["water_cup_dist_mean"] = water_cup_dist.mean()
        self.extras["log"]["pos_dist_mean"] = pos_dist.mean()
        self.extras["log"]["pour_water_in_bowl_frac"] = self.pour_water_in_bowl.mean()
        self.extras["log"]["pour_success_streak_mean"] = self._pour_success_streak.float().mean()

        return reward

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        r = self._shared_task_reward()
        return {"right_hand": r, "left_hand": r}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()
        fall_mask = self.cup_pos[:, 2] < self.cfg.fall_height
        oob_cup = self._out_of_bounds(self.cup_pos)
        oob_water = self._water_out_of_bounds(self.water_pos)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        split = int(self.cfg.pour_phase_split_step)
        p1 = self.episode_length_buf <= split
        water_cup_dist = torch.norm(self.water_pos - self.cup_pos, dim=-1)
        cup_goal_dist = torch.norm(self.goal_cup_pos - self.cup_pos, dim=-1)
        phase1_fail = p1 & (water_cup_dist > self.cfg.phase1_water_cup_max_dist)
        phase2_fail = (~p1) & (cup_goal_dist > self.cfg.phase2_cup_goal_max_dist)

        terminated = fall_mask #| oob_cup | oob_water | phase1_fail | phase2_fail
        term_d = {a: terminated for a in self.cfg.possible_agents}
        tout_d = {a: time_out for a in self.cfg.possible_agents}
        return term_d, tout_d

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() > 0:
            self._accumulate_episode_success_stats(env_ids)

        super()._reset_idx(env_ids)

        gx = sample_uniform(
            self.cfg.goal_cup_pos_x_range[0],
            self.cfg.goal_cup_pos_x_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        gy = sample_uniform(
            self.cfg.goal_cup_pos_y_range[0],
            self.cfg.goal_cup_pos_y_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        gz = sample_uniform(
            self.cfg.goal_cup_pos_z_range[0],
            self.cfg.goal_cup_pos_z_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        self.goal_cup_pos[env_ids] = torch.cat([gx, gy, gz], dim=1)
        self.goal_cup_rot[env_ids] = torch.tensor(
            self.cfg.goal_cup_rot_reset_wxyz, device=self.device, dtype=torch.float
        ).expand(len(env_ids), -1)
        tmin, tmax = self.cfg.goal_resample_time_range_s
        self._goal_time_left_s[env_ids] = sample_uniform(float(tmin), float(tmax), (len(env_ids),), device=self.device)

        self._sync_target_pose_to_goal(env_ids)

        cup_state = self.cup.data.default_nodal_state_w.clone()[env_ids]
        water_state = self.water.data.default_nodal_state_w.clone()[env_ids]
        if self.cfg.reset_cup_pos_noise > 0.0:
            n = env_ids.shape[0]
            noise_xy = sample_uniform(-1.0, 1.0, (n, 1, 2), device=self.device) * float(self.cfg.reset_cup_pos_noise)
            delta = torch.zeros((n, 1, 3), device=self.device)
            delta[:, :, 0:2] = noise_xy
            cup_state[:, :, 0:3] = cup_state[:, :, 0:3] + delta
            water_state[:, :, 0:3] = water_state[:, :, 0:3] + delta
        self.cup.write_nodal_state_to_sim(cup_state, env_ids)
        self.water.write_nodal_state_to_sim(water_state, env_ids)

        delta_max = self.robot_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        dof_pos_noise = sample_uniform(
            self.cfg.robot_reset_dof_pos_offset_range[0],
            self.cfg.robot_reset_dof_pos_offset_range[1],
            (len(env_ids), self.num_robot_dofs),
            device=self.device,
        )
        rand_delta = saturate(dof_pos_noise, delta_min, delta_max)
        dof_pos_r = self.right_hand.data.default_joint_pos[env_ids] + rand_delta
        dof_vel_r = sample_uniform(
            self.cfg.robot_reset_dof_vel_range[0],
            self.cfg.robot_reset_dof_vel_range[1],
            (len(env_ids), self.num_robot_dofs),
            device=self.device,
        )
        self.right_hand_prev_targets[env_ids] = dof_pos_r
        self.right_hand_curr_targets[env_ids] = dof_pos_r
        self.right_hand.set_joint_position_target(dof_pos_r, env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(dof_pos_r, dof_vel_r, env_ids=env_ids)

        delta_max = self.robot_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        dof_pos_noise = sample_uniform(
            self.cfg.robot_reset_dof_pos_offset_range[0],
            self.cfg.robot_reset_dof_pos_offset_range[1],
            (len(env_ids), self.num_robot_dofs),
            device=self.device,
        )
        rand_delta = saturate(dof_pos_noise, delta_min, delta_max)
        dof_pos_l = self.left_hand.data.default_joint_pos[env_ids] + rand_delta
        dof_vel_l = sample_uniform(
            self.cfg.robot_reset_dof_vel_range[0],
            self.cfg.robot_reset_dof_vel_range[1],
            (len(env_ids), self.num_robot_dofs),
            device=self.device,
        )
        self.left_hand_prev_targets[env_ids] = dof_pos_l
        self.left_hand_curr_targets[env_ids] = dof_pos_l
        self.left_hand.set_joint_position_target(dof_pos_l, env_ids=env_ids)
        self.left_hand.write_joint_state_to_sim(dof_pos_l, dof_vel_l, env_ids=env_ids)

        self.successes[env_ids] = 0.0
        self._success_streak[env_ids] = 0
        self.pour_water_in_bowl[env_ids] = 0.0
        self._pour_success_streak[env_ids] = 0

        self._compute_intermediate_values(advance_goal_timer=False)

        e = env_ids
        if e.numel() > 0:
            self.actions["right_hand"][e] = 0.0
            self.actions["left_hand"][e] = 0.0

    def _apply_action(self) -> None:
        for side, hand, prev, curr in (
            ("right", self.right_hand, self.right_hand_prev_targets, self.right_hand_curr_targets),
            ("left", self.left_hand, self.left_hand_prev_targets, self.left_hand_curr_targets),
        ):
            act = self.actions["right_hand" if side == "right" else "left_hand"]
            curr[:, self.actuated_dof_indices] = scale(
                act,
                self.act_dof_lower_limits,
                self.act_dof_upper_limits,
            )
            curr[:, self.actuated_dof_indices] = (
                self.cfg.act_moving_average * curr[:, self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * prev[:, self.actuated_dof_indices]
            )
            curr[:, self.actuated_dof_indices] = saturate(
                curr[:, self.actuated_dof_indices],
                self.act_dof_lower_limits,
                self.act_dof_upper_limits,
            )
            prev[:, self.actuated_dof_indices] = curr[:, self.actuated_dof_indices]
            hand.set_joint_position_target(curr[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower
