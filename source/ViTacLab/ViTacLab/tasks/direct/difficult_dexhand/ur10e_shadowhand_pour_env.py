"""UR10e + ShadowHand(Left) deformable cup pouring (aligned with :mod:`hand_pickup` patterns).

- **Two phases** (see ``pour_phase_split_step``): **goal position tracking + action penalties every step; rotation
  tracking vs ``goal_cup_rot`` only in phase 1** (phase 2 ignores cup–goal rotation for reward/success bonus).
  * Before split: fixed water–cup distance penalty (like ``fall_penalty``) when dist exceeds a threshold; hard fail
    beyond ``phase1_water_cup_max_dist``.
  * After split: extra water→bowl shaping; fail if ``‖goal_cup_pos−cup‖`` exceeds ``phase2_cup_goal_max_dist``.
- Policy obs (same layout as pickup): joint state, cup pose, goal, pos err, actions, action_rate; optional tactile.
- Logging: ``consecutive_successes`` (pose bonus steps), ``episode_success_rate`` (EMA / all-time) counts **episodes
  with at least one (pose + water-in-bowl) pour success**; see ``pour_water_in_bowl`` / ``_pour_success_streak``.
- Terminations: fall, cup OOB, water OOB, phase failures, timeout.
"""

from __future__ import annotations

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

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    UR10eShadowHandDirectBaseEnv,
)

from .ur10e_shadowhand_pour_env_cfg import UR10eShadowHandPourEnvCfg


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


class UR10eShadowHandPourEnv(UR10eShadowHandDirectBaseEnv):
    """UR10e + ShadowHand pour task (deformable cup + water + rigid target)."""

    cfg: UR10eShadowHandPourEnvCfg

    def __init__(self, cfg: UR10eShadowHandPourEnvCfg, render_mode: str | None = None, **kwargs):
        tactile_dim = (TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM) if cfg.use_full_tactile_obs else (5 * (1 + 2))
        # Match :class:`UR10eShadowHandPickupEnv`: 30+30 + obj(3+4) + goal(3) + pos_err(3) + act(30) + act_rate(30)
        base_obs_dim = 30 + 30 + 3 + 4 + 3 + 3 + 30 + 30
        cfg.observation_space = base_obs_dim + tactile_dim

        super().__init__(cfg, render_mode, **kwargs)

        # Spawn orientation for rigid target (bowl); position follows ``goal_cup_pos`` via cfg offset.
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

        self._prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._action_rate = torch.zeros_like(self._prev_actions)

        self._tactile_normal_force: torch.Tensor | None = None
        self._tactile_shear_force: torch.Tensor | None = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0
        self._tactile_normal_mean: torch.Tensor | None = None
        self._tactile_shear_mean: torch.Tensor | None = None

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Sticky + streak for **pour** success (pose + water in bowl) — used for episode success rate.
        self.pour_water_in_bowl = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._pour_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self._episode_success_count = 0
        self._episode_total_count = 0
        self._episode_success_rate_ema: float = 0.0

    def _sync_target_pose_to_goal(self, env_ids: torch.Tensor) -> None:
        """Place rigid ``target_cup``: xy track ``goal_cup_pos`` + offset; z fixed (see cfg)."""
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
        term = self.reset_terminated[env_ids]
        tout = self.reset_time_outs[env_ids]
        episode_done = term | tout
        if not episode_done.any():
            return
        buf = self.episode_length_buf[env_ids]
        split = int(self.cfg.pour_phase_split_step)
        # ``episode_length_buf`` is incremented before ``_get_dones``; ``<= split`` ⇒ first ``split`` steps = phase 1.
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
        # Episode "win" requires **water in bowl** (+ pose) at least once, or pour streak when max_cons>0.
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
        """Fraction of completed episodes that achieved **pour** success (pose + water in bowl) at least once."""
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

    # ------------------------------------------------------------------
    def _setup_task_scene(self) -> None:
        self.cup = DeformableObject(self.cfg.cup_cfg)
        self.water = DeformableObject(self.cfg.water_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        self.scene.deformable_objects["cup"] = self.cup
        self.scene.deformable_objects["water"] = self.water
        self.scene.rigid_objects["target_cup"] = self.target

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
        self._prev_actions[:] = self.actions
        super()._pre_physics_step(actions)
        self._action_rate[:] = self.actions - self._prev_actions

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
            if self._tactile_normal_mean is not None and self._tactile_shear_mean is not None:
                n_means, s_means = [], []
                for name in TACTILE_SENSOR_NAMES:
                    data = self.scene[name].data
                    nf = data.tactile_normal_force
                    sf = data.tactile_shear_force
                    n_means.append(nf.mean(dim=1, keepdim=True))
                    s_means.append(sf.mean(dim=1))
                self._tactile_normal_mean = torch.cat(n_means, dim=1)
                self._tactile_shear_mean = torch.cat(s_means, dim=1)

    def _compute_intermediate_values(self, advance_goal_timer: bool = True):
        """Refresh cached poses. Goal timer / resample / markers only once per env-step when ``advance_goal_timer``."""
        self.robot_dof_pos = self.robot.data.joint_pos
        self.robot_dof_vel = self.robot.data.joint_vel

        self.cup_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.water_pos = self.water.data.root_pos_w - self.scene.env_origins
        self.target_pos = self.target.data.root_pos_w - self.scene.env_origins
        # Deformable bodies have no root_quat; use first sim tet orientation as a coarse cup frame proxy.
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

    def _get_observations(self) -> dict:
        # ``_get_dones`` already advanced goal timer; only refresh kinematics for post-reset / new obs.
        self._compute_intermediate_values(advance_goal_timer=False)

        pos_err = self.goal_cup_pos - self.cup_pos

        obs_elems = [
            _unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self.actuated_dof_indices],
            self.cfg.vel_obs_scale * self.robot_dof_vel[:, self.actuated_dof_indices],
            self.cup_pos,
            self.cup_rot,
            self.goal_cup_pos,
            pos_err,
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

    def _get_rewards(self) -> torch.Tensor:
        # State computed in ``_get_dones`` (runs before rewards in ``DirectRLEnv.step``).
        split = int(self.cfg.pour_phase_split_step)
        p1 = self.episode_length_buf <= split

        pos_err = self.goal_cup_pos - self.cup_pos
        pos_dist = torch.norm(pos_err, p=2, dim=-1)
        pos_rew = (1.0 - torch.tanh(pos_dist / (self.cfg.pos_tracking_std + 1e-6))) * self.cfg.pos_tracking_weight

        rot_dist = _rotation_distance(self.cup_rot, self.goal_cup_rot)
        rot_rew = (1.0 - torch.tanh(rot_dist / (self.cfg.rot_tracking_std + 1e-6))) * self.cfg.rot_tracking_weight

        action_l2 = torch.sum(self.actions**2, dim=-1) * self.cfg.action_l2_weight
        action_rate_l2 = torch.sum(self._action_rate**2, dim=-1) * self.cfg.action_rate_l2_weight

        # Position tracking always; rotation vs goal only **phase 1** (phase 2: pour / water, not cup tilt vs goal).
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
        # Phase 1: position + rotation vs goal; phase 2: position only (ignore cup–goal rotation).
        succ_track = torch.where(p1, pos_ok & rot_ok, pos_ok)
        succ_water = (wt_xy <= self.cfg.phase2_success_water_xy_tol) & (wt_z <= self.cfg.phase2_success_water_z_tol)
        pour_full = succ_track & succ_water
        # Phase 1: goal pose (+rot). Phase 2: goal position + water near bowl (no rot gate).
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

        # Phase 1: fixed penalty (+ optional reward scale) when water drifts from cup — same pattern as ``fall_penalty``.
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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

        # First tensor: failure termination (not timeout); second: time-out only (Isaac Lab / cartpole convention).
        terminated = fall_mask | oob_cup | oob_water | phase1_fail | phase2_fail
        return terminated, time_out

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
            # xy only (no z); one shared translation for cup + water so water moves with the cup.
            noise_xy = sample_uniform(-1.0, 1.0, (n, 1, 2), device=self.device) * float(self.cfg.reset_cup_pos_noise)
            delta = torch.zeros((n, 1, 3), device=self.device)
            delta[:, :, 0:2] = noise_xy
            cup_state[:, :, 0:3] = cup_state[:, :, 0:3] + delta
            water_state[:, :, 0:3] = water_state[:, :, 0:3] + delta
        self.cup.write_nodal_state_to_sim(cup_state, env_ids)
        self.water.write_nodal_state_to_sim(water_state, env_ids)

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
        self.pour_water_in_bowl[env_ids] = 0.0
        self._pour_success_streak[env_ids] = 0

        self._compute_intermediate_values(advance_goal_timer=False)
