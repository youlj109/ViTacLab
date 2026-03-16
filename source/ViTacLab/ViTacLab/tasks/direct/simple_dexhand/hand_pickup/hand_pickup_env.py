from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import re

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    sample_uniform,
    saturate,
)

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from .hand_pickup_env_cfg import (
    UR10eShadowHandPickupEnvCfg,
    UR10eShadowHandPickupSceneCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs.ui import ViewerCfg


TACTILE_SENSOR_NAMES = ("tactile_sensor_ff", "tactile_sensor_lf", "tactile_sensor_mf", "tactile_sensor_rf", "tactile_sensor_th")


@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class UR10eShadowHandPickupEnv(DirectRLEnv):
    """UR10e + ShadowHand rigid-object pickup environment with tactile sensing."""

    cfg: UR10eShadowHandPickupEnvCfg

    def __init__(self, cfg: UR10eShadowHandPickupEnvCfg, render_mode: str | None = None, **kwargs):
        # approximate observation dimension (robot + object + goal + actions + tactile)
        base_obs_dim = 0
        base_obs_dim += 64  # robot DOF positions / velocities
        base_obs_dim += 16  # object pose
        base_obs_dim += 8   # goal pose
        base_obs_dim += 32  # actions
        # simple tactile summary: per-sensor normal/shear mean
        tactile_dim = len(TACTILE_SENSOR_NAMES) * (1 + 2)
        full_obs_dim = base_obs_dim + tactile_dim
        cfg.observation_space = full_obs_dim

        super().__init__(cfg, render_mode, **kwargs)

        # joint indices and buffers
        self.num_robot_dofs = self.robot.num_joints
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device
        )
        self.prev_targets = torch.zeros_like(self.robot_dof_targets)
        self.cur_targets = torch.zeros_like(self.robot_dof_targets)

        # selectable DOFs (UR10 arm + ShadowHand fingers)
        self.actuated_dof_indices: list[int] = []
        for i, name in enumerate(self.robot.joint_names):
            if re.match(self.cfg.arm_joint_expr, name) or re.match(self.cfg.hand_joint_expr, name):
                self.actuated_dof_indices.append(i)
        if not self.actuated_dof_indices:
            self.actuated_dof_indices = list(range(self.num_robot_dofs))
        self.actuated_dof_indices.sort()
        self.num_actions = len(self.actuated_dof_indices)

        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]

        # object / goal buffers
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # goal directly above initial object position in XY
        self.goal_object_pos = torch.tensor(
            [2, 0.5, self.cfg.goal_height], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.goal_object_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

        # tactile summary buffers
        self._tactile_normal_summary: torch.Tensor | None = None
        self._tactile_shear_summary: torch.Tensor | None = None

        # success stats
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # ---------------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------------

    def _setup_scene(self):
        # robot + object
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        # ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone envs
        self.scene.clone_environments(copy_from_source=False)

        # register entities
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        # simple dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------------
    # RL API
    # ---------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.to(device=self.device)
        actions = torch.clamp(actions, -1.0, 1.0)

        # map actions to all joints
        self.actions = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device)
        self.actions[:, self.actuated_dof_indices] = actions

        # PD target in joint space
        delta = 0.05 * self.actions
        self.prev_targets[:] = self.cur_targets
        self.cur_targets[:] = self.cur_targets + delta
        self.cur_targets = torch.clamp(self.cur_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        self.robot_dof_targets[:] = self.cur_targets
        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        obs_elems = [
            _unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits),
            self.robot_dof_vel,
            self.object_pos,
            self.object_rot,
            self.goal_object_pos,
            self.goal_object_rot,
            self.actions[:, self.actuated_dof_indices],
        ]

        # simple tactile summary: per-sensor mean normal / shear magnitude
        if self._tactile_normal_summary is not None and self._tactile_shear_summary is not None:
            obs_elems.append(self._tactile_normal_summary)
            obs_elems.append(self._tactile_shear_summary)

        obs = torch.cat(obs_elems, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # lift height reward
        height_error = self.goal_object_pos[:, 2] - self.object_pos[:, 2]
        height_rew = -torch.abs(height_error) * self.cfg.lift_reward_scale

        # xy distance penalty
        xy_error = self.object_pos[:, 0:2] - self.goal_object_pos[:, 0:2]
        xy_dist = torch.norm(xy_error, p=2, dim=-1)
        xy_rew = -xy_dist * self.cfg.lift_reward_scale

        action_penalty = torch.sum(self.actions**2, dim=-1) * self.cfg.action_penalty_scale

        reward = height_rew + xy_rew + action_penalty

        # success bonus
        success_mask = (
            torch.abs(height_error) <= self.cfg.success_tolerance_height
        ) & (xy_dist <= self.cfg.success_tolerance_xy)
        reward = torch.where(success_mask, reward + self.cfg.success_bonus, reward)

        self.successes = torch.where(success_mask, torch.ones_like(self.successes), self.successes)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminate if object falls below table or timeout
        fall_mask = self.object_pos[:, 2] < self.cfg.fall_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = fall_mask | time_out
        return done, time_out

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # reset object pose near initial spawn with small noise
        obj_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        obj_state[:, 0:3] = (
            obj_state[:, 0:3]
            + 0.01 * pos_noise
            + self.scene.env_origins[env_ids]
        )
        obj_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

        # reset robot joints around default with small noise
        delta_max = self.robot_dof_upper_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.robot.data.default_joint_pos[env_ids] + 0.5 * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        dof_vel = self.robot.data.default_joint_vel[env_ids] + 0.5 * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.robot_dof_targets[env_ids] = dof_pos

        self.robot.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

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

        # object pose
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w

        # update goal marker (one sphere per env)
        goal_pos = self.goal_object_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_object_rot)


