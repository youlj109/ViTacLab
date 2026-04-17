"""UR10e + ShadowHand Factory peg / gear / nut (Factory assets/rewards/randomization; sim/control like hand_pickup)."""

from __future__ import annotations

from collections.abc import Sequence

import carb
import torch
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate

from isaaclab_tasks.direct.factory import factory_utils

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10eShadowHandTacSLSceneCfg,
    build_ur10e_shadowhand_tactile_sensor_cfgs,
    build_ur10e_shadowhand_third_person_camera_cfg,
)
from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    UR10eShadowHandDirectBaseEnv,
    _scale,
    spawn_factory_table,
)

from .ur10e_shadowhand_forge_env_cfg import UR10eShadowHandForgeEnvCfg


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


# Joint position [lower, upper] -> [-1, 1] for policy observations (inverse of :func:`_scale`).
@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class UR10eShadowHandForgeEnv(UR10eShadowHandDirectBaseEnv):
    """Factory forge tasks with UR10e + Shadow Hand (joint-space control)."""

    cfg: UR10eShadowHandForgeEnvCfg

    def __init__(self, cfg: UR10eShadowHandForgeEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_task = cfg.task
        if not getattr(cfg, "enable_cameras", False):
            cfg.use_full_tactile_obs = False
        tactile_dim = (TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM) if cfg.use_full_tactile_obs else (5 * (1 + 2))
        base_obs_dim = 30 + 30 + 3 + 4 + 3 + 4 + 1 + 30 + 30
        cfg.observation_space = base_obs_dim + tactile_dim

        super().__init__(cfg, render_mode, **kwargs)

        factory_utils.set_body_inertias(self.robot, self.scene.num_envs)
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self.robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

        self._ee_body_idx = self.robot.body_names.index(self.cfg.ee_body_name)

        arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_expr)
        if len(arm_ids) != 6:
            raise RuntimeError(f"Forge dexhand: expected 6 arm joints, got {len(arm_ids)}")
        self._arm_joint_ids = [int(i) for i in arm_ids]

        self._identity_quat_w = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        self._prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._action_rate = torch.zeros_like(self._prev_actions)

        self._tactile_normal_force: torch.Tensor | None = None
        self._tactile_shear_force: torch.Tensor | None = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0
        self._tactile_normal_mean: torch.Tensor | None = None
        self._tactile_shear_mean: torch.Tensor | None = None

        self._keypoint_dist = torch.zeros((self.num_envs,), device=self.device)
        self._held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._held_base_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self._target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_held_base_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        self.last_update_timestamp = 0.0

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_factory_table()

        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        if getattr(self.cfg, "enable_cameras", False):
            if "third_person_camera" not in self.scene.sensors:
                cam_cfg = build_ur10e_shadowhand_third_person_camera_cfg()
                self.scene.sensors["third_person_camera"] = cam_cfg.class_type(cam_cfg)

            if isinstance(self.cfg.scene, UR10eShadowHandTacSLSceneCfg):
                sensor_cfgs = build_ur10e_shadowhand_tactile_sensor_cfgs(self.cfg.scene)
                for name, sensor_cfg in sensor_cfgs.items():
                    if name not in self.scene.sensors:
                        self.scene.sensors[name] = sensor_cfg.class_type(sensor_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if getattr(self.cfg, "enable_cameras", False):
            self._maybe_init_tacsl_nominal_render()

    def _setup_task_scene(self) -> None:
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

    def _wrist_pose_env(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
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

        self.fingertip_midpoint_pos, self.fingertip_midpoint_quat = self._wrist_pose_env()

        self._held_base_pos, self._held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
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

        self.last_update_timestamp = self.robot._data._sim_timestamp

        self._update_tactile()

    def _update_tactile(self) -> None:
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

    def _compute_keypoint_reward_terms(self, curr_successes: torch.Tensor) -> tuple[dict, dict]:
        keypoint_dist = self._keypoint_dist
        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        action_penalty_ee = torch.norm(self.actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self._action_rate, p=2, dim=-1)
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
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions[:] = self.actions
        self.actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        self._action_rate[:] = self.actions - self._prev_actions

    def _apply_action(self) -> None:
        # Must match ``UR10eShadowHandDirectBaseEnv`` / teleop: ``joint = _scale(action, lo, hi)`` for actions in [-1, 1].
        joint_ids = self.actuated_dof_indices
        self.cur_targets[:, joint_ids] = _scale(
            self.actions,
            self.robot_dof_lower_limits[:, joint_ids],
            self.robot_dof_upper_limits[:, joint_ids],
        )
        self.cur_targets[:, joint_ids] = (
            self.cfg.act_moving_average * self.cur_targets[:, joint_ids]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, joint_ids]
        )
        self.cur_targets[:, joint_ids] = saturate(
            self.cur_targets[:, joint_ids],
            self.robot_dof_lower_limits[:, joint_ids],
            self.robot_dof_upper_limits[:, joint_ids],
        )
        self.prev_targets[:, joint_ids] = self.cur_targets[:, joint_ids]
        self.robot.set_joint_position_target(self.cur_targets[:, joint_ids], joint_ids=joint_ids)

    def _get_observations(self) -> dict:
        if self.last_update_timestamp < self.robot._data._sim_timestamp:
            self._compute_intermediate_values()
        elif self._tactile_normal_force is not None:
            self._update_tactile()

        wrist_p = self.fingertip_midpoint_pos
        wrist_q = self.fingertip_midpoint_quat
        noisy_fixed = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        held_rel = self.held_pos - noisy_fixed
        wrist_rel = wrist_p - noisy_fixed

        obs_elems = [
            _unscale(
                self.robot.data.joint_pos[:, self.actuated_dof_indices],
                self.robot_dof_lower_limits[:, self.actuated_dof_indices],
                self.robot_dof_upper_limits[:, self.actuated_dof_indices],
            ),
            self.cfg.vel_obs_scale * self.robot.data.joint_vel[:, self.actuated_dof_indices],
            held_rel,
            self.held_quat,
            wrist_rel,
            wrist_q,
            self._keypoint_dist.unsqueeze(-1),
            self.actions,
            self._action_rate,
        ]

        if self.cfg.use_full_tactile_obs and self._tactile_normal_force is not None:
            obs_elems.append(self._tactile_normal_force)
            obs_elems.append(self._tactile_shear_force)
        elif self._tactile_normal_mean is not None:
            obs_elems.append(self._tactile_normal_mean)
            obs_elems.append(self._tactile_shear_mean)
        else:
            td = TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM if self.cfg.use_full_tactile_obs else (5 * (1 + 2))
            obs_elems.append(torch.zeros((self.num_envs, td), device=self.device))

        obs = torch.cat(obs_elems, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        if self.last_update_timestamp < self.robot._data._sim_timestamp:
            self._compute_intermediate_values()
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        rew_dict, rew_scales = self._compute_keypoint_reward_terms(curr_successes)
        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf = rew_buf + rew * rew_scales[rew_name]

        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.last_update_timestamp < self.robot._data._sim_timestamp:
            self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def step_sim_no_action(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def _write_art_root_pose(
        self,
        articulation: Articulation,
        env_ids: torch.Tensor,
        pos_env: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        n = len(env_ids)
        eo = self.scene.env_origins[env_ids]
        pos = torch.tensor(pos_env, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        quat = torch.tensor(quat_wxyz, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        st = articulation.data.default_root_state.clone()[env_ids]
        st[:, 0:3] = eo + pos
        st[:, 3:7] = quat
        st[:, 7:] = 0.0
        articulation.write_root_pose_to_sim(st[:, 0:7], env_ids=env_ids)
        articulation.write_root_velocity_to_sim(st[:, 7:], env_ids=env_ids)
        articulation.reset()

    def _set_robot_default_reset_pose(self, env_ids: torch.Tensor) -> None:
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(jp)
        self.robot.set_joint_position_target(jp, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        self.robot.reset()
        self.step_sim_no_action()

    def randomize_initial_state(self, env_ids: torch.Tensor) -> None:
        """Place task articulations from :attr:`cfg.object_poses` (env frame); obs ref from ``fixed_obs_ref_pos``."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
        op = self.cfg.object_poses

        self._write_art_root_pose(self._fixed_asset, env_ids, op.fixed_pos, op.fixed_quat)
        self._write_art_root_pose(self._held_asset, env_ids, op.held_pos, op.held_quat)

        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            sp = op.small_gear_pos if op.small_gear_pos is not None else op.fixed_pos
            sq = op.small_gear_quat if op.small_gear_quat is not None else op.fixed_quat
            lp = op.large_gear_pos if op.large_gear_pos is not None else op.fixed_pos
            lq = op.large_gear_quat if op.large_gear_quat is not None else op.fixed_quat
            self._write_art_root_pose(self._small_gear_asset, env_ids, sp, sq)
            self._write_art_root_pose(self._large_gear_asset, env_ids, lp, lq)

        n = len(env_ids)
        ref = torch.tensor(self.cfg.fixed_obs_ref_pos, dtype=torch.float32, device=self.device)
        self.fixed_pos_obs_frame[env_ids] = ref.unsqueeze(0).expand(n, -1)
        std = torch.tensor(self.cfg.fixed_obs_noise_std, dtype=torch.float32, device=self.device)
        self.init_fixed_pos_obs_noise[env_ids] = torch.randn(n, 3, device=self.device) * std.unsqueeze(0)

        self.step_sim_no_action()

        grasp_time = 0.0
        hand_ids = [i for i in self.actuated_dof_indices if i not in self._arm_joint_ids]
        hand_ids_t = torch.tensor(hand_ids, device=self.device, dtype=torch.long)
        closed_tgt = None
        if hand_ids:
            lo = self.robot_dof_lower_limits[:, hand_ids_t]
            hi = self.robot_dof_upper_limits[:, hand_ids_t]
            t = float(self.cfg.reset_grasp_hand_joint_lerp)
            closed_tgt = lo + t * (hi - lo)
        while grasp_time < 0.25:
            if closed_tgt is not None:
                jp = self.robot.data.joint_pos.clone()
                jp[:, hand_ids_t] = closed_tgt
                self.robot.set_joint_position_target(jp)
                self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.actions.zero_()
        self._prev_actions.zero_()

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            super()._reset_idx(slice(None))
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super()._reset_idx(env_ids)

        self._set_robot_default_reset_pose(env_ids_t)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids_t)

        self.prev_targets[env_ids_t] = self.robot.data.joint_pos[env_ids_t]
        self.cur_targets[env_ids_t] = self.robot.data.joint_pos[env_ids_t]

        self.ep_succeeded[env_ids_t] = 0
        self.ep_success_times[env_ids_t] = 0
