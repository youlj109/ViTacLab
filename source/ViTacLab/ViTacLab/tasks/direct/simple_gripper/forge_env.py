# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor import VisuoTactileSensor
from . import forge_utils
from .forge_env_cfg import ForgeEnvCfg


def _pick_random_usd_from_dir(dir_path: str) -> str | None:
    """Return a random USD file path from the given directory, or None if none found."""
    if not dir_path or not os.path.isdir(dir_path):
        return None
    exts = (".usd", ".usda")
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(exts)]
    if not files:
        return None
    name = random.choice(files)
    return os.path.join(dir_path, name)


def _apply_random_objects(cfg: ForgeEnvCfg) -> None:
    """Set held_asset and fixed_asset USD paths: random from folders when randomize_objects is True, else defaults."""
    scene = getattr(cfg, "scene", None)
    if scene is None:
        return

    def set_asset_usd_path(asset_name: str, path: str) -> bool:
        if not path:
            return False
        asset_cfg = getattr(scene, asset_name, None)
        if asset_cfg is None:
            return False
        spawn = getattr(asset_cfg, "spawn", None)
        if spawn is None or not hasattr(spawn, "usd_path"):
            return False
        spawn.usd_path = path
        return True

    if getattr(cfg, "randomize_objects", False):
        held_dir = getattr(cfg, "random_objects_usd_dir", "") or ""
        if held_dir:
            path = _pick_random_usd_from_dir(held_dir)
            set_asset_usd_path("held_asset", path or "")
        fixed_dir = getattr(cfg, "random_fixed_objects_usd_dir", "") or ""
        if fixed_dir:
            path = _pick_random_usd_from_dir(fixed_dir)
            set_asset_usd_path("fixed_asset", path or "")
    else:
        default_held = getattr(cfg, "default_held_asset_usd_path", "") or ""
        if default_held:
            set_asset_usd_path("held_asset", default_held)
        default_fixed = getattr(cfg, "default_fixed_asset_usd_path", "") or ""
        if default_fixed:
            set_asset_usd_path("fixed_asset", default_fixed)


class ForgeEnv(FactoryEnv):
    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        # Object randomization: pick random USD(s) from folder(s) before scene is built
        _apply_random_objects(cfg)

        # Update obs_order and state_order based on observation mode
        # This must be done BEFORE calling super().__init__() because FactoryEnv.__init__()
        # uses these to calculate observation_space and state_space dimensions
        if cfg.obs_mode == "reduce":
            # Reduce mode: exclude tactile information
            cfg.obs_order = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "force_threshold"]
            cfg.state_order = [
                "fingertip_pos",
                "fingertip_quat",
                "ee_linvel",
                "ee_angvel",
                "joint_pos",
                "held_pos",
                "held_pos_rel_fixed",
                "held_quat",
                "fixed_pos",
                "fixed_quat",
                "force_threshold",
            ]
        elif cfg.obs_mode == "full":
            # Full mode: include all tactile information
            cfg.obs_order = [
                "fingertip_pos_rel_fixed",
                "fingertip_quat",
                "ee_linvel",
                "ee_angvel",
                "force_threshold",
                "tactile_normal_force",
                "tactile_shear_force",
                "tactile_rgb_image",
            ]
            cfg.state_order = [
                "fingertip_pos",
                "fingertip_quat",
                "ee_linvel",
                "ee_angvel",
                "joint_pos",
                "held_pos",
                "held_pos_rel_fixed",
                "held_quat",
                "fixed_pos",
                "fixed_quat",
                "force_threshold",
                "tactile_normal_force",
                "tactile_shear_force",
                "tactile_rgb_image",
            ]
        else:
            raise ValueError(f"Unknown obs_mode: {cfg.obs_mode}. Must be 'reduce' or 'full'.")
        
        super().__init__(cfg, render_mode, **kwargs)
        print(f"num_envs: {self.num_envs}")
        print(f"Observation mode: {cfg.obs_mode}")
        # Success prediction.
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Flip quaternions.
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Tactile sensor information.
        # Get tactile sensor array size from scene config
        self.tactile_array_size = self.cfg.scene.tactile_sensor_left.tactile_array_size
        self.tactile_array_total = self.tactile_array_size[0] * self.tactile_array_size[1]  # 20 * 25 = 500
        
        # Get RGB image size from scene config
        self.tactile_image_height = self.cfg.scene.tactile_sensor_left.render_cfg.image_height  # 240
        self.tactile_image_width = self.cfg.scene.tactile_sensor_left.render_cfg.image_width  # 320
        self.tactile_image_channels = 3  # RGB
        self.tactile_image_total = self.tactile_image_height * self.tactile_image_width * self.tactile_image_channels  # 240 * 320 * 3 = 230400
        
        # Initialize tactile sensor data buffers
        self.tactile_normal_force = torch.zeros(
            (self.num_envs, 2 * self.tactile_array_total), device=self.device
        )
        self.tactile_shear_force = torch.zeros(
            (self.num_envs, 2 * self.tactile_array_total * 2), device=self.device
        )
        self.tactile_rgb_image = torch.zeros(
            (self.num_envs, 2 * self.tactile_image_total), device=self.device
        )

        # Set nominal dynamics parameters for randomization.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

    def _setup_scene(self):
        """Setup scene - tactile sensors are automatically created from ForgeSceneCfg."""
        # Call parent setup first - this will create the scene with tactile sensors from ForgeSceneCfg
        super()._setup_scene()
        
        # Initialize nominal tactile render for camera-based tactile sensing
        # This must be called after sim.reset() but before the first scene.update()
        # According to reference code, get_initial_render() should be called after sim.reset()
        # DirectRLEnv.__init__() calls sim.reset() after _setup_scene(), so we need to call
        # sim.reset() here first, then get_initial_render(), before DirectRLEnv calls scene.update()
        import builtins
        from isaaclab.sim.utils.stage import use_stage
        
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            # Reset simulation to activate physics handles
            with use_stage(self.sim.get_initial_stage()):
                self.sim.reset()
            
            # Get initial render for tactile sensors
            if "tactile_sensor_left" in self.scene.sensors:
                tactile_sensor_left = self.scene["tactile_sensor_left"]
                if tactile_sensor_left.cfg.enable_camera_tactile:
                    tactile_sensor_left.get_initial_render()
            
            if "tactile_sensor_right" in self.scene.sensors:
                tactile_sensor_right = self.scene["tactile_sensor_right"]
                if tactile_sensor_right.cfg.enable_camera_tactile:
                    tactile_sensor_right.get_initial_render()

    def _compute_intermediate_values(self, dt):
        """Add noise to observations for force sensing."""
        super()._compute_intermediate_values(dt)

        # Add noise to fingertip position.
        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy fingertip positions.
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        # Get tactile sensor data from both sensors using VisuoTactileSensorData
        tactile_data_left: VisuoTactileSensorData = self.scene["tactile_sensor_left"].data
        tactile_data_right: VisuoTactileSensorData = self.scene["tactile_sensor_right"].data

        # Combine tactile forces from both sensors
        # According to VisuoTactileSensorData:
        # - tactile_normal_force: (num_envs, nrows * ncols) - already flattened
        # - tactile_shear_force: (num_envs, nrows * ncols, 2)
        if tactile_data_left.tactile_normal_force is not None and tactile_data_right.tactile_normal_force is not None:
            # Combine normal forces from both sensors: (num_envs, 2 * tactile_array_total)
            self.tactile_normal_force = torch.cat(
                [
                    tactile_data_left.tactile_normal_force,
                    tactile_data_right.tactile_normal_force,
                ],
                dim=1,
            )
        else:
            self.tactile_normal_force = torch.zeros(
                (self.num_envs, 2 * self.tactile_array_total), device=self.device
            )

        if tactile_data_left.tactile_shear_force is not None and tactile_data_right.tactile_shear_force is not None:
            # Combine shear forces from both sensors
            # Flatten each sensor's shear force from (num_envs, tactile_array_total, 2) to (num_envs, tactile_array_total * 2)
            # Then concatenate: (num_envs, 2 * tactile_array_total * 2)
            self.tactile_shear_force = torch.cat(
                [
                    tactile_data_left.tactile_shear_force.view(self.num_envs, -1),
                    tactile_data_right.tactile_shear_force.view(self.num_envs, -1),
                ],
                dim=1,
            )
        else:
            self.tactile_shear_force = torch.zeros(
                (self.num_envs, 2 * self.tactile_array_total * 2), device=self.device
            )

        # Get tactile RGB images from both sensors
        # According to VisuoTactileSensorData:
        # - tactile_rgb_image: (num_envs, height, width, 3) or None
        if tactile_data_left.tactile_rgb_image is not None and tactile_data_right.tactile_rgb_image is not None:
            # Flatten RGB images from (num_envs, height, width, 3) to (num_envs, height * width * 3)
            # Then concatenate: (num_envs, 2 * height * width * 3)
            self.tactile_rgb_image = torch.cat(
                [
                    tactile_data_left.tactile_rgb_image.view(self.num_envs, -1),
                    tactile_data_right.tactile_rgb_image.view(self.num_envs, -1),
                ],
                dim=1,
            )
            # Normalize to [0, 1] if needed (assuming images are in [0, 255] range)
            if self.tactile_rgb_image.max() > 1.0:
                self.tactile_rgb_image = self.tactile_rgb_image / 255.0
        else:
            self.tactile_rgb_image = torch.zeros(
                (self.num_envs, 2 * self.tactile_image_total), device=self.device
            )

        # Apply visual disturbance to third_person_camera images if enabled
        if self.cfg.visual_disturbance and "third_person_camera" in self.scene.sensors:
            self._apply_visual_disturbance()

    def _apply_visual_disturbance(self):
        """Apply Gaussian noise or Gaussian blur to third_person_camera RGB output."""
        cam = self.scene["third_person_camera"]
        if "rgb" not in cam.data.output:
            return
        rgb = cam.data.output["rgb"]  # (num_envs, height, width, 3)
        if rgb is None or rgb.numel() == 0:
            return
        dtype = rgb.dtype
        device = rgb.device
        # Normalize to [0, 1] for processing if needed
        need_denorm = rgb.max() > 1.0
        if need_denorm:
            img = rgb.float() / 255.0
        else:
            img = rgb.float()

        if self.cfg.visual_disturbance_type == "gaussian_noise":
            std = self.cfg.visual_disturbance_noise_std
            img = img + std * torch.randn_like(img, device=device, dtype=img.dtype)
        elif self.cfg.visual_disturbance_type == "gaussian_blur":
            k = self.cfg.visual_disturbance_blur_kernel_size
            sigma = self.cfg.visual_disturbance_blur_sigma
            if k % 2 == 0:
                k += 1
            # Build 2D Gaussian kernel (same for all channels)
            x = torch.arange(k, device=device, dtype=img.dtype) - k // 2
            g = torch.exp(-(x**2) / (2 * sigma**2))
            g = g / g.sum()
            kernel_2d = (g.unsqueeze(0) * g.unsqueeze(1)).reshape(1, 1, k, k)
            N, H, W, C = img.shape
            img = img.permute(0, 3, 1, 2)  # (N, C, H, W)
            kernel = kernel_2d.expand(C, 1, k, k)  # (C, 1, k, k)
            img = F.conv2d(img, kernel, padding=k // 2, groups=C)
            img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
        else:
            return

        img = torch.clamp(img, 0.0, 1.0)
        if need_denorm:
            rgb.copy_(img.mul(255.0).to(dtype))
        else:
            rgb.copy_(img.to(dtype))

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        # Base observations (always included)
        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        # Add tactile information only in full mode
        if self.cfg.obs_mode == "full":
            obs_dict.update(
                {
                    "tactile_normal_force": self.tactile_normal_force,
                    "tactile_shear_force": self.tactile_shear_force,
                    "tactile_rgb_image": self.tactile_rgb_image,
                }
            )

        # State dictionary
        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        # Add tactile information to state only in full mode
        if self.cfg.obs_mode == "full":
            state_dict.update(
                {
                    "tactile_normal_force": self.tactile_normal_force,
                    "tactile_shear_force": self.tactile_shear_force,
                    "tactile_rgb_image": self.tactile_rgb_image,
                }
            )

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees
        # (so that angles appear on a continuous span uninterrupted by the joint limit)
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Use same base rewards as Factory.
        rew_buf = super()._get_rewards()

        rew_dict, rew_scales = {}, {}
        # Calculate action penalty for the asset-relative action space.
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]
        # Contact penalty based on tactile sensor forces.
        # Compute total contact force magnitude from tactile sensors
        # Sum of absolute normal forces across all tactile points
        total_normal_force = torch.sum(torch.abs(self.tactile_normal_force), dim=-1)
        # Sum of shear force magnitudes across all tactile points
        # Reshape shear force from (num_envs, 2 * tactile_array_total * 2) to (num_envs, 2 * tactile_array_total, 2)
        # Then compute L2 norm for each point and sum
        shear_force_reshaped = self.tactile_shear_force.view(self.num_envs, 2 * self.tactile_array_total, 2)
        total_shear_force = torch.sum(torch.norm(shear_force_reshaped, p=2, dim=-1), dim=-1)
        contact_force = total_normal_force + total_shear_force
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        # Add success prediction rewards.
        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # rescale from [-1, 1] to [0, 1]
        success_pred_error = (true_successes.float() - policy_success_pred).abs()
        # Delay success prediction penalty until some successes have occurred.
        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        # Add new FORGE reward terms.
        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
        }
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)

        # Compute initial action for correct EMA computation.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        # Set initial gains for the episode.
        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reset_buffers(self, env_ids):
        """Reset additional logging metrics."""
        super()._reset_buffers(env_ids)
        # Reset success pred metrics.
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            # Only log at the end.
            if torch.any(self.reset_buf):
                # Log prediction delay.
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                # Log early-term success rate (for all episodes we have "stopped", did we succeed?).
                pred_success_idxs = first_success_tx != 0  # Episodes which we have predicted success.

                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,  # Success has actually occurred.
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],  # Success occurred before we predicted it.
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
