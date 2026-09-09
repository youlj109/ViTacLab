# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Forge gripper shear validation: W100 procedural weight grasped by parallel GelSight fingers."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject

from ViTacLab.tasks.direct.simple_gripper.forge_env import ForgeEnv
from ViTacLab.tasks.direct.vitacsim_validation.gripper_shear_env_cfg import GripperShearForgeTask


class GripperShearValidationEnv(ForgeEnv):
    """Factory grasp pipeline with validation W100 rigid body instead of peg HeldAsset."""

    cfg: object

    def _setup_scene(self):
        super()._setup_scene()
        weight_cfg = getattr(self.cfg, "_weight_rigid_cfg")
        self._validation_weight = RigidObject(weight_cfg)
        self.scene.rigid_objects["contact_object"] = self._validation_weight

    def randomize_initial_state(self, env_ids):
        """Run Factory reset but place W100 in gripper and hide peg articulation."""
        super().randomize_initial_state(env_ids)
        if not hasattr(self, "_validation_weight"):
            return

        peg_pos = self._held_asset.data.root_pos_w.clone()
        peg_quat = self._held_asset.data.root_quat_w.clone()
        peg_vel = torch.zeros_like(self._held_asset.data.root_lin_vel_w)

        w_state = self._validation_weight.data.default_root_state.clone()
        w_state[:, 0:3] = peg_pos
        w_state[:, 3:7] = peg_quat
        w_state[:, 7:] = 0.0
        self._validation_weight.write_root_pose_to_sim(w_state[:, 0:7], env_ids=None)
        self._validation_weight.write_root_velocity_to_sim(w_state[:, 7:], env_ids=None)
        self._validation_weight.reset()

        hide = self._held_asset.data.default_root_state.clone()
        hide[:, 2] -= 10.0
        hide[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(hide[:, 0:7], env_ids=None)
        self._held_asset.write_root_velocity_to_sim(hide[:, 7:], env_ids=None)

        # Re-close gripper on W100.
        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        for name in ("tactile_sensor_left", "tactile_sensor_right"):
            if name in self.scene.sensors:
                sensor = self.scene[name]
                if sensor.cfg.enable_camera_tactile:
                    sensor.get_initial_render()
