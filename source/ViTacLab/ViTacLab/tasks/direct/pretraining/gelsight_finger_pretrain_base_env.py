# Copyright (c) 2022-2026, The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Base DirectRLEnv: GelSight short finger + contact rigid + one TacSL sensor."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils.stage import use_stage

from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from .gelsight_finger_pretrain_base_cfg import (
    GelsightFingerPretrainSceneCfg,
    build_gelsight_finger_tacsl_sensor_cfg,
    format_tacsl_cfg_paths,
)

TACTILE_SENSOR_NAME: str = "tactile_sensor"
TACTILE_POINTS_PER_SENSOR: int = 20 * 25
TACTILE_NORMAL_DIM: int = TACTILE_POINTS_PER_SENSOR
TACTILE_SHEAR_DIM: int = TACTILE_POINTS_PER_SENSOR * 2


class GelsightFingerPretrainBaseEnv(DirectRLEnv):
    """Spawns finger USD, ground, optional rigid object, TacSL after env clone."""

    robot: Articulation

    def _refresh_articulation_physx_views_after_tacsl(self) -> None:
        """Recreate articulation PhysX views after TacSL init (see ``tacsl_sensor_gelsight_finger_short``)."""

        for _name, art in self.scene.articulations.items():
            if not art.is_initialized:
                continue
            art._invalidate_initialize_callback(None)
            art._initialize_impl()
            art._is_initialized = True

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.robot

        self._setup_task_rigid_object()

        if getattr(self.cfg, "enable_cameras", True):
            sensor_cfg = build_gelsight_finger_tacsl_sensor_cfg(self.cfg.scene)
            format_tacsl_cfg_paths(sensor_cfg, self.scene.env_regex_ns)
            self.scene.sensors[TACTILE_SENSOR_NAME] = sensor_cfg.class_type(sensor_cfg)

        self._refresh_articulation_physx_views_after_tacsl()

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        if getattr(self.cfg, "enable_cameras", True):
            self._maybe_init_tacsl_nominal_render()

        self._init_tactile_buffers()

    def _setup_task_rigid_object(self) -> None:
        """Override in subclass to set ``self.contact_object`` and register ``scene.rigid_objects``."""

        raise NotImplementedError

    def _maybe_init_tacsl_nominal_render(self) -> None:
        if not isinstance(self.cfg.scene, GelsightFingerPretrainSceneCfg):
            return
        with use_stage(self.sim.get_initial_stage()):
            self.sim.reset()

        if TACTILE_SENSOR_NAME not in self.scene.sensors:
            return
        sensor = self.scene[TACTILE_SENSOR_NAME]
        if not getattr(sensor.cfg, "enable_camera_tactile", False):
            return
        try:
            sensor.get_initial_render()
        except Exception:
            pass

    def _init_tactile_buffers(self) -> None:
        self._tactile_normal_force: torch.Tensor | None = None
        self._tactile_shear_force: torch.Tensor | None = None
        self._tactile_normal_mean: torch.Tensor | None = None
        self._tactile_shear_mean: torch.Tensor | None = None

        if TACTILE_SENSOR_NAME not in self.scene.sensors or VisuoTactileSensorData is None:
            return

        first = self.scene[TACTILE_SENSOR_NAME]
        sz = first.cfg.tactile_array_size
        n_pts = sz[0] * sz[1]
        self._tactile_normal_force = torch.zeros((self.num_envs, n_pts), device=self.device)
        self._tactile_shear_force = torch.zeros((self.num_envs, n_pts * 2), device=self.device)
        self._tactile_normal_mean = torch.zeros((self.num_envs, 1), device=self.device)
        self._tactile_shear_mean = torch.zeros((self.num_envs, 2), device=self.device)

    def _update_tactile_data(self) -> None:
        if self._tactile_normal_force is None or TACTILE_SENSOR_NAME not in self.scene.sensors:
            return
        data = self.scene[TACTILE_SENSOR_NAME].data
        if getattr(data, "tactile_normal_force", None) is None or getattr(data, "tactile_shear_force", None) is None:
            return
        nf = data.tactile_normal_force
        sf = data.tactile_shear_force.view(self.num_envs, -1)
        self._tactile_normal_force = nf
        self._tactile_shear_force = sf
        if self._tactile_normal_mean is not None and self._tactile_shear_mean is not None:
            sf_hw = data.tactile_shear_force
            self._tactile_normal_mean = nf.mean(dim=1, keepdim=True)
            self._tactile_shear_mean = sf_hw.mean(dim=1)
