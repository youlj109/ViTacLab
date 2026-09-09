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

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

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
        self._expected_tactile_sensor_names = (TACTILE_SENSOR_NAME,)
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

    def _build_record_dict(self) -> dict[str, torch.Tensor]:
        """Return the canonical single-GelSight pose/force/RGB record schema.

        Compact or dense tactile policy observations remain controlled by
        ``use_full_tactile_obs`` in each pretraining task.  This method is an
        independent data/acceptance path and therefore does not change the
        registered observation dimension.
        """

        self._update_tactile_data()
        record: dict[str, torch.Tensor] = {
            "joint_pos": self.robot.data.joint_pos.detach().cpu(),
        }

        sensor = self.scene.sensors.get(TACTILE_SENSOR_NAME)
        pose = None
        if sensor is not None:
            data = sensor.data
            pos_w = getattr(data, "pos_w", None)
            quat_w = getattr(data, "quat_w_ros", None)
            if quat_w is None:
                quat_w = getattr(data, "quat_w", None)
            if pos_w is not None and quat_w is not None:
                pose = torch.cat((pos_w - self.scene.env_origins, quat_w), dim=-1)
        if pose is None:
            # TacSL V2 does not expose a standard pose field on every build.
            # The short-finger articulation's final rigid body is the closest
            # stable physical pose source for the elastomer.
            body_idx = max(0, len(self.robot.body_names) - 1)
            pose = torch.cat(
                (
                    self.robot.data.body_pos_w[:, body_idx] - self.scene.env_origins,
                    self.robot.data.body_quat_w[:, body_idx],
                ),
                dim=-1,
            )
        record["tactile_pos"] = pose.unsqueeze(1).detach().cpu()

        if sensor is None:
            return record

        data = sensor.data
        height, width = tuple(sensor.cfg.tactile_array_size)
        normal = getattr(data, "tactile_normal_force", None)
        shear = getattr(data, "tactile_shear_force", None)
        rgb = getattr(data, "tactile_rgb_image", None)
        if normal is not None:
            record["tactile_normal_force"] = (
                normal.reshape(self.num_envs, 1, height, width, 1).detach().cpu()
            )
        if shear is not None:
            record["tactile_shear_force"] = (
                shear.reshape(self.num_envs, 1, height, width, 2).detach().cpu()
            )
        if rgb is not None:
            rgb_u8 = rgb.detach()
            if rgb_u8.dtype != torch.uint8:
                if rgb_u8.numel() and float(rgb_u8.max().item()) <= 1.5:
                    rgb_u8 = rgb_u8 * 255.0
                rgb_u8 = torch.clamp(rgb_u8, 0.0, 255.0).to(torch.uint8)
            image_h = int(sensor.cfg.render_cfg.image_height)
            image_w = int(sensor.cfg.render_cfg.image_width)
            if rgb_u8.ndim == 2 and rgb_u8.shape[1] == image_h * image_w * 3:
                rgb_u8 = rgb_u8.reshape(self.num_envs, image_h, image_w, 3)
            record["tactile_rgb_image"] = rgb_u8.unsqueeze(1).cpu()
        return record
