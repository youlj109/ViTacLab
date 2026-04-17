"""Direct MARL base env: dual UR10e + ShadowHand, shared table / ground / optional camera."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    spawn_factory_table,
)

from .ur10e_dual_shadowhand_direct_base_cfg import (
    UR10eDualShadowHandTacSLSceneCfg,
    build_ur10e_dual_shadowhand_tacsl_sensor_cfgs,
    build_ur10e_dual_shadowhand_third_person_camera_cfg,
)


class UR10eDualShadowHandDirectMARLBaseEnv(DirectMARLEnv):
    """Spawns two UR10e+ShadowHand articulations, Seattle lab table, and ground (after clone).

    Subclasses implement :meth:`_setup_task_scene` to attach task rigid bodies / markers once
    the articulated roots are replicated.
    """

    right_hand: Articulation
    left_hand: Articulation

    def _setup_scene(self) -> None:
        self.right_hand = Articulation(self.cfg.right_robot_cfg)
        self.left_hand = Articulation(self.cfg.left_robot_cfg)

        spawn_factory_table()
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["right_robot"] = self.right_hand
        self.scene.articulations["left_robot"] = self.left_hand

        if getattr(self.cfg, "enable_cameras", False):
            if getattr(self.cfg, "enable_third_person_camera", True) and "third_person_camera" not in self.scene.sensors:
                cam_cfg = build_ur10e_dual_shadowhand_third_person_camera_cfg()
                self.scene.sensors["third_person_camera"] = cam_cfg.class_type(cam_cfg)

            if isinstance(self.cfg.scene, UR10eDualShadowHandTacSLSceneCfg):
                for arm_prefix, prim_name in (("left_", "LeftRobot"), ("right_", "RightRobot")):
                    sensor_cfgs = build_ur10e_dual_shadowhand_tacsl_sensor_cfgs(
                        self.cfg.scene, robot_prim_name=prim_name
                    )
                    for name, sensor_cfg in sensor_cfgs.items():
                        key = f"{arm_prefix}{name}"
                        if key not in self.scene.sensors:
                            self.scene.sensors[key] = sensor_cfg.class_type(sensor_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._setup_task_scene()

        if getattr(self.cfg, "enable_cameras", False) and isinstance(
            self.cfg.scene, UR10eDualShadowHandTacSLSceneCfg
        ):
            self._maybe_init_tacsl_nominal_render()

    def _setup_task_scene(self) -> None:
        """Override to register rigid objects / task-specific scene nodes."""
        pass

    def _maybe_init_tacsl_nominal_render(self) -> None:
        from isaaclab.sim.utils.stage import use_stage

        with use_stage(self.sim.get_initial_stage()):
            self.sim.reset()

        for name in self.scene.sensors:
            if "tactile_sensor" not in name:
                continue
            sensor = self.scene[name]
            if not getattr(sensor.cfg, "enable_camera_tactile", False):
                continue
            try:
                sensor.get_initial_render()
            except Exception:
                pass
