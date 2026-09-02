"""Direct MARL base env: dual UR10e + ShadowHand, shared table / ground / optional camera."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    spawn_factory_table,
    spawn_high_fidelity_scene_if_enabled,
)

from .ur10e_dual_shadowhand_direct_base_cfg import (
    UR10eDualShadowHandTacSLSceneCfg,
    build_ur10e_dual_shadowhand_tacsl_sensor_cfgs,
    build_ur10e_dual_shadowhand_third_person_camera_cfg,
)

# Order matches :func:`build_ur10e_dual_shadowhand_tacsl_sensor_cfgs` per arm, left then right.
_DUAL_TACSL_SENSOR_KEYS: tuple[str, ...] = tuple(
    f"{arm_prefix}{name}"
    for arm_prefix in ("left_", "right_")
    for name in (
        "tactile_sensor_ff",
        "tactile_sensor_lf",
        "tactile_sensor_mf",
        "tactile_sensor_rf",
        "tactile_sensor_th",
    )
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
        spawn_high_fidelity_scene_if_enabled(self.cfg)
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

    @staticmethod
    def _ee_body_index_for_record(robot: Articulation) -> int:
        body_names_src = getattr(robot, "body_names", None)
        if body_names_src is None:
            body_names_src = getattr(robot.data, "body_names", [])
        body_names = [str(name).lower() for name in body_names_src]
        ee_idx = 0
        for key in ("wrist_3", "wrist3", "hand", "palm"):
            try:
                ee_idx = next(i for i, name in enumerate(body_names) if key in name)
                break
            except StopIteration:
                continue
        return ee_idx

    def _get_observations(self) -> dict:
        base: dict = {}

        r_idx = getattr(self, "_record_ee_body_idx_right", None)
        if r_idx is None:
            self._record_ee_body_idx_right = self._ee_body_index_for_record(self.right_hand)
            r_idx = self._record_ee_body_idx_right
        l_idx = getattr(self, "_record_ee_body_idx_left", None)
        if l_idx is None:
            self._record_ee_body_idx_left = self._ee_body_index_for_record(self.left_hand)
            l_idx = self._record_ee_body_idx_left

        ee_pos_r = self.right_hand.data.body_pos_w[:, r_idx] - self.scene.env_origins
        ee_quat_r = self.right_hand.data.body_quat_w[:, r_idx]
        ee_pos_l = self.left_hand.data.body_pos_w[:, l_idx] - self.scene.env_origins
        ee_quat_l = self.left_hand.data.body_quat_w[:, l_idx]

        num_tactile = len(_DUAL_TACSL_SENSOR_KEYS)
        tactile_pos = torch.zeros((self.num_envs, num_tactile, 7), device=self.device, dtype=torch.float32)
        pose_list: list[torch.Tensor] = []
        for key in _DUAL_TACSL_SENSOR_KEYS:
            if key not in self.scene.sensors:
                pose_list = []
                break
            sensor_obj = self.scene[key]
            sensor_data = sensor_obj.data
            pos_w = getattr(sensor_data, "pos_w", None)
            quat_w = getattr(sensor_data, "quat_w_ros", None)
            if quat_w is None:
                quat_w = getattr(sensor_data, "quat_w", None)
            if pos_w is None:
                pos_w = getattr(sensor_obj, "pos_w", None)
            if quat_w is None:
                quat_w = getattr(sensor_obj, "quat_w_ros", None)
            if quat_w is None:
                quat_w = getattr(sensor_obj, "quat_w", None)
            if pos_w is None or quat_w is None:
                pose_list = []
                break
            pose_list.append(torch.cat((pos_w - self.scene.env_origins, quat_w), dim=-1))
        if len(pose_list) == num_tactile:
            tactile_pos = torch.stack(pose_list, dim=1)

        record_dict: dict[str, torch.Tensor] = {
            "joint_pos_right": self.right_hand.data.joint_pos.detach().cpu(),
            "joint_pos_left": self.left_hand.data.joint_pos.detach().cpu(),
            "tactile_pos": tactile_pos.detach().cpu(),
            "ee_pos_env_right": ee_pos_r.detach().cpu(),
            "ee_quat_env_right": ee_quat_r.detach().cpu(),
            "ee_pos_env_left": ee_pos_l.detach().cpu(),
            "ee_quat_env_left": ee_quat_l.detach().cpu(),
        }

        tactile_hw = (20, 25)
        if hasattr(self.cfg.scene, "_tactile_params"):
            tactile_hw = self.cfg.scene._tactile_params().get("tactile_array_size", (20, 25))

        norm_list: list[torch.Tensor] = []
        shear_list: list[torch.Tensor] = []
        rgb_list: list[torch.Tensor] = []
        for key in _DUAL_TACSL_SENSOR_KEYS:
            if key not in self.scene.sensors:
                continue
            data = self.scene[key].data
            nf = getattr(data, "tactile_normal_force", None)
            sf = getattr(data, "tactile_shear_force", None)
            rgb = getattr(data, "tactile_rgb_image", None)
            if nf is not None:
                norm_list.append(nf.reshape(self.num_envs, tactile_hw[0], tactile_hw[1], 1))
            if sf is not None:
                shear_list.append(sf.reshape(self.num_envs, tactile_hw[0], tactile_hw[1], 2))
            if rgb is not None:
                rgb_u8 = torch.clamp(rgb * 255.0, 0.0, 255.0).to(torch.uint8)
                rgb_list.append(rgb_u8)
        if len(norm_list) == num_tactile:
            record_dict["tactile_normal_force"] = torch.stack(norm_list, dim=1).detach().cpu()
        if len(shear_list) == num_tactile:
            record_dict["tactile_shear_force"] = torch.stack(shear_list, dim=1).detach().cpu()
        if len(rgb_list) == num_tactile:
            record_dict["tactile_rgb_image"] = torch.stack(rgb_list, dim=1).detach().cpu()

        if "third_person_camera" in self.scene.sensors:
            cam = self.scene["third_person_camera"]
            cam_out = cam.data.output
            rgb = cam_out.get("rgb", None)
            if rgb is not None:
                record_dict["third_person_camera"] = rgb.detach().cpu()
            cam_quat_world = getattr(cam.data, "quat_w_world", None)
            if cam_quat_world is None:
                cam_quat_world = getattr(cam.data, "quat_w", None)
            cam_quat_ros = getattr(cam.data, "quat_w_ros", None)
            if cam_quat_ros is not None:
                record_dict["third_person_camera_pos"] = (
                    torch.cat(((cam.data.pos_w - self.scene.env_origins), cam_quat_ros), dim=-1)
                    .unsqueeze(1)
                    .detach()
                    .cpu()
                )

        if "twist_camera" in self.scene.sensors:
            cam = self.scene["twist_camera"]
            cam_out = cam.data.output
            rgb = cam_out.get("rgb", None)
            if rgb is not None:
                record_dict["twist_camera"] = rgb.detach().cpu()
            cam_quat_world = getattr(cam.data, "quat_w_world", None)
            if cam_quat_world is None:
                cam_quat_world = getattr(cam.data, "quat_w", None)
            cam_quat_ros = getattr(cam.data, "quat_w_ros", None)
            if cam_quat_ros is not None:
                record_dict["twist_camera_pos"] = (
                    torch.cat(((cam.data.pos_w - self.scene.env_origins), cam_quat_ros), dim=-1)
                    .unsqueeze(1)
                    .detach()
                    .cpu()
                )

        base["record"] = record_dict
        return base

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
