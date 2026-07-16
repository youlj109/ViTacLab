from collections.abc import Sequence
from pathlib import Path
import re

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .ur10e_shadowhand_direct_base_cfg import (
    UR10eShadowHandTacSLSceneCfg,
    build_ur10e_shadowhand_tactile_sensor_cfgs,
    build_ur10e_shadowhand_third_person_camera_cfg,
)


@torch.jit.script
def _scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def _tacsl_to_batched_flat(t: torch.Tensor, num_envs: int) -> torch.Tensor:
    """TacSL tensors may be (N, F) or (N, F, 1); return a flattened (N, *)."""
    if t.ndim <= 1:
        return t.reshape(num_envs, -1)
    if t.ndim == 2:
        return t
    return t.reshape(num_envs, -1)


# Names must match ``build_ur10e_shadowhand_tactile_sensor_cfgs`` keys.
_TACSL_SENSOR_NAMES: tuple[str, ...] = (
    "tactile_sensor_ff",
    "tactile_sensor_lf",
    "tactile_sensor_mf",
    "tactile_sensor_rf",
    "tactile_sensor_th",
)


def spawn_factory_table(prim_path: str = "/World/envs/env_.*/Table") -> None:
    """Spawn the same Seattle lab table as Factory/Forge."""
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func(
        prim_path,
        cfg,
        translation=(0.55, 0.0, 0.0),
        orientation=(0.70711, 0.0, 0.0, 0.70711),
    )


def _resolve_asset_usd_path(usd_path: str) -> str:
    """Resolve repo-relative USD paths (e.g. source/ViTacLab/...) to an absolute file path."""
    raw = str(usd_path or "").strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    candidates: list[Path] = [Path.cwd() / p]
    cur = Path.cwd()
    for _ in range(12):
        if (cur / "source" / "ViTacLab").is_dir():
            candidates.append(cur / p)
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return raw


def spawn_high_fidelity_scene_if_enabled(cfg) -> None:
    """Optionally spawn a high-fidelity scene USD under each env root."""
    if not getattr(cfg, "enable_high_fidelity_scene", False):
        return

    usd_path = getattr(cfg, "high_fidelity_scene_usd_path", "")
    if not usd_path:
        return

    resolved_usd = _resolve_asset_usd_path(usd_path)
    prim_path = getattr(cfg, "high_fidelity_scene_prim_path", "/World/envs/env_.*/HighFidelityScene")
    translation = getattr(cfg, "high_fidelity_scene_translation", (0.0, 0.0, 0.0))
    orientation = getattr(cfg, "high_fidelity_scene_orientation", (1.0, 0.0, 0.0, 0.0))
    scale = getattr(cfg, "high_fidelity_scene_scale", (1.0, 1.0, 1.0))

    scene_spawn_cfg = sim_utils.UsdFileCfg(usd_path=resolved_usd, scale=scale)
    print(
        f"[INFO] Spawning high-fidelity scene: usd={resolved_usd} prim={prim_path} "
        f"scale={scale} translation={translation}"
    )
    scene_spawn_cfg.func(
        prim_path,
        scene_spawn_cfg,
        translation=translation,
        orientation=orientation,
    )
    print(f"[INFO] High-fidelity scene prim spawned (verify in Stage under {prim_path}).")


class UR10eShadowHandDirectBaseEnv(DirectRLEnv):
    """Base DirectRLEnv for UR10e arm + ShadowHand tasks."""

    robot: Articulation

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_robot_dofs = self.robot.num_joints
        self.prev_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros_like(self.prev_targets)

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

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_factory_table()
        spawn_high_fidelity_scene_if_enabled(self.cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.robot
        self._setup_task_scene()

        # Cameras / TacSL need rendering; skip in headless training when cfg.enable_cameras is False.
        if getattr(self.cfg, "enable_cameras", False):
            # Create third-person camera AFTER cloning environments.
            if "third_person_camera" not in self.scene.sensors:
                cam_cfg = build_ur10e_shadowhand_third_person_camera_cfg()
                self.scene.sensors["third_person_camera"] = cam_cfg.class_type(cam_cfg)

            # Create TacSL sensors AFTER cloning environments so sensor initialization sees all env prims.
            if isinstance(self.cfg.scene, UR10eShadowHandTacSLSceneCfg):
                sensor_cfgs = build_ur10e_shadowhand_tactile_sensor_cfgs(self.cfg.scene)
                for name, sensor_cfg in sensor_cfgs.items():
                    if name not in self.scene.sensors:
                        # InteractiveScene expects cfg.class_type(cfg) objects in its sensor dict.
                        self.scene.sensors[name] = sensor_cfg.class_type(sensor_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if getattr(self.cfg, "enable_cameras", False):
            self._maybe_init_tacsl_nominal_render()

    def _maybe_init_tacsl_nominal_render(self) -> None:
        """Nominal camera render for TacSL (``get_initial_render``) when ``enable_cameras`` is True.

        TacSL / Forge reference: call after ``sim.reset()`` so GPU handles and buffers are valid before the first
        ``scene.update()``. ``DirectRLEnv`` will reset again after ``_setup_scene()``; an extra reset here matches
        :class:`ForgeEnv` and avoids relying on external scripts for initialization.
        """
        if not isinstance(self.cfg.scene, UR10eShadowHandTacSLSceneCfg):
            return
        from isaaclab.sim.utils.stage import use_stage

        with use_stage(self.sim.get_initial_stage()):
            self.sim.reset()

        for name in _TACSL_SENSOR_NAMES:
            if name not in self.scene.sensors:
                continue
            sensor = self.scene[name]
            if not getattr(sensor.cfg, "enable_camera_tactile", False):
                continue
            try:
                sensor.get_initial_render()
            except Exception as e:
                print(f"[WARN] TacSL get_initial_render failed for {name}: {e}")

    def _setup_task_scene(self) -> None:
        raise NotImplementedError

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.to(device=self.device), -1.0, 1.0)

    def _apply_action(self) -> None:
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

    def _reset_robot_joints(
        self,
        env_ids: Sequence[int],
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
    ) -> None:
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.robot.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    def _resolve_body_index_by_keywords(self, keywords: Sequence[str]) -> int:
        """Resolve robot body index by fuzzy-name keyword search."""
        body_names_src = getattr(self.robot, "body_names", None)
        if body_names_src is None:
            body_names_src = getattr(self.robot.data, "body_names", [])
        body_names = [str(name).lower() for name in body_names_src]
        for key in keywords:
            try:
                return next(i for i, name in enumerate(body_names) if key in name)
            except StopIteration:
                continue
        return 0

    def _get_ee_pose_env(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return EE pose in env frame (pos) and world quaternion."""
        ee_idx = getattr(self, "_ee_body_idx", None)
        if ee_idx is None:
            ee_idx = self._resolve_body_index_by_keywords(("wrist_3", "wrist3", "hand", "palm"))
            self._ee_body_idx = ee_idx
        ee_pos_env = self.robot.data.body_pos_w[:, ee_idx] - self.scene.env_origins
        ee_quat_env = self.robot.data.body_quat_w[:, ee_idx]
        return ee_pos_env, ee_quat_env

    def _build_tactile_pose_tensor(self, sensor_names: Sequence[str], num_tactile: int) -> torch.Tensor:
        """Return tactile poses (N, num_tactile, 7) with TacSL/body fallback."""
        tactile_pos = torch.zeros((self.num_envs, num_tactile, 7), device=self.device, dtype=torch.float32)
        tactile_pos_source = "zero_no_sensor"
        if num_tactile <= 0:
            return tactile_pos

        if all(name in self.scene.sensors for name in sensor_names):
            pose_list: list[torch.Tensor] = []

            def _extract_pos_quat(sensor_obj, sensor_data):
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
                return pos_w, quat_w

            for name in sensor_names:
                sensor_obj = self.scene[name]
                sensor_data = sensor_obj.data
                pos_w, quat_w = _extract_pos_quat(sensor_obj, sensor_data)
                if pos_w is None or quat_w is None:
                    pose_list = []
                    break
                pose_list.append(torch.cat((pos_w - self.scene.env_origins, quat_w), dim=-1))

            if len(pose_list) == num_tactile:
                tactile_pos = torch.stack(pose_list, dim=1)
                tactile_pos_source = "tacsl_pose"
            else:
                body_names_src = getattr(self.robot, "body_names", None)
                if body_names_src is None:
                    body_names_src = getattr(self.robot.data, "body_names", [])
                body_names = [str(name).lower() for name in body_names_src]
                body_idx_list: list[int] = []
                key_map = {
                    "ff": ("ffdistal", "ff_tip", "ff"),
                    "lf": ("lfdistal", "lf_tip", "lf"),
                    "mf": ("mfdistal", "mf_tip", "mf"),
                    "rf": ("rfdistal", "rf_tip", "rf"),
                    "th": ("thdistal", "th_tip", "th"),
                }
                for sensor_name in sensor_names:
                    finger_key = str(sensor_name).split("_")[-1].lower()
                    search_keys = key_map.get(finger_key, (finger_key,))
                    idx_found = None
                    for sk in search_keys:
                        try:
                            idx_found = next(i for i, bname in enumerate(body_names) if sk in bname)
                            break
                        except StopIteration:
                            continue
                    if idx_found is None:
                        body_idx_list = []
                        break
                    body_idx_list.append(idx_found)

                if len(body_idx_list) == num_tactile:
                    pose_list = []
                    for body_idx in body_idx_list:
                        pose_list.append(
                            torch.cat(
                                (
                                    self.robot.data.body_pos_w[:, body_idx] - self.scene.env_origins,
                                    self.robot.data.body_quat_w[:, body_idx],
                                ),
                                dim=-1,
                            )
                        )
                    tactile_pos = torch.stack(pose_list, dim=1)
                    tactile_pos_source = "robot_body_fallback"
                else:
                    tactile_pos_source = "zero_no_body_match"

        if not getattr(self, "_printed_tactile_pos_source_once", False):
            print(f"[{self.__class__.__name__}] tactile_pos source: {tactile_pos_source}")
            self._printed_tactile_pos_source_once = True
        return tactile_pos

    def _append_camera_record(self, record_dict: dict, camera_name: str, rgb_key: str, depth_key: str, pose_key: str) -> None:
        """Append RGB/depth/pose entries from a camera sensor to record dict."""
        if camera_name not in self.scene.sensors:
            return
        camera = self.scene[camera_name]
        cam_out = camera.data.output
        if "rgb" in cam_out:
            record_dict[rgb_key] = cam_out["rgb"].detach().cpu()
        if "distance_to_image_plane" in cam_out:
            record_dict[depth_key] = cam_out["distance_to_image_plane"].detach().cpu()
        cam_pos_env = camera.data.pos_w - self.scene.env_origins
        cam_quat_w = getattr(camera.data, "quat_w_ros", None)
        if cam_quat_w is None:
            cam_quat_w = getattr(camera.data, "quat_w", None)
        if cam_quat_w is not None:
            record_dict[pose_key] = torch.cat((cam_pos_env, cam_quat_w), dim=-1).unsqueeze(1).detach().cpu()

    def _build_pickup_style_record_dict(
        self,
        *,
        joint_pos: torch.Tensor,
        tactile_sensor_names: Sequence[str],
        tactile_sensor_count: int,
        tactile_normal_force: torch.Tensor | None = None,
        tactile_shear_force: torch.Tensor | None = None,
        tactile_rgb_image: torch.Tensor | None = None,
        tactile_array_size: tuple[int, int] | None = None,
        tactile_image_hw: tuple[int, int] | None = None,
    ) -> dict:
        """Build pickup-v1 compatible ``record`` payload."""
        ee_pos_env, ee_quat_env = self._get_ee_pose_env()
        tactile_pos = self._build_tactile_pose_tensor(tuple(tactile_sensor_names), int(tactile_sensor_count))

        record_dict: dict = {
            "joint_pos": joint_pos.detach().cpu(),
            "tactile_pos": tactile_pos.detach().cpu(),
            "ee_pos_env": ee_pos_env.detach().cpu(),
            "ee_quat_env": ee_quat_env.detach().cpu(),
        }
        if (
            tactile_sensor_count > 0
            and tactile_normal_force is not None
            and tactile_shear_force is not None
            and tactile_array_size is not None
        ):
            h, w = int(tactile_array_size[0]), int(tactile_array_size[1])
            record_dict["tactile_normal_force"] = tactile_normal_force.detach().cpu().reshape(
                self.num_envs, tactile_sensor_count, h, w, 1
            )
            record_dict["tactile_shear_force"] = tactile_shear_force.detach().cpu().reshape(
                self.num_envs, tactile_sensor_count, h, w, 2
            )
            if tactile_rgb_image is not None and tactile_image_hw is not None:
                img_h, img_w = int(tactile_image_hw[0]), int(tactile_image_hw[1])
                record_dict["tactile_rgb_image"] = (
                    tactile_rgb_image.detach().cpu().reshape(self.num_envs, tactile_sensor_count, img_h, img_w, 3) * 255.0
                ).to(torch.uint8)

        self._append_camera_record(
            record_dict,
            camera_name="third_person_camera",
            rgb_key="third_person_camera",
            depth_key="third_person_camera_depth",
            pose_key="third_person_camera_pos",
        )
        self._append_camera_record(
            record_dict,
            camera_name="twist_camera",
            rgb_key="twist_camera",
            depth_key="twist_camera_depth",
            pose_key="twist_camera_pos",
        )
        return record_dict

