"""Canonical ViTacLab module for UR10e + ShadowHand direct base environments.

Maintained stack for RL training, data collection, and policy inference.
Uses local VisuoTactileSensorV2 (PhysX sparse anchors + depth-camera TacSL).
"""

from collections.abc import Sequence
from pathlib import Path
import re

import isaaclab.sim as sim_utils
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

try:
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData
except ImportError:
    VisuoTactileSensorData = None  # type: ignore

from .ur10e_shadowhand_direct_base_cfg import (
    UR10E_SHADOWHAND_TACTILE_SENSOR_NAMES,
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
        # TacSL nominal backgrounds must be captured only after DirectRLEnv has
        # completed its normal scene/EventManager/simulation-start lifecycle.
        self._tacsl_nominal_render_initialized = False
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
        self._use_rl_control = True

        self._shadow_render_sensors_enabled = bool(getattr(self.cfg, "enable_cameras", False))
        self._ur10e_stacked_tacsl_names = self._resolve_ur10e_stacked_tacsl_sensor_names()
        self._init_ur10e_stacked_tacsl_buffers()
        self._maybe_init_tacsl_nominal_render()

    def _maybe_init_tacsl_nominal_render(self) -> None:
        """Capture TacSL nominal backgrounds once, after the normal simulation start."""

        if self._tacsl_nominal_render_initialized or not self._shadow_render_sensors_enabled:
            return

        for name in self._ur10e_stacked_tacsl_names:
            if name not in self.scene.sensors:
                continue
            tactile = self.scene[name]
            if getattr(tactile.cfg, "enable_camera_tactile", False):
                try:
                    tactile.get_initial_render()
                except Exception as e:
                    print(f"[WARN] TacSL get_initial_render failed for {name}: {e}")
        self._tacsl_nominal_render_initialized = True

    def _initialize_deferred_tacsl_nominal_render(self) -> None:
        """Backward-compatible alias for the canonical deferred initializer."""

        self._maybe_init_tacsl_nominal_render()

    def _resolve_ur10e_stacked_tacsl_sensor_names(self) -> tuple[str, ...]:
        """Ordered TacSL sensor keys (matches :func:`build_ur10e_shadowhand_tactile_sensor_cfgs`)."""
        scene_has_tactile = isinstance(self.cfg.scene, UR10eShadowHandTacSLSceneCfg) or hasattr(
            type(self.cfg.scene), "_tactile_params"
        )
        if scene_has_tactile:
            return tuple(build_ur10e_shadowhand_tactile_sensor_cfgs(self.cfg.scene).keys())
        return UR10E_SHADOWHAND_TACTILE_SENSOR_NAMES

    def _init_ur10e_stacked_tacsl_buffers(self) -> None:
        """Forge-style stacked tactile tensors (``tactile_*``); safe alongside task-specific buffers."""
        names = self._ur10e_stacked_tacsl_names
        self._ur10e_stacked_n = len(names)
        n = self._ur10e_stacked_n

        try:
            tactile_hw = tuple(type(self.cfg.scene)._tactile_params().get("tactile_array_size", (20, 25)))
        except (AttributeError, TypeError):
            tactile_hw = (20, 25)
        self.tactile_array_size = (int(tactile_hw[0]), int(tactile_hw[1]))

        if n > 0 and names[0] in self.scene.sensors:
            first = self.scene[names[0]]
            self.tactile_array_size = tuple(int(value) for value in first.cfg.tactile_array_size)
            self._ur10e_stacked_array_total = int(self.tactile_array_size[0]) * int(self.tactile_array_size[1])
            self.tactile_image_height = int(first.cfg.render_cfg.image_height)
            self.tactile_image_width = int(first.cfg.render_cfg.image_width)
        else:
            self._ur10e_stacked_array_total = int(self.tactile_array_size[0]) * int(self.tactile_array_size[1])
            self.tactile_image_height = int(GELSIGHT_R15_CFG.image_height)
            self.tactile_image_width = int(GELSIGHT_R15_CFG.image_width)
        self.tactile_array_total = self._ur10e_stacked_array_total
        self.tactile_image_channels = 3
        self._ur10e_stacked_image_total = self.tactile_image_height * self.tactile_image_width * self.tactile_image_channels
        self.tactile_normal_force = torch.zeros((self.num_envs, n * self._ur10e_stacked_array_total), device=self.device)
        self.tactile_shear_force = torch.zeros((self.num_envs, n * self._ur10e_stacked_array_total * 2), device=self.device)
        self.tactile_rgb_image = torch.zeros((self.num_envs, n * self._ur10e_stacked_image_total), device=self.device)

    def _update_stacked_tacsl_tactile_from_scene(self) -> None:
        """Read all TacSL sensors and fill ``tactile_{normal,shear,rgb}_image``."""
        n_sens = self._ur10e_stacked_n
        flat_n = n_sens * self._ur10e_stacked_array_total
        flat_s = n_sens * self._ur10e_stacked_array_total * 2
        flat_rgb = n_sens * self._ur10e_stacked_image_total
        names = self._ur10e_stacked_tacsl_names
        if (
            self._shadow_render_sensors_enabled
            and VisuoTactileSensorData is not None
            and n_sens > 0
            and all(name in self.scene.sensors for name in names)
        ):
            chunks_n: list[torch.Tensor] = []
            chunks_s: list[torch.Tensor] = []
            chunks_rgb: list[torch.Tensor] = []
            ok_n = ok_s = ok_rgb = True
            for name in names:
                sensor = self.scene[name]
                try:
                    td = sensor.data
                except RuntimeError as e:
                    if "source and destination dtypes match" in str(e):
                        try:
                            sdata = getattr(sensor, "_data", None)
                            if sdata is not None and hasattr(sdata, "tactile_rgb_image"):
                                tri = getattr(sdata, "tactile_rgb_image")
                                if torch.is_tensor(tri) and tri.dtype != torch.uint8:
                                    sdata.tactile_rgb_image = tri.to(torch.uint8)
                            td = sensor.data
                        except Exception:
                            ok_n = ok_s = ok_rgb = False
                            continue
                    else:
                        ok_n = ok_s = ok_rgb = False
                        continue
                if td.tactile_normal_force is not None:
                    chunks_n.append(
                        torch.nan_to_num(td.tactile_normal_force, nan=0.0, posinf=0.0, neginf=0.0)
                    )
                else:
                    ok_n = False
                if td.tactile_shear_force is not None:
                    chunks_s.append(
                        torch.nan_to_num(td.tactile_shear_force, nan=0.0, posinf=0.0, neginf=0.0)
                    )
                else:
                    ok_s = False
                if td.tactile_rgb_image is not None:
                    chunks_rgb.append(td.tactile_rgb_image)
                else:
                    ok_rgb = False

            if ok_n and len(chunks_n) == n_sens:
                self.tactile_normal_force = torch.cat(chunks_n, dim=1)
            else:
                self.tactile_normal_force = torch.zeros((self.num_envs, flat_n), device=self.device)

            if ok_s and len(chunks_s) == n_sens:
                self.tactile_shear_force = torch.cat(chunks_s, dim=1)
            else:
                self.tactile_shear_force = torch.zeros((self.num_envs, flat_s), device=self.device)

            if ok_rgb and len(chunks_rgb) == n_sens:
                self.tactile_rgb_image = torch.cat(chunks_rgb, dim=1)
                if self.tactile_rgb_image.max() > 1.0:
                    self.tactile_rgb_image = self.tactile_rgb_image / 255.0
            else:
                self.tactile_rgb_image = torch.zeros((self.num_envs, flat_rgb), device=self.device)
        else:
            self.tactile_normal_force = torch.zeros((self.num_envs, flat_n), device=self.device)
            self.tactile_shear_force = torch.zeros((self.num_envs, flat_s), device=self.device)
            self.tactile_rgb_image = torch.zeros((self.num_envs, flat_rgb), device=self.device)

    def _apply_visual_disturbance(self) -> None:
        """Corrupt ``third_person_camera`` RGB in-place; fields are optional and read via ``getattr``."""
        if not getattr(self.cfg, "visual_disturbance", False):
            return
        if "third_person_camera" not in self.scene.sensors:
            return
        cam = self.scene["third_person_camera"]
        if "rgb" not in cam.data.output:
            return
        rgb = cam.data.output["rgb"]
        if rgb is None or rgb.numel() == 0:
            return
        dtype = rgb.dtype
        device = rgb.device
        need_denorm = bool(rgb.max() > 1.0)
        img = rgb.float() / 255.0 if need_denorm else rgb.float()

        vtype = getattr(self.cfg, "visual_disturbance_type", "gaussian_noise")
        if vtype == "gaussian_noise":
            std = float(getattr(self.cfg, "visual_disturbance_noise_std", 0.08))
            img = img + std * torch.randn_like(img, device=device, dtype=img.dtype)
        elif vtype == "gaussian_blur":
            k = int(getattr(self.cfg, "visual_disturbance_blur_kernel_size", 5))
            sigma = float(getattr(self.cfg, "visual_disturbance_blur_sigma", 1.0))
            if k % 2 == 0:
                k += 1
            x = torch.arange(k, device=device, dtype=img.dtype) - k // 2
            g = torch.exp(-(x**2) / (2 * sigma**2))
            g = g / g.sum()
            kernel_2d = (g.unsqueeze(0) * g.unsqueeze(1)).reshape(1, 1, k, k)
            _n, _h, _w, c = img.shape
            img = img.permute(0, 3, 1, 2)
            kernel = kernel_2d.expand(c, 1, k, k)
            img = F.conv2d(img, kernel, padding=k // 2, groups=c)
            img = img.permute(0, 2, 3, 1)
        else:
            return

        img = torch.clamp(img, 0.0, 1.0)
        if need_denorm:
            rgb.copy_(img.mul(255.0).to(dtype))
        else:
            rgb.copy_(img.to(dtype))

    def _sync_tacsl_tactile_and_third_person_visual(self) -> None:
        """Call this at the end of task ``_compute_intermediate_values`` when needed."""
        self._update_stacked_tacsl_tactile_from_scene()
        self._apply_visual_disturbance()

    def _build_record_dict(self) -> dict[str, torch.Tensor]:
        """Build the canonical single-arm record schema for collection/deployment."""

        self._update_stacked_tacsl_tactile_from_scene()
        record: dict[str, torch.Tensor] = {"joint_pos": self.robot.data.joint_pos.detach().cpu()}

        sensor_names = self._ur10e_stacked_tacsl_names
        tactile_poses: list[torch.Tensor] = []
        for name in sensor_names:
            if name not in self.scene.sensors:
                tactile_poses = []
                break
            sensor = self.scene[name]
            data = sensor.data
            pos_w = getattr(data, "pos_w", getattr(sensor, "pos_w", None))
            quat_w = getattr(data, "quat_w_ros", None)
            if quat_w is None:
                quat_w = getattr(data, "quat_w", getattr(sensor, "quat_w", None))
            if pos_w is None or quat_w is None:
                tactile_poses = []
                break
            tactile_poses.append(torch.cat((pos_w - self.scene.env_origins, quat_w), dim=-1))
        if not tactile_poses and sensor_names and all(name in self.scene.sensors for name in sensor_names):
            # TacSL V2 may not expose pose tensors; fall back to physical fingertip rigid-body poses.
            body_names = [str(name).lower() for name in self.robot.body_names]
            for sensor_name in sensor_names:
                finger = str(sensor_name).rsplit("_", maxsplit=1)[-1].lower()
                search_terms = (f"{finger}distal", f"{finger}_tip", finger)
                body_idx = next(
                    (
                        index
                        for term in search_terms
                        for index, body_name in enumerate(body_names)
                        if term in body_name
                    ),
                    None,
                )
                if body_idx is None:
                    tactile_poses = []
                    break
                tactile_poses.append(
                    torch.cat(
                        (
                            self.robot.data.body_pos_w[:, body_idx] - self.scene.env_origins,
                            self.robot.data.body_quat_w[:, body_idx],
                        ),
                        dim=-1,
                    )
                )
        if tactile_poses:
            record["tactile_pos"] = torch.stack(tactile_poses, dim=1).detach().cpu()

        n = self._ur10e_stacked_n
        if n > 0:
            rows, cols = self.tactile_array_size
            record["tactile_normal_force"] = (
                self.tactile_normal_force.reshape(self.num_envs, n, rows, cols, 1).detach().cpu()
            )
            record["tactile_shear_force"] = (
                self.tactile_shear_force.reshape(self.num_envs, n, rows, cols, 2).detach().cpu()
            )
            record["tactile_rgb_image"] = (
                self.tactile_rgb_image.reshape(
                    self.num_envs, n, self.tactile_image_height, self.tactile_image_width, 3
                )
                .mul(255.0)
                .clamp(0.0, 255.0)
                .to(torch.uint8)
                .detach()
                .cpu()
            )

        for camera_name in ("third_person_camera", "twist_camera"):
            if camera_name not in self.scene.sensors:
                continue
            camera = self.scene[camera_name]
            rgb = camera.data.output.get("rgb", None)
            if rgb is not None:
                record[camera_name] = rgb.detach().cpu()
            quat_w = getattr(camera.data, "quat_w_ros", None)
            if quat_w is None:
                quat_w = getattr(camera.data, "quat_w", None)
            pos_w = getattr(camera.data, "pos_w", None)
            if pos_w is not None and quat_w is not None:
                record[f"{camera_name}_pos"] = (
                    torch.cat((pos_w - self.scene.env_origins, quat_w), dim=-1).unsqueeze(1).detach().cpu()
                )
        return record

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
                if hasattr(self.cfg, "third_person_camera_pos"):
                    cam_cfg.offset.pos = tuple(getattr(self.cfg, "third_person_camera_pos"))
                if hasattr(self.cfg, "third_person_camera_rot"):
                    cam_cfg.offset.rot = tuple(getattr(self.cfg, "third_person_camera_rot"))
                if hasattr(self.cfg, "third_person_camera_width"):
                    cam_cfg.width = int(getattr(self.cfg, "third_person_camera_width"))
                if hasattr(self.cfg, "third_person_camera_height"):
                    cam_cfg.height = int(getattr(self.cfg, "third_person_camera_height"))
                self.scene.sensors["third_person_camera"] = cam_cfg.class_type(cam_cfg)

            scene_has_tactile = isinstance(self.cfg.scene, UR10eShadowHandTacSLSceneCfg) or hasattr(
                type(self.cfg.scene), "_tactile_params"
            )
            if scene_has_tactile:
                sensor_cfgs = build_ur10e_shadowhand_tactile_sensor_cfgs(self.cfg.scene)
                for name, sensor_cfg in sensor_cfgs.items():
                    if name not in self.scene.sensors:
                        self.scene.sensors[name] = sensor_cfg.class_type(sensor_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_task_scene(self) -> None:
        raise NotImplementedError

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if not self._use_rl_control:
            return
        self.actions = torch.clamp(actions.to(device=self.device), -1.0, 1.0)

    def _apply_action(self) -> None:
        if not self._use_rl_control:
            return
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

    def apply_joint_targets(self, joint_pos: torch.Tensor) -> None:
        """Apply absolute full-joint or actuated-joint targets for offline-policy playback."""

        targets = joint_pos.to(device=self.device, dtype=self.robot.data.joint_pos.dtype)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if targets.shape[0] == 1 and self.num_envs > 1:
            targets = targets.expand(self.num_envs, -1)
        if targets.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} target rows, got {targets.shape[0]}.")

        if targets.shape[-1] == self.num_robot_dofs:
            joint_ids = list(range(self.num_robot_dofs))
        elif targets.shape[-1] == len(self.actuated_dof_indices):
            joint_ids = self.actuated_dof_indices
        else:
            raise ValueError(
                f"Joint target dim must be full={self.num_robot_dofs} or actuated={len(self.actuated_dof_indices)}, "
                f"got {targets.shape[-1]}."
            )
        lower = self.robot_dof_lower_limits[:, joint_ids]
        upper = self.robot_dof_upper_limits[:, joint_ids]
        targets = saturate(targets, lower, upper)
        self.cur_targets[:, joint_ids] = targets
        self.prev_targets[:, joint_ids] = targets
        self.robot.set_joint_position_target(targets, joint_ids=joint_ids)

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

