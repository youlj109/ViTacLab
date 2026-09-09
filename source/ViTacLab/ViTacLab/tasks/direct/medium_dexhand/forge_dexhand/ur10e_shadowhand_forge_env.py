"""UR10e + ShadowHand Factory peg / gear / nut (Factory assets/rewards/randomization; sim/control like hand_pickup)."""

from __future__ import annotations

from collections.abc import Sequence

import carb
import torch
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    saturate,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
    axis_angle_from_quat,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import sample_uniform

from isaaclab_tasks.direct.factory import factory_utils

from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_sensor_data import (
    VisuoTactileSensorData,
)

from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_cfg import (
    UR10eShadowHandTacSLSceneCfg,
    build_ur10e_shadowhand_tactile_sensor_cfgs,
    build_ur10e_shadowhand_third_person_camera_cfg,
)
from ViTacLab.assets.robot.ur10e_shadowhand_direct_base_single.ur10e_shadowhand_direct_base_env import (
    UR10eShadowHandDirectBaseEnv,
    _scale,
    spawn_factory_table,
    spawn_high_fidelity_scene_if_enabled,
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
SHADOW_HAND_CANONICAL_ORDER = (
    "WRJ2",
    "WRJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
)


def _resolve_body_index_by_keywords(body_names: list[str], keywords: tuple[str, ...]) -> int | None:
    names = [str(n).lower() for n in body_names]
    for key in keywords:
        for i, nm in enumerate(names):
            if key in nm:
                return i
    return None


# Joint position [lower, upper] -> [-1, 1] for policy observations (inverse of :func:`_scale`).
@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class UR10eShadowHandForgeEnv(UR10eShadowHandDirectBaseEnv):
    """Factory forge tasks with UR10e + Shadow Hand (joint-space control)."""

    cfg: UR10eShadowHandForgeEnvCfg

    def __init__(
        self, cfg: UR10eShadowHandForgeEnvCfg, render_mode: str | None = None, **kwargs
    ):
        self.cfg_task = cfg.task
        if not getattr(cfg, "enable_cameras", False):
            cfg.use_full_tactile_obs = False
        tactile_dim = (
            (TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM)
            if cfg.use_full_tactile_obs
            else (5 * (1 + 2))
        )
        base_obs_dim = 30 + 30 + 3 + 4 + 3 + 4 + 1 + 4 + 4
        cfg.observation_space = base_obs_dim + tactile_dim

        super().__init__(cfg, render_mode, **kwargs)

        factory_utils.set_body_inertias(self.robot, self.scene.num_envs)
        # Task-level defaults can be overridden from env cfg for easier tuning.
        self.cfg_task.held_asset_cfg.friction = float(
            getattr(self.cfg, "held_object_friction", self.cfg_task.held_asset_cfg.friction)
        )
        self.cfg_task.robot_cfg.friction = float(
            getattr(self.cfg, "robot_friction", self.cfg_task.robot_cfg.friction)
        )
        factory_utils.set_friction(
            self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs
        )
        factory_utils.set_friction(
            self._fixed_asset,
            self.cfg_task.fixed_asset_cfg.friction,
            self.scene.num_envs,
        )
        factory_utils.set_friction(
            self.robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs
        )

        body_names = [str(n) for n in self.robot.body_names]
        ee_idx: int | None = None
        if bool(getattr(self.cfg, "prefer_palm_as_ee", True)):
            ee_idx = _resolve_body_index_by_keywords(body_names, ("palm", "hand", "grasp"))
        if ee_idx is None:
            try:
                ee_idx = body_names.index(str(self.cfg.ee_body_name))
            except ValueError:
                ee_idx = _resolve_body_index_by_keywords(body_names, ("wrist_3", "wrist3", "wrist"))
        if ee_idx is None:
            raise RuntimeError(
                f"Forge dexhand: failed to resolve EE body index from names: {body_names}"
            )
        self._ee_body_idx = int(ee_idx)

        arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_expr)
        if len(arm_ids) != 6:
            raise RuntimeError(
                f"Forge dexhand: expected 6 arm joints, got {len(arm_ids)}"
            )
        self._arm_joint_ids = [int(i) for i in arm_ids]
        self._arm_joint_ids_t = torch.tensor(
            self._arm_joint_ids, device=self.device, dtype=torch.long
        )
        hand_ids = [
            i for i in self.actuated_dof_indices if i not in self._arm_joint_ids
        ]
        self._hand_joint_ids = [int(i) for i in hand_ids]
        self._hand_joint_ids_t = torch.tensor(
            self._hand_joint_ids, device=self.device, dtype=torch.long
        )
        self._hand_shadow_index_by_joint: list[int] = []
        for jid in self._hand_joint_ids:
            jn = str(self.robot.joint_names[jid])
            sh_idx = -1
            for i, sh_name in enumerate(SHADOW_HAND_CANONICAL_ORDER):
                if sh_name in jn or jn.endswith(sh_name):
                    sh_idx = i
                    break
            self._hand_shadow_index_by_joint.append(sh_idx)

        self._identity_quat_w = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.ep_succeeded = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self.ep_success_times = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.fixed_pos_env_random = torch.zeros((self.num_envs, 3), device=self.device)
        self.held_pos_env_random = torch.zeros((self.num_envs, 3), device=self.device)

        self._prev_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self._action_rate = torch.zeros_like(self._prev_actions)
        self._last_task_actions = torch.zeros((self.num_envs, 4), device=self.device)
        self._last_task_action_rate = torch.zeros_like(self._last_task_actions)
        self._joint_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self._use_joint_action_mode = False

        self._tactile_normal_force: torch.Tensor | None = None
        self._tactile_shear_force: torch.Tensor | None = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0
        self._tactile_normal_mean: torch.Tensor | None = None
        self._tactile_shear_mean: torch.Tensor | None = None
        self._printed_tactile_status = False

        self._keypoint_dist = torch.zeros((self.num_envs,), device=self.device)
        self._held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._held_base_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self._target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_held_base_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.last_update_timestamp = 0.0
        # Re-run tactile buffer binding after local fields are initialized, since
        # `_setup_scene()` may have already prepared sensors during `super().__init__()`.
        self._setup_task_scene()
        self._init_ik_controller()
        self._maybe_debug_print_pose()

    def _init_ik_controller(self) -> None:
        if self.robot.is_fixed_base:
            self._ee_jacobi_idx = self._ee_body_idx - 1
            jac_joint_ids = self._arm_joint_ids
        else:
            self._ee_jacobi_idx = self._ee_body_idx
            jac_joint_ids = [int(j) + 6 for j in self._arm_joint_ids]
        self._ik_arm_joint_ids_t = torch.tensor(
            self._arm_joint_ids, device=self.device, dtype=torch.long
        )
        self._ik_jac_joint_ids_t = torch.tensor(
            jac_joint_ids, device=self.device, dtype=torch.long
        )
        diff_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=str(getattr(self.cfg, "ik_method", "dls")),
            ik_params={"lambda_val": float(getattr(self.cfg, "ik_lambda", 0.01))},
        )
        self._diff_ik_controller = DifferentialIKController(
            diff_cfg, num_envs=self.num_envs, device=str(self.device)
        )

    def _hand_targets_from_preset(self, pose_name: str) -> torch.Tensor:
        if len(self._hand_joint_ids) == 0:
            return torch.zeros((self.num_envs, 0), device=self.device)
        if str(pose_name) == "open":
            pose_vals = getattr(self.cfg, "hand_open_joint_pos_shadow_order", ())
        else:
            pose_vals = getattr(self.cfg, "hand_close_joint_pos_shadow_order", ())
        if not isinstance(pose_vals, (list, tuple)) or len(pose_vals) == 0:
            raise RuntimeError(
                f"Hand pose preset '{pose_name}' is empty. "
                "Please provide it in scripts/rsl_rl/full_tra/hand_pose_presets/..."
            )
        vals = torch.tensor(pose_vals, dtype=torch.float32, device=self.device)
        if vals.numel() < len(SHADOW_HAND_CANONICAL_ORDER):
            raise RuntimeError(
                f"Hand pose preset '{pose_name}' length {vals.numel()} is smaller than "
                f"required canonical ShadowHand DOFs {len(SHADOW_HAND_CANONICAL_ORDER)}."
            )

        # Map canonical shadow order to actual robot hand-joint order by joint name,
        # matching record_full_tra_single.py behavior.
        tgt_1env = torch.zeros((len(self._hand_joint_ids),), dtype=torch.float32, device=self.device)
        for local_i, sh_idx in enumerate(self._hand_shadow_index_by_joint):
            if sh_idx < 0:
                raise RuntimeError(
                    f"Failed to map robot hand joint '{self.robot.joint_names[self._hand_joint_ids[local_i]]}' "
                    "to ShadowHand canonical order."
                )
            tgt_1env[local_i] = vals[sh_idx]
        tgt = tgt_1env.unsqueeze(0).expand(self.num_envs, -1).clone()

        lo = self.robot_dof_lower_limits[:, self._hand_joint_ids_t]
        hi = self.robot_dof_upper_limits[:, self._hand_joint_ids_t]
        return saturate(tgt, lo, hi)

    def _set_hand_pose_for_envs(self, env_ids_t: torch.Tensor, *, closed: bool) -> None:
        if env_ids_t.numel() == 0 or len(self._hand_joint_ids) == 0:
            return
        if closed:
            hand_tgt = self._hand_targets_from_preset("close")[env_ids_t]
        else:
            hand_tgt = self._hand_targets_from_preset("open")[env_ids_t]
        jp = self.robot.data.joint_pos[env_ids_t].clone()
        jp[:, self._hand_joint_ids_t] = hand_tgt
        jv = torch.zeros_like(jp)
        self.robot.set_joint_position_target(
            jp[:, self._hand_joint_ids_t],
            joint_ids=self._hand_joint_ids,
            env_ids=env_ids_t,
        )
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids_t)

    def _servo_pregrasp_pose(self, env_ids_t: torch.Tensor) -> None:
        """IK servo wrist_3_link to a pre-grasp target defined in held-asset local frame."""
        if env_ids_t.numel() == 0:
            return

        n = int(env_ids_t.numel())
        target_pos_w_all = self.robot.data.body_pos_w[:, self._ee_body_idx].clone()
        target_quat_w_all = self.robot.data.body_quat_w[:, self._ee_body_idx].clone()
        pos_off = torch.tensor(
            self.cfg.pregrasp_offset_pos, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(n, 1)
        quat_off = torch.tensor(
            self.cfg.pregrasp_offset_quat, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(n, 1)

        held_pos_w = self._held_asset.data.root_pos_w[env_ids_t]
        held_quat_w = self._held_asset.data.root_quat_w[env_ids_t]
        target_quat_w_env, target_pos_w_env = torch_utils.tf_combine(
            held_quat_w, held_pos_w, quat_off, pos_off
        )

        pos_noise_std = torch.tensor(
            self.cfg.pregrasp_pos_noise, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        if torch.any(pos_noise_std > 0):
            target_pos_w_env = target_pos_w_env + torch.randn_like(target_pos_w_env) * pos_noise_std

        yaw_noise_deg = float(self.cfg.pregrasp_yaw_noise_deg)
        if yaw_noise_deg > 0:
            yaw = (
                (torch.rand((n,), device=self.device) * 2.0 - 1.0)
                * torch.deg2rad(torch.tensor(yaw_noise_deg, device=self.device))
            )
            yaw_quat = torch_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
            )
            target_quat_w_env = torch_utils.quat_mul(target_quat_w_env, yaw_quat)

        target_pos_w_all[env_ids_t] = target_pos_w_env
        target_quat_w_all[env_ids_t] = target_quat_w_env

        max_steps = int(self.cfg.pregrasp_ik_max_steps)
        pos_tol = float(self.cfg.pregrasp_ik_pos_tol)
        rot_tol = torch.deg2rad(
            torch.tensor(float(self.cfg.pregrasp_ik_rot_tol_deg), device=self.device)
        )
        pending_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        pending_mask[env_ids_t] = True
        for _ in range(max_steps):
            if not pending_mask.any():
                break
            pending = torch.nonzero(pending_mask, as_tuple=False).squeeze(-1)

            root_pose_w = self.robot.data.root_pose_w
            ee_pos_w_all = self.robot.data.body_pos_w[:, self._ee_body_idx]
            ee_quat_w_all = self.robot.data.body_quat_w[:, self._ee_body_idx]
            cmd_pos_w = torch.where(
                pending_mask.unsqueeze(-1), target_pos_w_all, ee_pos_w_all
            )
            cmd_quat_w = torch.where(
                pending_mask.unsqueeze(-1), target_quat_w_all, ee_quat_w_all
            )
            tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], cmd_pos_w, cmd_quat_w
            )
            cmd = torch.cat((tgt_pos_b, tgt_quat_b), dim=-1)
            self._diff_ik_controller.set_command(cmd)

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_w_all, ee_quat_w_all
            )
            jac_all = self.robot.root_physx_view.get_jacobians()
            jac = jac_all[:, self._ee_jacobi_idx][:, :, self._ik_jac_joint_ids_t]
            base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jac[:, :3, :] = torch.bmm(base_rot_matrix, jac[:, :3, :])
            jac[:, 3:, :] = torch.bmm(base_rot_matrix, jac[:, 3:, :])
            joint_pos_arm = self.robot.data.joint_pos[:, self._ik_arm_joint_ids_t]
            joint_des_arm = self._diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jac, joint_pos_arm
            )
            joint_des_arm = saturate(
                joint_des_arm,
                self.robot_dof_lower_limits[:, self._ik_arm_joint_ids_t],
                self.robot_dof_upper_limits[:, self._ik_arm_joint_ids_t],
            )

            self.cur_targets[
                pending.unsqueeze(-1), self._ik_arm_joint_ids_t.unsqueeze(0)
            ] = joint_des_arm[pending]
            self.cur_targets[
                pending.unsqueeze(-1), self._hand_joint_ids_t.unsqueeze(0)
            ] = self._hand_targets_from_preset("open")[pending]
            self.prev_targets[pending] = self.cur_targets[pending]
            self.robot.set_joint_position_target(
                self.cur_targets[pending][:, self.actuated_dof_indices],
                joint_ids=self.actuated_dof_indices,
                env_ids=pending,
            )
            self.step_sim_no_action()

            ee_pos_w = self.robot.data.body_pos_w[pending, self._ee_body_idx]
            ee_quat_w = self.robot.data.body_quat_w[pending, self._ee_body_idx]
            tgt_pos = target_pos_w_all[pending]
            tgt_quat = target_quat_w_all[pending]
            pos_err = torch.linalg.norm(ee_pos_w - tgt_pos, dim=-1)
            q_err = torch_utils.quat_mul(tgt_quat, torch_utils.quat_conjugate(ee_quat_w))
            ang_err = torch.linalg.norm(axis_angle_from_quat(q_err), dim=-1)
            ok = (pos_err <= pos_tol) & (ang_err <= rot_tol)
            pending_mask[pending[ok]] = False

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_factory_table()
        spawn_high_fidelity_scene_if_enabled(self.cfg)

        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -1.05),
        )

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
                if hasattr(self.cfg, "third_person_camera_pos"):
                    cam_cfg.offset.pos = tuple(self.cfg.third_person_camera_pos)
                if hasattr(self.cfg, "third_person_camera_rot"):
                    cam_cfg.offset.rot = tuple(self.cfg.third_person_camera_rot)
                if hasattr(self.cfg, "third_person_camera_width"):
                    cam_cfg.width = int(self.cfg.third_person_camera_width)
                if hasattr(self.cfg, "third_person_camera_height"):
                    cam_cfg.height = int(self.cfg.third_person_camera_height)
                self.scene.sensors["third_person_camera"] = cam_cfg.class_type(cam_cfg)

            if bool(getattr(self.cfg, "enable_twist_camera", True)) and "twist_camera" not in self.scene.sensors:
                twist_cfg = TiledCameraCfg(
                    prim_path=str(getattr(self.cfg, "twist_camera_prim_path", "/World/envs/env_.*/Robot/twist_camera")),
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        convention="None",
                    ),
                    data_types=list(
                        getattr(
                            self.cfg,
                            "twist_camera_data_types",
                            ("rgb", "distance_to_image_plane"),
                        )
                    ),
                    spawn=None,
                    width=int(getattr(self.cfg, "twist_camera_width", 320)),
                    height=int(getattr(self.cfg, "twist_camera_height", 240)),
                )
                self.scene.sensors["twist_camera"] = twist_cfg.class_type(twist_cfg)

            # Register TacSL sensors robustly (do not rely on runtime isinstance checks).
            sensor_cfgs = build_ur10e_shadowhand_tactile_sensor_cfgs(self.cfg.scene)
            for name, sensor_cfg in sensor_cfgs.items():
                if name not in self.scene.sensors:
                    self.scene.sensors[name] = sensor_cfg.class_type(sensor_cfg)

        # Initialize tactile buffers after sensors are created.
        self._setup_task_scene()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if getattr(self.cfg, "enable_cameras", False):
            self._maybe_init_tacsl_nominal_render()

    def _setup_task_scene(self) -> None:
        available_tactile = [n for n in TACTILE_SENSOR_NAMES if n in self.scene.sensors]
        if available_tactile:
            first = self.scene[available_tactile[0]]
            sz = first.cfg.tactile_array_size
            self._tactile_array_total = sz[0] * sz[1]
            self._num_tactile_sensors = len(available_tactile)
            self._tactile_normal_force = torch.zeros(
                (self.num_envs, self._num_tactile_sensors * self._tactile_array_total),
                device=self.device,
            )
            self._tactile_shear_force = torch.zeros(
                (
                    self.num_envs,
                    self._num_tactile_sensors * self._tactile_array_total * 2,
                ),
                device=self.device,
            )
            self._tactile_normal_mean = torch.zeros(
                (self.num_envs, self._num_tactile_sensors), device=self.device
            )
            self._tactile_shear_mean = torch.zeros(
                (self.num_envs, self._num_tactile_sensors * 2), device=self.device
            )
        else:
            self._num_tactile_sensors = 0
            self._tactile_array_total = 0
            self._tactile_normal_force = None
            self._tactile_shear_force = None
            print(
                f"[ForgeDexhand][Tactile][WARN] no tactile sensors found in scene.sensors. "
                f"available_keys={list(self.scene.sensors.keys())}",
                flush=True,
            )

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

        self.fingertip_midpoint_pos, self.fingertip_midpoint_quat = (
            self._wrist_pose_env()
        )

        self._held_base_pos, self._held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos,
            self.held_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        self._target_held_base_pos, self._target_held_base_quat = (
            factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
        )
        self._keypoint_dist = self._compute_keypoint_dist_from_bases(
            self._held_base_pos,
            self._held_base_quat,
            self._target_held_base_pos,
            self._target_held_base_quat,
        )

        self.last_update_timestamp = self.robot._data._sim_timestamp

        self._update_tactile()

    def _maybe_debug_print_pose(self) -> None:
        iv = int(getattr(self.cfg, "debug_pose_print_interval", 0))
        if iv <= 0:
            return
        if int(self.common_step_counter) % iv != 0:
            return
        i = int(getattr(self.cfg, "debug_pose_print_env_index", 0))
        i = max(0, min(i, self.num_envs - 1))
        ee_pos = self.robot.data.body_pos_w[i, self._ee_body_idx].detach().cpu().tolist()
        ee_quat = self.robot.data.body_quat_w[i, self._ee_body_idx].detach().cpu().tolist()
        held_pos = self._held_asset.data.root_pos_w[i].detach().cpu().tolist()
        held_quat = self._held_asset.data.root_quat_w[i].detach().cpu().tolist()
        print(
            "[ForgeDexhand][PoseDebug] "
            f"step={int(self.common_step_counter):6d} env={i:3d} | "
            f"ee_pos_w={[round(v, 5) for v in ee_pos]} ee_quat_wxyz={[round(v, 5) for v in ee_quat]} | "
            f"obj_pos_w={[round(v, 5) for v in held_pos]} obj_quat_wxyz={[round(v, 5) for v in held_quat]}",
            flush=True,
        )

    def _maybe_debug_print_success(self) -> None:
        iv = int(getattr(self.cfg, "debug_success_print_interval", 0))
        if iv <= 0:
            return
        if int(self.common_step_counter) % iv != 0:
            return
        i = int(getattr(self.cfg, "debug_success_print_env_index", 0))
        i = max(0, min(i, self.num_envs - 1))

        held_base_pos = self._held_base_pos
        target_held_base_pos = self._target_held_base_pos
        xy_dist = torch.linalg.vector_norm(
            target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
        )
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = float(fixed_cfg.height * self.cfg_task.success_threshold)
        elif self.cfg_task.name == "nut_thread":
            height_threshold = float(
                fixed_cfg.thread_pitch * self.cfg_task.success_threshold
            )
        else:
            height_threshold = float("nan")

        check_rot = self.cfg_task.name == "nut_thread"
        yaw_val = float("nan")
        yaw_gate = float("nan")
        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            yaw_val = float(curr_yaw[i].item())
            yaw_gate = float(self.cfg_task.ee_success_yaw)

        curr_success = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        success_i = bool(curr_success[i].item())

        print(
            "[ForgeDexhand][SuccessDebug] "
            f"step={int(self.common_step_counter):6d} env={i:3d} task={self.cfg_task.name} | "
            f"xy_dist={float(xy_dist[i].item()):.6f} (gate<0.002500) "
            f"z_disp={float(z_disp[i].item()):.6f} (gate<{height_threshold:.6f}) "
            f"yaw={yaw_val:.6f} (gate<{yaw_gate:.6f}) "
            f"success={int(success_i)}",
            flush=True,
        )

    def _update_tactile(self) -> None:
        if not self._printed_tactile_status:
            print(
                f"[ForgeDexhand][Tactile] enable_cameras={bool(getattr(self.cfg, 'enable_cameras', False))} "
                f"num_sensors={int(self._num_tactile_sensors)} "
                f"visuotactile_data_cls={'ok' if VisuoTactileSensorData is not None else 'none'}",
                flush=True,
            )
            self._printed_tactile_status = True
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
        if (
            len(norm_list) == self._num_tactile_sensors
            and len(shear_list) == self._num_tactile_sensors
        ):
            self._tactile_normal_force = torch.cat(norm_list, dim=1)
            self._tactile_shear_force = torch.cat(shear_list, dim=1)
            if (
                self._tactile_normal_mean is not None
                and self._tactile_shear_mean is not None
            ):
                n_means, s_means = [], []
                for name in TACTILE_SENSOR_NAMES:
                    data = self.scene[name].data
                    nf = data.tactile_normal_force
                    sf = data.tactile_shear_force
                    n_means.append(nf.mean(dim=1, keepdim=True))
                    s_means.append(sf.mean(dim=1))
                self._tactile_normal_mean = torch.cat(n_means, dim=1)
                self._tactile_shear_mean = torch.cat(s_means, dim=1)

    def _compute_keypoint_reward_terms(
        self, curr_successes: torch.Tensor
    ) -> tuple[dict, dict]:
        keypoint_dist = self._keypoint_dist
        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        action_penalty_ee = torch.norm(self._last_task_actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self._last_task_action_rate, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(
            success_threshold=self.cfg_task.engage_threshold, check_rot=False
        )

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

    def _get_curr_successes(
        self, success_threshold: float, check_rot: bool
    ) -> torch.Tensor:
        held_base_pos = self._held_base_pos
        target_held_base_pos = self._target_held_base_pos

        xy_dist = torch.linalg.vector_norm(
            target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
        )
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

    def _log_factory_metrics(
        self, rew_dict: dict, curr_successes: torch.Tensor
    ) -> None:
        if "episode" not in self.extras:
            self.extras["episode"] = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        if torch.any(self.reset_buf):
            success_rate = (
                torch.count_nonzero(curr_successes) / self.num_envs
            )
            self.extras["successes"] = success_rate
            # RSL-RL console logger reads episode/log channels.
            self.extras["episode"]["successes"] = success_rate
            self.extras["log"]["successes"] = success_rate

        first_success = torch.logical_and(curr_successes, self.ep_succeeded == 0)
        self.ep_succeeded = torch.where(
            curr_successes, torch.ones_like(self.ep_succeeded), self.ep_succeeded
        )

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[
            first_success_ids
        ]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(
                nonzero_success_ids
            )
            self.extras["success_times"] = success_times
            self.extras["episode"]["success_times"] = success_times
            self.extras["log"]["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        a = actions.to(self.device)
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if a.shape[0] == 1 and self.num_envs > 1:
            a = a.expand(self.num_envs, -1)
        if a.shape[-1] == self.num_actions:
            # Full joint-space mode (arm + hand), e.g. teleoperation pipeline.
            self._use_joint_action_mode = True
            self._joint_actions[:, :] = torch.clamp(a[: self.num_envs], -1.0, 1.0)
            return

        # Task-space mode: consume [dx, dy, dz, yaw_world_z].
        self._use_joint_action_mode = False
        if a.shape[-1] >= 4:
            a4 = a[:, :4]
        else:
            a4 = torch.zeros((a.shape[0], 4), device=self.device, dtype=a.dtype)
            a4[:, : a.shape[-1]] = a
        task_actions = torch.zeros((self.num_envs, 4), device=self.device)
        task_actions[:, :] = torch.clamp(a4[: self.num_envs], -1.0, 1.0)
        self._last_task_action_rate[:] = task_actions - self._last_task_actions
        self._last_task_actions[:] = task_actions

    def _apply_action(self) -> None:
        if self.last_update_timestamp < self.robot._data._sim_timestamp:
            self._compute_intermediate_values()

        if self._use_joint_action_mode:
            joint_ids = self.actuated_dof_indices
            self.cur_targets[:, joint_ids] = _scale(
                self._joint_actions,
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
            self.robot.set_joint_position_target(
                self.cur_targets[:, joint_ids], joint_ids=joint_ids
            )
            return

        pos_scale = torch.tensor(
            self.cfg.ee_pos_action_bounds, device=self.device, dtype=torch.float32
        )
        pos_actions = self._last_task_actions[:, 0:3] * pos_scale
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        target_pos_preclip = fixed_pos_action_frame + pos_actions
        delta_pos = target_pos_preclip - self.fingertip_midpoint_pos
        pos_clip = torch.tensor(
            self.cfg.ee_pos_step_clip, device=self.device, dtype=torch.float32
        ).view(1, 3)
        delta_pos = torch.clamp(delta_pos, -pos_clip, pos_clip)
        target_pos = self.fingertip_midpoint_pos + delta_pos

        yaw_lo, yaw_hi = self.cfg.ee_yaw_world_range_deg
        yaw_cmd_deg = 0.5 * (self._last_task_actions[:, 3] + 1.0) * (yaw_hi - yaw_lo) + yaw_lo
        yaw_cmd = torch.deg2rad(yaw_cmd_deg)
        yaw_clip = torch.deg2rad(
            torch.tensor(float(self.cfg.ee_yaw_world_step_clip_deg), device=self.device)
        )
        yaw_delta = torch.clamp(yaw_cmd, -yaw_clip, yaw_clip)
        world_z_axis = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        world_z_axis[:, 2] = 1.0
        q_delta = torch_utils.quat_from_angle_axis(yaw_delta, world_z_axis)
        target_quat = torch_utils.quat_mul(q_delta, self.fingertip_midpoint_quat)

        root_pose_w = self.robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos, target_quat
        )
        cmd = torch.cat((target_pos_b, target_quat_b), dim=-1)
        self._diff_ik_controller.set_command(cmd)

        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_w, ee_quat_w
        )
        jac = self.robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._ik_jac_joint_ids_t
        ]
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jac[:, :3, :] = torch.bmm(base_rot_matrix, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(base_rot_matrix, jac[:, 3:, :])
        joint_pos_arm = self.robot.data.joint_pos[:, self._ik_arm_joint_ids_t]
        joint_des_arm = self._diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jac, joint_pos_arm
        )
        joint_des_arm = saturate(
            joint_des_arm,
            self.robot_dof_lower_limits[:, self._ik_arm_joint_ids_t],
            self.robot_dof_upper_limits[:, self._ik_arm_joint_ids_t],
        )
        self.cur_targets[:, self._ik_arm_joint_ids_t] = joint_des_arm

        if bool(getattr(self.cfg, "hand_target_from_default_pose", True)):
            # Keep hand exactly at reset/default keyframe hand pose (e.g. pose_003).
            default_hand_tgt = self.robot.data.default_joint_pos[:, self._hand_joint_ids_t]
            self.cur_targets[:, self._hand_joint_ids_t] = default_hand_tgt
        else:
            close_tgt = self._hand_targets_from_preset("close")
            self.cur_targets[:, self._hand_joint_ids_t] = close_tgt

        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average)
            * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[:, self.actuated_dof_indices],
            self.robot_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[
            :, self.actuated_dof_indices
        ]
        self.robot.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _get_observations(self) -> dict:
        if self.last_update_timestamp < self.robot._data._sim_timestamp:
            print("[INFO] Computing intermediate values.")
            self._compute_intermediate_values()
        elif self._tactile_normal_force is not None:
            self._update_tactile()
        self._maybe_debug_print_pose()
        self._maybe_debug_print_success()

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
            self.cfg.vel_obs_scale
            * self.robot.data.joint_vel[:, self.actuated_dof_indices],
            held_rel,
            self.held_quat,
            wrist_rel,
            wrist_q,
            self._keypoint_dist.unsqueeze(-1),
            self._last_task_actions,
            self._last_task_action_rate,
        ]

        if self.cfg.use_full_tactile_obs and self._tactile_normal_force is not None:
            obs_elems.append(self._tactile_normal_force)
            obs_elems.append(self._tactile_shear_force)
        elif self._tactile_normal_mean is not None:
            obs_elems.append(self._tactile_normal_mean)
            obs_elems.append(self._tactile_shear_mean)
        else:
            td = (
                TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM
                if self.cfg.use_full_tactile_obs
                else (5 * (1 + 2))
            )
            obs_elems.append(torch.zeros((self.num_envs, td), device=self.device))

        obs = torch.cat(obs_elems, dim=-1)
        tactile_array_size: tuple[int, int] | None = None
        tactile_image_hw: tuple[int, int] | None = None
        if self._num_tactile_sensors > 0 and TACTILE_SENSOR_NAMES[0] in self.scene.sensors:
            first_sensor = self.scene[TACTILE_SENSOR_NAMES[0]]
            sz = first_sensor.cfg.tactile_array_size
            tactile_array_size = (int(sz[0]), int(sz[1]))
            cam_cfg = getattr(first_sensor.cfg, "camera_cfg", None)
            if cam_cfg is not None:
                tactile_image_hw = (int(cam_cfg.height), int(cam_cfg.width))
        tactile_rgb_image: torch.Tensor | None = None
        if self._num_tactile_sensors > 0 and tactile_image_hw is not None:
            rgb_list: list[torch.Tensor] = []
            for name in TACTILE_SENSOR_NAMES:
                if name not in self.scene.sensors:
                    continue
                rgb = getattr(self.scene[name].data, "tactile_rgb_image", None)
                if rgb is None:
                    continue
                rgb = torch.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
                if rgb.dtype == torch.uint8:
                    rgb_f32 = rgb.to(torch.float32) / 255.0
                elif torch.is_floating_point(rgb):
                    rgb_f32 = rgb.to(torch.float32)
                    if float(rgb_f32.max()) > 1.0:
                        rgb_f32 = torch.clamp(rgb_f32 / 255.0, 0.0, 1.0)
                    else:
                        rgb_f32 = torch.clamp(rgb_f32, 0.0, 1.0)
                else:
                    continue
                rgb_list.append(rgb_f32)
            if len(rgb_list) == self._num_tactile_sensors:
                tactile_rgb_image = torch.cat(
                    [img.reshape(self.num_envs, -1) for img in rgb_list], dim=1
                )
        record_dict = self._build_pickup_style_record_dict(
            joint_pos=self.robot.data.joint_pos,
            tactile_sensor_names=TACTILE_SENSOR_NAMES,
            tactile_sensor_count=int(self._num_tactile_sensors),
            tactile_normal_force=self._tactile_normal_force if self._num_tactile_sensors > 0 else None,
            tactile_shear_force=self._tactile_shear_force if self._num_tactile_sensors > 0 else None,
            tactile_rgb_image=tactile_rgb_image,
            tactile_array_size=tactile_array_size,
            tactile_image_hw=tactile_image_hw,
        )
        # Expose reset randomization offsets for recorder/replayer compensation.
        record_dict["fixed_pos_env_random"] = self.fixed_pos_env_random.detach().cpu()
        record_dict["held_pos_env_random"] = self.held_pos_env_random.detach().cpu()
        return {"policy": obs, "record": record_dict}

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
        pos = (
            torch.tensor(pos_env, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(n, -1)
        )
        quat = (
            torch.tensor(quat_wxyz, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(n, -1)
        )
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
        n = len(env_ids)

        dx = sample_uniform(
            self.cfg.fixed_reset_pos_x_range[0],
            self.cfg.fixed_reset_pos_x_range[1],
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        dy = sample_uniform(
            self.cfg.fixed_reset_pos_y_range[0],
            self.cfg.fixed_reset_pos_y_range[1],
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        dz = sample_uniform(
            self.cfg.fixed_reset_pos_z_range[0],
            self.cfg.fixed_reset_pos_z_range[1],
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        fixed_offset = torch.stack((dx, dy, dz), dim=-1)
        hx = sample_uniform(
            self.cfg.held_reset_pos_x_range[0],
            self.cfg.held_reset_pos_x_range[1],
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        hy = sample_uniform(
            self.cfg.held_reset_pos_y_range[0],
            self.cfg.held_reset_pos_y_range[1],
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        hz = sample_uniform(
            self.cfg.held_reset_pos_z_range[0],
            self.cfg.held_reset_pos_z_range[1],
            (n, 1),
            device=self.device,
        ).squeeze(-1)
        held_offset = torch.stack((hx, hy, hz), dim=-1)

        fixed_pos_t = torch.tensor(
            op.fixed_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(n, -1) + fixed_offset
        fixed_quat_t = torch.tensor(
            op.fixed_quat, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(n, -1)

        def _write_art_root_pose_tensor(
            articulation: Articulation,
            env_ids_local: torch.Tensor,
            pos_env_t: torch.Tensor,
            quat_wxyz_t: torch.Tensor,
        ) -> None:
            eo = self.scene.env_origins[env_ids_local]
            st = articulation.data.default_root_state.clone()[env_ids_local]
            st[:, 0:3] = eo + pos_env_t
            st[:, 3:7] = quat_wxyz_t
            st[:, 7:] = 0.0
            articulation.write_root_pose_to_sim(st[:, 0:7], env_ids=env_ids_local)
            articulation.write_root_velocity_to_sim(st[:, 7:], env_ids=env_ids_local)
            articulation.reset()

        _write_art_root_pose_tensor(self._fixed_asset, env_ids, fixed_pos_t, fixed_quat_t)
        held_pos_t = torch.tensor(
            op.held_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(n, -1) + held_offset
        held_quat_t = torch.tensor(
            op.held_quat, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(n, -1)
        _write_art_root_pose_tensor(self._held_asset, env_ids, held_pos_t, held_quat_t)

        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            sp = op.small_gear_pos if op.small_gear_pos is not None else op.fixed_pos
            sq = op.small_gear_quat if op.small_gear_quat is not None else op.fixed_quat
            lp = op.large_gear_pos if op.large_gear_pos is not None else op.fixed_pos
            lq = op.large_gear_quat if op.large_gear_quat is not None else op.fixed_quat
            sp_t = torch.tensor(sp, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1) + fixed_offset
            sq_t = torch.tensor(sq, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
            lp_t = torch.tensor(lp, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1) + fixed_offset
            lq_t = torch.tensor(lq, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
            _write_art_root_pose_tensor(self._small_gear_asset, env_ids, sp_t, sq_t)
            _write_art_root_pose_tensor(self._large_gear_asset, env_ids, lp_t, lq_t)

        ref = torch.tensor(
            self.cfg.fixed_obs_ref_pos, dtype=torch.float32, device=self.device
        )
        self.fixed_pos_obs_frame[env_ids] = ref.unsqueeze(0).expand(n, -1) + fixed_offset
        self.fixed_pos_env_random[env_ids] = fixed_offset
        self.held_pos_env_random[env_ids] = held_offset
        std = torch.tensor(
            self.cfg.fixed_obs_noise_std, dtype=torch.float32, device=self.device
        )
        self.init_fixed_pos_obs_noise[env_ids] = torch.randn(
            n, 3, device=self.device
        ) * std.unsqueeze(0)

        # Always start reset from hand-open preset.
        self._set_hand_pose_for_envs(env_ids, closed=False)
        self.step_sim_no_action()

        if bool(getattr(self.cfg, "use_pregrasp_reset", True)):
            # Optionally run pre-grasp IK after enforcing open hand at reset.
            self._servo_pregrasp_pose(env_ids)

        self.actions[env_ids].zero_()
        self._prev_actions[env_ids].zero_()
        self._last_task_actions[env_ids].zero_()
        self._last_task_action_rate[env_ids].zero_()

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            super()._reset_idx(slice(None))
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
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
