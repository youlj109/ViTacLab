from __future__ import annotations

import torch
from isaaclab.assets import RigidObject

from ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env import TACTILE_SENSOR_NAMES, UR10eShadowHandPickupEnv

from .blind_grasp_env_cfg import UR10eShadowHandBlindGraspEnvCfg


def _quat_wxyz_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = quat.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        (
            torch.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1),
            torch.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), dim=-1),
            torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ),
        dim=-2,
    )


def _depth_to_pointcloud_env(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_quat_world: torch.Tensor,
    env_origins: torch.Tensor,
    num_points: int = 2048,
) -> torch.Tensor:
    depth = depth.squeeze(-1) if depth.ndim == 4 and depth.shape[-1] == 1 else depth
    num_envs, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.unsqueeze(0).expand(num_envs, -1, -1)
    ys = ys.unsqueeze(0).expand(num_envs, -1, -1)

    fx = intrinsic[:, 0, 0].view(-1, 1, 1)
    fy = intrinsic[:, 1, 1].view(-1, 1, 1)
    cx = intrinsic[:, 0, 2].view(-1, 1, 1)
    cy = intrinsic[:, 1, 2].view(-1, 1, 1)

    z = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    x = (xs - cx) * z / fx.clamp_min(1e-6)
    y = (ys - cy) * z / fy.clamp_min(1e-6)
    pts_cam = torch.stack((x, y, z), dim=-1).reshape(num_envs, -1, 3)

    rot = _quat_wxyz_to_rotmat(cam_quat_world)
    pts_world = torch.matmul(pts_cam, rot.transpose(1, 2)) + cam_pos_w.unsqueeze(1)
    pts_env = pts_world - env_origins.unsqueeze(1)

    total = pts_env.shape[1]
    if total >= num_points:
        idx = torch.linspace(0, total - 1, num_points, device=device).long()
        pts_env = pts_env[:, idx]
    else:
        pad = pts_env[:, -1:, :].expand(-1, num_points - total, -1) if total > 0 else torch.zeros(num_envs, num_points, 3, device=device, dtype=dtype)
        pts_env = torch.cat((pts_env, pad), dim=1)
    return pts_env


class UR10eShadowHandBlindGraspEnv(UR10eShadowHandPickupEnv):
    """Pickup with a kinematic garbage can; the cube resets inside the bin (same rewards/obs as hand_pickup)."""

    cfg: UR10eShadowHandBlindGraspEnvCfg

    def __init__(self, cfg: UR10eShadowHandBlindGraspEnvCfg, render_mode: str | None = None, **kwargs):
        choice = max(0, min(int(getattr(cfg, "object_init_choice", 0)), len(cfg.object_init_pos_candidates) - 1))
        object_init_pos = cfg.resolve_object_init_pos(choice)
        cfg.object_init_choice = choice
        cfg.object_cfg = cfg.object_cfg.replace(
            init_state=cfg.object_cfg.init_state.replace(pos=object_init_pos)
        )
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_object_pos.zero_()
        self._goal_time_left_s.fill_(1.0e9)
        self.successes.zero_()
        self._success_streak.zero_()

    def _setup_task_scene(self) -> None:
        spawn = self.cfg.trash_can_cfg.spawn.replace(scale=self.cfg.trash_can_scale)
        tc_cfg = self.cfg.trash_can_cfg.replace(spawn=spawn)
        self.trash_can = RigidObject(tc_cfg)
        self.scene.rigid_objects["trash_can"] = self.trash_can
        super()._setup_task_scene()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        self.goal_object_pos[env_ids_t] = 0.0
        self._goal_time_left_s[env_ids_t] = 1.0e9
        self.successes[env_ids_t] = 0.0
        self._success_streak[env_ids_t] = 0
        obj_state = self.object.data.default_root_state.clone()[env_ids_t]
        base_pos = torch.tensor(self.cfg.resolve_object_init_pos(self.cfg.object_init_choice), device=self.device, dtype=torch.float).view(1, 3)
        obj_state[:, 0:3] = base_pos + self.scene.env_origins[env_ids_t]
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids_t)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids_t)

    def _resample_goals(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.goal_object_pos[env_ids] = 0.0
        self._goal_time_left_s[env_ids] = 1.0e9

    def _compute_intermediate_values(self):
        self.robot_dof_pos = self.robot.data.joint_pos
        self.robot_dof_vel = self.robot.data.joint_vel
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w

    def _get_observations(self) -> dict:
        base = super()._get_observations()
        num_tactile = len(TACTILE_SENSOR_NAMES)
        record_dict = self._build_pickup_style_record_dict(
            joint_pos=self.robot.data.joint_pos[:, self.actuated_dof_indices],
            tactile_sensor_names=TACTILE_SENSOR_NAMES,
            tactile_sensor_count=num_tactile,
        )

        tactile_hw = self.cfg.scene._tactile_params()["tactile_array_size"]
        norm_list: list[torch.Tensor] = []
        shear_list: list[torch.Tensor] = []
        rgb_list: list[torch.Tensor] = []
        for name in TACTILE_SENSOR_NAMES:
            if name not in self.scene.sensors:
                continue
            data = self.scene[name].data
            nf = getattr(data, "tactile_normal_force", None)
            sf = getattr(data, "tactile_shear_force", None)
            rgb = getattr(data, "tactile_rgb_image", None)
            if nf is not None:
                nf = torch.nan_to_num(nf, nan=0.0, posinf=0.0, neginf=0.0)
                norm_list.append(nf.reshape(self.num_envs, tactile_hw[0], tactile_hw[1], 1))
            if sf is not None:
                sf = torch.nan_to_num(sf, nan=0.0, posinf=0.0, neginf=0.0)
                shear_list.append(sf.reshape(self.num_envs, tactile_hw[0], tactile_hw[1], 2))
            if rgb is not None:
                if rgb.dtype == torch.uint8:
                    rgb_u8 = rgb
                elif rgb.max().item() <= 1.0:
                    rgb_u8 = torch.clamp(rgb * 255.0, 0.0, 255.0).to(torch.uint8)
                else:
                    rgb_u8 = torch.clamp(rgb, 0.0, 255.0).to(torch.uint8)
                rgb_list.append(rgb_u8)
        if len(norm_list) == num_tactile:
            record_dict["tactile_normal_force"] = torch.stack(norm_list, dim=1).detach().cpu()
        if len(shear_list) == num_tactile:
            record_dict["tactile_shear_force"] = torch.stack(shear_list, dim=1).detach().cpu()
        if len(rgb_list) == num_tactile:
            record_dict["tactile_rgb_image"] = torch.stack(rgb_list, dim=1).detach().cpu()

        base["record"] = record_dict
        return base

    def _get_rewards(self) -> torch.Tensor:
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = 0.0
        self.extras["log"]["episode_success_rate"] = 0.0
        self.extras["log"]["episode_success_rate_all_time"] = 0.0
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
