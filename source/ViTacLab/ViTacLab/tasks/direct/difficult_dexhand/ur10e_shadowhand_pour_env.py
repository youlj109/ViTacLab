"""UR10e + ShadowHand(Left) deformable paper-cup pouring task with tactile sensing.

本环境在 IsaacLab DirectRLEnv 框架下，实现如下组合任务：

- 机器人：UR10e 机械臂 + 左手 ShadowHand（带 TacSL 触觉）
- 操作对象：可变形 Deformable 纸杯（这里用 Deformable Cuboid 近似纸杯形状）
- 任务目标：控制机械臂与手完成“将纸杯移至目标杯上方并倾倒”的倒水动作

实现严格参考以下代码结构与配置（在对应注释处标明）：

- 关节控制 / 限位 / 机器人初始化：
  - 参考 `tacex_assets.robots.ur10e_shadowhand.ur10e_shadowhand_gelsighthand` 中
    `UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG` 的定义（本文件内重新实现，无外部依赖）。
- 任务环境与倒水逻辑结构：
  - 参考 LeRobot 倒水任务 `lehome/tasks/livingroom/loft_water.py`，使用单 env、单 articulation 的结构，
    将机器人、可倒水对象、目标容器加入同一场景，并在 `_get_rewards` / `_get_dones` 中实现任务判据。
- 触觉传感器配置与读取：
  - 参考 ViTacLab 中 `simple_dexhand/inhand_manipulation/inhand_manipulation_env.py`
    以及 `shadow_hand_env_cfg.py` 中的 `ShadowHandSceneCfg` 与 TacSL 传感器定义。

注意：
- 机器人 USD 模型路径按需求硬编码为：
  `source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/ur10e/ur10e_shadow_left_hand_glb_withtac.usd`
- 纸杯使用 IsaacLab 自带的 `DeformableObjectCfg + MeshCuboidCfg` 生成（无需外部 USD 文件），
  满足“deformable（可变形）”的要求，与参考任务中 rigid 杯子区分开。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import re

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

from .ur10e_shadowhand_pour_env_cfg import (
    UR10eShadowHandPourEnvCfg,
    UR10eShadowHandTactileSceneCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs.ui import ViewerCfg


##
# Tactile sensor constants (参考 simple_dexhand/inhand_manipulation)
##

TACTILE_SENSOR_NAMES = ("tactile_sensor_ff", "tactile_sensor_lf", "tactile_sensor_mf", "tactile_sensor_rf", "tactile_sensor_th")
TACTILE_POINTS_PER_SENSOR = 20 * 25  # 参考 ShadowHandSceneCfg 中 tactile_array_size=(20, 25)
TACTILE_NORMAL_DIM = 5 * TACTILE_POINTS_PER_SENSOR  # 2500
TACTILE_SHEAR_DIM = 5 * TACTILE_POINTS_PER_SENSOR * 2  # 5000


@torch.jit.script
def _scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def _unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def _randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


@torch.jit.script
def _rotation_distance(object_rot, target_rot):
    """Orientation distance between current cup pose and goal pose."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


##
# Environment implementation
##


class UR10eShadowHandPourEnv(DirectRLEnv):
    """UR10e + ShadowHand deformable paper-cup pouring environment with tactile sensing."""

    cfg: UR10eShadowHandPourEnvCfg

    def __init__(self, cfg: UR10eShadowHandPourEnvCfg, render_mode: str | None = None, **kwargs):
        # 在 super() 之前，根据是否启用触觉设置 observation_space 尺寸
        # 这里按 reduced 模式（不拼接触觉）+ 可选触觉两种情况计算。
        base_obs_dim = 0
        # 机器人关节（UR10e + ShadowHand，大约 6 + 24 = 30 DOF）
        base_obs_dim += 64  # 富余设置，实际会在运行时根据具体数量拼接
        # 杯子 / 目标位姿 + 线速度等
        base_obs_dim += 32
        # 动作
        base_obs_dim += 32

        full_obs_dim = base_obs_dim + TACTILE_NORMAL_DIM + TACTILE_SHEAR_DIM
        # 默认使用 reduced 版本，是否拼接触觉由 cfg 控制（可根据需要扩展）。
        cfg.observation_space = full_obs_dim

        super().__init__(cfg, render_mode, **kwargs)

        # 关节数量与关节索引
        self.num_robot_dofs = self.robot.num_joints
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device
        )
        self.prev_targets = torch.zeros_like(self.robot_dof_targets)
        self.cur_targets = torch.zeros_like(self.robot_dof_targets)

        # 选择所有可控关节（使用名字正则，与 TacEx 一致）
        self.actuated_dof_indices: list[int] = []
        for i, name in enumerate(self.robot.joint_names):
            if re.match(self.cfg.arm_joint_expr, name) or re.match(self.cfg.hand_joint_expr, name):
                self.actuated_dof_indices.append(i)
        if not self.actuated_dof_indices:
            # 回退：如果正则匹配不到，则全部关节可控
            self.actuated_dof_indices = list(range(self.num_robot_dofs))
        self.actuated_dof_indices.sort()
        self.num_actions = len(self.actuated_dof_indices)

        # 关节限位
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]

        # 单位向量
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # 杯子与目标的中间缓存
        self.cup_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cup_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.cup_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.cup_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_rot = torch.zeros((self.num_envs, 4), device=self.device)

        # 目标姿态（用于奖励与可视化）
        self.goal_cup_pos = torch.tensor(self.cfg.goal_cup_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.goal_cup_rot = torch.tensor(self.cfg.goal_cup_rot, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

        # 触觉缓存
        self._tactile_normal_force: torch.Tensor | None = None
        self._tactile_shear_force: torch.Tensor | None = None
        self._num_tactile_sensors = 0
        self._tactile_array_total = 0

        # 成功统计
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # ---------------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------------

    def _setup_scene(self):
        # 机器人、纸杯、目标容器
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup = DeformableObject(self.cfg.cup_cfg)
        self.target = RigidObject(self.cfg.target_cfg)

        # 地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 复制 envs
        self.scene.clone_environments(copy_from_source=False)

        # 注册到 scene，便于 randomization / 管理
        self.scene.articulations["robot"] = self.robot
        self.scene.deformable_objects["cup"] = self.cup
        self.scene.rigid_objects["target_cup"] = self.target

        # 环境光照
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 初始化 TacSL 触觉 buffers（参考 inhand_manipulation）
        if TACTILE_SENSOR_NAMES[0] in self.scene.sensors and VisuoTactileSensorData is not None:
            first = self.scene[TACTILE_SENSOR_NAMES[0]]
            sz = first.cfg.tactile_array_size
            self._tactile_array_total = sz[0] * sz[1]
            self._num_tactile_sensors = len(TACTILE_SENSOR_NAMES)
            self._tactile_normal_force = torch.zeros(
                (self.num_envs, self._num_tactile_sensors * self._tactile_array_total),
                device=self.device,
            )
            self._tactile_shear_force = torch.zeros(
                (self.num_envs, self._num_tactile_sensors * self._tactile_array_total * 2),
                device=self.device,
            )

    # ---------------------------------------------------------------------
    # Control
    # ---------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 将输入 action 映射为 [-1, 1]，并缓存到 self.actions
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 关节位置目标（与 inhand_manipulation 中的手部控制逻辑保持一致）
        self.cur_targets[:, self.actuated_dof_indices] = _scale(
            self.actions,
            self.robot_dof_lower_limits[:, self.actuated_dof_indices],
            self.robot_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[:, self.actuated_dof_indices],
            self.robot_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.robot.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    # ---------------------------------------------------------------------
    # Tactile data
    # ---------------------------------------------------------------------

    def _update_tactile_data(self) -> None:
        """Read 5 TacSL sensors and fill _tactile_normal_force / _tactile_shear_force."""
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
        if len(norm_list) == self._num_tactile_sensors and len(shear_list) == self._num_tactile_sensors:
            self._tactile_normal_force = torch.cat(norm_list, dim=1)
            self._tactile_shear_force = torch.cat(shear_list, dim=1)

    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------

    def _compute_intermediate_values(self):
        # 机器人关节状态
        self.robot_dof_pos = self.robot.data.joint_pos
        self.robot_dof_vel = self.robot.data.joint_vel

        # 杯子与目标姿态
        # deformable 对象目前仅使用位置（质心）信息；旋转和速度近似为零
        self.cup_pos = self.cup.data.root_pos_w - self.scene.env_origins
        # 旋转近似为单位四元数，线速度/角速度近似为零
        self.cup_rot[:] = 0.0
        self.cup_rot[:, 0] = 1.0
        self.cup_linvel.zero_()
        self.cup_angvel.zero_()

        self.target_pos = self.target.data.root_pos_w - self.scene.env_origins
        self.target_rot = self.target.data.root_quat_w

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        # 机器人 + 杯子 + 目标 + 动作
        obs_elems = [
            _unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits),
            self.cfg.vel_obs_scale * self.robot_dof_vel,
            self.cup_pos,
            self.cup_rot,
            self.cup_linvel,
            self.cfg.vel_obs_scale * self.cup_angvel,
            self.target_pos,
            self.target_rot,
            self.goal_cup_pos,
            self.goal_cup_rot,
            quat_mul(self.cup_rot, quat_conjugate(self.goal_cup_rot)),
            self.actions,
        ]
        obs = torch.cat(obs_elems, dim=-1)

        # 追加触觉（policy obs）
        if self._tactile_normal_force is not None:
            self._update_tactile_data()
            obs = torch.cat([obs, self._tactile_normal_force, self._tactile_shear_force], dim=-1)

        observations = {"policy": obs}
        return observations

    # ---------------------------------------------------------------------
    # Rewards / terminations
    # ---------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # 目标：纸杯移动到目标上方且倾斜到倒水姿态
        # 位置距离（杯子质心 vs goal 位置）
        cup_pos_error = self.cup_pos - self.goal_cup_pos
        cup_pos_dist = torch.norm(cup_pos_error, p=2, dim=-1)

        # 姿态距离（杯子当前姿态 vs goal 姿态）
        rot_dist = _rotation_distance(self.cup_rot, self.goal_cup_rot)

        # 奖励项
        pos_rew = cup_pos_dist * self.cfg.cup_pos_reward_scale
        rot_rew = 1.0 / (torch.abs(rot_dist) + 1e-3) * self.cfg.cup_rot_reward_scale
        action_penalty = torch.sum(self.actions**2, dim=-1) * self.cfg.action_penalty_scale

        reward = pos_rew + rot_rew + action_penalty

        # 成功条件：位置和姿态都在容差范围内
        success_mask = (cup_pos_dist <= self.cfg.success_tolerance_pos) & (
            torch.abs(rot_dist) <= self.cfg.success_tolerance_rot
        )
        reward = torch.where(success_mask, reward + self.cfg.success_bonus, reward)

        # 掉落惩罚：杯子高度低于一定阈值
        fall_mask = self.cup_pos[:, 2] < self.cfg.fall_height
        reward = torch.where(fall_mask, reward + self.cfg.fall_penalty, reward)

        # 记录成功
        self.successes = torch.where(success_mask, torch.ones_like(self.successes), self.successes)

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["success_rate"] = self.successes.mean()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 掉落或时间到即结束
        fall_mask = self.cup_pos[:, 2] < self.cfg.fall_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = fall_mask | time_out
        return done, time_out

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # 重置目标杯
        target_state = self.target.data.default_root_state.clone()[env_ids]
        target_state[:, 0:3] = target_state[:, 0:3] + self.scene.env_origins[env_ids]
        target_state[:, 7:] = 0.0
        self.target.write_root_pose_to_sim(target_state[:, :7], env_ids)
        self.target.write_root_velocity_to_sim(target_state[:, 7:], env_ids)

        # 重置机器人关节
        delta_max = self.robot_dof_upper_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]
        delta_min = self.robot_dof_lower_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.robot.data.default_joint_pos[env_ids] + self.cfg.reset_robot_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
        dof_vel = self.robot.data.default_joint_vel[env_ids] + self.cfg.reset_robot_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.robot_dof_targets[env_ids] = dof_pos

        self.robot.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # 重置统计量
        self.successes[env_ids] = 0.0

        # 更新中间量
        self._compute_intermediate_values()

