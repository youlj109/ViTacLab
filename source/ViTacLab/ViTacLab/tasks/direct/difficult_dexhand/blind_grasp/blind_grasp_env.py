from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject
from isaaclab.utils.math import sample_uniform

from ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env import TACTILE_SENSOR_NAMES, UR10eShadowHandPickupEnv

from .blind_grasp_env_cfg import UR10eShadowHandBlindGraspEnvCfg


class UR10eShadowHandBlindGraspEnv(UR10eShadowHandPickupEnv):
    """Level-3 blind bin grasp: object is in slot A or B; actor sees only joints + tactile.

    Object pose is used only for reset randomization and external benchmark
    scoring. It never enters actor observations, rewards, or termination.
    """

    cfg: UR10eShadowHandBlindGraspEnvCfg

    def __init__(self, cfg: UR10eShadowHandBlindGraspEnvCfg, render_mode: str | None = None, **kwargs):
        choice = int(getattr(cfg, "object_init_choice", -1))
        spawn = cfg.resolve_object_init_pos(choice)
        cfg.object_cfg = cfg.object_cfg.replace(init_state=cfg.object_cfg.init_state.replace(pos=spawn))
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_object_pos.zero_()
        self._goal_time_left_s.fill_(1.0e9)
        self.successes.zero_()
        self._success_streak.zero_()
        self.object_slot_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tactile_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tactile_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_tactile_shear_peak = torch.zeros(
            (self.num_envs, len(TACTILE_SENSOR_NAMES)), dtype=torch.float, device=self.device
        )

    def _num_slots(self) -> int:
        n_cands = len(self.cfg.object_init_pos_candidates)
        n_cfg = int(getattr(self.cfg, "object_slot_count", 2) or 2)
        return max(1, min(n_cfg, n_cands))

    def _sample_slot_indices(self, n: int) -> torch.Tensor:
        pinned = int(getattr(self.cfg, "object_init_choice", -1))
        n_slots = self._num_slots()
        if pinned >= 0:
            idx = max(0, min(pinned, n_slots - 1))
            return torch.full((n,), idx, device=self.device, dtype=torch.long)
        return torch.randint(0, n_slots, (n,), device=self.device)

    def _setup_task_scene(self) -> None:
        spawn = self.cfg.trash_can_cfg.spawn.replace(scale=self.cfg.trash_can_scale)
        tc_cfg = self.cfg.trash_can_cfg.replace(spawn=spawn)
        self.trash_can = RigidObject(tc_cfg)
        self.scene.rigid_objects["trash_can"] = self.trash_can
        super()._setup_task_scene()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids_t = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        if env_ids_t.numel() == 0:
            return
        if not hasattr(self, "object_slot_index"):
            self.object_slot_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_object_pos[env_ids_t] = 0.0
        self._goal_time_left_s[env_ids_t] = 1.0e9
        self.successes[env_ids_t] = 0.0
        self._success_streak[env_ids_t] = 0
        if hasattr(self, "tactile_successes"):
            self.tactile_successes[env_ids_t] = False
            self._tactile_success_streak[env_ids_t] = 0
            self._prev_tactile_shear_peak[env_ids_t] = 0.0

        n = int(env_ids_t.numel())
        slots = self._sample_slot_indices(n)
        self.object_slot_index[env_ids_t] = slots
        cands = torch.tensor(
            self.cfg.object_init_pos_candidates[: self._num_slots()],
            device=self.device,
            dtype=torch.float,
        )
        base = cands[slots]
        dx = sample_uniform(
            self.cfg.object_reset_pos_x_range[0],
            self.cfg.object_reset_pos_x_range[1],
            (n, 1),
            device=self.device,
        )
        dy = sample_uniform(
            self.cfg.object_reset_pos_y_range[0],
            self.cfg.object_reset_pos_y_range[1],
            (n, 1),
            device=self.device,
        )
        dz = sample_uniform(
            self.cfg.object_reset_pos_z_range[0],
            self.cfg.object_reset_pos_z_range[1],
            (n, 1),
            device=self.device,
        )
        obj_state = self.object.data.root_state_w[env_ids_t].clone()
        obj_state[:, 0:3] = base + torch.cat([dx, dy, dz], dim=1) + self.scene.env_origins[env_ids_t]
        obj_state[:, 7:] = 0.0
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

    def _build_tactile_only_record_dict(self) -> dict[str, torch.Tensor]:
        """Policy/IL record: joint proprioception plus tactile arrays only."""
        record: dict[str, torch.Tensor] = {
            "joint_pos": self.robot.data.joint_pos[:, self.actuated_dof_indices].detach().cpu(),
        }
        num_tactile = len(TACTILE_SENSOR_NAMES)
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
            record["tactile_normal_force"] = torch.stack(norm_list, dim=1).detach().cpu()
        if len(shear_list) == num_tactile:
            record["tactile_shear_force"] = torch.stack(shear_list, dim=1).detach().cpu()
        if len(rgb_list) == num_tactile:
            record["tactile_rgb_image"] = torch.stack(rgb_list, dim=1).detach().cpu()
        return record

    def _build_record_dict(self) -> dict[str, torch.Tensor]:
        """Keep IL/deployment records on the same tactile-only contract as actor obs."""
        return self._build_tactile_only_record_dict()

    def _get_record(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        record = self._build_tactile_only_record_dict()
        if env_ids is None:
            return record
        ids = torch.as_tensor(env_ids, device="cpu", dtype=torch.long)
        return {key: value[ids] for key, value in record.items()}

    def _get_observations(self) -> dict:
        base = super()._get_observations()
        base["record"] = self._build_tactile_only_record_dict()
        return base

    def _get_rewards(self) -> torch.Tensor:
        self._update_tactile_data()
        if self._tactile_normal_force is None or self._tactile_shear_force is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        num_sensors = len(TACTILE_SENSOR_NAMES)
        normal = torch.nan_to_num(self._tactile_normal_force, nan=0.0, posinf=0.0, neginf=0.0)
        shear = torch.nan_to_num(self._tactile_shear_force, nan=0.0, posinf=0.0, neginf=0.0)
        normal = normal.reshape(self.num_envs, num_sensors, -1).clamp_min(0.0)
        shear = shear.reshape(self.num_envs, num_sensors, -1, 2)

        normal_peak = normal.amax(dim=-1)
        shear_peak = torch.linalg.vector_norm(shear, dim=-1).amax(dim=-1)
        contact_threshold = max(float(self.cfg.tactile_contact_threshold), 1.0e-9)
        load_threshold = max(float(self.cfg.tactile_load_threshold), 1.0e-9)
        contact_mask = normal_peak >= contact_threshold
        contact_count = contact_mask.sum(dim=-1)

        contact_score = torch.tanh(normal_peak / contact_threshold).mean(dim=-1)
        min_fingers = max(int(self.cfg.tactile_success_min_fingers), 1)
        multi_finger_score = ((contact_count.float() - 1.0) / max(min_fingers - 1, 1)).clamp(0.0, 1.0)
        stable_contact = contact_count >= min_fingers
        load_score = torch.tanh(shear_peak / load_threshold).mean(dim=-1) * stable_contact.float()

        previous_contact = self._prev_tactile_shear_peak > 0.0
        persistent_contact = contact_mask & previous_contact
        shear_delta = (shear_peak - self._prev_tactile_shear_peak).abs() / load_threshold
        slip_score = (shear_delta * persistent_contact.float()).mean(dim=-1).clamp(max=10.0)
        self._prev_tactile_shear_peak.copy_(shear_peak.detach())

        reward = (
            float(self.cfg.tactile_contact_reward_weight) * contact_score
            + float(self.cfg.tactile_multi_finger_reward_weight) * multi_finger_score
            + float(self.cfg.tactile_load_reward_weight) * load_score
            + float(self.cfg.tactile_slip_penalty_weight) * slip_score
            + float(self.cfg.action_l2_weight) * torch.sum(self.actions.square(), dim=-1)
            + float(self.cfg.action_rate_l2_weight) * torch.sum(self._action_rate.square(), dim=-1)
        )

        tactile_success_now = stable_contact & (shear_peak.amax(dim=-1) >= load_threshold)
        self._tactile_success_streak = torch.where(
            tactile_success_now,
            self._tactile_success_streak + 1,
            torch.zeros_like(self._tactile_success_streak),
        )
        self.tactile_successes |= self._tactile_success_streak >= int(self.cfg.tactile_success_hold_steps)

        # External benchmark metric only. It is deliberately computed after the
        # tactile reward and is not consumed by policy/control logic.
        self._compute_intermediate_values()
        lift_ok = self.object_pos[:, 2] > float(self.cfg.benchmark_lift_height)
        # A raised cube without live multi-finger contact/load (for example a
        # toss) is not a successful grasp. Do not use latched tactile_successes:
        # they may refer to contact that was lost before the lift.
        benchmark_success_now = lift_ok & tactile_success_now
        self.successes = torch.where(benchmark_success_now, torch.ones_like(self.successes), self.successes)
        self._success_streak = torch.where(
            benchmark_success_now, self._success_streak + 1, torch.zeros_like(self._success_streak)
        )
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["consecutive_successes"] = self._success_streak.float().mean()
        self.extras["log"]["episode_success_rate"] = float(self._episode_success_rate_ema)
        self.extras["log"]["episode_success_rate_all_time"] = self.get_episode_success_rate()
        self.extras["log"]["tactile_contact_count"] = contact_count.float().mean()
        self.extras["log"]["tactile_contact_score"] = contact_score.mean()
        self.extras["log"]["tactile_load_score"] = load_score.mean()
        self.extras["log"]["tactile_slip_score"] = slip_score.mean()
        self.extras["log"]["tactile_success_rate"] = self.tactile_successes.float().mean()
        self.extras["log"]["object_slot_index"] = self.object_slot_index.float().mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use timeout only; object geometry cannot alter the control horizon."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out
