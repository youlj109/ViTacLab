# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-only RL: map hand actions to full robot commands via GPU batched differential IK (UR10e arm).

For environments that **only** expose hand joints in ``actuated_dof_indices`` (e.g. in-hand manipulation with a
fixed arm pose), :class:`ArmIkHandActionExpander` skips IK and passes actions through unchanged.

Task-agnostic trajectory: phases ``target:env_steps:use_rotation`` (see :class:`TrajectoryPhase`) — used only when
the arm is part of the action space (pickup / pour style).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciR

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

logger = logging.getLogger(__name__)


_TRAJ_TARGET_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def parse_trajectory_phases(spec: str) -> tuple["TrajectoryPhase", ...]:
    """Parse ``"object:80:0,goal:-1:0"`` or ``"cup:150:0,goal_cup:-1:0"`` → tuple of :class:`TrajectoryPhase`.

    ``target`` is an env field name (see :class:`TrajectoryPhase`). ``env_steps`` ``-1`` means until episode end.
    ``use_rotation`` is ``0``/``1``.
    """
    phases: list[TrajectoryPhase] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise ValueError(f"Bad trajectory segment {part!r}; expected target:steps:use_rot")
        target, steps_s, rot_s = bits[0].strip(), bits[1].strip(), bits[2].strip()
        steps = int(steps_s)
        use_rot = bool(int(rot_s))
        if not _TRAJ_TARGET_RE.match(target):
            raise ValueError(
                f"Invalid trajectory target {target!r}; use a Python identifier (e.g. object, cup, goal_cup)."
            )
        phases.append(TrajectoryPhase(target=target, env_steps=steps, use_rotation=use_rot))
    if not phases:
        raise ValueError("trajectory must contain at least one phase")
    return tuple(phases)


@dataclass(frozen=True)
class TrajectoryPhase:
    """One segment of the EE schedule.

    ``target``: name used to resolve **world-frame** anchor position & orientation:

    1. **Rigid / deformable asset** — if ``env.<target>`` exists and has ``.data.root_pos_w`` /
       ``.data.root_quat_w``, those are used (e.g. ``object``, ``cup``, ``target``).
    2. **Tensor pair (env-local position)** — else if ``env.<target>_pos`` and ``env.<target>_rot`` exist,
       position is ``<target>_pos + env_origins``; rotation is world ``<target>_rot`` (e.g. ``goal_cup``).
    3. **Legacy** — if ``target == "goal"`` and (2) fails, use ``goal_object_pos`` / ``goal_object_rot``.

    ``env_steps``: duration in env steps (``DirectRLEnv.episode_length_buf``); ``-1`` = hold until reset.
    ``use_rotation``: if True, offset is applied in anchor frame and palm orientation includes anchor rotation;
    if False, offset is world + default palm orientation (fixed / pickup_down).
    """

    target: str
    env_steps: int
    use_rotation: bool


@dataclass
class IkRlHandArmCfg:
    """Minimal IK + palm configuration for :class:`ArmIkHandActionExpander`."""

    # --- palm chain
    # World offset when ``use_rotation`` is False; anchor-frame offset when True (same numeric tuple).
    object_to_palm_offset: tuple[float, float, float] = (0.0, 0.0, 0.05)
    # Palm origin expressed in wrist_3 (tool) frame: translation (m) + euler xyz (rad).
    palm_in_wrist_pos: tuple[float, float, float] = (0.0, 0.0, 0.08)
    palm_in_wrist_euler_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Default palm orientation when phase ``use_rotation`` is False.
    palm_orientation_mode: Literal["fixed", "pickup_down"] = "pickup_down"
    palm_euler_xyz: tuple[float, float, float] = (0.0, 2.2, 0.0)
    palm_normal_in_palm_frame: tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_down: tuple[float, float, float] = (0.0, 0.0, -1.0)
    palm_yaw_offset_rad: float = 0.0

    # When phase ``use_rotation`` is True: palm orientation = anchor_quat * euler(palm_euler_in_anchor_frame).
    palm_euler_in_anchor_frame: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- trajectory (task-agnostic; pickup: object then goal)
    trajectory: tuple[TrajectoryPhase, ...] = (
        TrajectoryPhase("object", 80, False),
        TrajectoryPhase("goal", -1, False),
    )

    # --- differential IK (GPU)
    ee_body_name: str = "wrist_3_link"
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls"
    ik_lambda: float | None = None


def _euler_palm_pickup_down(
    palm_normal_in_palm_frame: np.ndarray,
    world_down: np.ndarray,
    yaw_about_world_z: float,
) -> np.ndarray:
    n = np.asarray(palm_normal_in_palm_frame, dtype=np.float64).ravel()
    n = n / (np.linalg.norm(n) + 1e-12)
    d = np.asarray(world_down, dtype=np.float64).ravel()
    d = d / (np.linalg.norm(d) + 1e-12)
    _ret = SciR.align_vectors(d.reshape(1, 3), n.reshape(1, 3))
    r_align = _ret[0] if isinstance(_ret, tuple) else _ret
    r_yaw = SciR.from_euler("z", float(yaw_about_world_z), degrees=False)
    r_total = r_yaw * r_align
    return r_total.as_euler("xyz")


def _se3_from_pos_quat(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    r = matrix_from_quat(quat_wxyz)
    n_b = pos.shape[0]
    t = torch.zeros((n_b, 4, 4), device=pos.device, dtype=pos.dtype)
    t[:, :3, :3] = r
    t[:, :3, 3] = pos
    t[:, 3, 3] = 1.0
    return t


def _se3_inv(T: torch.Tensor) -> torch.Tensor:
    r = T[:, :3, :3]
    p = T[:, :3, 3]
    rt = r.transpose(-1, -2)
    pinv = -torch.bmm(rt, p.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(T)
    out[:, :3, :3] = rt
    out[:, :3, 3] = pinv
    out[:, 3, 3] = 1.0
    return out


def _wrist_pose_from_palm_batch(
    palm_pos_w: torch.Tensor,
    palm_quat_w: torch.Tensor,
    wrist_in_palm_pos: torch.Tensor,
    wrist_in_palm_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``T_world_wrist = T_world_palm @ inv(T_wrist_palm)``."""
    t_wp = _se3_from_pos_quat(
        wrist_in_palm_pos.expand(palm_pos_w.shape[0], -1),
        wrist_in_palm_quat.expand(palm_pos_w.shape[0], -1),
    )
    t_pw = _se3_inv(t_wp)
    t_wp_world = _se3_from_pos_quat(palm_pos_w, palm_quat_w)
    t_ww = torch.bmm(t_wp_world, t_pw)
    wpos = t_ww[:, :3, 3]
    wquat = quat_from_matrix(t_ww[:, :3, :3])
    return wpos, wquat


class ArmIkHandActionExpander:
    """Maps normalized hand actions to full (arm+hand) normalized actions for :class:`DirectRLEnv`.

    If the env already lists **only hand** DOFs as actuated (fixed arm in USD), runs in **hand-only** mode: no IK,
    :meth:`expand` returns ``hand_actions`` unchanged.
    """

    def __init__(self, env: DirectRLEnv, cfg: IkRlHandArmCfg):
        self._env = env
        self._cfg = cfg
        self._device = env.device
        self._num_envs = env.num_envs

        arm_re = re.compile(env.cfg.arm_joint_expr)
        hand_re = re.compile(env.cfg.hand_joint_expr)
        names = env.robot.joint_names
        actuated = list(env.actuated_dof_indices)

        self._arm_slots: list[int] = []
        self._hand_slots: list[int] = []
        self._arm_joint_indices: list[int] = []
        self._hand_joint_indices: list[int] = []

        for slot, ji in enumerate(actuated):
            n = names[ji]
            if arm_re.match(n):
                self._arm_slots.append(slot)
                self._arm_joint_indices.append(ji)
            elif hand_re.match(n):
                self._hand_slots.append(slot)
                self._hand_joint_indices.append(ji)

        self.num_hand = len(self._hand_slots)
        self.num_arm = len(self._arm_slots)
        self._full_dim = len(actuated)

        # In-hand tasks: only hand joints are actuated; arm pose is fixed in the scene / default joint targets.
        if self.num_arm == 0 and self.num_hand > 0 and self.num_hand == self._full_dim:
            self._hand_only_mode = True
            lower = env.robot_dof_lower_limits
            upper = env.robot_dof_upper_limits
            self._lower = lower[:, actuated].clone()
            self._upper = upper[:, actuated].clone()
            logger.info(
                "[ik_rl] ArmIkHandActionExpander: hand-only actuated DOFs (no arm in action space); "
                "skipping differential IK — use train.py for pure joint-space in-hand if preferred."
            )
            return

        self._hand_only_mode = False
        if self.num_hand == 0 or self.num_arm != 6:
            raise RuntimeError(
                f"ArmIkHandActionExpander: expected 6 arm DOFs and >0 hand DOFs; "
                f"got arm={self.num_arm}, hand={self.num_hand}, actuated={actuated}"
            )

        lower = env.robot_dof_lower_limits
        upper = env.robot_dof_upper_limits
        self._lower = lower[:, actuated].clone()
        self._upper = upper[:, actuated].clone()

        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        ik_params = None
        if cfg.ik_method == "dls" and cfg.ik_lambda is not None:
            ik_params = {"lambda_val": float(cfg.ik_lambda)}
        diff_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=cfg.ik_method,
            ik_params=ik_params,
        )
        self._diff_ik_controller = DifferentialIKController(
            diff_cfg, num_envs=self._num_envs, device=str(self._device)
        )
        body_ids, _ = env.robot.find_bodies(cfg.ee_body_name)
        if len(body_ids) < 1:
            raise RuntimeError(
                f"IK: no body matching {cfg.ee_body_name!r}. Sample: {env.robot.body_names[:30]}"
            )
        self._ee_body_idx = int(body_ids[0])
        j_ids, j_names = env.robot.find_joints(env.cfg.arm_joint_expr)
        if len(j_ids) != 6:
            raise RuntimeError(f"IK: expected 6 arm joints, got {len(j_ids)}: {j_names}")
        if set(j_ids) != set(self._arm_joint_indices):
            raise RuntimeError("IK: arm joint indices mismatch between find_joints and actuated list")

        if env.robot.is_fixed_base:
            self._ee_jacobi_idx = self._ee_body_idx - 1
            jac_joint_ids = list(j_ids)
        else:
            self._ee_jacobi_idx = self._ee_body_idx
            jac_joint_ids = [int(j) + 6 for j in j_ids]
        self._diff_ik_joint_pos_ids_t = torch.tensor(j_ids, device=self._device, dtype=torch.long)
        self._diff_ik_jac_joint_ids_t = torch.tensor(jac_joint_ids, device=self._device, dtype=torch.long)
        self._diff_ik_slot_per_col = [self._arm_slots[self._arm_joint_indices.index(jid)] for jid in j_ids]

        # Fixed T_wrist_palm (palm origin in wrist frame)
        pe = torch.tensor(cfg.palm_in_wrist_euler_xyz, device=self._device, dtype=torch.float32).view(1, 3)
        self._wrist_in_palm_quat = quat_from_euler_xyz(pe[:, 0], pe[:, 1], pe[:, 2])
        self._wrist_in_palm_pos = torch.tensor(cfg.palm_in_wrist_pos, device=self._device, dtype=torch.float32).view(1, 3)

        self._offset_vec = torch.tensor(cfg.object_to_palm_offset, device=self._device, dtype=torch.float32).view(1, 3)

        self._phases = cfg.trajectory
        self._validate_trajectory()

    def _validate_trajectory(self) -> None:
        for p in self._phases:
            try:
                self._resolve_anchor_world(p.target)
            except RuntimeError as e:
                raise RuntimeError(f"Trajectory phase target={p.target!r} is invalid: {e}") from e
        if getattr(self._env, "episode_length_buf", None) is None:
            logger.warning("[ik_rl] env has no episode_length_buf; trajectory timing may be wrong.")

    def _resolve_anchor_world(self, target_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame anchor (N,3), (N,4) wxyz from env by ``target_name`` (see :class:`TrajectoryPhase`)."""
        env = self._env
        origins = env.scene.env_origins

        # 1) Rigid / deformable asset on env
        if hasattr(env, target_name):
            obj = getattr(env, target_name)
            if obj is not None and hasattr(obj, "data"):
                d = obj.data
                if hasattr(d, "root_pos_w") and hasattr(d, "root_quat_w"):
                    return d.root_pos_w, d.root_quat_w

        # 2) Tensor pair: <name>_pos (env-local) + origins, <name>_rot (world quat, optional)
        pos_attr = f"{target_name}_pos"
        rot_attr = f"{target_name}_rot"
        if hasattr(env, pos_attr):
            pos = getattr(env, pos_attr)
            gpos = pos.to(device=origins.device, dtype=origins.dtype) + origins
            if hasattr(env, rot_attr):
                gquat = getattr(env, rot_attr).to(device=origins.device, dtype=origins.dtype)
            else:
                gquat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gpos.device, dtype=gpos.dtype).expand(
                    self._num_envs, -1
                )
            return gpos, gquat

        # 3) Legacy: "goal" -> goal_object_pos / goal_object_rot (pickup-style)
        if target_name == "goal":
            if hasattr(env, "goal_object_pos"):
                gp = env.goal_object_pos.to(device=origins.device, dtype=origins.dtype) + origins
                if hasattr(env, "goal_object_rot"):
                    gq = env.goal_object_rot.to(device=origins.device, dtype=origins.dtype)
                else:
                    gq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gp.device, dtype=gp.dtype).expand(
                        self._num_envs, -1
                    )
                return gp, gq

        raise RuntimeError(
            f"expected env.{target_name} (asset with .data.root_pos_w / .data.root_quat_w), "
            f"or env.{pos_attr} [+ env.{rot_attr}], or for target 'goal' use goal_object_pos[/rot]."
        )

    @property
    def num_actuated(self) -> int:
        return self._full_dim

    def _phase_id(self, buf: torch.Tensor) -> torch.Tensor:
        """Map ``episode_length_buf`` → phase index per env."""
        n = buf.shape[0]
        device = buf.device
        pid = torch.zeros(n, dtype=torch.long, device=device)
        cum = 0
        for i, p in enumerate(self._phases):
            if p.env_steps < 0:
                pid = torch.where(buf >= cum, torch.full_like(pid, i), pid)
                return pid
            nxt = cum + int(p.env_steps)
            m = (buf >= cum) & (buf < nxt)
            pid = torch.where(m, torch.full_like(pid, i), pid)
            cum = nxt
        pid = torch.where(buf >= cum, torch.full_like(pid, len(self._phases) - 1), pid)
        return pid

    def _anchor_pos_quat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Full batch anchor poses in world frame (N,3), (N,4)."""
        p0, q0 = self._resolve_anchor_world(self._phases[0].target)
        ap = torch.zeros_like(p0)
        aq = torch.zeros_like(q0)

        buf = getattr(self._env, "episode_length_buf", torch.zeros(self._num_envs, device=p0.device, dtype=torch.long))
        pid = self._phase_id(buf)

        for i, ph in enumerate(self._phases):
            m = pid == i
            if not m.any():
                continue
            pos_i, quat_i = self._resolve_anchor_world(ph.target)
            ap = torch.where(m.unsqueeze(-1), pos_i, ap)
            aq = torch.where(m.unsqueeze(-1), quat_i, aq)
        return ap, aq

    def _use_rotation_mask(self) -> torch.Tensor:
        buf = getattr(self._env, "episode_length_buf", torch.zeros(self._num_envs, device=self._device, dtype=torch.long))
        pid = self._phase_id(buf)
        ur = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        for i, ph in enumerate(self._phases):
            m = pid == i
            if ph.use_rotation:
                ur = ur | m
        return ur

    def _default_palm_quat_world(self) -> torch.Tensor:
        """(N,4) when not using anchor rotation."""
        cfg = self._cfg
        if cfg.palm_orientation_mode == "fixed":
            e = torch.tensor(cfg.palm_euler_xyz, device=self._device, dtype=torch.float32).view(1, 3)
            return quat_from_euler_xyz(
                e[:, 0].expand(self._num_envs),
                e[:, 1].expand(self._num_envs),
                e[:, 2].expand(self._num_envs),
            )
        euler = _euler_palm_pickup_down(
            np.array(cfg.palm_normal_in_palm_frame, dtype=np.float64),
            np.array(cfg.world_down, dtype=np.float64),
            float(cfg.palm_yaw_offset_rad),
        )
        e = torch.tensor(euler, device=self._device, dtype=torch.float32).view(1, 3)
        return quat_from_euler_xyz(
            e[:, 0].expand(self._num_envs),
            e[:, 1].expand(self._num_envs),
            e[:, 2].expand(self._num_envs),
        )

    def _compute_wrist_world_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_pos, anchor_quat = self._anchor_pos_quat()
        use_rot = self._use_rotation_mask()
        off = self._offset_vec.expand(self._num_envs, 3)

        pos_world = anchor_pos + off
        pos_body = anchor_pos + quat_apply(anchor_quat, off)
        palm_pos = torch.where(use_rot.unsqueeze(-1), pos_body, pos_world)

        q_def = self._default_palm_quat_world()
        e_rel = torch.tensor(self._cfg.palm_euler_in_anchor_frame, device=self._device, dtype=torch.float32).view(1, 3)
        q_rel = quat_from_euler_xyz(e_rel[:, 0], e_rel[:, 1], e_rel[:, 2])
        q_rot = quat_mul(anchor_quat, q_rel.expand(self._num_envs, -1))
        palm_quat = torch.where(use_rot.unsqueeze(-1), q_rot, q_def)

        return _wrist_pose_from_palm_batch(
            palm_pos,
            palm_quat,
            self._wrist_in_palm_pos,
            self._wrist_in_palm_quat,
        )

    def _expand_diff_ik(self, out: torch.Tensor) -> torch.Tensor:
        wrist_pos_w, wrist_quat_w = self._compute_wrist_world_batch()
        robot = self._env.robot
        root_pose_w = robot.data.root_pose_w
        wrist_pos_b, wrist_quat_cmd = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], wrist_pos_w, wrist_quat_w
        )
        cmd = torch.cat([wrist_pos_b, wrist_quat_cmd], dim=-1)
        self._diff_ik_controller.set_command(cmd)
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[:, self._ee_body_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_w, ee_quat_w
        )
        jac = robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._diff_ik_jac_joint_ids_t]
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jac[:, :3, :] = torch.bmm(base_rot_matrix, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(base_rot_matrix, jac[:, 3:, :])
        joint_pos = robot.data.joint_pos[:, self._diff_ik_joint_pos_ids_t]
        joint_des = self._diff_ik_controller.compute(ee_pos_b, ee_quat_b, jac, joint_pos)
        for col in range(6):
            slot = self._diff_ik_slot_per_col[col]
            lo = self._lower[:, slot]
            hi = self._upper[:, slot]
            out[:, slot] = (2.0 * joint_des[:, col] - hi - lo) / (hi - lo + 1e-8)
        return torch.clamp(out, -1.0, 1.0)

    def expand(self, hand_actions: torch.Tensor) -> torch.Tensor:
        hand_actions = hand_actions.to(device=self._device, dtype=torch.float32)
        if getattr(self, "_hand_only_mode", False):
            return hand_actions
        out = torch.zeros((self._num_envs, self._full_dim), device=self._device, dtype=torch.float32)
        h_slots = torch.tensor(self._hand_slots, device=self._device, dtype=torch.long)
        out[:, h_slots] = hand_actions
        return self._expand_diff_ik(out)


class IkHandRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Policy outputs hand joints only; arm is filled by :class:`ArmIkHandActionExpander`."""

    def __init__(
        self,
        env: gym.Env,
        clip_actions: float | None,
        expander: ArmIkHandActionExpander,
    ):
        self._expander = expander
        self._num_hand = expander.num_hand
        super().__init__(env, clip_actions)
        self.num_actions = self._num_hand
        low = -1.0 if clip_actions is None else -float(clip_actions)
        high = 1.0 if clip_actions is None else float(clip_actions)
        self._policy_single_action_space = gym.spaces.Box(
            low=low, high=high, shape=(self._num_hand,), dtype=np.float32
        )
        self._policy_action_space = gym.vector.utils.batch_space(
            self._policy_single_action_space, expander._num_envs
        )

    @property
    def action_space(self) -> gym.Space:
        return self._policy_action_space

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        full = self._expander.expand(actions)
        return super().step(full)
