# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-only RL: map hand actions to full robot commands via GPU batched differential IK (UR10e arm).

Each arm follows waypoints for ``ee_body_name`` (e.g. ``wrist_3_link``): ``pos`` (m), ``quat`` (**Isaac Lab wxyz**),
``steps`` (env steps; ``-1`` = until episode end).

**Multi-env:** by default ``pos`` is **env-local** (same convention as task obs: root at ``scene.env_origins``). The
expander adds ``env.scene.env_origins`` to obtain **simulation world** targets for differential IK. Set
``IkRlHandArmCfg.add_env_origins_to_waypoint_pos=False`` only if your YAML stores **global** world ``pos`` (single-env
debug).

If the env exposes ``ik_rl_trajectory_xyz_offset`` (N,3), it is **added** in env-local space before the origins shift.

Waypoint phase uses ``env.episode_length_buf`` (steps since reset). IK+RL training scripts set
``init_at_random_ep_len=False`` so RSL-RL does not randomize that buffer at rollout start (which would desync phases).

**IK speed:** `dls` uses damping ``lambda_val`` (ik_rl default **0.005** when `ik_lambda` is unset; Isaac Lab **0.01**).
Use `ik_delta_scale` > 1 in YAML for an extra per-step joint-space gain.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EeWaypoint:
    """One EE target segment: ``ee_body`` pose; ``quat_wxyz`` is Isaac Lab (w,x,y,z); ``pos`` is env-local unless cfg disables origins."""

    pos_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    env_steps: int


def parse_waypoints_list(raw: list | tuple) -> tuple[EeWaypoint, ...]:
    """Parse YAML trajectory: list of ``{pos: [x,y,z], quat: [w,x,y,z], steps: int}`` with **wxyz** quaternions."""

    if not raw:
        raise ValueError("trajectory must be a non-empty list of {pos, quat, steps}")
    out: list[EeWaypoint] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"trajectory[{i}] must be a dict, got {type(item)}")
        pos = item.get("pos")
        quat = item.get("quat")
        steps = item.get("steps")
        if pos is None or quat is None or steps is None:
            raise KeyError(f"trajectory[{i}] needs keys pos, quat, steps")
        pos_t = tuple(float(x) for x in pos)
        quat_t = tuple(float(x) for x in quat)
        if len(pos_t) != 3:
            raise ValueError(f"trajectory[{i}].pos must have length 3")
        if len(quat_t) != 4:
            raise ValueError(f"trajectory[{i}].quat must be length-4 Isaac Lab wxyz [w,x,y,z]")
        qn = np.asarray(quat_t, dtype=np.float64).ravel()
        qn = qn / (np.linalg.norm(qn) + 1e-12)
        quat_n = tuple(float(x) for x in qn.tolist())
        out.append(EeWaypoint(pos_xyz=pos_t, quat_wxyz=quat_n, env_steps=int(steps)))
    return tuple(out)


@dataclass
class IkRlHandArmCfg:
    """Differential IK: EE targets from explicit waypoints only."""

    waypoints: tuple[EeWaypoint, ...]
    #: If True (default), waypoint ``pos`` is env-local; add ``env.scene.env_origins`` for world-frame IK.
    add_env_origins_to_waypoint_pos: bool = True
    ee_body_name: str = "wrist_3_link"
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls"
    ik_lambda: float | None = None
    #: pinv / svd / trans: scales Jacobian output (Isaac default ``k_val`` = 1.0). Ignored for ``dls``.
    ik_k_val: float | None = None
    #: Multiplies the joint-space delta from the controller each step (>1 = faster tracking, may overshoot).
    ik_delta_scale: float = 1.0


class ArmIkHandActionExpander:
    """Maps normalized hand actions to full (arm+hand) normalized actions for :class:`DirectRLEnv`.

    If the env already lists **only hand** DOFs as actuated (fixed arm in USD), runs in **hand-only** mode: no IK,
    :meth:`expand` returns ``hand_actions`` unchanged.
    """

    def __init__(self, env: DirectRLEnv, cfg: IkRlHandArmCfg, robot: "Articulation | None" = None):
        self._env = env
        self._cfg = cfg
        self._robot = robot if robot is not None else getattr(env, "robot", None)
        if self._robot is None:
            raise RuntimeError(
                "ArmIkHandActionExpander: env has no `robot`; pass robot=<Articulation> "
                "(e.g. env.right_hand / env.left_hand after multi_agent_to_single_agent)."
            )
        self._device = env.device
        self._num_envs = env.num_envs

        if not cfg.waypoints:
            raise ValueError("IkRlHandArmCfg.waypoints must be non-empty")

        arm_re = re.compile(env.cfg.arm_joint_expr)
        hand_re = re.compile(env.cfg.hand_joint_expr)
        names = self._robot.joint_names
        actuated = list(env.actuated_dof_indices)

        self._arm_slots: list[int] = []
        self._hand_slots: list[int] = []
        self._arm_joint_indices: list[int] = []

        for slot, ji in enumerate(actuated):
            n = names[ji]
            if arm_re.match(n):
                self._arm_slots.append(slot)
                self._arm_joint_indices.append(ji)
            elif hand_re.match(n):
                self._hand_slots.append(slot)

        self.num_hand = len(self._hand_slots)
        self.num_arm = len(self._arm_slots)
        self._full_dim = len(actuated)

        if self.num_arm == 0 and self.num_hand > 0 and self.num_hand == self._full_dim:
            self._hand_only_mode = True
            self._hand_slots_t = None
            lower = env.robot_dof_lower_limits
            upper = env.robot_dof_upper_limits
            self._lower = lower[:, actuated].clone()
            self._upper = upper[:, actuated].clone()
            logger.info(
                "[ik_rl] ArmIkHandActionExpander: hand-only actuated DOFs (no arm in action space); "
                "skipping differential IK."
            )
            return

        self._hand_only_mode = False
        self._hand_slots_t = torch.tensor(self._hand_slots, device=self._device, dtype=torch.long)
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

        ik_params: dict[str, float] | None = None
        if cfg.ik_method == "dls":
            # Lower damping ⇒ larger joint steps per env tick (Isaac default lambda is 0.01).
            lam = float(cfg.ik_lambda) if cfg.ik_lambda is not None else 0.005
            ik_params = {"lambda_val": lam}
        elif cfg.ik_method in ("pinv", "svd", "trans") and cfg.ik_k_val is not None:
            ik_params = {"k_val": float(cfg.ik_k_val)}
        diff_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=cfg.ik_method,
            ik_params=ik_params,
        )
        self._diff_ik_controller = DifferentialIKController(
            diff_cfg, num_envs=self._num_envs, device=str(self._device)
        )
        body_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        if len(body_ids) < 1:
            raise RuntimeError(
                f"IK: no body matching {cfg.ee_body_name!r}. Sample: {self._robot.body_names[:30]}"
            )
        self._ee_body_idx = int(body_ids[0])
        j_ids, j_names = self._robot.find_joints(env.cfg.arm_joint_expr)
        if len(j_ids) != 6:
            raise RuntimeError(f"IK: expected 6 arm joints, got {len(j_ids)}: {j_names}")
        if set(j_ids) != set(self._arm_joint_indices):
            raise RuntimeError("IK: arm joint indices mismatch between find_joints and actuated list")

        if self._robot.is_fixed_base:
            self._ee_jacobi_idx = self._ee_body_idx - 1
            jac_joint_ids = list(j_ids)
        else:
            self._ee_jacobi_idx = self._ee_body_idx
            jac_joint_ids = [int(j) + 6 for j in j_ids]
        self._diff_ik_joint_pos_ids_t = torch.tensor(j_ids, device=self._device, dtype=torch.long)
        self._diff_ik_jac_joint_ids_t = torch.tensor(jac_joint_ids, device=self._device, dtype=torch.long)
        self._diff_ik_slot_per_col = [self._arm_slots[self._arm_joint_indices.index(jid)] for jid in j_ids]

        self._waypoints = cfg.waypoints
        if getattr(self._env, "episode_length_buf", None) is None:
            logger.warning("[ik_rl] env has no episode_length_buf; trajectory timing may be wrong.")

    @property
    def num_actuated(self) -> int:
        return self._full_dim

    def _phase_id(self, buf: torch.Tensor) -> torch.Tensor:
        """Map ``episode_length_buf`` → waypoint index per env.

        Segment ``i`` with finite ``env_steps`` holds for ``buf in [cum, cum + env_steps)`` (half-open).
        ``env_steps < 0`` means hold that waypoint until episode end.

        **Dual-arm:** same ``episode_length_buf`` for both arms; different ``trajectory_right`` / ``trajectory_left``
        ``steps`` ⇒ the two arms switch to waypoint 2 at different times.
        """
        n = buf.shape[0]
        device = buf.device
        pid = torch.zeros(n, dtype=torch.long, device=device)
        cum = 0
        for i, w in enumerate(self._waypoints):
            if w.env_steps < 0:
                pid = torch.where(buf >= cum, torch.full_like(pid, i), pid)
                return pid
            seg = int(w.env_steps)
            if seg <= 0:
                continue
            nxt = cum + seg
            m = (buf >= cum) & (buf < nxt)
            pid = torch.where(m, torch.full_like(pid, i), pid)
            cum = nxt
        pid = torch.where(buf >= cum, torch.full_like(pid, len(self._waypoints) - 1), pid)
        return pid

    def _ee_pose_world_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame EE pose (N,3), (N,4) wxyz from waypoints + optional per-env xyz offset."""
        buf = getattr(
            self._env,
            "episode_length_buf",
            torch.zeros(self._num_envs, device=self._device, dtype=torch.long),
        )
        pid = self._phase_id(buf)
        pos_out = torch.zeros(self._num_envs, 3, device=self._device, dtype=torch.float32)
        quat_out = torch.zeros(self._num_envs, 4, device=self._device, dtype=torch.float32)

        for i, w in enumerate(self._waypoints):
            m = pid == i
            if not m.any():
                continue
            p = torch.tensor(w.pos_xyz, device=self._device, dtype=torch.float32).view(1, 3).expand(self._num_envs, -1)
            q = torch.tensor(w.quat_wxyz, device=self._device, dtype=torch.float32).view(1, 4).expand(self._num_envs, -1)
            pos_out = torch.where(m.unsqueeze(-1), p, pos_out)
            quat_out = torch.where(m.unsqueeze(-1), q, quat_out)

        offset = getattr(self._env, "ik_rl_trajectory_xyz_offset", None)
        if offset is not None:
            pos_out = pos_out + offset.to(device=pos_out.device, dtype=pos_out.dtype)
        if self._cfg.add_env_origins_to_waypoint_pos:
            scene = getattr(self._env, "scene", None)
            origins = getattr(scene, "env_origins", None) if scene is not None else None
            if origins is not None:
                pos_out = pos_out + origins.to(device=pos_out.device, dtype=pos_out.dtype)
        return pos_out, quat_out

    def _expand_diff_ik(self, out: torch.Tensor) -> torch.Tensor:
        wrist_pos_w, wrist_quat_w = self._ee_pose_world_batch()
        robot = self._robot
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
        s = float(self._cfg.ik_delta_scale)
        if abs(s - 1.0) > 1e-9:
            joint_des = joint_pos + (joint_des - joint_pos) * s
        for col in range(6):
            slot = self._diff_ik_slot_per_col[col]
            lo = self._lower[:, slot]
            hi = self._upper[:, slot]
            out[:, slot] = (2.0 * joint_des[:, col] - hi - lo) / (hi - lo + 1e-8)
        return torch.clamp(out, -1.0, 1.0)

    def expand(self, hand_actions: torch.Tensor) -> torch.Tensor:
        hand_actions = hand_actions.to(device=self._device, dtype=torch.float32)
        if self._hand_only_mode:
            return hand_actions
        assert self._hand_slots_t is not None
        out = torch.zeros((self._num_envs, self._full_dim), device=self._device, dtype=torch.float32)
        out[:, self._hand_slots_t] = hand_actions
        return self._expand_diff_ik(out)


class DualArmIkHandActionExpander:
    """Two :class:`ArmIkHandActionExpander` (right / left) for MARL dual-arm after ``multi_agent_to_single_agent``."""

    def __init__(self, env: DirectRLEnv, cfg_right: IkRlHandArmCfg, cfg_left: IkRlHandArmCfg):
        self._right = ArmIkHandActionExpander(env, cfg_right, robot=env.right_hand)
        self._left = ArmIkHandActionExpander(env, cfg_left, robot=env.left_hand)
        wr = cfg_right.waypoints
        wl = cfg_left.waypoints
        if wr and wl and wr[0].env_steps >= 0 and wl[0].env_steps >= 0 and wr[0].env_steps != wl[0].env_steps:
            logger.info(
                "[ik_rl] trajectory_right[0].steps (%s) != trajectory_left[0].steps (%s): "
                "both arms share episode_length_buf, so they switch to waypoint 2 at different times.",
                wr[0].env_steps,
                wl[0].env_steps,
            )
        self.num_hand = self._right.num_hand + self._left.num_hand
        self._num_envs = self._right._num_envs

    @property
    def num_actuated(self) -> int:
        return self._right.num_actuated + self._left.num_actuated

    def expand(self, hand_actions: torch.Tensor) -> torch.Tensor:
        hand_actions = hand_actions.to(device=self._right._device, dtype=torch.float32)
        nr, nl = self._right.num_hand, self._left.num_hand
        fr = self._right.expand(hand_actions[:, :nr])
        fl = self._left.expand(hand_actions[:, nr : nr + nl])
        return torch.cat([fr, fl], dim=-1)


class IkHandRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Policy outputs hand joints only; arm is filled by :class:`ArmIkHandActionExpander`."""

    def __init__(
        self,
        env: gym.Env,
        clip_actions: float | None,
        expander: ArmIkHandActionExpander | DualArmIkHandActionExpander,
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


def build_ik_cfg_from_trajectory_args(args: Any, *, arm: Literal["single", "right", "left"] = "single") -> IkRlHandArmCfg:
    """Build :class:`IkRlHandArmCfg` from argparse namespace (YAML-merged)."""

    if arm == "single":
        raw = getattr(args, "trajectory", None)
    elif arm == "right":
        raw = getattr(args, "trajectory_right", None) or getattr(args, "trajectory", None)
    else:
        raw = getattr(args, "trajectory_left", None) or getattr(args, "trajectory", None)
    if raw is None:
        raise ValueError(
            "IK trajectory missing: set `trajectory` in the ik_rl YAML (list of {pos, quat, steps}), "
            "and for dual-arm optionally `trajectory_right` / `trajectory_left`."
        )
    waypoints = parse_waypoints_list(raw)
    ee = getattr(args, "ee_body", None) or "wrist_3_link"
    return IkRlHandArmCfg(
        waypoints=waypoints,
        add_env_origins_to_waypoint_pos=not bool(getattr(args, "ik_waypoints_world_frame", False)),
        ee_body_name=str(ee),
        ik_method=getattr(args, "ik_method", "dls"),
        ik_lambda=getattr(args, "ik_lambda", None),
        ik_k_val=getattr(args, "ik_k_val", None),
        ik_delta_scale=float(getattr(args, "ik_delta_scale", 1.0)),
    )
