# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phased pregrasp / grasp + pour (GPU IK + trajectory).

- **joint_targets** (default): fixed ``arm_yaml`` + hand per phase (world-open-loop).
- **cup_relative**: wrist target = cup pose × offset each step (works with ``reset_cup_pos_noise``); uses ikpy
  :class:`video_teleop.core.video_teleop_control.VideoTeleopControl` (CPU loop per env — keep ``num_envs`` modest).
  Deformable cups use ``sim_element_quat_w`` as a frame proxy — it jitters each sim step; by default
  ``cup_relative_stable_cup_rotation`` freezes that orientation at episode reset while **position** still tracks
  the cup each step (stops arm “wandering”).

Optional **freeze_hand_after_script**: hand fixed from YAML during IK; policy dim = 1 dummy.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import yaml

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_mul
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from ik_rl_hand_vec_env import ArmIkHandActionExpander, IkRlHandArmCfg

logger = logging.getLogger(__name__)


class _PhaseSpec(NamedTuple):
    n_steps: int
    mode: str
    cup_relative: bool
    ik_trajectory_hand_only: bool
    static_tensor: torch.Tensor | None
    wrist_pos_cup: np.ndarray
    wrist_euler_cup: np.ndarray
    hvec: np.ndarray


def _root_pose_w_to_T_44(pq: torch.Tensor) -> np.ndarray:
    """World pose of robot root: ``(7,)`` pos + quat wxyz → ``(4,4)`` base-to-world."""
    from scipy.spatial.transform import Rotation as R

    pq = pq.detach().cpu().numpy().ravel()
    pos = pq[:3].astype(np.float64)
    w, x, y, z = [float(v) for v in pq[3:7]]
    rm = R.from_quat([x, y, z, w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rm
    T[:3, 3] = pos
    return T



class TimeOffsetArmIkHandExpander(ArmIkHandActionExpander):
    """Same as :class:`ArmIkHandActionExpander`, but IK trajectory phase index uses ``episode_length_buf - offset``.

    Kept in ``full_ik`` so ``ik_rl`` stays unchanged; used after scripted pregrasp/grasp steps.
    """

    def __init__(self, env: DirectRLEnv, cfg: IkRlHandArmCfg, trajectory_step_offset: int) -> None:
        super().__init__(env, cfg)
        self._full_ik_traj_off = max(0, int(trajectory_step_offset))

    def _buf_for_ik_traj(self) -> torch.Tensor:
        buf = getattr(
            self._env,
            "episode_length_buf",
            torch.zeros(self._num_envs, device=self._device, dtype=torch.long),
        )
        if self._full_ik_traj_off <= 0:
            return buf
        return torch.clamp(buf - self._full_ik_traj_off, min=0)

    def _anchor_pos_quat(self) -> tuple[torch.Tensor, torch.Tensor]:
        p0, q0 = self._resolve_anchor_world(self._phases[0].target)
        ap = torch.zeros_like(p0)
        aq = torch.zeros_like(q0)
        buf = self._buf_for_ik_traj()
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
        buf = self._buf_for_ik_traj()
        pid = self._phase_id(buf)
        ur = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        for i, ph in enumerate(self._phases):
            m = pid == i
            if ph.use_rotation:
                ur = ur | m
        return ur


def _project_root_from_this_file() -> Path:
    # .../ViTacLab/scripts/rsl_rl/full_ik/utils/full_ik_hand_vec_env.py -> parents[4] = ViTacLab
    return Path(__file__).resolve().parents[4]


def _ensure_source_on_path(project_root: Path) -> None:
    sd = project_root / "source"
    if sd.is_dir() and str(sd) not in sys.path:
        sys.path.insert(0, str(sd))


def _resolve_path(project_root: Path, p: str) -> Path:
    q = Path(p).expanduser()
    if q.is_absolute():
        return q
    return (project_root / q).resolve()


def _load_arm_joint_pos(yaml_path: Path) -> dict[str, float]:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    jp = data.get("joint_pos")
    if jp is None:
        jp = data.get("arm_joint_pos")
    if jp is None or not isinstance(jp, dict):
        raise ValueError(f"{yaml_path}: expected top-level joint_pos or arm_joint_pos (UR10e rad)")
    return {str(k): float(v) for k, v in jp.items()}


def _load_hand_shadow_order(yaml_path: Path) -> np.ndarray:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    seq = data.get("hand_joint_pos_shadow_order")
    if not isinstance(seq, list) or len(seq) != 24:
        raise ValueError(f"{yaml_path}: need hand_joint_pos_shadow_order: [24 floats]")
    return np.array([float(x) for x in seq], dtype=np.float64)


def _hand_joint_for_robot_name(name: str, hand_joints: np.ndarray, sh_names: list[str]) -> float:
    for sh_idx, sh_name in enumerate(sh_names):
        if sh_name in name or name.endswith(sh_name):
            return float(hand_joints[sh_idx])
    return 0.0


def joints_rad_to_normalized_action(
    env: DirectRLEnv,
    arm_joint_pos: dict[str, float],
    hand_vec24: np.ndarray,
    *,
    sh_names: list[str],
) -> torch.Tensor:
    """Build normalized actuated action (1, num_actuated) from arm dict + 24-dim Shadow order hand."""
    robot = env.robot
    joint_names = list(robot.joint_names)
    arm_re = re.compile(env.cfg.arm_joint_expr)
    hand_re = re.compile(env.cfg.hand_joint_expr)
    arm_indices = [i for i, n in enumerate(joint_names) if arm_re.match(n)]
    hand_indices = [i for i, n in enumerate(joint_names) if hand_re.match(n)]
    actuated = list(env.actuated_dof_indices)

    arm_list = [float(arm_joint_pos.get(joint_names[j], 0.0)) for j in arm_indices]
    full_dof = np.zeros(robot.num_joints, dtype=np.float64)
    for i, idx in enumerate(arm_indices):
        if i < len(arm_list):
            full_dof[idx] = arm_list[i]
    h = np.asarray(hand_vec24, dtype=np.float64).ravel()
    if h.size != 24:
        raise ValueError(f"hand_vec24 must have length 24, got {h.size}")
    for idx in hand_indices:
        full_dof[idx] = _hand_joint_for_robot_name(joint_names[idx], h, sh_names)

    actuated_np = full_dof[np.array(actuated, dtype=np.int64)]
    lower = env.robot_dof_lower_limits[0, actuated].detach().cpu().numpy()
    upper = env.robot_dof_upper_limits[0, actuated].detach().cpu().numpy()
    eps = 1e-6
    scale = np.where(upper - lower > eps, 2.0 * (actuated_np - lower) / (upper - lower) - 1.0, 0.0)
    t = torch.tensor(scale, dtype=torch.float32, device=env.device).unsqueeze(0)
    return torch.clamp(t, -1.0, 1.0)


def hand_shadow_to_normalized_slots(
    env: DirectRLEnv,
    hand_vec24: np.ndarray,
    *,
    sh_names: list[str],
    hand_joint_indices: list[int],
    hand_slots: list[int],
    joint_names: list[str],
) -> torch.Tensor:
    """Map 24-dim Shadow hand vector to normalized action values for hand slots only."""
    if len(hand_joint_indices) != len(hand_slots):
        raise ValueError("hand_joint_indices and hand_slots length mismatch")
    h = np.asarray(hand_vec24, dtype=np.float64).ravel()
    if h.size != 24:
        raise ValueError(f"hand_vec24 must have length 24, got {h.size}")
    lower = env.robot_dof_lower_limits[0].detach().cpu().numpy()
    upper = env.robot_dof_upper_limits[0].detach().cpu().numpy()
    out = np.zeros(len(hand_slots), dtype=np.float64)
    for i, ji in enumerate(hand_joint_indices):
        nm = joint_names[ji]
        val = _hand_joint_for_robot_name(nm, h, sh_names)
        lo = float(lower[ji])
        hi = float(upper[ji])
        if hi - lo > 1e-6:
            out[i] = 2.0 * (val - lo) / (hi - lo) - 1.0
        else:
            out[i] = 0.0
    t = torch.tensor(out, dtype=torch.float32, device=env.device).unsqueeze(0)
    return torch.clamp(t, -1.0, 1.0)


class PhasedArmIkHandExpander:
    """Scripted phases (joint replay and/or cup-relative wrist IK), then GPU IK pour + optional frozen hand."""

    def __init__(
        self,
        env: DirectRLEnv,
        ik_cfg: IkRlHandArmCfg,
        phase_schedule: list[dict[str, Any]],
        *,
        project_root: Path,
        freeze_hand_after_script: bool = False,
        freeze_hand_yaml: str | None = None,
        cup_relative_stable_cup_rotation: bool = True,
    ) -> None:
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs
        _ensure_source_on_path(project_root)
        from video_teleop.core.shadowhand_joints import shadowhand_joint_names

        self._sh_names = shadowhand_joint_names()
        self._freeze_hand_after_script = bool(freeze_hand_after_script)
        self._cup_relative_stable_cup_rotation = bool(cup_relative_stable_cup_rotation)
        self._fixed_hand_norm: torch.Tensor | None = None
        self._vc = None
        self._cup_ik_prev_arm: list[np.ndarray | None] | None = None
        self._cup_rel_stable_quat: torch.Tensor | None = None

        robot = env.robot
        joint_names = list(robot.joint_names)
        arm_re = re.compile(env.cfg.arm_joint_expr)
        self._joint_names = joint_names
        self._arm_indices = [i for i, n in enumerate(joint_names) if arm_re.match(n)]

        phase_specs: list[_PhaseSpec] = []
        first_arm_yaml: str | None = None
        any_cup_rel = False
        arm_override_total = 0

        for ph in phase_schedule:
            n = int(ph.get("env_steps", 0))
            if n < 0:
                raise ValueError("phase_schedule env_steps must be >= 0")
            mode = str(ph.get("mode", "joint_targets")).strip().lower().replace("-", "_")
            if mode in ("cup_relative", "relative_to_cup"):
                if not hasattr(env, "cup"):
                    raise ValueError(
                        f"phase {ph.get('name')!r}: mode=cup_relative needs env.cup (e.g. Pour deformable task)"
                    )
                wpc = ph.get("wrist_pos_in_cup_frame")
                if not isinstance(wpc, (list, tuple)) or len(wpc) != 3:
                    raise ValueError(
                        f"phase {ph.get('name')!r}: cup_relative needs wrist_pos_in_cup_frame: [x, y, z] (meters)"
                    )
                wec = ph.get("wrist_euler_in_cup_frame", [0.0, 0.0, 0.0])
                if not isinstance(wec, (list, tuple)) or len(wec) != 3:
                    raise ValueError(f"phase {ph.get('name')!r}: wrist_euler_in_cup_frame must be length-3 (rad)")
                hand_spec = ph.get("hand", "open")
                hand_yaml = ph.get("hand_yaml")
                if hand_yaml:
                    hvec = _load_hand_shadow_order(_resolve_path(project_root, str(hand_yaml)))
                elif str(hand_spec).lower() in ("open", "zeros", "zero"):
                    hvec = np.zeros(24, dtype=np.float64)
                else:
                    raise ValueError(f"phase {ph.get('name')!r}: set hand: open or hand_yaml")
                phase_specs.append(
                    _PhaseSpec(
                        n,
                        "cup_relative",
                        True,
                        False,
                        None,
                        np.array([float(wpc[0]), float(wpc[1]), float(wpc[2])], dtype=np.float64),
                        np.array([float(wec[0]), float(wec[1]), float(wec[2])], dtype=np.float64),
                        hvec,
                    )
                )
                any_cup_rel = True
                arm_override_total += n
                logger.info(
                    "[full_ik] phase %r: cup_relative steps=%d pos_cup=%s euler_cup=%s",
                    ph.get("name", "?"),
                    n,
                    wpc,
                    wec,
                )
            elif mode in ("ik_trajectory", "trajectory", "ik"):
                hand_spec = ph.get("hand", "open")
                hand_yaml = ph.get("hand_yaml")
                if hand_yaml:
                    hvec = _load_hand_shadow_order(_resolve_path(project_root, str(hand_yaml)))
                elif str(hand_spec).lower() in ("open", "zeros", "zero"):
                    hvec = np.zeros(24, dtype=np.float64)
                else:
                    raise ValueError(
                        f"phase {ph.get('name')!r}: mode=ik_trajectory supports hand: open or hand_yaml: path"
                    )
                phase_specs.append(
                    _PhaseSpec(
                        n,
                        "ik_trajectory",
                        False,
                        True,
                        None,
                        np.zeros(3),
                        np.zeros(3),
                        hvec,
                    )
                )
                logger.info(
                    "[full_ik] phase %r: ik_trajectory hand-only steps=%d hand=%s",
                    ph.get("name", "?"),
                    n,
                    ("open" if hand_yaml is None else hand_yaml),
                )
            else:
                arm_p = ph.get("arm_yaml") or ph.get("arm_yaml_path")
                if not arm_p:
                    raise ValueError(
                        f"phase {ph.get('name')!r}: missing arm_yaml (or use mode: cup_relative)"
                    )
                if first_arm_yaml is None:
                    first_arm_yaml = str(arm_p)
                arm_dict = _load_arm_joint_pos(_resolve_path(project_root, str(arm_p)))
                hand_spec = ph.get("hand", "open")
                hand_yaml = ph.get("hand_yaml")
                if hand_yaml:
                    hvec = _load_hand_shadow_order(_resolve_path(project_root, str(hand_yaml)))
                elif str(hand_spec).lower() in ("open", "zeros", "zero"):
                    hvec = np.zeros(24, dtype=np.float64)
                else:
                    raise ValueError(f"phase {ph.get('name')!r}: set hand: open or hand_yaml: path")
                t = joints_rad_to_normalized_action(env, arm_dict, hvec, sh_names=self._sh_names)
                phase_specs.append(
                    _PhaseSpec(n, "joint_targets", False, False, t, np.zeros(3), np.zeros(3), hvec),
                )
                arm_override_total += n
                logger.info(
                    "[full_ik] scripted phase %r: steps=%d arm_yaml=%s",
                    ph.get("name", "?"),
                    n,
                    arm_p,
                )

        self._phase_specs = phase_specs
        self._scripted_lens = [p.n_steps for p in phase_specs]
        self._scripted_total = int(sum(self._scripted_lens))
        self._arm_override_total = int(arm_override_total)
        if self._scripted_total < 0:
            raise RuntimeError("invalid scripted horizon")

        if any_cup_rel:
            self._cup_ik_prev_arm = [None] * self._num_envs
            if self._cup_relative_stable_cup_rotation:
                q0 = env.cup.data.sim_element_quat_w[:, 0, :].clone().to(device=self._device, dtype=torch.float32)
                self._cup_rel_stable_quat = q0 / q0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                logger.info(
                    "[full_ik] cup_relative: stable cup rotation ON (snapshot sim_element_quat_w[:,0] each reset)"
                )
            else:
                logger.warning(
                    "[full_ik] cup_relative: stable cup rotation OFF — deformable tet quat may jitter; arm can oscillate"
                )
            logger.warning(
                "[full_ik] cup_relative runs CPU ikpy once per env per step — prefer small --num_envs (e.g. ≤16)"
            )

        if self._arm_override_total > 0:
            self._base = TimeOffsetArmIkHandExpander(env, ik_cfg, self._arm_override_total)
        else:
            self._base = ArmIkHandActionExpander(env, ik_cfg)
        self.num_hand = self._base.num_hand
        self.num_actuated = self._base.num_actuated
        self._full_dim = self._base._full_dim  # noqa: SLF001
        self._num_envs_b = self._num_envs
        self._hand_slots = list(self._base._hand_slots)  # noqa: SLF001
        self._hand_joint_indices = list(self._base._hand_joint_indices)  # noqa: SLF001
        self._phase_hand_slots: list[torch.Tensor | None] = []
        for ph in self._phase_specs:
            if ph.ik_trajectory_hand_only:
                hs = hand_shadow_to_normalized_slots(
                    env,
                    ph.hvec,
                    sh_names=self._sh_names,
                    hand_joint_indices=self._hand_joint_indices,
                    hand_slots=self._hand_slots,
                    joint_names=self._joint_names,
                )
                self._phase_hand_slots.append(hs.to(device=self._device, dtype=torch.float32))
            else:
                self._phase_hand_slots.append(None)

        if self._freeze_hand_after_script:
            rel = (freeze_hand_yaml or "").strip()
            if not rel:
                for ph in reversed(phase_schedule):
                    hy = ph.get("hand_yaml")
                    if hy:
                        rel = str(hy).strip()
                        break
            if not rel:
                raise ValueError(
                    "freeze_hand_after_script=True requires freeze_hand_yaml in full_ik YAML "
                    "or hand_yaml on at least one phase_schedule entry"
                )
            hpath = _resolve_path(project_root, rel)
            hvec = _load_hand_shadow_order(hpath)
            if first_arm_yaml:
                first_arm = _load_arm_joint_pos(_resolve_path(project_root, first_arm_yaml))
            else:
                first_arm = {
                    joint_names[j]: float(env.robot.data.joint_pos[0, j].item()) for j in self._arm_indices
                }
            full = joints_rad_to_normalized_action(env, first_arm, hvec, sh_names=self._sh_names)
            h_slots = list(self._base._hand_slots)  # noqa: SLF001
            self._fixed_hand_norm = full[:, h_slots].to(device=self._device).clone()
            logger.info("[full_ik] freeze_hand_after_script: IK phase hand from %s", hpath)

    def _ensure_vc(self):
        if self._vc is None:
            from video_teleop.core.video_teleop_control import VideoTeleopControl

            self._vc = VideoTeleopControl()

    def _cup_relative_actuated_batch(
        self, ph: _PhaseSpec, ik_fallback: torch.Tensor, cup_quat_w: torch.Tensor
    ) -> torch.Tensor:
        """Normalized actuated actions (N, full_dim) from per-env cup pose + fixed offset in cup frame.

        Args:
            cup_quat_w: (N,4) wxyz orientation defining the cup frame (stable-per-episode or live).
        """
        self._ensure_vc()
        env = self._env
        n = self._num_envs
        dev = self._device
        cup_pos = env.cup.data.root_pos_w
        cup_quat = cup_quat_w.clone()
        cup_quat = cup_quat / cup_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        off = torch.tensor(ph.wrist_pos_cup, device=dev, dtype=torch.float32).view(1, 3).expand(n, 3)
        pos_w = cup_pos + quat_apply(cup_quat, off)
        r = torch.full((n,), float(ph.wrist_euler_cup[0]), device=dev, dtype=torch.float32)
        p = torch.full((n,), float(ph.wrist_euler_cup[1]), device=dev, dtype=torch.float32)
        y = torch.full((n,), float(ph.wrist_euler_cup[2]), device=dev, dtype=torch.float32)
        q_off = quat_from_euler_xyz(r, p, y)
        q_wrist = quat_mul(cup_quat, q_off)
        roll, pitch, yaw = euler_xyz_from_quat(q_wrist)

        out = ik_fallback.clone()
        root = env.robot.data.root_pose_w
        for i in range(n):
            self._vc.T_world_arm_base = _root_pose_w_to_T_44(root[i])
            if self._cup_ik_prev_arm is not None:
                self._vc._last_arm_joints = self._cup_ik_prev_arm[i]
            pos = pos_w[i].detach().cpu().numpy().astype(np.float64)
            eu = np.array(
                [float(roll[i].item()), float(pitch[i].item()), float(yaw[i].item())],
                dtype=np.float64,
            )
            targ = self._vc.compute(pos, eu, ph.hvec)
            if targ is None:
                continue
            if self._cup_ik_prev_arm is not None:
                self._cup_ik_prev_arm[i] = targ.arm_joints
            ad = {
                self._joint_names[self._arm_indices[j]]: float(targ.arm_joints[j])
                for j in range(min(len(targ.arm_joints), len(self._arm_indices)))
            }
            row = joints_rad_to_normalized_action(env, ad, targ.hand_joints, sh_names=self._sh_names)
            out[i] = row[0]
        return out

    def expand(self, hand_actions: torch.Tensor) -> torch.Tensor:
        hand_actions = hand_actions.to(device=self._device, dtype=torch.float32)
        if getattr(self._base, "_hand_only_mode", False):
            return hand_actions

        buf = getattr(
            self._env,
            "episode_length_buf",
            torch.zeros(self._num_envs, device=self._device, dtype=torch.long),
        )
        n_env = buf.shape[0]
        if self._cup_ik_prev_arm is not None:
            fresh = buf == 0
            if fresh.any():
                for i in torch.where(fresh)[0].tolist():
                    self._cup_ik_prev_arm[i] = None

        if (
            self._cup_rel_stable_quat is not None
            and hasattr(self._env, "cup")
            and self._env.cup is not None
        ):
            fresh = buf == 0
            if fresh.any():
                cq = self._env.cup.data.sim_element_quat_w[:, 0, :].clone().to(
                    device=self._device, dtype=torch.float32
                )
                cq = cq / cq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                self._cup_rel_stable_quat = torch.where(fresh.unsqueeze(-1), cq, self._cup_rel_stable_quat)

        if self._freeze_hand_after_script and self._fixed_hand_norm is not None:
            hand_for_ik = self._fixed_hand_norm.expand(n_env, -1)
        else:
            hand_for_ik = hand_actions
        ik_out = self._base.expand(hand_for_ik)

        if self._scripted_total <= 0 or not self._phase_specs:
            return ik_out

        out = ik_out.clone()
        in_ik = torch.ones(n_env, dtype=torch.bool, device=self._device)
        cum = 0
        for i, ph in enumerate(self._phase_specs):
            lo, hi = cum, cum + ph.n_steps
            m = (buf >= lo) & (buf < hi)
            cum = hi
            if not m.any():
                continue
            if ph.cup_relative:
                if self._cup_rel_stable_quat is not None:
                    cqw = self._cup_rel_stable_quat
                else:
                    lq = self._env.cup.data.sim_element_quat_w[:, 0, :].clone().to(
                        device=self._device, dtype=torch.float32
                    )
                    cqw = lq / lq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                full_b = self._cup_relative_actuated_batch(ph, ik_out, cqw)
                out = torch.where(m.unsqueeze(-1), full_b, out)
            elif ph.ik_trajectory_hand_only:
                hs = self._phase_hand_slots[i]
                if hs is not None:
                    hs_b = hs.to(device=self._device, dtype=torch.float32).expand(n_env, -1)
                    h_prev = out[:, self._hand_slots]
                    h_new = torch.where(m.unsqueeze(-1), hs_b, h_prev)
                    out[:, self._hand_slots] = h_new
            else:
                assert ph.static_tensor is not None
                row = ph.static_tensor.to(device=self._device, dtype=torch.float32).expand(n_env, -1)
                out = torch.where(m.unsqueeze(-1), row, out)
            if ph.cup_relative or (ph.static_tensor is not None):
                in_ik = in_ik & ~m

        out = torch.where(in_ik.unsqueeze(-1), ik_out, out)
        return out


class PhasedIkHandRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Phased expander: policy hand vector, or 1-dim dummy when hand is frozen after scripted phases."""

    def __init__(
        self,
        env,
        clip_actions: float | None,
        expander: PhasedArmIkHandExpander,
        *,
        freeze_hand_after_script: bool = False,
    ):
        self._expander = expander
        self._num_hand = expander.num_hand
        self._freeze_hand_after_script = bool(freeze_hand_after_script)
        super().__init__(env, clip_actions)
        # Parent sets num_actions from full env (arm+hand); override to policy-facing dim.
        self.num_actions = 1 if self._freeze_hand_after_script else self._num_hand
        if self.clip_actions is not None:
            self._modify_action_space()
        import gymnasium as gym

        low = -1.0 if clip_actions is None else -float(clip_actions)
        high = 1.0 if clip_actions is None else float(clip_actions)
        pol_dim = 1 if self._freeze_hand_after_script else self._num_hand
        self._policy_single_action_space = gym.spaces.Box(
            low=low, high=high, shape=(pol_dim,), dtype=np.float32
        )
        self._policy_action_space = gym.vector.utils.batch_space(
            self._policy_single_action_space,
            expander._num_envs_b,
        )

    @property
    def action_space(self):
        return self._policy_action_space

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        full = self._expander.expand(actions)
        return super().step(full)
