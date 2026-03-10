from __future__ import annotations

from dataclasses import MISSING
from typing import Optional, Sequence

import torch

from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from ViTacLab.utils.cutMeshNode import cut_mesh_with_world_plane


@configclass
class BreakableObjectCfg:
    """Config for a breakable rigid object with an attached contact sensor."""

    # required sub-configs: marked with dataclasses.MISSING as default
    rigid_cfg: RigidObjectCfg = MISSING
    contact_cfg: ContactSensorCfg = MISSING

    # Which mesh root to cut; by default use the rigid body's prim path.
    mesh_root_path: Optional[str] = None

    # breaking / cutting parameters
    break_force_threshold: float = 50.0
    max_cuts_per_env: int = 1
    success_min_cuts: int = 1
    # "force" | "opposite_force" | "world_z"
    cut_direction_mode: str = "force"


class BreakableObject:
    """Runtime wrapper around a rigid object + contact sensor + breaking logic."""

    def __init__(self, cfg: BreakableObjectCfg):
        self.cfg = cfg
        # underlying assets (created later in scene using these cfgs)
        self.rigid = RigidObject(cfg.rigid_cfg)
        self.contact = ContactSensor(cfg.contact_cfg)

        # per-env cut counters; lazily sized when we know num_envs/device
        self._cut_counts: Optional[torch.Tensor] = None

    #
    # Scene registration helpers
    #
    def register_to_scene(self, scene, rigid_name: str, contact_name: str):
        """Register internal objects to the InteractiveScene."""
        scene.rigid_objects[rigid_name] = self.rigid
        scene.sensors[contact_name] = self.contact

    def initialize_counters(self, num_envs: int, device: torch.device | str):
        """Allocate per-env cut counters."""
        self._cut_counts = torch.zeros(
            (num_envs,), dtype=torch.int32, device=device
        )

    #
    # Core breaking logic
    #
    def step_breaking(self):
        """Check contact forces and cut mesh if breaking conditions are met."""
        if self._cut_counts is None:
            # counters not initialized; nothing to do
            return

        contact_data = self.contact.data
        net_forces_w: torch.Tensor = contact_data.net_forces_w
        contact_pos_w: torch.Tensor = contact_data.contact_pos_w

        if net_forces_w is None or contact_pos_w is None:
            return

        force_threshold = self.cfg.break_force_threshold
        max_cuts_per_env = max(int(self.cfg.max_cuts_per_env), 1)
        cut_mode = self.cfg.cut_direction_mode or "force"

        force_norm = torch.linalg.norm(net_forces_w, dim=-1)  # [E, B]
        num_envs, num_bodies = force_norm.shape

        for env_id in range(num_envs):
            if self._cut_counts[env_id] >= max_cuts_per_env:
                continue

            max_val, max_body = torch.max(force_norm[env_id], dim=0)
            if max_val.item() < force_threshold:
                continue

            body_contact_pos = contact_pos_w[env_id, max_body]  # [F, 3]
            valid_mask = ~torch.isnan(body_contact_pos[..., 0])
            if not torch.any(valid_mask):
                continue

            contact_idx = torch.nonzero(valid_mask, as_tuple=False)[0, 0].item()
            contact_point = body_contact_pos[contact_idx]
            contact_force = net_forces_w[env_id, max_body]

            if cut_mode == "opposite_force":
                dir_vec = -contact_force
            elif cut_mode == "world_z":
                dir_vec = torch.tensor(
                    [0.0, 0.0, 1.0],
                    device=contact_force.device,
                    dtype=contact_force.dtype,
                )
            else:
                dir_vec = contact_force

            dir_norm = torch.linalg.norm(dir_vec)
            if dir_norm.item() <= 1e-6:
                continue

            plane_center = contact_point.detach().cpu().numpy()
            plane_normal = (dir_vec / dir_norm).detach().cpu().numpy()

            mesh_root_path = (
                self.cfg.mesh_root_path or self.cfg.rigid_cfg.prim_path
            )
            cut_mesh_with_world_plane(mesh_root_path, plane_center, plane_normal)

            self._cut_counts[env_id] += 1

    #
    # Episode helpers
    #
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset rigid object pose and clear cut counters for given envs."""
        if env_ids is None:
            env_ids = self.rigid._ALL_INDICES

        # reset rigid object
        root_state = self.rigid.data.default_root_state[env_ids].clone()
        self.rigid.write_root_state_to_sim(root_state, env_ids=env_ids)

        # reset counters
        if self._cut_counts is not None:
            self._cut_counts[env_ids] = 0

    @property
    def cut_counts(self) -> Optional[torch.Tensor]:
        return self._cut_counts

    def is_success(self) -> torch.Tensor:
        """Return boolean tensor per env indicating success."""
        if self._cut_counts is None:
            return torch.zeros(0, dtype=torch.bool)
        min_cuts = max(int(self.cfg.success_min_cuts), 1)
        return self._cut_counts >= min_cuts

