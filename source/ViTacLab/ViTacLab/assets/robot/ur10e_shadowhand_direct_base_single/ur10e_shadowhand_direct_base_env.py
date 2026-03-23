from collections.abc import Sequence
import re

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from .ur10e_shadowhand_direct_base_cfg import (
    UR10eShadowHandTacSLSceneCfg,
    build_ur10e_shadowhand_tactile_sensor_cfgs,
    build_ur10e_shadowhand_third_person_camera_cfg,
)


@torch.jit.script
def _scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def spawn_factory_table(prim_path: str = "/World/envs/env_.*/Table") -> None:
    """Spawn the same Seattle lab table as Factory/Forge."""
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func(
        prim_path,
        cfg,
        translation=(0.55, 0.0, 0.0),
        orientation=(0.70711, 0.0, 0.0, 0.70711),
    )


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

