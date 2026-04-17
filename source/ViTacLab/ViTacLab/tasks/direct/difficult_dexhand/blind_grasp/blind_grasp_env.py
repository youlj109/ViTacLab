from __future__ import annotations

from isaaclab.assets import RigidObject

from ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env import UR10eShadowHandPickupEnv

from .blind_grasp_env_cfg import UR10eShadowHandBlindGraspEnvCfg


class UR10eShadowHandBlindGraspEnv(UR10eShadowHandPickupEnv):
    """Pickup with a kinematic garbage can; the cube resets inside the bin (same rewards/obs as hand_pickup)."""

    cfg: UR10eShadowHandBlindGraspEnvCfg

    def _setup_task_scene(self) -> None:
        spawn = self.cfg.trash_can_cfg.spawn.replace(scale=self.cfg.trash_can_scale)
        tc_cfg = self.cfg.trash_can_cfg.replace(spawn=spawn)
        self.trash_can = RigidObject(tc_cfg)
        self.scene.rigid_objects["trash_can"] = self.trash_can
        super()._setup_task_scene()
