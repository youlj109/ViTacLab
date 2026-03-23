from __future__ import annotations

from typing import Sequence

from ViTacLab.assets.object.breakable import BreakableObject
from ViTacLab.tasks.direct.simple_gripper.forge_env import ForgeEnv


class ForgeBreakableEnv(ForgeEnv):
    """ForgeEnv 扩展版本，在 HeldAsset 上附加 BreakableObject 破坏逻辑。

    复用 ForgeEnv 的观测、控制和奖励逻辑，仅增加:
    - breakable: BreakableObject
    - 基于接触力的网格切割
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # 提前定义属性，避免父类构造期间访问时报错
        self.breakable: BreakableObject | None = None

        # 先按 simple_gripper 的逻辑构建场景与任务（包括 held_asset）
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # 构建 breakable 包装：复用现有 held_asset 的 prim 路径
        if hasattr(self.cfg, "breakable") and getattr(self.cfg, "breakable") is not None:
            held = self.scene.rigid_objects.get("held_asset", None)
            if held is not None:
                # 将 BreakableObjectCfg 绑定到已有 HeldAsset 上
                self.cfg.breakable.rigid_cfg.prim_path = held.cfg.prim_path
                self.cfg.breakable.contact_cfg.prim_path = held.cfg.prim_path
                if not self.cfg.breakable.mesh_root_path:
                    self.cfg.breakable.mesh_root_path = held.cfg.prim_path

                self.breakable = BreakableObject(self.cfg.breakable)
                self.breakable.initialize_counters(self.num_envs, self.device)
                self.breakable.register_to_scene(
                    self.scene, rigid_name="breakable_object", contact_name="breakable_contact"
                )

    def _apply_action(self) -> None:
        # 完全复用 ForgeEnv 的动作应用与奖励前向计算
        super()._apply_action()

        # 追加 breakable 破坏逻辑（只改变几何，不改动原任务奖励/终止逻辑）
        if self.breakable is not None:
            self.breakable.step_breaking()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 先按原逻辑重置机器人/任务物体/随机化
        super()._reset_idx(env_ids)

        # 再重置 breakable 的计数和刚体状态，使网格回到未切割形态
        if self.breakable is not None:
            self.breakable.reset(env_ids)

