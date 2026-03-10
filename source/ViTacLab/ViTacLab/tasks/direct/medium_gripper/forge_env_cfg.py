from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ViTacLab.assets.object.breakable import BreakableObjectCfg
from ViTacLab.tasks.direct.simple_gripper.forge_env_cfg import (
    ForgeTaskGearMeshCfg,
    ForgeTaskNutThreadCfg,
    ForgeTaskPegInsertCfg,
)


@configclass
class ForgeBreakableMixinCfg:
    """为 simple_gripper 任务增加 BreakableObjectCfg 配置的 Mixin。

    使用 HeldAsset 作为可破坏刚体，仅通过 prim_path 绑定已有资产，不额外 spawn。
    """

    breakable: BreakableObjectCfg = BreakableObjectCfg(
        rigid_cfg=RigidObjectCfg(
            prim_path="/World/envs/env_.*/HeldAsset",
            spawn=None,  # 仅创建句柄，复用 FactoryEnv 已经创建的 HeldAsset prim
        ),
        contact_cfg=ContactSensorCfg(
            prim_path="/World/envs/env_.*/HeldAsset",
            update_period=0.0,
            history_length=4,
            track_contact_points=True,
            debug_vis=False,
            # 简单起见，对 /World 下的所有物体开放接触统计
            filter_prim_paths_expr=["/World/.*"],
        ),
        mesh_root_path="/World/envs/env_.*/HeldAsset",
        break_force_threshold=5.0,
        max_cuts_per_env=1,
        success_min_cuts=1,
        cut_direction_mode="force",
    )


@configclass
class ForgeTaskPegInsertBreakableCfg(ForgeTaskPegInsertCfg, ForgeBreakableMixinCfg):
    """PegInsert 任务的可破坏版本 EnvCfg。

    继承原 ForgeTaskPegInsertCfg 的任务/控制配置，仅追加 breakable 字段。
    """


@configclass
class ForgeTaskGearMeshBreakableCfg(ForgeTaskGearMeshCfg, ForgeBreakableMixinCfg):
    """GearMesh 任务的可破坏版本 EnvCfg。"""


@configclass
class ForgeTaskNutThreadBreakableCfg(ForgeTaskNutThreadCfg, ForgeBreakableMixinCfg):
    """NutThread 任务的可破坏版本 EnvCfg。"""

