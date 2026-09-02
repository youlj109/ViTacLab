# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert, RobotCfg


@configclass
class ForgeTask(FactoryTask): 
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    robot_cfg: RobotCfg = RobotCfg(franka_fingerpad_length=0.005)      # 必须在子类中显式覆盖 # 0.01760
    contact_penalty_scale: float = 0.2
    hand_init_pos: list = [0.0, 0.0, 0.1]


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05
