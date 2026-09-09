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
    # PegInsert MRO 先于 ForgeTask：须在子类显式覆盖的项写在这里。
    # 位置随机沿用 PegInsert.hand_init_pos_noise（不写成全 0，与 env 钉点/任务解耦）。
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    # Reset 后 peg 相对手爪的抖动关掉；否则世界系下几何仍多一层随机。
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05
    hand_init_pos: list = [0.0, 0.0, 0.05]
    # PegInsert MRO 先于 ForgeTask：须在子类显式覆盖的项写在这里。
    # 位置随机沿用 PegInsert.hand_init_pos_noise（不写成全 0，与 env 钉点/任务解耦）。
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    # Reset 后 peg 相对手爪的抖动关掉；否则世界系下几何仍多一层随机。
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    success_threshold: float = 0.2


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05
    hand_init_pos: list = [0.0, 0.0, 0.05]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    # Reset 后 peg 相对手爪的抖动关掉；否则世界系下几何仍多一层随机。
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    success_threshold: float = 0.7
    ee_success_yaw: float = 1.0
