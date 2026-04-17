# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi-Peg: dual UR10e + Shadow Hand; Cosmos washer + nail as rigid bodies; peg-style keypoint rewards (no ForgePegInsert)."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FixedAssetCfg, HeldAssetCfg, RobotCfg

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_cfg import (
    UR10E_DUAL_SHADOWHAND_LEFT_CFG,
    UR10E_DUAL_SHADOWHAND_RIGHT_CFG,
    UR10eDualShadowHandDirectSceneCfg,
)

WASHER_USD_PATH: str = "source/ViTacLab/ViTacLab/assets/data/Objects/cosmos_assets/b_Washers/b_Washers.usd"
NAIL_USD_PATH: str = "source/ViTacLab/ViTacLab/assets/data/Objects/cosmos_assets/b_Nails/b_Nails.usd"


def _ur10e_bi_peg_right_arm_joint_pos() -> dict[str, float]:
    """Default right-arm analytic pose (teleop-style); override via ``right_robot_cfg`` for your layout."""

    return {
        "shoulder_pan_joint": -0.2381820505749035,
        "shoulder_lift_joint": -0.7457377055913127,
        "elbow_joint": 0.7105199793481264,
        "wrist_1_joint": -1.5355786007566066,
        "wrist_2_joint": -1.5707963265900005,
        "wrist_3_joint": -1.8089821043693861,
    }


def _ur10e_bi_peg_left_arm_joint_pos() -> dict[str, float]:
    """Default left-arm analytic pose; override via ``left_robot_cfg``."""

    return {
        "shoulder_pan_joint": -0.0647053798972387,
        "shoulder_lift_joint": -0.6931192918684786,
        "elbow_joint": 0.9688935001938827,
        "wrist_1_joint": -1.8465705353261026,
        "wrist_2_joint": -1.5707963265904117,
        "wrist_3_joint": 1.5060872198977679,
    }


UR10E_DUAL_BIPEG_NUM_ACTUATED_DOFS: int = 30
_BIPEG_OBS_TACTILE_PAD: int = 5 * (1 + 2)


def _bi_peg_obs_dim_per_agent() -> int:
    return 30 + 30 + 3 + 4 + 3 + 4 + 1 + 30 + 30 + _BIPEG_OBS_TACTILE_PAD


def _bi_peg_state_dim() -> int:
    return 2 * _bi_peg_obs_dim_per_agent()


@configclass
class BiPegTaskCfg:
    """Geometry + reward knobs for :func:`isaaclab_tasks.direct.factory.factory_utils` peg_insert-style terms."""

    name: str = "peg_insert"

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg(
        usd_path=WASHER_USD_PATH,
        diameter=0.01,
        height=0.025,
        base_height=0.0,
        friction=0.75,
        mass=0.05,
    )
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg(
        usd_path=NAIL_USD_PATH,
        diameter=0.008,
        height=0.05,
        friction=0.75,
        mass=0.02,
    )
    robot_cfg: RobotCfg = RobotCfg(friction=0.75)

    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: tuple[float, float] = (5, 4)
    keypoint_coef_coarse: tuple[float, float] = (50, 2)
    keypoint_coef_fine: tuple[float, float] = (100, 0)
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.1
    ee_success_yaw: float = 0.0


@configclass
class BiPegObjectPosesCfg:
    """Root poses in **environment frame** (m, wxyz). Written on reset."""

    fixed_pos: tuple[float, float, float] = (0.6, 0.0, 0.2)
    fixed_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    held_pos: tuple[float, float, float] = (0.8, 0.0, 0.20)
    held_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@configclass
class UR10eDualShadowHandBiPegEnvCfg(DirectMARLEnvCfg):
    """Dual-arm peg-style task: rigid hole + peg USDs; rewards use Factory keypoint squashing (not ``ForgePegInsert``)."""

    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {
        "right_hand": UR10E_DUAL_BIPEG_NUM_ACTUATED_DOFS,
        "left_hand": UR10E_DUAL_BIPEG_NUM_ACTUATED_DOFS,
    }
    observation_spaces = {
        "right_hand": _bi_peg_obs_dim_per_agent(),
        "left_hand": _bi_peg_obs_dim_per_agent(),
    }
    state_space = _bi_peg_state_dim()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    scene: UR10eDualShadowHandDirectSceneCfg = UR10eDualShadowHandDirectSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    right_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_RIGHT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_RIGHT_CFG.init_state.replace(
            joint_pos=_ur10e_bi_peg_right_arm_joint_pos(),
        ),
    )
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_ur10e_bi_peg_left_arm_joint_pos(),
        ),
    )

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"

    ee_body_name: str = "wrist_3_link"

    task: BiPegTaskCfg = BiPegTaskCfg()
    object_poses: BiPegObjectPosesCfg = BiPegObjectPosesCfg()

    # Rigid bodies (Cosmos USD); tune scale/mass in sim.
    hole_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/hole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WASHER_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.12),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    peg_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=NAIL_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.50, 0.0, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    fixed_obs_ref_pos: tuple[float, float, float] = (0.55, 0.0, 0.12)
    fixed_obs_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)

    vel_obs_scale: float = 0.2
    act_moving_average: float = 0.3
    enable_cameras: bool = False
    # When ``enable_cameras`` is True: spawn TacSL sensors; third-person RGB is optional (see ``difficult_dexhand.bi_blind_peg``).
    enable_third_person_camera: bool = True

    reset_grasp_hand_joint_lerp: float = 0.55
