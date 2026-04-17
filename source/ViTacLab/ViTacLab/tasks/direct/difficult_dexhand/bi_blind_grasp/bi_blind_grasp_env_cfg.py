import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_cfg import (
    UR10E_DUAL_SHADOWHAND_LEFT_CFG,
    UR10E_DUAL_SHADOWHAND_RIGHT_CFG,
    UR10eDualShadowHandDirectSceneCfg,
)
from ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env_cfg import GARBAGE_CAN_USD_PATH
from ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env_cfg import NAIL_USD_PATH, WASHER_USD_PATH


def _ur10e_bi_blind_right_arm_joint_pos() -> dict[str, float]:
    return {
        "shoulder_pan_joint": -0.2381820505749035,
        "shoulder_lift_joint": -0.7457377055913127,
        "elbow_joint": 0.7105199793481264,
        "wrist_1_joint": -1.5355786007566066,
        "wrist_2_joint": -1.5707963265900005,
        "wrist_3_joint": -1.8089821043693861,
    }


def _ur10e_bi_blind_left_arm_joint_pos() -> dict[str, float]:
    return {
        "shoulder_pan_joint": -0.0647053798972387,
        "shoulder_lift_joint": -0.6931192918684786,
        "elbow_joint": 0.9688935001938827,
        "wrist_1_joint": -1.8465705353261026,
        "wrist_2_joint": -1.5707963265904117,
        "wrist_3_joint": 1.5060872198977679,
    }


UR10E_DUAL_BI_BLIND_NUM_ACTUATED_DOFS: int = 30


def _obs_dim_per_agent() -> int:
    # q(30) + dq(30) + wrist(3+4) + hole(3+4) + peg(3+4) + act(30) + dact(30)
    return 141


def _state_dim() -> int:
    return 2 * _obs_dim_per_agent()


@configclass
class UR10eDualShadowHandBiBlindGraspEnvCfg(DirectMARLEnvCfg):
    """Dual-arm blind grasp in a trash can: left hand grasps washer(hole), right hand grasps nail(peg)."""

    decimation = 2
    episode_length_s = 8.0
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {
        "right_hand": UR10E_DUAL_BI_BLIND_NUM_ACTUATED_DOFS,
        "left_hand": UR10E_DUAL_BI_BLIND_NUM_ACTUATED_DOFS,
    }
    observation_spaces = {
        "right_hand": _obs_dim_per_agent(),
        "left_hand": _obs_dim_per_agent(),
    }
    state_space = _state_dim()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    scene: UR10eDualShadowHandDirectSceneCfg = UR10eDualShadowHandDirectSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    right_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_RIGHT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_RIGHT_CFG.init_state.replace(joint_pos=_ur10e_bi_blind_right_arm_joint_pos()),
    )
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_LEFT_CFG.init_state.replace(joint_pos=_ur10e_bi_blind_left_arm_joint_pos()),
    )

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"
    ee_body_name: str = "wrist_3_link"
    act_moving_average: float = 0.3
    vel_obs_scale: float = 0.2
    enable_cameras: bool = False

    trash_can_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    trash_can_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trash_can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GARBAGE_CAN_USD_PATH,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.001),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.88, -0.15, 0.14), rot=(1.0, 0.0, 0.0, 0.0)),
    )

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.84, -0.20, 0.10), rot=(1.0, 0.0, 0.0, 0.0)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.92, -0.20, 0.10), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    object_reset_pos_x_range: tuple[float, float] = (-0.01, 0.01)
    object_reset_pos_y_range: tuple[float, float] = (-0.01, 0.01)
    object_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
    object_reset_yaw_range: tuple[float, float] = (-3.14159, 3.14159)

    wrist_to_object_reward_scale: float = 4.0
    success_dist_threshold: float = 0.06
    success_bonus: float = 5.0
    action_l2_weight: float = -0.003
    action_rate_l2_weight: float = -0.003
    fall_height: float = -0.02
