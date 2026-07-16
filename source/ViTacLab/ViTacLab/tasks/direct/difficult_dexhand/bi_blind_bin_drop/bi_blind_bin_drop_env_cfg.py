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
from ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env_cfg import (
    GARBAGE_CAN_USD_PATH,
    _blind_grasp_robot_init_joint_pos,
)

BI_BLIND_BIN_DROP_OBJECT_INIT_POS_CANDIDATES: tuple[tuple[float, float, float], ...] = (
    (0.3, 0.12, 0.1),
    (0.3, 0.0, 0.1),
    (0.3, -0.08, 0.1),
)


UR10E_DUAL_BI_BLIND_NUM_ACTUATED_DOFS: int = 30


def _obs_dim_per_agent() -> int:
    # q(30) + dq(30) + wrist(3+4) + object(3+4) + trash_can(3+4) + act(30) + dact(30)
    return 134


def _state_dim() -> int:
    return 2 * _obs_dim_per_agent()


@configclass
class UR10eDualShadowHandBiBlindBinDropEnvCfg(DirectMARLEnvCfg):
    """Dual-arm blind bin-drop: cube starts outside the bin; objective is to drop it into the bin."""

    decimation = 2
    episode_length_s = 10.0
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
        init_state=UR10E_DUAL_SHADOWHAND_RIGHT_CFG.init_state.replace(
            joint_pos=_blind_grasp_robot_init_joint_pos(),
        ),
    )
    left_robot_cfg: ArticulationCfg = UR10E_DUAL_SHADOWHAND_LEFT_CFG.replace(
        init_state=UR10E_DUAL_SHADOWHAND_LEFT_CFG.init_state.replace(
            joint_pos=_blind_grasp_robot_init_joint_pos(),
        ),
    )

    arm_joint_expr: str = ".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"
    hand_joint_expr: str = ".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"
    ee_body_name: str = "wrist_3_link"
    act_moving_average: float = 0.3
    vel_obs_scale: float = 0.2
    enable_cameras: bool = False

    object_init_pos_candidates: tuple[tuple[float, float, float], ...] = BI_BLIND_BIN_DROP_OBJECT_INIT_POS_CANDIDATES
    object_init_choice: int = 0

    # Bin (kinematic): acts as the target receptacle.
    trash_can_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    trash_can_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trash_can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GARBAGE_CAN_USD_PATH,
            scale=trash_can_scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=-0.002),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 0.07), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # Randomize bin X position around the nominal spawn every reset.
    trash_can_reset_pos_x_range: tuple[float, float] = (-0.1, 0.1)

    # Cube starts outside the bin.
    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/SdfCube/SdfCube.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             retain_accelerations=False,
    #             enable_gyroscopic_forces=False,
    #         ),
    #         articulation_props=None,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.74, -0.18, 0.11)),
    # )
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/ViTacLab/ViTacLab/assets/data/Objects/cosmos_assets/1_object_A/lemon/lemon.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 放在机械臂前方稍远处，避免一开始就贴太近
            pos=(0.74, -0.18, 0.11),
        ),
    )

    @staticmethod
    def resolve_object_init_pos(choice: int) -> tuple[float, float, float]:
        idx = max(0, min(int(choice), len(BI_BLIND_BIN_DROP_OBJECT_INIT_POS_CANDIDATES) - 1))
        return tuple(float(v) for v in BI_BLIND_BIN_DROP_OBJECT_INIT_POS_CANDIDATES[idx])

    object_reset_pos_x_range: tuple[float, float] = (-0.02, 0.02)
    object_reset_pos_y_range: tuple[float, float] = (-0.02, 0.02)
    object_reset_pos_z_range: tuple[float, float] = (0.0, 0.0)
    object_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    # "Inside bin" geometric thresholds in env frame around can center.
    bin_success_xy_radius: float = 0.08
    bin_success_z_min: float = -0.02
    bin_success_z_max: float = 0.14
    min_success_steps: int = 3

    wrist_to_object_reward_scale: float = 2.5
    object_to_bin_reward_scale: float = 6.0
    success_bonus: float = 5.0
    action_l2_weight: float = -0.003
    action_rate_l2_weight: float = -0.003
    fall_height: float = -0.02
    out_of_bound_x: tuple[float, float] = (0.25, 1.10)
    out_of_bound_y: tuple[float, float] = (-0.70, 0.70)
    out_of_bound_z: tuple[float, float] = (-0.05, 0.90)
