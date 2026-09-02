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
    BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES,
    _blind_grasp_robot_init_joint_pos,
)
from ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env_cfg import WASHER_USD_PATH

# Half separation along X between washer (hole) and nail (peg); matches legacy (0.84 vs 0.92).
BI_BLIND_GRASP_HOLE_PEG_HALF_SEP_X: float = 0.04


UR10E_DUAL_BI_BLIND_NUM_ACTUATED_DOFS: int = 30


def _obs_dim_per_agent() -> int:
    # q(30) + dq(30) + wrist(3+4) + hole(3+4) + peg(3+4) + act(30) + dact(30)
    return 141


def _state_dim() -> int:
    return 2 * _obs_dim_per_agent()


@configclass
class UR10eDualShadowHandBiBlindGraspEnvCfg(DirectMARLEnvCfg):
    """Dual-arm blind grasp: left hand grasps washer(hole), right hand grasps a procedural cylinder peg."""

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

    # Same blind-grasp-safe arm posture as :class:`UR10eShadowHandBlindGraspEnvCfg` (v2) on both roots.
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

    object_init_pos_candidates: tuple[tuple[float, float, float], ...] = BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES
    object_init_choice: int = 0

    hole_scale: tuple[float, float, float] = (3.0, 3.0, 10.0)
    peg_radius_m: float = 0.01
    peg_height_m: float = 0.06

    hole_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/hole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WASHER_USD_PATH,
            scale=hole_scale,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    peg_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/peg",
        spawn=sim_utils.CylinderCfg(
            radius=peg_radius_m,
            height=peg_height_m,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.65, 0.65, 0.75),
            ),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.66, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    @staticmethod
    def resolve_object_center(choice: int) -> tuple[float, float, float]:
        idx = max(0, min(int(choice), len(BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES) - 1))
        return tuple(float(v) for v in BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES[idx])

    @classmethod
    def resolve_hole_init_pos(cls, choice: int) -> tuple[float, float, float]:
        idx = max(0, min(int(choice), len(BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES) - 1))
        cx, cy, cz = (float(v) for v in BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES[idx])
        h = BI_BLIND_GRASP_HOLE_PEG_HALF_SEP_X
        return (cx - h, cy, cz)

    @classmethod
    def resolve_peg_init_pos(cls, choice: int) -> tuple[float, float, float]:
        idx = max(0, min(int(choice), len(BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES) - 1))
        cx, cy, cz = (float(v) for v in BLIND_GRASP_OBJECT_INIT_POS_CANDIDATES[idx])
        h = BI_BLIND_GRASP_HOLE_PEG_HALF_SEP_X
        return (cx + h, cy, cz)

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
