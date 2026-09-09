#!/usr/bin/env python3
"""Visual demo for ViTacSim alignment / attribution behavior.

Focus case: interference-only contact (target object lifted away, interference object touches pad).
Produces a video with tactile RGB, contact/slip masks, and force heatmaps + runtime metrics overlay.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate ViTacSim alignment visual demo video.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=260)
parser.add_argument("--log_interval", type=int, default=40)
parser.add_argument("--strict_target_attribution", action="store_true")
parser.add_argument("--output_video", type=str, default="logs/alignment_visuotactile_v2/interference_visual.mp4")
parser.add_argument("--force_exit", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.sensors import GELSIGHT_R15_CFG

from ViTacLab.assets.sensor import VisuoTactileSensorV2Cfg

_TACTILE_ARRAY = (20, 25)
_TACTILE_MARGIN = 0.005
_SETTLE_STEPS = 8
_INTERFERENCE_CONTACT_Z_OFFSET = 0.0


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(12):
        if (p / "source").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def _finger_usd() -> str:
    return str(
        (
            _repo_root()
            / "source/ViTacLab/ViTacLab/assets/data/Sensors/Gelsight_finger/gelsight_r15_finger_short_v2.usd"
        ).resolve()
    )


def _make_robot_cfg(usd_path: str) -> ArticulationCfg:
    spawn = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        physics_material_prim_path="elastomer",
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.0005),
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.45), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}),
        actuators={},
    )


def _make_object_cfg(prim_path: str, z: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _make_sensor_cfg(strict_target_attribution: bool) -> VisuoTactileSensorV2Cfg:
    return VisuoTactileSensorV2Cfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=_TACTILE_ARRAY,
        tactile_margin=_TACTILE_MARGIN,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        depth_penetration_deadband=0.0,
        normal_contact_stiffness=1.0,
        tangential_stiffness=0.1,
        friction_coefficient=2.0,
        enable_normal_correction=True,
        enable_slip_stick_reconstruction=True,
        use_physx_sparse_anchors=True,
        strict_target_contact_attribution=bool(strict_target_attribution),
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def _format_sensor_paths(sensor_cfg: VisuoTactileSensorV2Cfg, scene: InteractiveScene) -> None:
    ns = scene.env_regex_ns
    sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.camera_cfg is not None:
        sensor_cfg.camera_cfg.prim_path = sensor_cfg.camera_cfg.prim_path.format(ENV_REGEX_NS=ns)
    if sensor_cfg.contact_object_prim_path_expr is not None:
        sensor_cfg.contact_object_prim_path_expr = sensor_cfg.contact_object_prim_path_expr.format(ENV_REGEX_NS=ns)


def _register_sensor(scene: InteractiveScene, strict_target_attribution: bool) -> None:
    cfg = _make_sensor_cfg(strict_target_attribution)
    _format_sensor_paths(cfg, scene)
    scene.sensors["tactile_sensor"] = cfg.class_type(cfg)


def _write_object_pose(
    scene: InteractiveScene,
    object_name: str,
    xy_offset: tuple[float, float] = (0.0, 0.0),
    z_offset: float = 0.0,
) -> None:
    rigid = scene.rigid_objects[object_name]
    state = rigid.data.default_root_state.clone()
    state[:, 0] += float(xy_offset[0])
    state[:, 1] += float(xy_offset[1])
    state[:, 2] += float(z_offset)
    state[:, 0:3] += scene.env_origins
    rigid.write_root_pose_to_sim(state[:, :7], env_ids=None)
    rigid.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]), env_ids=None)


def _reset_robot_and_scene(scene: InteractiveScene) -> None:
    scene["robot"].reset()
    scene.rigid_objects["contact_object"].reset()
    scene.rigid_objects["interference_object"].reset()
    robot = scene["robot"]
    state = robot.data.default_root_state.clone()
    state[:, 0:3] += scene.env_origins
    robot.write_root_pose_to_sim(state[:, :7], env_ids=None)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=None)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone(), env_ids=None)
    scene.write_data_to_sim()


def _extract_physx(ts, env_id: int = 0) -> tuple[float, float]:
    view = getattr(ts, "_contact_physx_view", None)
    if view is None:
        return 0.0, 0.0
    num_filters = int(view.filter_count)
    forces, _points, _normals, _seps, buffer_count, buffer_start_indices = view.get_contact_data(dt=ts._sim_physics_dt)
    friction_forces, _fp, buffer_count_f, buffer_start_indices_f = view.get_friction_data(dt=ts._sim_physics_dt)
    counts = buffer_count.view(ts._num_envs, num_filters)
    starts = buffer_start_indices.view(ts._num_envs, num_filters)
    counts_f = buffer_count_f.view(ts._num_envs, num_filters)
    starts_f = buffer_start_indices_f.view(ts._num_envs, num_filters)

    fn_total = 0.0
    ft_total = 0.0
    for fi in range(num_filters):
        cnt = int(counts[env_id, fi].item())
        if cnt > 0:
            sl = slice(int(starts[env_id, fi].item()), int(starts[env_id, fi].item()) + cnt)
            fn_total += float(torch.clamp(forces[sl].reshape(-1), min=0.0).sum().item())
        cntf = int(counts_f[env_id, fi].item())
        if cntf > 0:
            slf = slice(int(starts_f[env_id, fi].item()), int(starts_f[env_id, fi].item()) + cntf)
            ft_total += float(torch.norm(friction_forces[slf], dim=-1).sum().item())
    return fn_total, ft_total


def _to_uint8_rgb(img: torch.Tensor | None, h: int, w: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = img.detach().cpu().numpy()
    if arr.dtype in (np.float32, np.float64):
        if float(arr.max()) <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr


def _heatmap(vals: torch.Tensor | None, rows: int, cols: int, h: int, w: int) -> np.ndarray:
    if vals is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    v = vals.detach().cpu().numpy().reshape(rows, cols)
    vmax = float(np.max(v))
    if vmax <= 1e-12:
        n = np.zeros_like(v, dtype=np.float32)
    else:
        n = (v / vmax).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap((n * 255.0).astype(np.uint8), cv2.COLORMAP_JET)


def _mask_overlay(contact: torch.Tensor | None, slip: torch.Tensor | None, rows: int, cols: int, h: int, w: int) -> np.ndarray:
    base = np.zeros((h, w, 3), dtype=np.uint8)
    if contact is not None:
        c = contact.detach().cpu().numpy().reshape(rows, cols).astype(np.uint8)
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_NEAREST)
        base[:, :, 1] = np.maximum(base[:, :, 1], c * 180)  # green
    if slip is not None:
        s = slip.detach().cpu().numpy().reshape(rows, cols).astype(np.uint8)
        s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
        base[:, :, 2] = np.maximum(base[:, :, 2], s * 220)  # red
    return base


@configclass
class VisualSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot = _make_robot_cfg(_finger_usd())
    contact_object = _make_object_cfg("{ENV_REGEX_NS}/contact_object", 0.5)
    interference_object = _make_object_cfg("{ENV_REGEX_NS}/interference_object", 0.5)


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=[0.5, 0.6, 1.0], target=[-0.1, 0.1, 0.5])
    scene = InteractiveScene(VisualSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2))
    _register_sensor(scene, bool(args_cli.strict_target_attribution))
    sim.reset()

    ts = scene["tactile_sensor"]
    _reset_robot_and_scene(scene)
    # Nominal baseline must be captured with NO contact for all objects.
    _write_object_pose(scene, "contact_object", (0.0, 0.0), z_offset=0.35)
    _write_object_pose(scene, "interference_object", (0.0, 0.0), z_offset=0.35)
    scene.write_data_to_sim()

    sim_dt = sim.get_physics_dt()
    for _ in range(max(1, _SETTLE_STEPS)):
        _write_object_pose(scene, "contact_object", (0.0, 0.0), z_offset=0.35)
        _write_object_pose(scene, "interference_object", (0.0, 0.0), z_offset=0.35)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    ts.get_initial_render()

    # Start interference-only rollout: target lifted, interference lowered to contact.
    _write_object_pose(scene, "contact_object", (0.0, 0.0), z_offset=0.35)
    _write_object_pose(scene, "interference_object", (0.0, 0.0), z_offset=_INTERFERENCE_CONTACT_Z_OFFSET)
    scene.write_data_to_sim()

    out_path = Path(args_cli.output_video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = int(GELSIGHT_R15_CFG.image_height)
    w = int(GELSIGHT_R15_CFG.image_width)
    panel_h, panel_w = h * 2, w * 2
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (panel_w, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    rows, cols = _TACTILE_ARRAY
    interference_obj = scene["interference_object"]
    force = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    false_activation = 0
    raw_contact_steps = 0

    for step in range(1, int(args_cli.steps) + 1):
        torque.zero_()
        # Keep target object lifted, but let interference object evolve under physics
        # (do not teleport it every step, otherwise contact dynamics can be suppressed).
        _write_object_pose(scene, "contact_object", (0.0, 0.0), z_offset=0.35)
        torque[:, 0, 2] = 12.0
        interference_obj.permanent_wrench_composer.set_forces_and_torques(force, torque)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        d = ts.data
        rgb = _to_uint8_rgb(d.tactile_rgb_image[0] if d.tactile_rgb_image is not None else None, h, w)
        contact = d.contact_mask[0] if d.contact_mask is not None else None
        slip = d.slip_mask[0] if d.slip_mask is not None else None
        mask = _mask_overlay(contact, slip, rows, cols, h, w)
        fn = _heatmap(d.tactile_normal_force[0] if d.tactile_normal_force is not None else None, rows, cols, h, w)
        ft_mag = None
        if d.tactile_shear_force is not None:
            ft_mag = torch.norm(d.tactile_shear_force[0], dim=-1)
        ft = _heatmap(ft_mag, rows, cols, h, w)

        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[0:h, 0:w] = rgb[:, :, ::-1]  # RGB->BGR for cv2
        panel[0:h, w : 2 * w] = mask
        panel[h : 2 * h, 0:w] = fn
        panel[h : 2 * h, w : 2 * w] = ft

        fn_physx, ft_physx = _extract_physx(ts, env_id=0)
        fn_sensor = float(torch.clamp(d.tactile_normal_force[0], min=0.0).sum().item()) if d.tactile_normal_force is not None else 0.0
        ft_sensor = float(torch.norm(d.tactile_shear_force[0], dim=-1).sum().item()) if d.tactile_shear_force is not None else 0.0
        sensor_contact = float(d.contact_mask[0].float().mean().item()) > 0.01 if d.contact_mask is not None else False
        if sensor_contact:
            false_activation += 1
        false_rate = float(false_activation) / float(step)
        raw_depth_ratio = 0.0
        if d.tactile_depth_image is not None and isinstance(getattr(ts, "_nominal_tactile", None), dict):
            nominal = getattr(ts, "_nominal_tactile")
            depth_key = "distance_to_image_plane" if "distance_to_image_plane" in nominal else ("depth" if "depth" in nominal else None)
            if depth_key is not None and depth_key in nominal:
                raw_delta = torch.clamp(nominal[depth_key][0].squeeze(-1) - d.tactile_depth_image[0].squeeze(-1), min=0.0)
                raw_depth_ratio = float((raw_delta > 1e-5).float().mean().item())
                if raw_depth_ratio > 0.005:
                    raw_contact_steps += 1
        raw_contact_rate = float(raw_contact_steps) / float(step)

        title = (
            f"interference_only | strict_attr={bool(args_cli.strict_target_attribution)} | step={step} | "
            f"physx_fn={fn_physx:.3f} sensor_fn={fn_sensor:.3f} | physx_ft={ft_physx:.3f} sensor_ft={ft_sensor:.3f} | "
            f"target_false_activation_rate={false_rate:.3f} raw_depth_contact_rate={raw_contact_rate:.3f}"
        )
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Top-Left: tactile RGB", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, "Top-Right: contact(green)/slip(red)", (w + 8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, "Bottom-Left: normal force heatmap", (8, panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, "Bottom-Right: shear magnitude heatmap", (w + 8, panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(panel)

        if args_cli.log_interval > 0 and (step % int(args_cli.log_interval) == 0 or step == int(args_cli.steps)):
            print(
                f"[step {step:04d}] physx_fn={fn_physx:.4f} sensor_fn={fn_sensor:.4f} "
                f"physx_ft={ft_physx:.4f} sensor_ft={ft_sensor:.4f} false_rate={false_rate:.4f} "
                f"raw_depth_rate={raw_contact_rate:.4f}"
            )

    writer.release()
    print(f"[DONE] video saved: {out_path}")
    return 0


if __name__ == "__main__":
    code = 0
    try:
        code = int(main())
    except KeyboardInterrupt:
        code = 130
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
        if bool(getattr(args_cli, "force_exit", False)):
            os._exit(code)
    raise SystemExit(code)
