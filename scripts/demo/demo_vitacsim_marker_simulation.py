#!/usr/bin/env python3
"""ViTacSim marker simulation demo: FOTS-style marker overlay on Taxim RGB.

Modes:
  - ``--synthetic-only``: pure-Python marker test (no Isaac Sim), gray pad + Gaussian indent.
  - default: normal-force validation scene (W100), saves none/gelsight/xense comparison panel.

Usage (ViTacLab repo root)::

    python scripts/demo/demo_vitacsim_marker_simulation.py --synthetic-only

    ../IsaacLab/isaaclab.sh -p scripts/demo/demo_vitacsim_marker_simulation.py \\
        --headless --enable_cameras --device cuda:0 --weight-id W100
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_TACSL = _REPO / "source" / "ViTacLab" / "ViTacLab" / "assets" / "sensor" / "tacsl_sensor"


def _load_marker_module():
    path = _TACSL / "visuotactile_marker.py"
    mod_name = "ViTacLab.assets.sensor.tacsl_sensor.visuotactile_marker"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gaussian_indent(h: int, w: int, cx: float, cy: float, sigma: float, amp_mm: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return (amp_mm * g).astype(np.float32)


def run_synthetic_demo(out_dir: Path, *, image_h: int = 240, image_w: int = 320) -> dict:
    """Draw marker patterns on gray background using height-map displacements only."""
    import torch

    marker_mod = _load_marker_module()
    MarkerSimulator = marker_mod.MarkerSimulator

    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("gelsight", "xense")
    summary: dict = {"mode": "synthetic", "patterns": {}}
    panel_rows = []

    height_mm = torch.tensor(
        _gaussian_indent(image_h, image_w, image_w * 0.52, image_h * 0.48, sigma=22.0, amp_mm=1.8),
        dtype=torch.float32,
    )
    bg = torch.full((image_h, image_w, 3), 210, dtype=torch.uint8)

    for pattern in patterns:
        sim = MarkerSimulator(
            pattern=pattern,
            image_height=image_h,
            image_width=image_w,
            device="cpu",
        )
        disp = sim.displacements_from_height_mm(height_mm)
        pos = sim.rest_xy + disp
        rgb = sim.draw_markers_on_image(bg, pos)
        arr = rgb.numpy()
        disp_np = disp.numpy()

        np.save(out_dir / f"synthetic_rgb_{pattern}.npy", arr)
        np.save(out_dir / f"synthetic_disp_{pattern}.npy", disp_np)
        panel_rows.append(arr)
        summary["patterns"][pattern] = {
            "num_markers": int(sim.num_markers),
            "max_displacement_px": float(np.linalg.norm(disp_np, axis=-1).max()),
            "mean_displacement_px": float(np.linalg.norm(disp_np, axis=-1).mean()),
        }

    panel = np.concatenate(panel_rows, axis=1)
    try:
        import cv2

        cv2.imwrite(str(out_dir / "synthetic_panel_gelsight_xense.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    except ImportError:
        np.save(out_dir / "synthetic_panel.npy", panel)

    with (out_dir / "synthetic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[marker demo] synthetic outputs -> {out_dir}")
    return summary


def _parse_cli():
    parser = argparse.ArgumentParser(description="ViTacSim marker simulation demo.")
    parser.add_argument("--synthetic-only", action="store_true", help="Marker-only test without Isaac Sim.")
    parser.add_argument("--out-dir", type=str, default="logs/vitacsim_validation/marker_simulation")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--weight-id", type=str, default="W100", choices=("W200", "W100", "W050", "W020", "W010"))
    parser.add_argument("--settle-steps", type=int, default=180)
    parser.add_argument("--record-steps", type=int, default=40)
    parser.add_argument("--weight-rest-z", type=float, default=0.442)
    parser.add_argument("--weight-clearance-z", type=float, default=0.520)
    parser.add_argument("--weight-spawn-z", type=float, default=-1.0)
    parser.add_argument("--weight-drop-offset", type=float, default=0.012)
    parser.add_argument("--finger-root-z", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    pre = _parse_cli()
    args, _ = pre.parse_known_args()

    out_dir = Path(args.out_dir)
    run_synthetic_demo(out_dir)

    if args.synthetic_only:
        return 0

    if str(_REPO / "source" / "ViTacLab") not in sys.path:
        sys.path.insert(0, str(_REPO / "source" / "ViTacLab"))

    from isaaclab.app import AppLauncher

    app_parser = _parse_cli()
    AppLauncher.add_app_launcher_args(app_parser)
    args_cli = app_parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils import configclass

    from ViTacLab.assets.sensor import VisuoTactileSensorV2Cfg
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import calibrated_gelsight_r15_cfg
    from ViTacLab.tasks.direct.pretraining.gelsight_finger_pretrain_base_cfg import (
        GELSIGHT_FINGER_SHORT_USD,
        build_gelsight_finger_robot_cfg,
    )
    from ViTacLab.tasks.direct.vitacsim_validation.validation_beta_config import (
        DEFAULT_FINGER_ROOT_Z,
        resolve_force_render_k_ref,
    )
    from ViTacLab.tasks.direct.vitacsim_validation.validation_weight_spawner_cfg import validation_weight_spawner_cfg
    from ViTacLab.tasks.direct.vitacsim_validation.weight_spec import WEIGHT_MASS_KG

    _TACTILE_ARRAY = (20, 25)
    _NOMINAL_WARMUP = 8

    def _finger_root_z() -> float:
        z = float(args_cli.finger_root_z)
        return DEFAULT_FINGER_ROOT_Z if z < 0.0 else z

    def _spawn_z() -> float:
        if float(args_cli.weight_spawn_z) >= 0.0:
            return float(args_cli.weight_spawn_z)
        return float(args_cli.weight_rest_z) + float(args_cli.weight_drop_offset)

    def _make_weight_cfg(weight_id: str) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/contact_object",
            spawn=validation_weight_spawner_cfg(weight_id),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, float(args_cli.weight_clearance_z))),
        )

    def _make_sensor_cfg(marker_pattern: str) -> VisuoTactileSensorV2Cfg:
        render_cfg = calibrated_gelsight_r15_cfg(
            prefer_local=True,
            enable_marker_simulation=(marker_pattern != "none"),
            marker_pattern=marker_pattern,
        )
        k_ref = resolve_force_render_k_ref(args_cli.weight_id, 0.0)
        return VisuoTactileSensorV2Cfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
            history_length=0,
            render_cfg=render_cfg,
            enable_camera_tactile=True,
            enable_force_field=True,
            tactile_array_size=_TACTILE_ARRAY,
            tactile_margin=0.005,
            contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
            depth_penetration_deadband=0.0,
            normal_contact_stiffness=1.0,
            normal_correction_k_ref=k_ref,
            enable_normal_correction=True,
            enable_slip_stick_reconstruction=True,
            enable_corrected_force_render=True,
            corrected_force_render_blend=1.0,
            require_physx_sparse_anchors=True,
            strict_target_contact_attribution=True,
            camera_cfg=TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
                height=render_cfg.image_height,
                width=render_cfg.image_width,
                data_types=["distance_to_image_plane"],
                spawn=None,
            ),
        )

    def _format_paths(sensor_cfg: VisuoTactileSensorV2Cfg, scene: InteractiveScene) -> None:
        ns = scene.env_regex_ns
        sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=ns)
        if sensor_cfg.camera_cfg is not None:
            sensor_cfg.camera_cfg.prim_path = sensor_cfg.camera_cfg.prim_path.format(ENV_REGEX_NS=ns)
        if sensor_cfg.contact_object_prim_path_expr is not None:
            sensor_cfg.contact_object_prim_path_expr = sensor_cfg.contact_object_prim_path_expr.format(ENV_REGEX_NS=ns)

    def _write_weight_pose(scene: InteractiveScene, z_root: float) -> None:
        obj = scene.rigid_objects["contact_object"]
        state = obj.data.default_root_state.clone()
        state[:, 0] = scene.env_origins[:, 0]
        state[:, 1] = scene.env_origins[:, 1]
        state[:, 2] = scene.env_origins[:, 2] + float(z_root)
        state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device).expand(state.shape[0], 4)
        state[:, 7:] = 0.0
        obj.write_root_pose_to_sim(state[:, :7], env_ids=None)
        obj.write_root_velocity_to_sim(state[:, 7:], env_ids=None)

    @configclass
    class MarkerValidationSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot: ArticulationCfg = build_gelsight_finger_robot_cfg().replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, _finger_root_z()),
                rot=(0.70711, -0.70711, 0.0, 0.0),
                joint_pos={},
                joint_vel={},
            )
        )
        contact_object: RigidObjectCfg = _make_weight_cfg(args_cli.weight_id)

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    finger_usd = Path(GELSIGHT_FINGER_SHORT_USD)
    if not finger_usd.is_file():
        finger_usd = _REPO / GELSIGHT_FINGER_SHORT_USD
    if not finger_usd.is_file():
        print(f"[ERROR] GelSight finger USD not found: {GELSIGHT_FINGER_SHORT_USD}", file=sys.stderr)
        simulation_app.close()
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("none", "gelsight", "xense")
    summary: dict = {"mode": "sim", "weight_id": args_cli.weight_id, "patterns": {}}
    panel_rows = []

    for pattern in patterns:
        print(f"[INFO] sim pattern={pattern}")
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(MarkerValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.3))
        scfg = _make_sensor_cfg(pattern)
        _format_paths(scfg, scene)
        scene.sensors["tactile_sensor"] = scfg.class_type(scfg)
        sim.reset()

        sim_dt = sim.get_physics_dt()
        scene["robot"].reset()
        scene.rigid_objects["contact_object"].reset()
        ts = scene["tactile_sensor"]

        clearance_z = float(args_cli.weight_clearance_z)
        _write_weight_pose(scene, clearance_z)
        for _ in range(_NOMINAL_WARMUP):
            _write_weight_pose(scene, clearance_z)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        ts.get_initial_render()

        _write_weight_pose(scene, _spawn_z())
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        for _ in range(int(args_cli.settle_steps)):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        for _ in range(int(args_cli.record_steps)):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        rgb = ts.data.tactile_rgb_image[0].detach().cpu().numpy()
        disp = ts.data.tactile_marker_displacement
        disp_np = disp[0].detach().cpu().numpy() if disp is not None else None

        np.save(out_dir / f"sim_rgb_{pattern}_{args_cli.weight_id}.npy", rgb)
        if disp_np is not None:
            np.save(out_dir / f"sim_disp_{pattern}_{args_cli.weight_id}.npy", disp_np)
        panel_rows.append(rgb)
        mag = float(np.linalg.norm(disp_np, axis=-1).max()) if disp_np is not None and disp_np.size else 0.0
        summary["patterns"][pattern] = {
            "mass_kg": float(WEIGHT_MASS_KG[args_cli.weight_id]),
            "num_markers": int(ts._tactile_rgb_render.num_markers),
            "max_displacement_px": mag,
        }
        sim.clear()
        simulation_app.update()

    if panel_rows:
        panel = np.concatenate(panel_rows, axis=1)
        try:
            import cv2

            cv2.imwrite(
                str(out_dir / f"sim_panel_{args_cli.weight_id}_none_gelsight_xense.png"),
                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            )
        except ImportError:
            np.save(out_dir / f"sim_panel_{args_cli.weight_id}.npy", panel)

    with (out_dir / f"sim_summary_{args_cli.weight_id}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[marker demo] sim outputs -> {out_dir}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
