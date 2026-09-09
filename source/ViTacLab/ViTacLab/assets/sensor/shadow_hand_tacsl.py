"""Shared TacSL construction and record helpers for standalone Shadow Hands.

The UR10e task bases own their sensor lifecycle because their robot USD layout
is task-specific.  Standalone Shadow-Hand tasks use this module so Vision and
two-hand hand-over environments expose the same canonical data-collection
contract without duplicating sensor or tensor-conversion code.

The helpers do not append dense tactile tensors to policy observations.
Environments continue to control that independently through their task config.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.sensors import TiledCameraCfg

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from ViTacLab.assets.sensor.tacsl_sensor import VisuoTactileSensorV2Cfg as VisuoTactileSensorCfg


SHADOW_HAND_TACSL_FINGERS: tuple[str, ...] = ("ff", "lf", "mf", "rf", "th")


def shadow_hand_tacsl_sensor_keys(prefix: str = "") -> tuple[str, ...]:
    """Return canonical five-finger sensor keys, optionally prefixed per hand."""

    return tuple(f"{prefix}tactile_sensor_{finger}" for finger in SHADOW_HAND_TACSL_FINGERS)


def build_shadow_hand_tacsl_sensor_cfgs(
    *,
    robot_prim_name: str,
    contact_object_prim_path_expr: str = "/World/envs/env_.*/object",
    tactile_array_size: tuple[int, int] = (20, 25),
) -> dict[str, VisuoTactileSensorCfg]:
    """Build five real GelSight/TacSL sensors for one standalone Shadow Hand.

    Args:
        robot_prim_name: Replicated articulation root name, for example
            ``Robot``, ``RightRobot``, or ``LeftRobot``.
        contact_object_prim_path_expr: Regex-like prim expression containing
            every task object that may contact the elastomers.
        tactile_array_size: Dense normal/shear grid shape per fingertip.

    Returns:
        Mapping from unprefixed canonical sensor key to sensor config.  A
        caller hosting multiple hands must prefix the mapping keys when adding
        them to ``scene.sensors``.
    """

    def _make(finger: str) -> VisuoTactileSensorCfg:
        return VisuoTactileSensorCfg(
            prim_path=f"/World/envs/env_.*/{robot_prim_name}/gelsight_{finger}distal/elastomer/tactile_sensor",
            history_length=0,
            debug_vis=False,
            render_cfg=GELSIGHT_R15_CFG,
            enable_camera_tactile=True,
            enable_force_field=True,
            tactile_array_size=tactile_array_size,
            tactile_margin=0.005,
            contact_object_prim_path_expr=contact_object_prim_path_expr,
            normal_contact_stiffness=1.0,
            friction_coefficient=2.0,
            tangential_stiffness=0.1,
            camera_cfg=TiledCameraCfg(
                prim_path=f"/World/envs/env_.*/{robot_prim_name}/gelsight_{finger}distal/elastomer_tip/cam",
                height=GELSIGHT_R15_CFG.image_height,
                width=GELSIGHT_R15_CFG.image_width,
                data_types=["distance_to_image_plane"],
                spawn=None,
            ),
            trimesh_vis_tactile_points=False,
            visualize_sdf_closest_pts=False,
        )

    return {f"tactile_sensor_{finger}": _make(finger) for finger in SHADOW_HAND_TACSL_FINGERS}


def _body_pose_for_finger(env: Any, robot: Any, finger: str) -> torch.Tensor:
    """Return an environment-frame fingertip pose when TacSL exposes no pose field."""

    body_names = list(getattr(robot, "body_names", getattr(robot.data, "body_names", ())))
    candidates = (
        f"robot0_{finger}distal",
        f"{finger}distal",
    )
    body_idx = None
    for candidate in candidates:
        for index, name in enumerate(body_names):
            name_str = str(name)
            if name_str == candidate or name_str.endswith(candidate):
                body_idx = index
                break
        if body_idx is not None:
            break
    if body_idx is None:
        raise RuntimeError(
            f"Cannot resolve {finger!r} fingertip body for tactile pose; body_names={body_names!r}"
        )
    pos = robot.data.body_pos_w[:, body_idx] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, body_idx]
    return torch.cat((pos, quat), dim=-1)


def _sensor_pose(env: Any, sensor: Any) -> torch.Tensor | None:
    data = sensor.data
    pos_w = getattr(data, "pos_w", None)
    quat_w = getattr(data, "quat_w_ros", None)
    if quat_w is None:
        quat_w = getattr(data, "quat_w", None)
    if pos_w is None:
        pos_w = getattr(sensor, "pos_w", None)
    if quat_w is None:
        quat_w = getattr(sensor, "quat_w_ros", None)
    if quat_w is None:
        quat_w = getattr(sensor, "quat_w", None)
    if pos_w is None or quat_w is None:
        return None
    return torch.cat((pos_w - env.scene.env_origins, quat_w), dim=-1)


def _rgb_to_uint8(rgb: torch.Tensor) -> torch.Tensor:
    value = rgb.detach()
    if value.dtype == torch.uint8:
        return value
    if value.is_floating_point() and value.numel() and float(value.max().item()) <= 1.5:
        value = value * 255.0
    return torch.clamp(value, 0.0, 255.0).to(torch.uint8)


def build_shadow_hand_tactile_record(
    env: Any,
    groups: Sequence[tuple[Any, Sequence[str]]],
) -> dict[str, torch.Tensor]:
    """Build canonical pose/force/RGB record fields for one or more hands.

    Args:
        env: Environment owning ``scene``, ``num_envs``, and ``device``.
        groups: Ordered ``(articulation, sensor_keys)`` pairs.  Every group
            must contain keys in ``ff, lf, mf, rf, th`` order.

    Returns:
        CPU tensors. ``joint_pos`` and ``tactile_pos`` are always emitted.
        Force and RGB fields are emitted only when all expected real TacSL
        sensor outputs are available, making an incomplete runtime chain
        visible to diagnostics instead of silently substituting fake data.
    """

    expected_count = sum(len(keys) for _, keys in groups)
    joint_chunks = [robot.data.joint_pos for robot, _ in groups]
    record: dict[str, torch.Tensor] = {
        "joint_pos": torch.cat(joint_chunks, dim=-1).detach().cpu(),
    }

    poses: list[torch.Tensor] = []
    normals: list[torch.Tensor] = []
    shears: list[torch.Tensor] = []
    rgbs: list[torch.Tensor] = []

    for robot, keys in groups:
        if len(keys) != len(SHADOW_HAND_TACSL_FINGERS):
            raise ValueError(
                f"Expected {len(SHADOW_HAND_TACSL_FINGERS)} sensor keys per hand, got {len(keys)}"
            )
        for finger, key in zip(SHADOW_HAND_TACSL_FINGERS, keys, strict=True):
            sensor = env.scene.sensors.get(key)
            pose = _sensor_pose(env, sensor) if sensor is not None else None
            if pose is None:
                pose = _body_pose_for_finger(env, robot, finger)
            poses.append(pose)

            if sensor is None:
                continue
            data = sensor.data
            array_h, array_w = tuple(sensor.cfg.tactile_array_size)
            normal = getattr(data, "tactile_normal_force", None)
            shear = getattr(data, "tactile_shear_force", None)
            rgb = getattr(data, "tactile_rgb_image", None)
            if normal is not None:
                normals.append(normal.reshape(env.num_envs, array_h, array_w, 1))
            if shear is not None:
                shears.append(shear.reshape(env.num_envs, array_h, array_w, 2))
            if rgb is not None:
                rgb_u8 = _rgb_to_uint8(rgb)
                image_h = int(sensor.cfg.render_cfg.image_height)
                image_w = int(sensor.cfg.render_cfg.image_width)
                if rgb_u8.ndim == 2 and rgb_u8.shape[1] == image_h * image_w * 3:
                    rgb_u8 = rgb_u8.reshape(env.num_envs, image_h, image_w, 3)
                rgbs.append(rgb_u8)

    record["tactile_pos"] = torch.stack(poses, dim=1).detach().cpu()
    if len(normals) == expected_count:
        record["tactile_normal_force"] = torch.stack(normals, dim=1).detach().cpu()
    if len(shears) == expected_count:
        record["tactile_shear_force"] = torch.stack(shears, dim=1).detach().cpu()
    if len(rgbs) == expected_count:
        record["tactile_rgb_image"] = torch.stack(rgbs, dim=1).detach().cpu()
    return record


def initialize_tacsl_nominal_render(env: Any, sensor_keys: Sequence[str]) -> None:
    """Initialize TacSL camera backgrounds after the replicated scene exists."""

    from isaaclab.sim.utils.stage import use_stage

    with use_stage(env.sim.get_initial_stage()):
        env.sim.reset()

    for key in sensor_keys:
        if key not in env.scene.sensors:
            continue
        sensor = env.scene[key]
        if not getattr(sensor.cfg, "enable_camera_tactile", False):
            continue
        sensor.get_initial_render()
