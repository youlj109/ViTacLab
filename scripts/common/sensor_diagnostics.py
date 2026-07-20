"""Compact camera/TacSL diagnostics shared by zero and random smoke tests.

The helper is intentionally read-only: it reports registered scene sensors and
summarizes camera/tactile tensors already returned by an environment step.  It
does not alter the environment, render settings, observations, or record data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


_SENSOR_RECORD_KEYS = (
    "tactile_pos",
    "tactile_normal_force",
    "tactile_shear_force",
    "tactile_rgb_image",
    "third_person_camera",
    "third_person_camera_depth",
    "third_person_camera_pos",
    "twist_camera",
    "twist_camera_depth",
    "twist_camera_pos",
)

_REQUIRED_TACTILE_RECORD_KEYS = (
    "tactile_pos",
    "tactile_normal_force",
    "tactile_shear_force",
    "tactile_rgb_image",
)


def _tensor_summary(value: Any) -> str:
    """Return shape/dtype/range information without changing ``value``."""

    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.numel() == 0:
            return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} empty=True"
        finite = bool(torch.isfinite(tensor).all().item()) if tensor.is_floating_point() else True
        nonzero = int(torch.count_nonzero(tensor).item())
        minimum = tensor.min().item()
        maximum = tensor.max().item()
        return (
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
            f"finite={finite} nonzero={nonzero}/{tensor.numel()} min={minimum} max={maximum}"
        )

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"shape={value.shape} dtype={value.dtype} empty=True"
        finite = bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
        nonzero = int(np.count_nonzero(value))
        return (
            f"shape={value.shape} dtype={value.dtype} finite={finite} "
            f"nonzero={nonzero}/{value.size} min={value.min()} max={value.max()}"
        )

    return f"type={type(value).__name__}"


def print_sensor_observation_diagnostics(env: Any, observation: Any) -> None:
    """Print and enforce the camera/TacSL acceptance contract.

    This helper is called only for ``--enable_cameras`` smoke tests.  It raises
    ``RuntimeError`` when real sensors or canonical record fields are missing,
    preventing zero/random agents from reporting a misleading ``[PASS]``.
    MARL tasks may intentionally omit ``record`` from their policy observation;
    in that case the read-only ``_build_record_dict`` fallback is inspected.
    """

    unwrapped = env.unwrapped
    scene = getattr(unwrapped, "scene", None)
    scene_sensors = getattr(scene, "sensors", {}) if scene is not None else {}
    sensor_names = tuple(sorted(str(name) for name in scene_sensors.keys()))
    print(f"[SENSOR-DIAG]: scene sensors ({len(sensor_names)}): {sensor_names}", flush=True)

    registered_tactile = tuple(name for name in sensor_names if "tactile_sensor" in name)
    print(
        f"[SENSOR-DIAG]: registered real tactile sensors ({len(registered_tactile)}): "
        f"{registered_tactile}",
        flush=True,
    )

    expected_tactile = tuple(getattr(unwrapped, "_expected_tactile_sensor_names", ()))
    if not expected_tactile:
        expected_tactile = tuple(getattr(unwrapped, "_ur10e_stacked_tacsl_names", ()))
    if not expected_tactile:
        raise RuntimeError(
            "Camera/TacSL acceptance failed: environment declares no expected tactile sensor names."
        )

    present = tuple(name for name in expected_tactile if name in scene_sensors)
    missing = tuple(name for name in expected_tactile if name not in scene_sensors)
    print(f"[SENSOR-DIAG]: expected tactile present={present} missing={missing}", flush=True)
    if missing:
        raise RuntimeError(
            f"Camera/TacSL acceptance failed: missing real tactile sensors {missing}; "
            f"expected={expected_tactile}."
        )

    record: dict[str, Any] = {}
    if isinstance(observation, Mapping):
        observation_record = observation.get("record")
        if isinstance(observation_record, Mapping):
            record.update(observation_record)
    else:
        print(
            f"[SENSOR-DIAG]: observation is {type(observation).__name__}; trying canonical fallback",
            flush=True,
        )

    fallback_builder = getattr(unwrapped, "_build_record_dict", None)
    if callable(fallback_builder):
        try:
            fallback_record = fallback_builder()
        except Exception as exc:
            raise RuntimeError(
                "Camera/TacSL acceptance failed while building canonical record fallback"
            ) from exc
        if isinstance(fallback_record, Mapping):
            for key, value in fallback_record.items():
                record.setdefault(key, value)

    if not record:
        raise RuntimeError(
            "Camera/TacSL acceptance failed: neither observation nor _build_record_dict supplied a record."
        )

    reported = 0
    for key in _SENSOR_RECORD_KEYS:
        if key not in record:
            continue
        print(f"[SENSOR-DIAG]: record[{key!r}] {_tensor_summary(record[key])}", flush=True)
        reported += 1
    if reported == 0:
        raise RuntimeError(
            "Camera/TacSL acceptance failed: record contains no canonical camera/tactile fields."
        )

    missing_record = tuple(key for key in _REQUIRED_TACTILE_RECORD_KEYS if key not in record)
    if missing_record:
        raise RuntimeError(
            f"Camera/TacSL acceptance failed: record is missing required tactile fields {missing_record}."
        )

    expected_count = len(expected_tactile)
    for key in _REQUIRED_TACTILE_RECORD_KEYS:
        value = record[key]
        shape = tuple(value.shape) if torch.is_tensor(value) or isinstance(value, np.ndarray) else ()
        if len(shape) < 2 or shape[1] != expected_count:
            raise RuntimeError(
                f"Camera/TacSL acceptance failed: record[{key!r}] shape={shape} does not have "
                f"sensor axis size {expected_count}."
            )
        if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"Camera/TacSL acceptance failed: record[{key!r}] contains NaN/Inf.")
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise RuntimeError(f"Camera/TacSL acceptance failed: record[{key!r}] contains NaN/Inf.")

    tactile_rgb = record["tactile_rgb_image"]
    rgb_nonzero = (
        int(torch.count_nonzero(tactile_rgb).item())
        if torch.is_tensor(tactile_rgb)
        else int(np.count_nonzero(tactile_rgb))
    )
    if rgb_nonzero == 0:
        raise RuntimeError(
            "Camera/TacSL acceptance failed: tactile RGB is entirely zero; real GelSight render was not produced."
        )
    rgb_min = tactile_rgb.min().item()
    rgb_max = tactile_rgb.max().item()
    if rgb_min == rgb_max:
        raise RuntimeError(
            f"Camera/TacSL acceptance failed: tactile RGB is constant ({rgb_min}); "
            "the GelSight image is uninitialized or was converted incorrectly."
        )

    print(
        f"[SENSOR-DIAG-PASS]: {expected_count} real tactile sensor(s) and canonical "
        "pose/normal/shear/RGB record fields validated.",
        flush=True,
    )
