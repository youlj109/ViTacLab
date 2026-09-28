"""Enable dual-arm recording sensors without replacing the local task scene.

The local dual-arm environment adds cameras/TacSL sensors after cloning only
when the scene has the TacSL marker type.  Promote a copy of the *existing*
scene to that type, retaining all instance fields instead of constructing a
fresh generic scene.  Object/robot assets, physics and task criteria are never
changed here.  This adapter belongs to the collection entry points, not the
simulation implementation.
"""

from __future__ import annotations

from copy import copy


def _recording_scene(scene, contact_prim_path: str, tactile_scene_type: type):
    """Retain scene state/methods and specialize only the sensor contact target."""
    original_type = type(scene)
    if issubclass(original_type, tactile_scene_type):
        bases = (original_type,)
    elif issubclass(tactile_scene_type, original_type):
        bases = (tactile_scene_type,)
    else:
        bases = (original_type, tactile_scene_type)

    original_params = getattr(original_type, "_tactile_params", tactile_scene_type._tactile_params)

    def tactile_params(cls):
        params = dict(original_params())
        params["contact_object_prim_path_expr"] = contact_prim_path
        return params

    recording_type = type(
        f"Collection{original_type.__name__}",
        bases,
        {"__module__": __name__, "_tactile_params": classmethod(tactile_params)},
    )
    result = copy(scene)
    result.__class__ = recording_type
    return result


def enable_dual_tactile_recording(env_cfg, *, contact_prim_path: str) -> None:
    """Enable local sensors, preserving current task assets and scene settings.

    Imports Isaac-dependent configuration only after the caller starts Kit.
    The contact target is the manipulated object for HandOver, the plate for
    BiStab; this changes only which body the recording sensors measure.
    """
    from ViTacLab.assets.robot.ur10e_dual_shadowhand_direct_base.ur10e_dual_shadowhand_direct_base_cfg import (
        UR10eDualShadowHandTacSLSceneCfg,
    )

    env_cfg.scene = _recording_scene(
        env_cfg.scene, contact_prim_path, UR10eDualShadowHandTacSLSceneCfg
    )
    env_cfg.enable_cameras = True
