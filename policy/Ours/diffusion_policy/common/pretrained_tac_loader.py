"""
Load tactile backbone weights from a PretrainVQGAN checkpoint into
``ViTacEncoder.key_model_map`` / compatible tactile encoder maps.

The VQGAN pretraining (see ``PretrainVQGAN/.../encoder_decoder.py``) only
trains the ResNet-18 trunk up to (but excluding) the adaptive pooling stage.
It stores weights under the following prefixes:

    state_dicts.ema_model:
        vqgan.tac_rgb_encoder.base_rgb_model.*
        vqgan.tac_force_encoder.base_rgb_model.*
    state_dicts.model (DDP-wrapped):
        module.vqgan.tac_rgb_encoder.base_rgb_model.*
        module.vqgan.tac_force_encoder.base_rgb_model.*

Note on normalization: the pretraining script already replaces ``BatchNorm2d``
with ``GroupNorm`` (matching the downstream encoder), so the tensor names are
``bn1.weight/bias`` without ``running_mean/var`` - identical to what
``BiViTacEncoder_v85``'s ``key_model_map[key]`` expects. The pretraining
encoder additionally sets ``avgpool = Identity`` and ``fc = Identity``;
both are parameter-free, so loading the trunk-only state dict into the
downstream ResNet (which keeps ``AdaptiveAvgPool2d`` + ``fc=Identity``)
is safe.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Dict, Iterable, Tuple

import dill
import torch
import torch.nn as nn


# Candidate prefixes to strip, in priority order. The first prefix that yields
# any matching keys wins for each tactile sub-encoder.
_RGB_PREFIXES = (
    "vqgan.tac_rgb_encoder.base_rgb_model.",
    "module.vqgan.tac_rgb_encoder.base_rgb_model.",
    # Legacy PreTrainMask-style checkpoints.
    "tac_rgb_encoder.base_rgb_model.",
    "module.tac_rgb_encoder.base_rgb_model.",
)
_FORCE_PREFIXES = (
    "vqgan.tac_force_encoder.base_rgb_model.",
    "module.vqgan.tac_force_encoder.base_rgb_model.",
    "tac_force_encoder.base_rgb_model.",
    "module.tac_force_encoder.base_rgb_model.",
)


def _tac_key_to_obs_type(shape_meta_obs: dict) -> Dict[str, str]:
    """Map tactile encoder keys to ``tac_rgb`` | ``tac_force``.

    Older encoders expand tactile observations by the first dimension, e.g.
    ``tac -> tac_0``. Newer encoders such as ``BiViTacEncoder_v93`` keep the
    original observation key, e.g. ``left_hand_tac``. Register both forms so
    the loader can work with either encoder layout.
    """
    out: Dict[str, str] = {}
    for base_key, attr in shape_meta_obs.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type not in ("tac_rgb", "tac_force"):
            continue
        out[base_key] = obs_type
        shape = tuple(attr["shape"])
        n = int(shape[0])
        for i in range(n):
            out[f"{base_key}_{i}"] = obs_type
    return out


def _resolve_tac_keys(obs_encoder: nn.Module, tac_type: Dict[str, str]) -> list[str]:
    keys = list()
    seen = set()
    for attr_name in ("tac_keys", "tac_rgb_keys", "tac_force_keys"):
        for key in getattr(obs_encoder, attr_name, []):
            if key in tac_type and key not in seen:
                keys.append(key)
                seen.add(key)
    if keys:
        return keys
    key_model_map = getattr(obs_encoder, "key_model_map", None)
    if key_model_map is None:
        return []
    return [key for key in key_model_map.keys() if key in tac_type]


def _infer_target_types(tac_keys: Iterable[str], tac_type: Dict[str, str]) -> Tuple[str, ...]:
    target_types = []
    seen = set()
    for key in tac_keys:
        typ = tac_type.get(key)
        if typ in ("tac_rgb", "tac_force") and typ not in seen:
            target_types.append(typ)
            seen.add(typ)
    return tuple(target_types)


def _strip_prefix(sd: dict, prefix: str) -> dict:
    plen = len(prefix)
    return {k[plen:]: v for k, v in sd.items() if k.startswith(prefix)}


def _extract_first_matching(sd: dict, prefixes: Tuple[str, ...]) -> Tuple[dict, str | None]:
    for pref in prefixes:
        sub = _strip_prefix(sd, pref)
        if sub:
            return sub, pref
    return {}, None


def _select_source_state_dict(payload: dict) -> Tuple[dict, str]:
    """Pick the most appropriate state dict from the checkpoint payload.

    Preference order (EMA weights are usually the right choice for downstream
    transfer):
        1. ``state_dicts.ema_model``
        2. ``state_dicts.model``
        3. ``state_dict``
        4. top-level dict (already a state_dict)
    """
    if "state_dicts" in payload and isinstance(payload["state_dicts"], dict):
        sd_group = payload["state_dicts"]
        for name in ("ema_model", "model"):
            if name in sd_group and isinstance(sd_group[name], dict):
                return sd_group[name], f"state_dicts.{name}"
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"], "state_dict"
    # Last resort: assume the payload itself is a flat state_dict.
    if all(isinstance(v, torch.Tensor) for v in payload.values() if v is not None):
        return payload, "<root>"
    raise KeyError(
        "Unexpected checkpoint format: expected 'state_dicts.ema_model', "
        "'state_dicts.model', or 'state_dict' containing tensor values."
    )


def load_pretrained_tac_key_models(
    obs_encoder: nn.Module,
    ckpt_path: str,
    *,
    map_location: str | torch.device = "cpu",
    target_types: Iterable[str] | None = None,
) -> Tuple[int, int]:
    """Load tactile ResNet backbone weights into ``obs_encoder.key_model_map``.

    Supports both the new VQGAN-style checkpoint (this file's main target)
    and legacy ``PreTrainMask``-style checkpoints.

    Returns
    -------
    (num_loaded_tac_keys, total_parameter_tensors_copied)
    """
    shape_meta = getattr(obs_encoder, "shape_meta", None)
    if not isinstance(shape_meta, Mapping) or "obs" not in shape_meta:
        raise AttributeError(
            "obs_encoder must expose shape_meta['obs'] so pretrained tac type "
            "can be inferred"
        )

    tac_type = _tac_key_to_obs_type(shape_meta["obs"])
    tac_keys = _resolve_tac_keys(obs_encoder, tac_type)
    if not tac_keys:
        print("[pretrained_tac] no tactile key_model_map entries found")
        return 0, 0

    if target_types is None:
        target_types = _infer_target_types(tac_keys, tac_type)
    else:
        target_types = tuple(target_types)
    unknown_types = set(target_types) - {"tac_rgb", "tac_force"}
    if unknown_types:
        raise ValueError(f"Unsupported pretrained tac target_types: {sorted(unknown_types)}")
    if not target_types:
        print("[pretrained_tac] no tactile target types inferred from obs_encoder")
        return 0, 0

    path = pathlib.Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"pretrained tac ckpt not found: {path}")

    payload = torch.load(path.open("rb"), map_location=map_location, pickle_module=dill)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint root must be dict, got {type(payload)}")

    sd, src_name = _select_source_state_dict(payload)
    if not isinstance(sd, dict):
        raise TypeError(f"{src_name} must be dict, got {type(sd)}")

    rgb_sd, rgb_prefix = (
        _extract_first_matching(sd, _RGB_PREFIXES)
        if "tac_rgb" in target_types
        else ({}, None)
    )
    force_sd, force_prefix = (
        _extract_first_matching(sd, _FORCE_PREFIXES)
        if "tac_force" in target_types
        else ({}, None)
    )
    if ("tac_rgb" in target_types and not rgb_sd) and (
        "tac_force" in target_types and not force_sd
    ):
        raise RuntimeError(
            f"No tac_rgb / tac_force backbone keys found in {path} ({src_name}). "
            f"Tried prefixes: {_RGB_PREFIXES + _FORCE_PREFIXES}"
        )
    if "tac_rgb" in target_types and not rgb_sd:
        raise RuntimeError(
            f"No tac_rgb backbone keys found in {path} ({src_name}). "
            f"Tried prefixes: {_RGB_PREFIXES}"
        )
    if "tac_force" in target_types and not force_sd:
        raise RuntimeError(
            f"No tac_force backbone keys found in {path} ({src_name}). "
            f"Tried prefixes: {_FORCE_PREFIXES}"
        )

    print(
        f"[pretrained_tac] using source={src_name} from {path.name} | "
        f"target_types={target_types}, "
        f"rgb_prefix={rgb_prefix!r} ({len(rgb_sd)} tensors), "
        f"force_prefix={force_prefix!r} ({len(force_sd)} tensors)"
    )

    loaded_keys = 0
    n_param_tensors = 0
    for key in tac_keys:
        typ = tac_type.get(key)
        if typ is None:
            print(f"[pretrained_tac] skip {key}: no tac type in shape_meta")
            continue
        if typ not in target_types:
            print(f"[pretrained_tac] skip {key}: type {typ} not requested")
            continue
        src = rgb_sd if typ == "tac_rgb" else force_sd
        if not src:
            print(f"[pretrained_tac] skip {key}: no weights for type {typ} in ckpt")
            continue
        if key not in obs_encoder.key_model_map:
            print(f"[pretrained_tac] skip {key}: not in key_model_map")
            continue
        module = obs_encoder.key_model_map[key]
        res = module.load_state_dict(src, strict=False)
        loaded_keys += 1
        n_param_tensors += len(src)
        if res.missing_keys:
            print(
                f"[pretrained_tac] {key} ({typ}) missing_keys ({len(res.missing_keys)}): "
                f"{res.missing_keys[:8]}{'...' if len(res.missing_keys) > 8 else ''}"
            )
        if res.unexpected_keys:
            print(
                f"[pretrained_tac] {key} ({typ}) unexpected_keys ({len(res.unexpected_keys)}): "
                f"{res.unexpected_keys[:8]}{'...' if len(res.unexpected_keys) > 8 else ''}"
            )
        print(
            f"[pretrained_tac] loaded {typ} backbone into key_model_map['{key}'] "
            f"from {path.name}"
        )

    return loaded_keys, n_param_tensors
