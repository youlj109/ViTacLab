"""Canonical camera-only Zarr dataset adapter for Diffusion Policy."""

from __future__ import annotations

import copy
from typing import Dict

import numba
import numpy as np
import torch

from diffusion_policy_core.common.normalize_util import get_image_range_normalizer
from diffusion_policy_core.common.replay_buffer import ReplayBuffer
from diffusion_policy_core.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy_core.dataset.base_dataset import BaseImageDataset
from diffusion_policy_core.model.common.normalizer import LinearNormalizer


class RobotImageDataset(BaseImageDataset):
    """Load head/twist camera episodes and expose model observation keys."""

    _ZARR_TO_MODEL = {"head_camera": "head_cam", "twist_camera": "twist_cam"}

    def __init__(
        self,
        zarr_path,
        shape_meta=None,
        camera_keys=("head_camera", "twist_camera"),
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.camera_keys = tuple(camera_keys)
        unsupported = [key for key in self.camera_keys if key not in self._ZARR_TO_MODEL]
        if unsupported:
            raise ValueError(f"Unsupported camera Zarr keys: {unsupported}")
        keys = [*self.camera_keys, "state", "action"]
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
        self._validate_shapes()

        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.batch_size = int(batch_size)
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            key: np.zeros((self.batch_size, sequence_length, *value.shape[1:]), dtype=value.dtype)
            for key, value in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {key: torch.from_numpy(value) for key, value in self.buffers.items()}
        for value in self.buffers_torch.values():
            value.pin_memory()

    def _validate_shapes(self) -> None:
        """Validate Zarr state/action/camera dimensions against the Hydra profile."""

        if self.shape_meta is None:
            return
        expected_state = tuple(self.shape_meta["obs"]["agent_pos"]["shape"])
        expected_action = tuple(self.shape_meta["action"]["shape"])
        if self.replay_buffer["state"].shape[1:] != expected_state:
            raise ValueError(
                f"Zarr state shape {self.replay_buffer['state'].shape[1:]} != config {expected_state}."
            )
        if self.replay_buffer["action"].shape[1:] != expected_action:
            raise ValueError(
                f"Zarr action shape {self.replay_buffer['action'].shape[1:]} != config {expected_action}."
            )
        for zarr_key in self.camera_keys:
            model_key = self._ZARR_TO_MODEL[zarr_key]
            expected_chw = tuple(self.shape_meta["obs"][model_key]["shape"])
            expected_hwc = (expected_chw[1], expected_chw[2], expected_chw[0])
            if self.replay_buffer[zarr_key].shape[1:] != expected_hwc:
                raise ValueError(
                    f"Zarr camera {zarr_key} shape {self.replay_buffer[zarr_key].shape[1:]} "
                    f"!= config HWC {expected_hwc}."
                )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": self.replay_buffer["action"], "agent_pos": self.replay_buffer["state"]},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        for key in self.camera_keys:
            normalizer[self._ZARR_TO_MODEL[key]] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError("Slice access is not supported by the optimized batch sampler.")
        if isinstance(idx, int):
            return {key: torch.from_numpy(value) for key, value in self.sampler.sample_sequence(idx).items()}
        if isinstance(idx, np.ndarray):
            if len(idx) != self.batch_size:
                raise ValueError(f"Expected an index batch of {self.batch_size}, got {len(idx)}")
            for key, value in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[key], value, self.sampler.indices, idx, self.sampler.sequence_length)
            return self.buffers_torch
        raise ValueError(f"Unsupported dataset index type: {type(idx).__name__}")

    def postprocess(self, samples, device):
        obs = {"agent_pos": samples["state"].to(device, non_blocking=True).float()}
        for key in self.camera_keys:
            obs[self._ZARR_TO_MODEL[key]] = (
                samples[key].to(device, non_blocking=True).float().movedim(-1, -3).contiguous() / 255.0
            )
        return {"obs": obs, "action": samples["action"].to(device, non_blocking=True).float()}


def _batch_sample_sequence(data, input_arr, indices, idx, sequence_length):
    for index in numba.prange(len(idx)):
        buffer_start, buffer_end, sample_start, sample_end = indices[idx[index]]
        data[index, sample_start:sample_end] = input_arr[buffer_start:buffer_end]
        if sample_start > 0:
            data[index, :sample_start] = data[index, sample_start]
        if sample_end < sequence_length:
            data[index, sample_end:] = data[index, sample_end - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(data, input_arr, indices, idx, sequence_length):
    batch_size = len(idx)
    expected = (batch_size, sequence_length, *input_arr.shape[1:])
    if data.shape != expected:
        raise ValueError(f"Batch buffer shape {data.shape} != expected {expected}")
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
