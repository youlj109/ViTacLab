"""Canonical Zarr dataset adapter for ViTacDP camera/tactile training data."""

from __future__ import annotations

import copy
from typing import Dict

import numba
import numpy as np
import torch

from vitacdp_core.common.normalize_util import get_image_range_normalizer
from vitacdp_core.common.replay_buffer import ReplayBuffer
from vitacdp_core.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from vitacdp_core.dataset.base_dataset import BaseImageDataset
from vitacdp_core.model.common.normalizer import LinearNormalizer


class RobotImageDataset(BaseImageDataset):
    """Load canonical ViTacDP Zarr data and expand sensor arrays into model keys."""

    def __init__(
        self,
        zarr_path,
        shape_meta,
        camera_num,
        tac_num,
        tac_type,
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
        self.camera_num = int(camera_num)
        self.tac_num = int(tac_num)
        self.tac_type = str(tac_type)
        if self.tac_type not in {"tac_rgb", "tac_force"}:
            raise ValueError(f"tac_type must be tac_rgb or tac_force, got {self.tac_type!r}")

        keys = ["camera", "camera_pos", self.tac_type, "tac_pos", "state", "action"]
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
        if self.replay_buffer["camera"].shape[1] != self.camera_num:
            raise ValueError(
                f"Zarr camera count {self.replay_buffer['camera'].shape[1]} != config {self.camera_num}."
            )
        if self.replay_buffer[self.tac_type].shape[1] != self.tac_num:
            raise ValueError(
                f"Zarr tactile count {self.replay_buffer[self.tac_type].shape[1]} != config {self.tac_num}."
            )
        if self.replay_buffer["camera_pos"].shape[1:] != (self.camera_num, 9):
            raise ValueError(f"camera_pos must be (T,{self.camera_num},9), got {self.replay_buffer['camera_pos'].shape}")
        if self.replay_buffer["tac_pos"].shape[1:] != (self.tac_num, 9):
            raise ValueError(f"tac_pos must be (T,{self.tac_num},9), got {self.replay_buffer['tac_pos'].shape}")
        expected_camera = tuple(self.shape_meta["obs"]["camera"]["shape"])
        expected_camera_hwc = (expected_camera[0], expected_camera[2], expected_camera[3], expected_camera[1])
        if self.replay_buffer["camera"].shape[1:] != expected_camera_hwc:
            raise ValueError(
                f"camera must be (T,{expected_camera_hwc}), got {self.replay_buffer['camera'].shape}."
            )
        expected_tactile = tuple(self.shape_meta["obs"][self.tac_type]["shape"])
        if self.tac_type == "tac_rgb":
            expected_tactile = (
                expected_tactile[0], expected_tactile[2], expected_tactile[3], expected_tactile[1]
            )
        if self.replay_buffer[self.tac_type].shape[1:] != expected_tactile:
            raise ValueError(
                f"{self.tac_type} shape {self.replay_buffer[self.tac_type].shape[1:]} != config {expected_tactile}."
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
        low_dim = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        for index in range(self.camera_num):
            low_dim[f"camera_pos_{index}"] = self.replay_buffer["camera_pos"][:, index]
        for index in range(self.tac_num):
            low_dim[f"tac_pos_{index}"] = self.replay_buffer["tac_pos"][:, index]
            if self.tac_type == "tac_force":
                low_dim[f"tac_{index}"] = self.replay_buffer[self.tac_type][:, index]

        normalizer = LinearNormalizer()
        normalizer.fit(data=low_dim, last_n_dims=1, mode=mode, **kwargs)
        for index in range(self.camera_num):
            normalizer[f"camera_{index}"] = get_image_range_normalizer()
        if self.tac_type == "tac_rgb":
            for index in range(self.tac_num):
                normalizer[f"tac_{index}"] = get_image_range_normalizer()
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
                batch_sample_sequence(
                    self.buffers[key],
                    value,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        raise ValueError(f"Unsupported dataset index type: {type(idx).__name__}")

    def postprocess(self, samples, device):
        camera = samples["camera"].to(device, non_blocking=True).float() / 255.0
        camera_pos = samples["camera_pos"].to(device, non_blocking=True).float()
        tactile = samples[self.tac_type].to(device, non_blocking=True).float()
        tactile_pos = samples["tac_pos"].to(device, non_blocking=True).float()
        if self.tac_type == "tac_rgb":
            tactile = tactile / 255.0

        obs = {"agent_pos": samples["state"].to(device, non_blocking=True).float()}
        for index in range(self.camera_num):
            obs[f"camera_{index}"] = camera[:, :, index].movedim(-1, -3).contiguous()
            obs[f"camera_pos_{index}"] = camera_pos[:, :, index]
        for index in range(self.tac_num):
            value = tactile[:, :, index]
            if self.tac_type == "tac_rgb":
                value = value.movedim(-1, -3).contiguous()
            obs[f"tac_{index}"] = value
            obs[f"tac_pos_{index}"] = tactile_pos[:, :, index]
        return {
            "obs": obs,
            "action": samples["action"].to(device, non_blocking=True).float(),
        }


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
