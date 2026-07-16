import copy
import pdb
from typing import Dict

import numba
import numpy as np
import torch
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from termcolor import cprint


class RobotImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        shape_meta,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
    ):

        super().__init__()
        # cprint(zarr_path, "red")
        # cprint(batch_size, "red")
        self.shape_meta = shape_meta
        self.tac_type = shape_meta["obs"]["tac"]["type"]
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            # keys=["head_camera", "twist_camera", "left_hand_tac_rgb", "right_hand_tac_rgb", "head_camera_pos", "twist_camera_pos", "left_hand_tac_pos", "right_hand_tac_pos", "state", "action"],
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

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
        camera_pos = self.replay_buffer["camera_pos"]  # N, Nc, D
        tac_pos = self.replay_buffer["tac_pos"]  # N, Nt, D
        tac_force = self.replay_buffer["tac_force"]  # N, Nt, ...

        n_camera = camera_pos.shape[1]
        n_tac = tac_pos.shape[1]
        shared_pos_data = np.concatenate(
            [camera_pos[:, i, :] for i in range(n_camera)]
            + [tac_pos[:, i, :] for i in range(n_tac)],
            axis=0,
        )
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "shared_camera_tac_pos": shared_pos_data,
        }
        if self.tac_type == "tac_force":
            for i in range(n_tac):
                data[f"tac_{i}"] = tac_force[:, i, ...]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        shared_pos_normalizer = normalizer["shared_camera_tac_pos"]

        # Different camera/tactile dimensions use different normalizer keys,
        # but they share fit statistics from all pose sources.
        for i in range(n_camera):
            normalizer[f"camera_pos_{i}"] = shared_pos_normalizer
            normalizer[f"camera_{i}"] = get_image_range_normalizer()
        for i in range(n_tac):
            normalizer[f"tac_pos_{i}"] = shared_pos_normalizer
        
        if self.tac_type == "tac_rgb":
            for i in range(n_tac):
                normalizer[f"tac_{i}"] = get_image_range_normalizer()  # tac_rgb image norm
        del normalizer.params_dict["shared_camera_tac_pos"]
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            # print(idx, len(idx))
            # print(self.batch_size)
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        camera = samples["camera"].to(device, non_blocking=True) / 255.0
        if self.tac_type == "tac_rgb":
            tac = samples["tac_rgb"].to(device, non_blocking=True) / 255.0
        elif self.tac_type == "tac_force":
            tac = samples["tac_force"].to(device, non_blocking=True)
        else:
            raise ValueError(f"Unsupported tac type: {self.tac_type}")
        action = samples["action"].to(device, non_blocking=True)
        camera_pos = samples["camera_pos"].to(device, non_blocking=True)
        tac_pos = samples["tac_pos"].to(device, non_blocking=True)
        n_camera = camera.shape[2]
        n_tac = tac.shape[2]
        obs = {
            "agent_pos": agent_pos,  # B, T, D
        }
        for i in range(n_camera):
            obs[f"camera_{i}"] = camera[:, :, i, ...]  # B, T, 3, H, W
            obs[f"camera_pos_{i}"] = camera_pos[:, :, i, ...]  # B, T, D
        for i in range(n_tac):
            obs[f"tac_{i}"] = tac[:, :, i, ...]  # B, T, ...
            obs[f"tac_pos_{i}"] = tac_pos[:, :, i, ...]  # B, T, D
        return {
            "obs": obs,
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[
            idx[i]
        ]
        data[i, sample_start_idx:sample_end_idx] = input_arr[
            buffer_start_idx:buffer_end_idx
        ]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(
    _batch_sample_sequence, nopython=True, parallel=False
)
_batch_sample_sequence_parallel = numba.jit(
    _batch_sample_sequence, nopython=True, parallel=True
)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(
            data, input_arr, indices, idx, sequence_length
        )
