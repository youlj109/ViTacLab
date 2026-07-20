import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vitacdp_core.common.pytorch_util import dict_apply
from vitacdp_core.model.common.normalizer import LinearNormalizer
from vitacdp_core.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vitacdp_core.model.diffusion.mask_generator import LowdimMaskGenerator
from vitacdp_core.policy.base_image_policy import BaseImagePolicy
from einops import rearrange, reduce


def _first_bad_index(finite_mask: torch.Tensor):
    bad = torch.nonzero(~finite_mask, as_tuple=False)
    if bad.numel() == 0:
        return None
    return tuple(int(value) for value in bad[0].detach().cpu().tolist())


def _tensor_stats(
    name: str,
    value: torch.Tensor,
    finite_mask: torch.Tensor | None = None,
) -> str:
    tensor = value.detach()
    if finite_mask is None:
        finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    total = int(tensor.numel())
    first_bad = _first_bad_index(finite_mask)
    summary = (
        f"stage={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} finite={finite_count}/{total}"
    )
    if finite_count:
        finite_values = tensor[finite_mask]
        summary += (
            f" min={float(finite_values.min().item()):.9g}"
            f" max={float(finite_values.max().item()):.9g}"
            f" mean={float(finite_values.float().mean().item()):.9g}"
        )
    if first_bad is not None:
        summary += f" first_bad_index={first_bad}"
    return summary


def _require_finite(name: str, value: torch.Tensor, *, report: bool = False) -> None:
    finite_mask = torch.isfinite(value)
    all_finite = bool(finite_mask.all().item())
    if report or not all_finite:
        summary = _tensor_stats(name, value, finite_mask=finite_mask)
    if report:
        print(f"[ViTacDP-NaN-DIAG] {summary}")
    if not all_finite:
        raise RuntimeError(f"[ViTacDP-NaN-DIAG] first non-finite tensor: {summary}")


def _iter_output_tensors(value, path="output"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_output_tensors(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_output_tensors(item, f"{path}[{index}]")


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: nn.Module,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            # global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        # Hydra may inject workspace-only flags under policy; they must not reach
        # diffusers DDPMScheduler.step(**kwargs) via self.kwargs.
        _kwargs = dict(kwargs)
        for _k in ("load_pretrained_tac", "pretrained_tac_ckpt"):
            _kwargs.pop(_k, None)
        self.kwargs = _kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self._nan_diag_enabled = os.environ.get("VITACDP_NAN_DIAG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._nan_diag_completed = False
        self._nan_diag_context = "not-set"

    def _validate_inference_state(self) -> None:
        parameter_tensors = 0
        parameter_elements = 0
        for name, value in self.named_parameters():
            _require_finite(f"model_parameter.{name}", value)
            parameter_tensors += 1
            parameter_elements += int(value.numel())

        buffer_tensors = 0
        buffer_elements = 0
        for name, value in self.named_buffers():
            _require_finite(f"model_buffer.{name}", value)
            buffer_tensors += 1
            buffer_elements += int(value.numel())

        for key, params in self.normalizer.params_dict.items():
            scale = params["scale"]
            offset = params["offset"]
            _require_finite(f"normalizer.{key}.scale", scale, report=True)
            _require_finite(f"normalizer.{key}.offset", offset, report=True)
            zero_scale = torch.nonzero(scale == 0, as_tuple=False)
            if zero_scale.numel():
                first_zero = tuple(
                    int(value) for value in zero_scale[0].detach().cpu().tolist()
                )
                raise RuntimeError(
                    "[ViTacDP-NaN-DIAG] zero normalizer scale would make unnormalization "
                    f"divide by zero: key={key!r}, first_zero_index={first_zero}, "
                    f"shape={tuple(scale.shape)}"
                )

        print(
            "[ViTacDP-NaN-DIAG] checkpoint tensors are finite: "
            f"parameters={parameter_tensors} ({parameter_elements} elements), "
            f"buffers={buffer_tensors} ({buffer_elements} elements)"
        )

    def _register_nan_hooks(self):
        handles = []

        def register_tree(root_name: str, root: nn.Module):
            for module_name, module in root.named_modules():
                qualified_name = root_name if not module_name else f"{root_name}.{module_name}"

                def check_output(_module, _inputs, output, qualified_name=qualified_name):
                    for output_path, tensor in _iter_output_tensors(output):
                        if not torch.isfinite(tensor).all():
                            _require_finite(
                                f"{self._nan_diag_context}.{qualified_name}.{output_path}",
                                tensor,
                            )

                handles.append(module.register_forward_hook(check_output))

        register_tree("obs_encoder", self.obs_encoder)
        register_tree("diffusion_model", self.model)
        return handles

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        diag_active = self._nan_diag_enabled and not self._nan_diag_completed
        if diag_active:
            _require_finite(
                "conditional_sample.initial_random_trajectory",
                trajectory,
                report=True,
            )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            timestep = (
                int(t.detach().cpu().item())
                if isinstance(t, torch.Tensor)
                else int(t)
            )
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            if diag_active:
                _require_finite(
                    f"conditional_sample.conditioned_trajectory[t={timestep}]",
                    trajectory,
                )

            # 2. predict model output
            self._nan_diag_context = f"conditional_sample.model_forward[t={timestep}]"
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )
            if diag_active:
                _require_finite(
                    f"conditional_sample.model_output[t={timestep}]",
                    model_output,
                )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample
            if diag_active:
                _require_finite(
                    f"conditional_sample.scheduler_prev_sample[t={timestep}]",
                    trajectory,
                )

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        diag_active = self._nan_diag_enabled and not self._nan_diag_completed
        hook_handles = []
        if diag_active:
            print("[ViTacDP-NaN-DIAG] enabled for this first predict_action call")
            self._validate_inference_state()
            hook_handles = self._register_nan_hooks()

        try:
            if diag_active:
                for key, value in obs_dict.items():
                    _require_finite(f"raw_observation.{key}", value, report=True)

            # normalize input
            nobs = self.normalizer.normalize(obs_dict)
            if diag_active:
                for key, value in nobs.items():
                    _require_finite(f"normalized_observation.{key}", value, report=True)
            value = next(iter(nobs.values()))
            B, To = value.shape[:2]
            T = self.horizon
            Da = self.action_dim
            Do = self.obs_feature_dim
            To = self.n_obs_steps

            # build input
            device = self.device
            dtype = self.dtype

            # handle different ways of passing observation
            local_cond = None
            global_cond = None
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            if diag_active:
                for key, value in this_nobs.items():
                    _require_finite(f"encoder_input.{key}", value, report=True)
                self._nan_diag_context = "predict_action.encoder_forward"

            if self.obs_as_global_cond:
                # condition through global feature
                nobs_features = self.obs_encoder(this_nobs)
                if diag_active:
                    _require_finite("encoder_output", nobs_features, report=True)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
                if diag_active:
                    _require_finite("global_condition", global_cond, report=True)
                # empty data for action
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                # condition through impainting
                nobs_features = self.obs_encoder(this_nobs)
                if diag_active:
                    _require_finite("encoder_output", nobs_features, report=True)
                # reshape back to B, T, Do
                nobs_features = nobs_features.reshape(B, To, -1)
                cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:, :To, Da:] = nobs_features
                cond_mask[:, :To, Da:] = True

            if diag_active:
                _require_finite("conditional_data", cond_data, report=True)

            # run sampling
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs,
            )
            if diag_active:
                _require_finite("diffusion_sample", nsample, report=True)

            # unnormalize prediction
            naction_pred = nsample[..., :Da]
            if diag_active:
                _require_finite("normalized_action_prediction", naction_pred, report=True)
            action_pred = self.normalizer["action"].unnormalize(naction_pred)
            if diag_active:
                _require_finite("unnormalized_action_prediction", action_pred, report=True)

            # get action
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
            if diag_active:
                _require_finite("returned_action_chunk", action, report=True)
                print("[ViTacDP-NaN-DIAG-PASS] first predict_action call remained finite")
                self._nan_diag_completed = True

            result = {"action": action, "action_pred": action_pred}
            return result
        finally:
            for handle in hook_handles:
                handle.remove()

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
