"""Canonical ViTacDP multi-camera and tactile observation encoder.

This file contains the single maintained encoder architecture used by ViTacDP.
It was consolidated from the historical experiment sequence and preserves the
final checkpoint-compatible module layout without retaining versioned classes.
"""

from __future__ import annotations

import copy
import os
import time

import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision

from vitacdp_core.common.pytorch_util import replace_submodules
from vitacdp_core.model.common.module_attr_mixin import ModuleAttrMixin


def print_params(model: nn.Module) -> None:
    """Print a concise parameter-count summary for the encoder."""
    params_by_part: dict[str, int] = {}
    total = sum(parameter.numel() for parameter in model.parameters())
    for name, parameter in model.named_parameters():
        part = name.split(".", 1)[0]
        params_by_part[part] = params_by_part.get(part, 0) + parameter.numel()
    print("----------------------------------")
    print(f"Class name: {model.__class__.__name__}")
    print(f"  Number of parameters: {total / 1e6:.4f}M")
    for part, count in params_by_part.items():
        ratio = count / total if total else 0.0
        print(f"   {part}: {count / 1e6:.4f}M ({ratio:.2%})")
    print("----------------------------------")

class SelfAttnMaxPoolCompressor(nn.Module):
    """Fuse tactile feature tokens with self-attention and max pooling."""

    def __init__(self, feature_dim, num_heads=4):
        """
        Args:
            feature_dim (int): 触觉特征的维度 D
            num_heads (int): 多头注意力的头数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2), nn.GELU(), nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, tactile_features_list):
        """
        Args:
            tactile_features_list: 包含 N 个触觉特征的 list，每个 shape 为 (Batch, Dim)
        Returns:
            compressed_feature: 融合并压缩后的单个特征，shape 为 (Batch, Dim)
        """
        if not tactile_features_list:
            raise ValueError("At least one tactile feature is required for ViTacDP fusion.")
        x = torch.stack(tactile_features_list, dim=1)
        attn_output, _ = self.self_attn(query=x, key=x, value=x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        compressed_feature, _ = torch.max(x, dim=1)
        return compressed_feature


class BiViTacEncoder(ModuleAttrMixin):
    """Encode cameras, tactile observations, sensor poses, and low-dimensional state."""

    def __init__(self, shape_meta: dict, num_fingers, in_dim, cond_dim, out_dim):
        """
        Keep BiTac constructor interface while matching MMDP
        MultiImageObsEncoder behavior under MMDP robot_dp defaults:
        - rgb_model: resnet18, weights=None
        - resize_shape: null
        - crop_shape: null
        - random_crop: True
        - use_group_norm: True
        - share_rgb_model: False
        - imagenet_norm: True
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()
        _ = (num_fingers, in_dim, cond_dim, out_dim)
        base_rgb_model = torchvision.models.resnet18(weights=None)
        base_rgb_model.fc = nn.Identity()
        rgb_keys = list()
        tac_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        key_pos_map = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                camera_num = shape[0]
                for i in range(camera_num):
                    key_i = key + f'_{i}'
                    key_shape_map[key_i] = shape[1:]
                    rgb_keys.append(key_i)
                    key_pos_map[key_i] = key + f'_pos_{i}'
                    this_model = copy.deepcopy(base_rgb_model)
                    this_model = replace_submodules(root_module=this_model, predicate=lambda x: isinstance(x, nn.BatchNorm2d), func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
                    key_model_map[key_i] = this_model
                    key_transform_map[key_i] = nn.Sequential(nn.Identity(), nn.Identity(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            elif obs_type == 'tac_rgb':
                tac_num = shape[0]
                for i in range(tac_num):
                    key_i = key + f'_{i}'
                    tac_keys.append(key_i)
                    key_shape_map[key_i] = shape[1:]
                    key_pos_map[key_i] = key + f'_pos_{i}'
                    this_model = copy.deepcopy(base_rgb_model)
                    this_model = replace_submodules(root_module=this_model, predicate=lambda x: isinstance(x, nn.BatchNorm2d), func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
                    key_model_map[key_i] = this_model
                    key_transform_map[key_i] = nn.Sequential(nn.Identity(), nn.Identity(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            elif obs_type == 'tac_force':
                tac_num = shape[0]
                for i in range(tac_num):
                    key_i = key + f'_{i}'
                    tac_keys.append(key_i)
                    key_shape_map[key_i] = shape[1:]
                    key_pos_map[key_i] = key + f'_pos_{i}'
                    this_model = copy.deepcopy(base_rgb_model)
                    this_model = replace_submodules(root_module=this_model, predicate=lambda x: isinstance(x, nn.BatchNorm2d), func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
                    this_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    this_model.maxpool = nn.Identity()
                    key_model_map[key_i] = this_model

                    class ToCHW(nn.Module):

                        def forward(self, x: torch.Tensor) -> torch.Tensor:
                            return x.movedim(-1, -3).contiguous()
                    key_transform_map[key_i] = nn.Sequential(ToCHW(), nn.Identity(), nn.Identity(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            elif obs_type == 'low_dim':
                key_shape_map[key] = shape
                low_dim_keys.append(key)
            else:
                pass
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = False
        self.rgb_keys = sorted(rgb_keys)
        self.tac_keys = sorted(tac_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map
        self.key_pos_map = key_pos_map
        self.fuser = SelfAttnMaxPoolCompressor(feature_dim=512)
        self.pos_encoder = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 512))
        self.debug_cam_enable = os.environ.get("VITACDP_CAM_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
        self.debug_cam_dir = os.path.abspath(
            os.path.expanduser(os.environ.get("VITACDP_CAM_DEBUG_DIR", "outputs/vitacdp_cam_debug"))
        )
        self._debug_cam_counter = 0
        if self.debug_cam_enable:
            os.makedirs(self.debug_cam_dir, exist_ok=True)
        self.model_summary_enable = os.environ.get("VITACDP_MODEL_SUMMARY", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if self.model_summary_enable:
            print_params(self)

    def _forward_resnet18_with_featmap(self, model: nn.Module, x: torch.Tensor):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        featmap = model.layer4(x)
        pooled = model.avgpool(featmap)
        feature = torch.flatten(pooled, 1)
        return (feature, featmap)

    def _save_cam_overlay_batch(self, key: str, raw_img: torch.Tensor, featmap: torch.Tensor, grad: torch.Tensor):
        with torch.no_grad():
            if raw_img.ndim != 4:
                print(f'[CAM_DEBUG] skip save {key}: unexpected raw_img ndim={raw_img.ndim}')
                return
            if raw_img.shape[1] == 3:
                img_for_vis = raw_img
            elif raw_img.shape[-1] == 3:
                img_for_vis = raw_img.permute(0, 3, 1, 2).contiguous()
            else:
                print(f'[CAM_DEBUG] skip save {key}: unsupported raw_img shape={tuple(raw_img.shape)}')
                return
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * featmap).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=img_for_vis.shape[-2:], mode='bilinear', align_corners=False)
            cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
            cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-06)
            img = img_for_vis.detach()
            img_min = img.amin(dim=(1, 2, 3), keepdim=True)
            img_max = img.amax(dim=(1, 2, 3), keepdim=True)
            img = (img - img_min) / (img_max - img_min + 1e-06)
            heat = torch.cat([cam, torch.zeros_like(cam), torch.zeros_like(cam)], dim=1)
            overlay = (0.6 * img + 0.4 * heat).clamp(0.0, 1.0)
            batch_size = overlay.shape[0]
            print(f'[CAM_DEBUG] start save key={key}, batch_size={batch_size}, dir={self.debug_cam_dir}')
            for b in range(batch_size):
                self._debug_cam_counter += 1
                unique_name = f'{time.time_ns()}_{os.getpid()}_{self._debug_cam_counter}_{key}_b{b}.png'
                out_path = os.path.join(self.debug_cam_dir, unique_name)
                torchvision.utils.save_image(overlay[b], out_path)
                print(f'[CAM_DEBUG] saved {out_path}')

    def forward(self, obs_dict):
        batch_size = None
        rgb_features = list()
        tac_features = list()
        tac_debug_records = list()
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == self.key_shape_map[key]
            img = self.key_transform_map[key](img)
            feature = self.key_model_map[key](img)
            pos_key = self.key_pos_map[key]
            pos = self.pos_encoder(obs_dict[pos_key])
            rgb_features.append(feature + pos)
        for key in self.tac_keys:
            raw_img = obs_dict[key]
            img = raw_img
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == self.key_shape_map[key]
            model = self.key_model_map[key]
            if self.debug_cam_enable:
                print(f'[CAM_DEBUG] tac_key={key}, grad_enabled={torch.is_grad_enabled()}')
            if self.debug_cam_enable:
                with torch.inference_mode(False):
                    with torch.enable_grad():
                        img = self.key_transform_map[key](img)
                        feature, featmap = self._forward_resnet18_with_featmap(model, img)
                        featmap.retain_grad()
                        pos_key = self.key_pos_map[key]
                        pos = self.pos_encoder(obs_dict[pos_key])
                        tac_features.append(feature + pos)
                        print(f'[CAM_DEBUG] capture featmap for {key}, featmap_shape={tuple(featmap.shape)}, grad_enabled_now={torch.is_grad_enabled()}')
                        tac_debug_records.append({'key': key, 'raw_img': raw_img, 'featmap': featmap})
            else:
                img = self.key_transform_map[key](img)
                feature = model(img)
                pos_key = self.key_pos_map[key]
                pos = self.pos_encoder(obs_dict[pos_key])
                tac_features.append(feature + pos)
        features = list()
        if self.debug_cam_enable:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    fuse_feat = self.fuser(tac_features)
        else:
            fuse_feat = self.fuser(tac_features)
        if self.debug_cam_enable:
            print(f'[CAM_DEBUG] after fuser: records={len(tac_debug_records)}, grad_enabled={torch.is_grad_enabled()}')
        if self.debug_cam_enable and tac_debug_records:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    self.zero_grad(set_to_none=True)
                    cam_target = fuse_feat.norm(dim=-1).sum()
                    print(f'[CAM_DEBUG] backward cam_target={cam_target.item():.6f}')
                    cam_target.backward(retain_graph=True)
                    for rec in tac_debug_records:
                        grad = rec['featmap'].grad
                        if grad is None:
                            print(f"[CAM_DEBUG] skip {rec['key']}: grad is None")
                            continue
                        print(f"[CAM_DEBUG] grad ready for {rec['key']}, grad_shape={tuple(grad.shape)}")
                        self._save_cam_overlay_batch(key=rec['key'], raw_img=rec['raw_img'], featmap=rec['featmap'], grad=grad)
        features.extend(rgb_features)
        features.append(fuse_feat)
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        # Shape inference needs one synthetic sample only; a large historical
        # batch could allocate multiple gigabytes for camera/tactile tensors.
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            if attr['type'] == 'rgb':
                camera_num = attr['shape'][0]
                for i in range(camera_num):
                    example_obs_dict[f'{key}_{i}'] = torch.zeros((batch_size, *attr['shape'][1:]), dtype=self.dtype, device=self.device)
                    example_obs_dict[f'{key}_pos_{i}'] = torch.zeros((batch_size, *obs_shape_meta[f'{key}_pos']['shape'][1:]), dtype=self.dtype, device=self.device)
            elif attr['type'] == 'tac_rgb':
                tac_num = attr['shape'][0]
                for i in range(tac_num):
                    example_obs_dict[f'{key}_{i}'] = torch.zeros((batch_size, *attr['shape'][1:]), dtype=self.dtype, device=self.device)
                    example_obs_dict[f'{key}_pos_{i}'] = torch.zeros((batch_size, *obs_shape_meta[f'{key}_pos']['shape'][1:]), dtype=self.dtype, device=self.device)
            elif attr['type'] == 'tac_force':
                tac_num = attr['shape'][0]
                for i in range(tac_num):
                    example_obs_dict[f'{key}_{i}'] = torch.zeros((batch_size, *attr['shape'][1:]), dtype=self.dtype, device=self.device)
                    example_obs_dict[f'{key}_pos_{i}'] = torch.zeros((batch_size, *obs_shape_meta[f'{key}_pos']['shape'][1:]), dtype=self.dtype, device=self.device)
            elif attr['type'] == 'low_dim':
                example_obs_dict[key] = torch.zeros((batch_size, attr['shape'][0]), dtype=self.dtype, device=self.device)
            elif attr['type'] == 'pos':
                pass
            else:
                raise ValueError(f"Unsupported obs type: {attr['type']}")
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        if self.model_summary_enable:
            print("BiViTacEncoder output_shape:", output_shape)
        return output_shape
