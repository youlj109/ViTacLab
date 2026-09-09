import random
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision
# from diffusion_policy.common.Attention_utils import DoubleCrossAttention, PositionEmbedding1D
# from diffusion_policy.common.Tac_encoder import TacEncoder_v1, TacEncoder_v2, TacEncoder_v3, TacEncoder_v4, TacEncoder_v5, TacEncoder_v6, TacEncoder_v7, TacEncoder_v8, TacEncoder_v9, TacEncoder_v10, TacEncoder_v11, TacEncoder_v12, TacEncoder_v13
# from diffusion_policy.common.MoT import MoT
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from termcolor import cprint
from typing import List, Type
import copy

def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        part_name = name.split(".")[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()

    cprint(f"----------------------------------", "cyan")
    cprint(f"Class name: {model.__class__.__name__}", "cyan")
    cprint(f"  Number of parameters: {all_num_param / 1e6:.4f}M", "cyan")
    for part_name, num_params in params_dict.items():
        cprint(
            f"   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})",
            "cyan",
        )
    cprint(f"----------------------------------", "cyan")

class SelfAttnMaxPoolCompressor(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        """
        Args:
            feature_dim (int): 触觉特征的维度 D
            num_heads (int): 多头注意力的头数
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Self-Attention 层
        # batch_first=True 让输入输出 shape 维持在 (Batch, Seq_len, Dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 2. 规范化层 (LayerNorm) - 稳定训练
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # 3. 前馈神经网络 (FFN) - 增加每个 Token 的非线性表达能力
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, tactile_features_list):
        """
        Args:
            tactile_features_list: 包含 N 个触觉特征的 list，每个 shape 为 (Batch, Dim)
        Returns:
            compressed_feature: 融合并压缩后的单个特征，shape 为 (Batch, Dim)
        """
        # 将传入的 N 个特征堆叠成集合形式: (Batch, N, Dim)
        # 这里 N 就是序列长度 Seq_len
        x = torch.stack(tactile_features_list, dim=1)
        
        # ==========================================
        # 阶段 1: Self-Attention (传感器之间的信息交互)
        # ==========================================
        # 在自注意力中，Query, Key, Value 都是 x 本身
        attn_output, _ = self.self_attn(query=x, key=x, value=x)
        
        # 残差连接 + LayerNorm
        x = self.norm1(x + attn_output)
        
        # ==========================================
        # 阶段 2: Feed Forward (特征非线性升维再降维)
        # ==========================================
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        # 此时 x 的 shape 依然是 (Batch, N, Dim)，但已经融合了全局信息
        
        # ==========================================
        # 阶段 3: Max-Pooling (提取最显著的特征)
        # ==========================================
        # 在 N 个传感器的维度 (dim=1) 上取最大值
        # 注意：torch.max 会返回一个元组 (values, indices)，我们只需要 values [0]
        compressed_feature, _ = torch.max(x, dim=1)
        
        # 最终输出 shape: (Batch, Dim)
        return compressed_feature

class ViTacEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        num_fingers,
        in_dim,
        cond_dim,
        out_dim,
    ):
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

        # Keep compatibility with existing BiTac hydra config signature.
        _ = (num_fingers, in_dim, cond_dim, out_dim)

        # Fixed settings aligned with IL_Baselines/Diffusion_Policy config.
        base_rgb_model = torchvision.models.resnet18(weights=None)
        base_rgb_model.fc = nn.Identity()
        rgb_keys = list()
        tac_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        key_pos_map = dict()
        
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")

            if obs_type == "rgb":
                camera_num = shape[0]
                for i in range(camera_num):
                    key_i = key + f"_{i}"
                    key_shape_map[key_i] = shape[1:]
                    rgb_keys.append(key_i)
                    key_pos_map[key_i] = key + f"_pos_{i}"
                    this_model = copy.deepcopy(base_rgb_model)
                    this_model = replace_submodules(
                        root_module=this_model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features // 16,
                            num_channels=x.num_features,
                        ),
                    )
                    key_model_map[key_i] = this_model

                    # MMDP defaults: no resize/crop, always imagenet normalize
                    key_transform_map[key_i] = nn.Sequential(
                        nn.Identity(),
                        nn.Identity(),
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    )
            elif obs_type == "tac_rgb":
                tac_num = shape[0]
                for i in range(tac_num):
                    key_i = key + f"_{i}"
                    tac_keys.append(key_i)
                    key_shape_map[key_i] = shape[1:]
                    key_pos_map[key_i] = key + f"_pos_{i}"
                    this_model = copy.deepcopy(base_rgb_model)
                    this_model = replace_submodules(
                        root_module=this_model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features // 16,
                            num_channels=x.num_features,
                        ),
                    )
                    key_model_map[key_i] = this_model

                    # MMDP defaults: no resize/crop, always imagenet normalize
                    key_transform_map[key_i] = nn.Sequential(
                        nn.Identity(),
                        nn.Identity(),
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    )
            elif obs_type == "tac_force":
                tac_num = shape[0]
                for i in range(tac_num):
                    key_i = key + f"_{i}"
                    tac_keys.append(key_i)
                    key_shape_map[key_i] = shape[1:]
                    key_pos_map[key_i] = key + f"_pos_{i}"
                    this_model = copy.deepcopy(base_rgb_model)
                    this_model = replace_submodules(
                        root_module=this_model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features // 16,
                            num_channels=x.num_features,
                        ),
                    )
                    this_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    this_model.maxpool = nn.Identity()
                    key_model_map[key_i] = this_model
                    
                    class ToCHW(nn.Module):
                        def forward(self, x: torch.Tensor) -> torch.Tensor:
                            return x.movedim(-1, -3).contiguous()

                    # MMDP defaults: no resize/crop, always imagenet normalize
                    key_transform_map[key_i] = nn.Sequential(
                        ToCHW(),
                        nn.Identity(),
                        nn.Identity(),
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    )
            elif obs_type == "low_dim":
                key_shape_map[key] = shape
                low_dim_keys.append(key)
            else:
                # raise RuntimeError(f"Unsupported obs type: {obs_type}")
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
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 512),
        )
        
        print_params(self)

    def forward(self, obs_dict):
        batch_size = None
        rgb_features = list()
        tac_features = list()

        # run each rgb obs through independent models
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
            tac_features.append(feature + pos)
        # v1 = features[0]
        # v1_pos = self.pos_encoder(obs_dict["head_camera_pos"])
        # v2 = features[3]
        # v2_pos = self.pos_encoder(obs_dict["twist_camera_pos"])
        # t1 = features[1]
        # t1_pos = self.pos_encoder(obs_dict["left_hand_tac_pos"])
        # t2 = features[2]
        # t2_pos = self.pos_encoder(obs_dict["right_hand_tac_pos"])
        
        features = list()
        fuse_feat = self.fuser(tac_features)
        features.extend(rgb_features)
        features.append(fuse_feat)
        
        # process low_dim input
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
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 192
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            if attr["type"] == "rgb" or attr["type"] == "tac_rgb" or attr["type"] == "tac_force" or attr["type"] == "pos":
                sensor_num = shape[0]
                for i in range(sensor_num):
                    example_obs_dict[f"{key}_{i}"] = torch.zeros(
                        (batch_size,) + shape[1:],
                        dtype=self.dtype,
                        device=self.device,
                    )
            else:
                example_obs_dict[key] = torch.zeros(
                    (batch_size,) + shape,
                    dtype=self.dtype,
                    device=self.device,
                )
        example_output = self.forward(example_obs_dict)
        return example_output.shape[1:]