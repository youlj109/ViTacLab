from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import torch

from omegaconf import OmegaConf

from lehome.lehome.assets.object.fluid import FluidObject


@dataclass
class WaterObjectCfg:
    """Config for a water (fluid) object used in pouring tasks.

    设计参考 Lehome_Marble 中的 `lehome/assets/object/fluid.py` 与
    `tasks/livingroom/loft_water.py`，但将资产路径与配置放在当前 ViTacLab 工程：

    - USD 资产: `source/ViTacLab/ViTacLab/assets/data/Objects/Water/water.usdc`
    - 配置 YAML: `source/ViTacLab/ViTacLab/assets/data/Objects/Water/config/fluid.yaml`

    以上文件需由用户根据自身水体资产实际情况放置到对应路径。
    """

    prim_path: str = "/World/Object/fluid_items/fluid_items_1"
    usd_path: str = "source/ViTacLab/ViTacLab/assets/data/Objects/Water/water.usdc"
    cfg_path: str = "source/ViTacLab/ViTacLab/assets/data/Objects/Water/config/fluid.yaml"
    use_container: bool = True


class WaterObject:
    """Thin wrapper around Lehome FluidObject, adapted for ViTacLab path layout.

    用法与 `lehome.lehome.assets.object.fluid.FluidObject` 基本一致，在 DirectRLEnv
    环境中创建一个 WaterObject 实例并在 `_setup_scene` 中调用。
    """

    def __init__(self, env_id: int, env_origin: torch.Tensor, cfg: WaterObjectCfg | None = None):
        self.cfg = cfg or WaterObjectCfg()

        # 解析配置文件
        cfg_path = self.cfg.cfg_path
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(os.getcwd(), self.cfg.cfg_path)
        fluid_cfg = OmegaConf.load(cfg_path)

        usd_path = self.cfg.usd_path
        if not os.path.isabs(usd_path):
            usd_path = os.path.join(os.getcwd(), self.cfg.usd_path)

        # 创建底层 FluidObject（Lehome 实现）
        self._fluid = FluidObject(
            env_id=env_id,
            env_origin=env_origin,
            prim_path=self.cfg.prim_path,
            usd_path=usd_path,
            config=fluid_cfg,
            use_container=self.cfg.use_container,
        )

    def initialize(self) -> None:
        """Initialize fluid state if needed."""
        if hasattr(self._fluid, "initialize"):
            self._fluid.initialize()

    def reset(self, soft: bool = True) -> None:
        """Reset fluid container / particles."""
        if hasattr(self._fluid, "reset"):
            self._fluid.reset(soft=soft)

    def get_all_pose(self) -> dict[str, Any]:
        """Expose fluid pose(s) for success metric or logging."""
        if hasattr(self._fluid, "get_all_pose"):
            return self._fluid.get_all_pose()
        return {}

