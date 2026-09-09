"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import os

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import pathlib

import hydra
import torch
import torch.distributed as dist
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")
    ),
)
def main(cfg: OmegaConf):

    OmegaConf.resolve(cfg)
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    try:
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
