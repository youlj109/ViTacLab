"""Unified training entry for Diffusion Policy and ViTacDP.

Hydra overrides are passed after ``--`` or as remaining ``key=value`` tokens.
Examples are documented in ``policy/README.md``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("Diffusion_Policy", "ViTacDP"),
        required=True,
        help="Policy package to train.",
    )
    parser.add_argument(
        "--config-name",
        default="robot_dp",
        help="Hydra config name below the selected policy package's config directory.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides, for example task=head_only task.agent_dim=30 task.dataset.zarr_path=...",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    package_dir = Path(__file__).resolve().parent / args.policy
    config_dir = package_dir / "config"
    if not config_dir.is_dir():
        raise SystemExit(f"Policy config directory not found: {config_dir}")
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    core_package = "diffusion_policy_core" if args.policy == "Diffusion_Policy" else "vitacdp_core"

    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    BaseWorkspace = importlib.import_module(f"{core_package}.workspace.base_workspace").BaseWorkspace

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    GlobalHydra.instance().clear()
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
