---
pretty_name: ViTacLab Runtime Assets
license: other
task_categories:
  - reinforcement-learning
tags:
  - robotics
  - tactile-sensing
  - isaac-sim
  - isaac-lab
---

# ViTacLab runtime assets

This dataset repository contains the large binary assets needed by the
[ViTacLab](https://github.com/youlj109/ViTacLab) simulation environments.
Its paths mirror the Git checkout, so users should download it directly into
the repository root.

## Contents

- `assets/data/Objects/`: task objects used by the canonical environments.
- `assets/data/Robots/`: UR10e, Shadow Hand, Franka, and Wuji Hand assets.
- `assets/data/Sensors/`: GelSight sensor USD, meshes, textures, and calibration.
- `xense_lab_data/`: Xense background, marker rest positions, and Taxim polycalib.
- `data/calibration/tactile/`: processed tactile calibration and validation data.
- `MANIFEST.json`: byte sizes and SHA256 hashes for every published file.

High-fidelity room backgrounds, generated logs, recorded rollouts, policy
checkpoints, duplicate archives, and backup files are intentionally excluded.
Isaac Lab Factory task assets and its standard Nucleus assets remain
dependencies of the matching Isaac Lab installation.

## Download

From the root of a ViTacLab Git checkout:

```bash
python -m venv .venv_hf
.venv_hf/bin/python -m pip install -U huggingface_hub
bash bash_command/download_hf_assets.sh Yanlj/ViTacLab-assets
```

The download script overlays the repository-relative paths and verifies every
file against `MANIFEST.json`.

## Compatibility

The release is intended for Linux with Isaac Sim 5.1.0 and Isaac Lab 2.3.2.
See `docs/REPRODUCIBILITY.md` in the Git repository for installation,
configuration, smoke tests, and the exact boundary between Git, this dataset,
and upstream Isaac assets.

## License

The repository contains assets with mixed provenance. The ViTacLab-authored
parts follow the code repository's license; bundled third-party assets retain
their original notices and terms. Users are responsible for complying with
the license or terms attached to each upstream asset.
