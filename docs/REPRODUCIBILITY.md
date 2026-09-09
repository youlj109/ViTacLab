# ViTacLab reproduction guide

This guide covers the supported release target: all canonical environments,
tactile sensors, calibration, and demos. Policy training datasets and policy
checkpoints are outside this release target.

## 1. Release layout

ViTacLab is intentionally split across three sources:

1. **GitHub** contains source code, text configuration, launch scripts, and
   documentation.
2. **Hugging Face Dataset** contains repository-relative binary assets under
   `source/ViTacLab/ViTacLab/assets/data/` and tactile calibration data.
3. **Isaac Lab / Isaac Sim** supplies the simulator, standard Nucleus assets,
   Factory task assets, and TacSL data under `ISAACLAB_NUCLEUS_DIR`.

Generated logs, videos, rollout NPZ files, policy checkpoints, duplicate ZIP
archives, and optional high-fidelity room backgrounds are not prerequisites
for the supported target.

## 2. Tested software contract

- Linux; development host uses Ubuntu 22.04.
- NVIDIA GPU and a driver supported by Isaac Sim 5.1.0.
- Isaac Sim **5.1.0**.
- Isaac Lab **2.3.2**.
- Python provided by the matching Isaac Sim / Isaac Lab environment. Do not
  use an unrelated system Python; the development machine's base Python 3.13
  is not the reproduction environment.

Isaac Sim, PyTorch, CUDA libraries, and core Isaac Lab packages must come from
one consistent Isaac Lab installation. Do not independently upgrade PyTorch
inside that environment.

## 3. Install

```bash
git clone https://github.com/youlj109/ViTacLab.git
cd ViTacLab

# Activate the environment in which Isaac Lab 2.3.2 is already usable.
conda activate <isaac-lab-environment>

python -m pip install -e source/ViTacLab
python -m pip install -r requirements/repro.txt

# Keep Hub tooling isolated from Isaac/Anaconda package constraints.
python -m venv .venv_hf
.venv_hf/bin/python -m pip install -U huggingface_hub

bash bash_command/download_hf_assets.sh Yanlj/ViTacLab-assets
```

The asset downloader preserves repository-relative paths and verifies sizes
and SHA256 hashes from the published `MANIFEST.json`.

Create a machine-local configuration when paths differ from the defaults:

```bash
cp config/local.env.example config/local.env
# Edit config/local.env, then:
set -a && source config/local.env && set +a
```

`config/local.env` is ignored by Git. Do not put tokens in it.

## 4. External Isaac assets

The code intentionally uses upstream assets that must match the installed
Isaac Lab / Isaac Sim release:

- `{ISAACLAB_NUCLEUS_DIR}/TacSL/` for standard TacSL render calibration.
- Isaac Lab Factory peg, gear, and nut assets for Forge tasks.
- `{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd`.

These are not mirrored to the ViTacLab Hugging Face repository. A machine that
cannot access the configured Isaac asset root must first install or cache the
corresponding upstream asset pack.

## 5. Static acceptance

Run this before launching Isaac Sim:

```bash
bash bash_command/verify_repro_env.sh
```

The preflight checks the launcher, GPU driver, downloaded assets and calibration
files, then runs `scripts/audit_project.py --verbose`. Acceptance requires zero
errors and exactly 31 unique canonical Gym registrations. Static success proves
source/layout consistency, not PhysX runtime correctness.

## 6. Runtime acceptance

First list the registered tasks:

```bash
python scripts/list_envs.py
```

For every task ID in `docs/ENVIRONMENT_MATRIX.md`, run:

```bash
python scripts/zero_agent.py \
  --task <TASK_ID> --num_envs 1 --max-steps 20 \
  --enable_cameras --headless
```

The required result is `[SENSOR-DIAG-PASS]`, finite tactile tensors, nonzero
and nonconstant tactile RGB, 20 completed steps, and a clean shutdown.

Then run the stricter ViTacSim/PhysX gate:

```bash
bash bash_command/visuotactile_physx_alignment_check.sh
```

See `docs/VITACSIM_PHYSX_VALIDATION.md` for metrics and failure interpretation.

## 7. Optional components

- Video teleoperation uses a separate sender environment; install
  `source/video_teleop/requirements.txt` there.
- Xense calibration fitting can use the external Taxim implementation described
  in `third_party/README.md`.
- High-fidelity room backgrounds under `assets/data/Scene/` are only used by
  `full_tra_high_fidelity.py` and `bash_command/high_env.sh`; they are not
  required by the canonical environment smoke tests.
- Diffusion Policy / ViTacDP training and inference require task-compatible
  datasets and checkpoints. They are not included in this environment release.

## 8. Maintainer upload

Do not paste a Hugging Face token into chat, scripts, or shell arguments.
Authenticate locally, then run the idempotent uploader:

```bash
python -m venv .venv_hf
.venv_hf/bin/python -m pip install -U huggingface_hub
.venv_hf/bin/hf auth login
.venv_hf/bin/hf auth whoami
bash bash_command/upload_hf_assets.sh Yanlj/ViTacLab-assets
```

The script creates a public Dataset repository if needed, uploads the curated
approximately 1.4 GB payload, and publishes a SHA256 manifest. Re-running the
same command resumes/skips content already present on the Hub.
