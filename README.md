# ViTacLab

Benchmarking and pretraining **visuo-tactile** representations for robotic manipulation, built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Overview

ViTacLab is an Isaac Lab **extension** that lives **outside** the upstream `IsaacLab` repo. It adds dexterous manipulation environments (e.g. Shadow Hand, UR10e + hand), tactile and vision sensors, and training / teleoperation tooling.

**Highlights**

- **Isolation** — Develop and version-control project code here without forking Isaac Lab.
- **Flexibility** — Run as a Python package (`pip install -e`) and as an Omniverse extension.

**Keywords:** Isaac Lab, visuo-tactile, dexterous manipulation, extension

---

## ViTacSim advisor release (Xense 400×700 + M2 nut)

> **Stable snapshot:** branch [`release/advisor-xense-v0.1`](https://github.com/youlj109/ViTacLab/tree/release/advisor-xense-v0.1) · tag [`v0.1-advisor-xense`](https://github.com/youlj109/ViTacLab/releases/tag/v0.1-advisor-xense)

Code-only release: **lab tactile videos / polycalib / fitted params are not in Git** (see below).

### Prerequisites

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) + Isaac Sim (tested with conda env `env_isaaclab_510test`)
- `python -m pip install -e source/ViTacLab`
- Lab data placed locally under `data/calibration/tactile/` (gitignored): advisor mp4, `correct.zip`, ball-calib video
- Render assets installed into `source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data/` — see [`xense_lab_data/README.md`](source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/xense_lab_data/README.md)

Optional: external [Taxim](https://github.com/TacTip/Taxim) repo for polycalib fitting (`TAXIM_REPO=...` in install scripts).

### Reproduce (Task 2 → Task 3)

```bash
# 1) Import real frames + bg/marker_rest into xense_lab_data/
python3 scripts/calibration/import_advisor_tactile_videos.py --install-bg

# 2) Sim NF sweep (TacSL + ViTacSim, advisor M2 nut G010–G210)
bash bash_command/run_vitacsim_calibration_sweep_dual.sh

# 3) Joint RGB + marker fit → data/calibration/tactile/fitted_params.json
bash bash_command/run_task2_advisor_calibration.sh

# 4) Re-sweep with fitted params, then validation panels
SKIP_EXISTING=0 FITTED_PARAMS=data/calibration/tactile/fitted_params.json \
  bash bash_command/run_vitacsim_calibration_sweep_dual.sh
bash bash_command/run_task3_advisor_validation.sh
```

**Outputs (local):** `logs/vitacsim_validation/task3/panel_nf_three_way.png`, `TASK3_VALIDATION_REPORT.md`

### Documentation

| Topic | File |
|-------|------|
| Calibration protocol (Task 2) | [`docs/VITACSIM_CALIBRATION.md`](docs/VITACSIM_CALIBRATION.md) |
| Marker simulation (FOTS) | [`docs/VITACSIM_MARKER_SIMULATION.md`](docs/VITACSIM_MARKER_SIMULATION.md) |
| ViTacSim V1/V2 principles | [`docs/VITACSIM_PRINCIPLES.md`](docs/VITACSIM_PRINCIPLES.md) |
| PhysX validation | [`docs/VITACSIM_PHYSX_VALIDATION.md`](docs/VITACSIM_PHYSX_VALIDATION.md) |

---

## Repository map

| Path | Description |
|------|-------------|
| `source/ViTacLab/` | Main extension: tasks, assets, agents |
| `source/video_teleop/` | Camera + MediaPipe → ZMQ; no Isaac dependency on the sender side |
| `scripts/rsl_rl/` | RSL-RL training / play (full-joint RL and IK-assisted hand RL) |
| `scripts/teleoperation/video_teleop/` | Video teleop launchers + calibration configs |
| `docs/` | Extra notes (e.g. cameras vs headless for RL) |

---

## Documentation index

| Topic | Location |
|-------|----------|
| **Quick install (中文)** | [`docs/QUICK_INSTALL.md`](docs/QUICK_INSTALL.md) |
| RSL-RL training (commands, Hydra, IK-RL) | [`scripts/rsl_rl/README.md`](scripts/rsl_rl/README.md), [`scripts/rsl_rl/QUICKSTART.md`](scripts/rsl_rl/QUICKSTART.md) |
| IK-RL team guide (env / reward / recording) | [`docs/ik_rl_modification_guide.md`](docs/ik_rl_modification_guide.md) |
| IK-RL YAML configs | [`scripts/rsl_rl/ik_rl/configs/README.md`](scripts/rsl_rl/ik_rl/configs/README.md) |
| Video teleop (calibration, sender/receiver, UR10e task) | [`scripts/teleoperation/video_teleop/QUICK_START.md`](scripts/teleoperation/video_teleop/QUICK_START.md), [`scripts/teleoperation/video_teleop/README.md`](scripts/teleoperation/video_teleop/README.md) |
| `video_teleop` package internals | [`source/video_teleop/docs/README.md`](source/video_teleop/docs/README.md), [`source/video_teleop/docs/ENGINEERING_SUMMARY.md`](source/video_teleop/docs/ENGINEERING_SUMMARY.md) |
| Headless training vs `enable_cameras` | [`docs/enable_cameras_headless_rl.md`](docs/enable_cameras_headless_rl.md) |
| **ViTacSim Xense calibration & validation** | [`docs/VITACSIM_CALIBRATION.md`](docs/VITACSIM_CALIBRATION.md) (see **Advisor release** above) |

---

## Installation

1. Install Isaac Lab using the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda or `uv` installs simplify calling Python from the terminal.

2. Clone this repository **separately** from the Isaac Lab installation (not inside the upstream `IsaacLab` tree).

3. With a Python interpreter that has Isaac Lab available, install the extension in editable mode:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in your venv/conda
    python -m pip install -e source/ViTacLab
    ```

4. **Verify** the installation:

    - List environments registered by this extension (all IDs whose entry point lives under `ViTacLab.tasks`):

        ```bash
        python scripts/list_envs.py
        ```

        Optional filter:

        ```bash
        python scripts/list_envs.py --keyword Pour
        ```

    - Run a task with the standard Isaac Lab training script (example):

        ```bash
        python scripts/rsl_rl/full_rl/train.py --task=<TASK_NAME>
        ```

        See [`scripts/rsl_rl/README.md`](scripts/rsl_rl/README.md) for IK-RL and other entry points.

    - **Dummy agents** (zero or random actions) are useful to sanity-check environments:

        ```bash
        python scripts/zero_agent.py --task=<TASK_NAME>
        python scripts/random_agent.py --task=<TASK_NAME>
        ```

---

## IDE setup (optional)

- In VS Code: `Ctrl+Shift+P` → **Tasks: Run Task** → `setup_python_env`. You will be prompted for the absolute path to your Isaac Sim installation.

If it succeeds, `.vscode/.python.env` contains Python paths for Omniverse/Isaac extensions and improves IntelliSense.

---

## Omniverse extension (optional)

Example UI code: `source/ViTacLab/ViTacLab/ui_extension_example.py`.

1. **Extension Manager** → `Window` → `Extensions` → hamburger menu → **Settings**.
2. Under **Extension Search Paths**, add the absolute path to this **repository’s `source` directory**, and (if needed) Isaac Lab’s `IsaacLab/source`.
3. Refresh, then enable the extension under **Third Party**.

---

## Code formatting

We use **pre-commit** for formatting:

```bash
pip install pre-commit
pre-commit run --all-files
```

---

## Troubleshooting

### Pylance missing extension indexing

Add your extension path under `python.analysis.extraPaths` in `.vscode/settings.json`, for example:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-this-repo>/source/ViTacLab"
    ]
}
```

### Pylance running out of memory

If indexing too many Omniverse packages, remove unused paths from `extraPaths`. Candidates to exclude often include:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"     // animation
"<path-to-isaac-sim>/extscache/omni.kit.*"      // kit UI
"<path-to-isaac-sim>/extscache/omni.graph.*"    // graph UI
"<path-to-isaac-sim>/extscache/omni.services.*" // services
```
