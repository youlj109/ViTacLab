# ViTacBench handoff

ViTacBench task/recording changes from `d5c5e2d` integrated onto main `2ca8b79`.
The current Xense force-depth reconstruction, calibration, measured-marker
optimization, website, and six-load results are preserved. This is not an
update to the Xense physical or optical parameters.

## Run (repo root, configured Isaac Lab environment)

The launchers accept `PYTHON_SIM=/path/to/python.sh` (or an activated Isaac Lab
Python), `ISAACLAB_SH=/path/to/isaaclab.sh`, and
`PYTHON_OFFLINE=/path/to/python`. The offline interpreter needs NumPy/OpenCV
and the dependencies of the requested exporter. No specific conda environment
name is required. The Isaac Lab Docker Python can be selected explicitly as
`PYTHON_SIM=/isaac-sim/python.sh` inside the container.

| Paper / suite task | Command |
| --- | --- |
| Pickup | `bash bash_command/demo_data_collection_replay.sh` |
| Pour | `bash bash_command/demo_pour_collection_replay.sh` |
| In-hand | `bash bash_command/demo_inhand_collection_replay.sh` |
| Handover | `bash bash_command/demo_handover_collection_replay.sh` |
| Forge DexHand (peg/gear/nut) | `bash bash_command/demo_forge_all_collection_replay.sh` |
| Forge Franka RL | `bash bash_command/demo_forge_franka_rl_all_collection_replay.sh` |
| Unscrew / bottle | `bash bash_command/demo_bottle_collection_replay.sh` |
| BiStab | `bash bash_command/demo_bistab_collection_replay.sh` |
| Blind Classification | `bash bash_command/demo_blind_classification_all_shapes.sh` |
| Blind Grasp | `bash bash_command/demo_blind_grasp_all_phases.sh` |
| Blind Targeted Grasp (Retrieval) | `bash bash_command/demo_blind_retrieval_collection_replay.sh` |
| Blind Drop + Blind Targeted Drop | `bash bash_command/demo_bi_blind_bin_drop_all_phases.sh` |

`MODE=collect|visualize|full`, `DEVICE=cuda:0`, `SAVE_OUTCOME=completed|success`.
NPZ + panel mp4 land under fresh run subdirectories of `logs/demo_<task>/`.
Existing recordings and videos are never pre-deleted. Each command prints its
actual output paths. For replay, set `MODE=visualize` and point `RECORD_DIR`
to the actual collection run directory, not its parent. Checkpoint-based demos
still require compatible local checkpoints; these are not bundled here.

`SAVE_OUTCOME=completed` does not mean benchmark success. Forge's optional
geometric threshold overrides are not calibrated benchmark settings; disabled
or relaxed success criteria must be reported as demonstration-only output.

## Integration notes

- The generic RL player gains recording arguments and bounded rollout support;
  cameras are enabled for explicit camera/recording requests instead of always.
- Blind task observation dimensions are resolved before Gym spaces are created.
  Disabling object-state observations removes privileged object/goal geometry;
  Retrieval retains its explicit target-class command.
- Pickup-family tactile buffers are initialized after sensor construction and
  retain a zero-filled observation layout when cameras are disabled, rather
  than silently dropping the declared tactile dimensions.
- The mainline strict target-contact attribution is retained. Retrieval's
  sensor builder targets the cube/sphere/cone paths. The donor's permissive
  `_tactile_policy` overrides did not connect to the actual `_tactile_params`
  construction path and are not used here.
- Imported scene changes include Blind Grasp bin size/friction/object mass,
  Handover's yellow sphere, and the bottle contact link/initial height. Existing
  checkpoints and old benchmark success rates must not be assumed compatible.
- `policy/prepare_dataset.py --profile tactile_only` prepares data; the included
  ViTacDP task YAML files remain templates. The existing training dataset loader
  expects camera/pose fields, so these templates are **not train-ready** without
  a separate loader/model adaptation.
- Existing repository audit checks are kept. The donor's new source-string
  matching checks were not imported; focused data/tensor regression tests cover
  the integration behavior instead.

## Blind Grasp blocker

Actor contract is in place: tactile + proprio only (`include_object_state_in_policy_obs=False`, `policy_record_profile=tactile_only`). Success is live multi-finger gel contact then cube `z > 0.35` (external metric, not in obs/reward).

The donor reports that keyframe replay and GUI teleop do not extract. Its
privileged expert (`scripts/data_collection/full_trajectory/collect_blind_grasp_privileged.py`)
can reach pose_008 gels but **has not produced a successful lift** with live
lift-tactile and `z>0.35`. Imported configuration requests cube mass `0.05 kg`
and sim friction `(3.0, 2.5)`. Our local startup warns that mass modification
could not be applied to `/object`; the requested mass is **not a verified
effective mass** of the loaded asset. No new successful lift is claimed here.

## IL contract

Use `scripts/debug/validate_play_record_npz.py --profile tactile_only` for
Blind Grasp's policy records. Classification/Retrieval demo archives still use
the canonical recording schema (which may contain diagnostic cameras/poses);
do not confuse that archive with actor observations. Use the tactile-only
converter profile when preparing restricted policy inputs.
`policy/prepare_dataset.py` has `canonical` / `tactile_only`. Placeholder ViTacDP task yamls: `policy/ViTacDP/config/task/single_arm_tactile_only_{force,rgb}.yaml`.

## Integration verification (2026-09-28)

- `python3 scripts/audit_project.py --repro-only`: passes asset/task/tactile checks.
  This is not a claim that the broader repository style audit is clean.
- `python3 scripts/debug/test_demo_launchers.py`: 24 stubbed full/visualize
  scenarios (12 launchers), preserving existing NPZ, MP4, summaries, and reports.
- `python3 scripts/debug/test_blind_retrieval_helpers.py`: four tests for
  task-local helpers, shared imports, and target-specific geometry without
  mutation of `env.object`. The Retrieval runner's `--help` also passes inside
  the Isaac container. This runner is explicitly a privileged geometry-assisted
  keyframe demo, not an autonomous blind-policy evaluation.
- `scripts/debug/test_advisor_dataset_offline.py`: 13 synthetic data tests;
  cross-checked in environments with Zarr and PyTorch3D respectively, including
  real RGB/force Zarr round trips and canonical pose conversion. Each environment
  skips only its unavailable optional dependency; together all cases are covered.
- Isaac Lab 2.3.2 Python: `scripts/debug/test_blind_observations_offline.py`
  passes six tensor/interface tests, including 12 observation configurations,
  target paths, no object-geometry leakage when disabled, and grasp success
  requiring both current tactile support and lift. The constructor lifecycle
  test additionally covers sensor-present/absent and full/summary observations.
- The existing Xense measured-marker CPU/CUDA regression and six-load published
  metric recomputation both pass. Sensor/calibration source and published
  Xense assets have no integration diff against main.
- `scripts/rsl_rl/full_rl/play.py --help` works in the Isaac container. General
  multi-environment recording semantics and checkpoint compatibility across
  changed scenes are not certified by this integration.
- Actual Isaac Lab 2.3.2 Docker / RTX 5090 smoke: Blind Grasp creates and steps
  for two zero-action control steps; policy shape is `(1, 7620)`, and finite
  joint, normal-force, shear-force, and tactile-RGB record fields are present.
  Log: `logs/advisor_integration_blindgrasp_smoke_final_20260928.log` (local),
  clean exit code 0. A deliberately invalid task exits with code 1 as expected.
  The requested-mass warning described above remains; this is not a mass or
  manipulation-success validation. The smoke helper reports the test exit code
  to Kit before shutdown so exceptions are not silently masked as success.

For a checkpoint-free two-step runtime smoke in the configured Isaac container:

```bash
/isaac-sim/python.sh scripts/debug/smoke_advisor_integration.py --headless --steps 2
```

This checks creation/stepping, observation dimensions, and finite tactile record
fields. It does not demonstrate a successful manipulation episode or estimate
a task success rate.
