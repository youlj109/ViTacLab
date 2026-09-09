# ViTacLab Executable Script Usage Reference

This file lists every repository-owned executable Python entry and every explicit `argparse` option declared by that entry. The option text is generated from the current source; the authoritative runtime view is always `python <script> --help` in the remote Isaac Lab environment. Isaac `AppLauncher` options such as `--headless`, `--device`, and `--enable_cameras`, and shared RSL-RL options injected by `scripts/common/rl/cli_args.py`, appear at runtime even when they are not declared directly in the entry file.

Static generation result: **48 executable Python entries in the repository**. All explicit `add_argument` calls contain `help=` (enforced by `scripts/audit_project.py`).

## `.vscode/tools/setup_vscode.py`

This script sets up the vs-code settings for the Isaac Lab project. This script merges the python.analysis.extraPaths from the "{ISAACSIM_DIR}/.vscode/settings.json" file into the ".vscode/settings.json" file. This is necessary because Isaac Sim 2022.2.1 onwards does not add the necessary python packages to the python path when the "setup_python_env.sh" is run as part of the vs-code launch configuration.

Run help: `python .vscode/tools/setup_vscode.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--isaac_path` | str | - | - | The absolute path to the Isaac Sim installation. |

## `policy/prepare_dataset.py`

Convert canonical ViTacLab episode NPZ files into policy-training Zarr data. This is the single maintained dataset conversion entry for both camera-only Diffusion Policy and multimodal ViTacDP. Input episodes are produced by the collectors under ``scripts/data_collection``.

Run help: `python policy/prepare_dataset.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `input_dir` | Path | - | - | Directory containing episode NPZ files. |
| `output_zarr` | Path | - | - | Destination Zarr directory. |
| `--policy` | str | - | ('Diffusion_Policy', 'ViTacDP') | Required. 'Output schema: camera-only Diffusion Policy or multimodal ViTacDP.' |
| `--episode-glob` | str | episode*.npz' | - | 'Glob used to find episodes below input_dir (default: episode*.npz). |
| `--max-episodes` | int | 0 | - | Maximum episodes to convert; 0 uses all. |
| `--stride` | int | 1 | - | Keep every Nth frame before transition alignment. |
| `--camera-keys` | str, nargs='+' | None | - | Ordered NPZ camera keys. Default auto-selects third_person_camera then twist_camera. |
| `--joint-key` | str | joint_pos' | - | 'NPZ key used for policy state. |
| `--action-source` | str | next_joint' | ('next_joint', 'recorded_action') | 'Use next-frame joint targets or the NPZ action field as training actions. |
| `--action-key` | str | action' | - | 'NPZ key used with --action-source recorded_action. |
| `--tactile-type` | str | rgb' | ('rgb', 'force') | 'ViTacDP tactile representation; ignored for Diffusion_Policy. |
| `--overwrite` | flag (store_true) | False | - | Replace output_zarr if it already exists. |
| `--validate-only` | flag (store_true) | False | - | Validate episode keys/shapes and print the resolved schema without writing Zarr. |

## `policy/train_policy.py`

Unified training entry for Diffusion Policy and ViTacDP. Hydra overrides are passed after ``--`` or as remaining ``key=value`` tokens. Examples are documented in ``policy/README.md``.

Run help: `python policy/train_policy.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--policy` | str | - | ('Diffusion_Policy', 'ViTacDP') | Required. 'Policy package to train.' |
| `--config-name` | str | 'robot_dp' | - | Hydra config name below the selected policy package's config directory. |
| `overrides` | str, nargs=argparse.REMAINDER | - | - | Hydra overrides, for example task=head_only task.agent_dim=30 task.dataset.zarr_path=... |

## `scripts/audit_project.py`

Run dependency-free static acceptance checks for the ViTacLab repository. This audit intentionally does not import Isaac Lab, Isaac Sim, CUDA, camera, or tactile packages. It verifies the repository invariants that can be checked on any machine: Python syntax, canonical Gym registrations, entry-point targets, the 31-task TacSL/GelSight sensor-count and record implementation manifest, script documentation, argparse help, collector placement, stale/versioned filenames, duplicate source files, and obsolete policy paths. Examples: python scripts/audit_project.py python scripts/audit_project.py --root /workspace/ViTacLab --verbose Exit status is zero only when every required check passes. Warnings identify items that require remote runtime validation but do not make the static audit fail.

Run help: `python scripts/audit_project.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--root` | Path | Path(__file__).resolve().parents[1] | - | ViTacLab repository root (default: inferred from this script). |
| `--verbose` | flag (store_true) | False | - | Print all informational audit counters. |

## `scripts/data_collection/full_trajectory/play_action.py`

Replay a normalized-action trajectory for any registered ViTacLab Gym action
space and optionally write canonical NPZ observations.

Run help: `python scripts/data_collection/full_trajectory/play_action.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--trajectory-file` | str | empty | - | JSON produced by `record_action.py`; required when executing. |
| `--task` | str | empty | - | Optional Gym task override; otherwise use JSON task. |
| `--num-envs` | int | 1 | - | Parallel env count; currently required to remain 1. |
| `--num-episodes` | int | 1 | - | Replay attempts. |
| `--max-steps` | int | 0 | - | Maximum steps per attempt; 0 uses all frame holds. |
| `--record-data` | flag | False | - | Write compressed episode NPZ. |
| `--record-path` | str | None | - | NPZ output directory. |
| `--record-env-index` | int | 0 | - | Environment stream to record. |
| `--record-step-interval` | int | 1 | - | Save one row every N steps. |
| `--save-outcome` | str | `success` | `success`, `completed`, `all` | Outcome filter. |
| `--seed` | int | None | - | Environment seed. |

## `scripts/data_collection/full_trajectory/record_action.py`

Record normalized Gym-action keyframes with generic action sliders. This is
the full-trajectory entry for Franka, standalone ShadowHand and pretraining
tasks that do not share the UR10e marker/IK schema.

Run help: `python scripts/data_collection/full_trajectory/record_action.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | None | - | Registered task; required when executing. |
| `--num-envs` | int | 1 | - | Currently required to remain 1. |
| `--fps` | float | 30 | - | Step frequency. |
| `--max-steps` | int | 0 | - | Stop after N steps. |
| `--record-dir` | str | `scripts/data_collection/full_trajectory/records_action` | - | JSON output directory. |
| `--record-name` | str | `action_trajectory` | - | Filename prefix. |
| `--hold-steps` | int | 30 | - | Default replay duration for each snapshot. |
| `--initial-action` | float | 0 | - | Initial normalized slider value. |
| `--gui`, `--no-gui` | mutually exclusive | GUI | - | Interactive sliders or headless constant-action smoke mode. |

## `scripts/data_collection/full_trajectory/play_dual_phase.py`

Play a canonical dual-UR10e full trajectory and record NPZ data. Use ``--phase 1`` through ``--phase 4`` for the built-in BinDrop branches, or pass a task-matched ``--trajectory-json`` from ``record_dual.py``. Canonical environment success signals are preferred over the BinDrop fallback.

Run help: `python scripts/data_collection/full_trajectory/play_dual_phase.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--phase` | int | 1 | (1, 2, 3, 4) | Dual-arm business phase selecting the keyframe branch and replay-cone setup (default: 1). |
| `--trajectory-json` | str | ' | - | 'Optional trajectory JSON. When provided, its frames override the built-in phase trajectory. |
| `--keyframe-json` | str | str(POSE_KEYFRAME_DUAL_JSON) | - | Path to the canonical keyframe JSON used to build the scripted trajectory. |
| `--task` | str | `bi_blind_bin_drop` | - | Dual-UR10e alias or registered dual-arm Gym ID. Non-BinDrop tasks require a matching `--trajectory-json`. |
| `--env` | str | ' | - | 'Optional environment entry point in module:Class form; must be supplied together with --cfg. |
| `--cfg` | str | ' | - | 'Optional environment-config entry point in module:Class form; must be supplied together with --env. |
| `--num-envs` | int | 1 | - | Number of parallel simulation environments. |
| `--fps` | float | 30.0 | - | Target control and recording loop frequency in frames per second. |
| `--num-episodes` | int | 1 | - | Exclusive upper episode index; with --start-episode 0 this is the number of attempts. |
| `--start-episode` | int | 0 | - | First episode index (0-based). Use with same --record-path to continue numbering. |
| `--resume` | flag (store_true) | False | - | Start at max(existing episode_*_success.npz)+1 under --record-path (>= --start-episode). |
| `--object-init-index` | int | -1 | - | Object initialization preset index; -1 uses the trajectory/keyframe default. |
| `--arm-pos-tol` | float | 0.05 | - | Maximum arm-joint absolute error in radians for a keyframe to count as reached. |
| `--hand-pos-tol` | float | 0.08 | - | Maximum hand-joint absolute error in radians for a keyframe to count as reached. |
| `--stable-steps` | int | 12 | - | Consecutive in-tolerance steps required before advancing to the next keyframe. |
| `--max-steps-per-frame` | int | 360 | - | Maximum simulation steps allowed for each trajectory keyframe. |
| `--post-arm-reached-steps` | int | 30 | - | Extra settling steps after the arm reaches a keyframe before advancing. |
| `--max-arm-joint-step` | float | 0.0 | - | Optional per-step arm-joint delta limit in radians; 0 disables the limit. |
| `--action-smoothing` | float | 0.75 | - | Exponential smoothing factor for commanded actions in [0, 1). |
| `--record-step-interval` | int | 4 | - | Record one observation every N simulation steps. |
| `--record-path` | str | None | - | Output directory for episode NPZ files and metadata. |
| `--record-env-index` | int | 0 | - | Parallel environment index whose observation stream is saved. |
| `--save-outcome` | str | `success` | `success`, `completed`, `all` | Select whether only canonical successes, structurally completed trajectories, or all attempts are written. |
| `--tactile-reset-settle-steps` | int | 16 | - | Zero-action steps after reset so tactile and camera sensors settle. |
| `--hand-noise` | float | 0.02 | - | Uniform random hand-joint perturbation magnitude in radians. |
| `--trash-can-reference-x` | float | 0.7 | - | World-frame trash-can x reference used by the bin-drop scripted phase. |
| `--left-touch-x-joint-gain` | float | 0.8 | - | Gain mapping left tactile x displacement to the scripted joint correction. |
| `--left-touch-max-elbow-delta` | float | 0.18 | - | Maximum absolute left-elbow correction applied by tactile alignment. |
| `--recompute-arm-ik` | flag (store_true) | False | - | Recompute arm IK online instead of replaying arm keyframe joint values. |
| `--camera-brightness-factor` | float | 0.25 | - | Multiplicative camera brightness augmentation factor. |
| `--camera-noise-std` | float | 28.0 | - | Standard deviation of additive camera pixel noise in 0-255 units. |
| `--camera-blur-kernel` | int | 7 | - | Odd Gaussian-blur kernel size; values <=1 disable blur. |

## `scripts/data_collection/full_trajectory/play_single.py`

Replay a canonical single-UR10e pure-control trajectory and record observations. The task is resolved through the shared single-task registry; canonical environment success signals are preferred over the legacy Pickup lift/goal-Z fallback.

Run help: `python scripts/data_collection/full_trajectory/play_single.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | `pickup` | - | Single-UR10e alias or registered Gym ID. |
| `--env` | str | ' | - | 'Optional environment entry point in module:Class form; must be supplied together with --cfg. |
| `--cfg` | str | ' | - | 'Optional environment-config entry point in module:Class form; must be supplied together with --env. |
| `--num_envs` | int | 1 | - | Number of parallel simulation environments. |
| `--seed` | int | None | - | Random seed for deterministic env resets/replay. |
| `--fps` | float | 30.0 | - | Target control and recording loop frequency in frames per second. |
| `--max-steps` | int | 0 | - | 0 = until app closed. |
| `--max-steps-per-trajectory` | int | 0 | - | If >0, max env.step count per episode while replaying one JSON; exceed => fail episode and reset. |
| `--max-episodes` | int | 0 | - | 0 = unlimited resets. |
| `--trajectory-file` | str | empty | - | JSON generated by GUI recorder; checked as required only when replay executes so `--help` remains usable with Isaac Lab AppLauncher. |
| `--arm-base-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Robot arm-base world position as X Y Z in meters. |
| `--arm-base-euler` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Robot arm-base XYZ Euler orientation in radians. |
| `--arm-pos-tol` | float | 0.05 | - | Arm joint convergence tolerance (rad). |
| `--hand-pos-tol` | float | 0.08 | - | Hand joint convergence tolerance (rad). |
| `--stable-steps` | int | 5 | - | Need N consecutive converged steps to switch frame. |
| `--max-steps-per-frame` | int | 240 | - | Force switch if a frame cannot converge. |
| `--post-arm-reached-steps` | int | 30 | - | After arm reaches target, wait at most N more steps before switching frame. |
| `--goal-z-tol` | float | 0.012 | - | Success z window: abs(object_z-goal_z) <= tol. |
| `--grasp-lift-min-dz` | float | 0.02 | - | Minimum lift above init z to consider object as grasped/lifted. |
| `--arm-max-step-rad` | float | 0.0 | - | Per sim step, cap each arm joint change toward IK (rad); 0=off. Try 0.012-0.04 to slow post-grasp lift. |
| `--arm-slew-only-when-lifted` | flag (store_true) | False | - | If set, --arm-max-step-rad applies only when object z is already lifted (grasp-lift-min-dz). |
| `--hand-max-step-rad` | float | 0.0 | - | Per sim step, cap each Shadow Hand joint toward IK (rad); 0=off. |
| `--record-data` | flag (store_true) | False | - | Record successful episodes to npz. |
| `--record-step-interval` | int | 3 | - | Within each episode, append one data row every N env.step (>=1). Default 2 = every other step. |
| `--record-path` | str | None | - | Output dir for npz records. |
| `--record-env-index` | int | 0 | - | Env index to record. |
| `--record-max-episodes` | int | 0 | - | Stop after saving N successful episodes. |
| `--save-outcome` | str | `success` | `success`, `completed`, `all` | Select whether only canonical successes, structurally completed trajectories, or all attempts are written. |

## `scripts/data_collection/full_trajectory/play_single_phase.py`

Play one canonical single-arm BlindGrasp business phase and record NPZ data. Use ``--phase 1`` through ``--phase 3``. Phases 1/2 use BlindGrasp keyframes; phase 3 uses the BlindGraspReplay keyframe/environment by default. All phases share this implementation and command-line interface. Canonical environment success signals are preferred; the legacy object-height criterion is used only when the environment exposes no success signal. ``--save-outcome all`` retains unsuccessful attempts for structural data-chain validation while the production default remains success-only.

Run help: `python scripts/data_collection/full_trajectory/play_single_phase.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--phase` | int | 1 | (1, 2, 3) | Single-arm business phase (default: 1). |
| `--keyframe-json` | str | ' | - | 'Optional keyframe JSON override; default is selected by --phase. |
| `--task` | str | ' | - | 'Optional task alias or registered Gym ID; default is selected by --phase. |
| `--num-envs` | int | 1 | - | Number of parallel simulation environments. |
| `--fps` | float | 30.0 | - | Target control and recording loop frequency in frames per second. |
| `--num-episodes` | int | 1 | - | Number of episodes to attempt and save. |
| `--object-init-index` | int | 0 | - | Object initialization preset index; -1 uses the trajectory/keyframe default. |
| `--object-xy-noise` | float | 0.03 | - | Uniform object-reset position noise in the world XY plane, in meters. |
| `--hand-noise` | float | 0.02 | - | Uniform random hand-joint perturbation magnitude in radians. |
| `--success-z-threshold` | float | 0.4 | - | Object world-height threshold in meters used to label successful pickup. |
| `--arm-base-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Robot arm-base world position as X Y Z in meters. |
| `--arm-base-euler` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Robot arm-base XYZ Euler orientation in radians. |
| `--arm-pos-tol` | float | 0.05 | - | Maximum arm-joint absolute error in radians for a keyframe to count as reached. |
| `--hand-pos-tol` | float | 0.08 | - | Maximum hand-joint absolute error in radians for a keyframe to count as reached. |
| `--stable-steps` | int | 15 | - | Consecutive in-tolerance steps required before advancing to the next keyframe. |
| `--max-steps-per-frame` | int | 240 | - | Maximum simulation steps allowed for each trajectory keyframe. |
| `--post-arm-reached-steps` | int | 30 | - | Extra settling steps after the arm reaches a keyframe before advancing. |
| `--action-smoothing` | float | 0.75 | - | Exponential smoothing factor for commanded actions in [0, 1). |
| `--record-step-interval` | int | 1 | - | Record one observation every N simulation steps. |
| `--tactile-reset-settle-steps` | int | 8 | - | Zero-action steps after reset so tactile and camera sensors settle. |
| `--record-path` | str | None | - | Output directory for episode NPZ files and metadata. |
| `--record-env-index` | int | 0 | - | Parallel environment index whose observation stream is saved. |
| `--save-outcome` | str | `success` | `success`, `completed`, `all` | Save canonical successes only, also structurally completed trajectories, or every attempted outcome. |

## `scripts/data_collection/full_trajectory/record_dual.py`

Record dual-arm full trajectory with marker-driven IK + hand sliders.

Run help: `python scripts/data_collection/full_trajectory/record_dual.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | bi_blind_grasp' | - | 'Supported task alias used to select the canonical environment/config pair. |
| `--env` | str | ' | - | 'module:EnvClass |
| `--cfg` | str | ' | - | 'module:CfgClass |
| `--num-envs, --num_envs` | int | 1 | - | Number of parallel simulation environments. |
| `--fps` | float | 30.0 | - | Target control and recording loop frequency in frames per second. |
| `--object-init-index` | int | 0 | - | Object initialization preset index; -1 uses the trajectory/keyframe default. |
| `--record-dir` | str | ./scripts/data_collection/full_trajectory/records_dual' | - | 'Directory in which the recorded trajectory JSON is written. |
| `--record-name` | str | traj_dual' | - | 'Output trajectory filename; omit to generate a timestamped name. |
| `--hand-joints` | str | sim' | ['zeros', 'sim'] | 'Initial hand source: zeros or current simulator joint values. |
| `--hand-gui` | flag (store_true) | False | - | Show Shadow Hand joint sliders (default). |
| `--no-hand-gui` | flag (store_false) | True | - | Disable hand sliders. |
| `--hold-initial-pose` | flag (store_true) | False | - | Keep both arms at reset pose until either marker is moved or recording starts. |
| `--print-every` | int | 0 | - | Print arm/hand command info every N steps (0=disable). |
| `--print-on-change` | flag (store_true) | False | - | Print when commands change. |
| `--print-hand-rad` | flag (store_true) | False | - | Also print hand target joints in rad. |
| `--marker-right-pos` | float, nargs=3 | (0.3, 0.0, 0.58) | - | Initial right-arm IK marker world position X Y Z in meters. |
| `--marker-right-euler` | float, nargs=3 | (0.0, 1.57, 0.0) | - | Initial right-arm IK marker XYZ Euler orientation in radians. |
| `--marker-left-pos` | float, nargs=3 | (1, 0.0, 0.58) | - | Initial left-arm IK marker world position X Y Z in meters. |
| `--marker-left-euler` | float, nargs=3 | (0.0, 1.57, 0.0) | - | Initial left-arm IK marker XYZ Euler orientation in radians. |
| `--max-steps` | int | 0 | - | Maximum control-loop steps; 0 runs until quit or environment termination. |
| `--show_rgb` | flag (store_true) | False | - | Reserved for compatibility with single-arm CLI. |
| `--show_ff` | flag (store_true) | False | - | Reserved for compatibility with single-arm CLI. |
| `--manual-reset-only` | flag (store_true) | False | - | Disable auto-reset in env.step; only reset from UI Reset button. |
| `--auto-reset` | flag (store_false) | True | - | Allow the environment to auto-reset when episode ends. |

## `scripts/data_collection/full_trajectory/record_single.py`

Tune UR10e arm joint targets from a visual marker pose + IK (no random arm actions). Loads the same task presets as ``scripts/data_collection/manual/record_observations.py``. Spawns a small ``VisualCuboid`` under ``/World/Debug/ArmIkTarget``. **Move/orient this prim in the viewport** (e.g. with the move/rotate gizmo); each step reads its **world** pose, runs the same pipeline as video teleop via ``VideoTeleopControl.compute``, and steps the env. Use this to find a good arm pose, then copy the printed joint dict into your cfg. A docked **Shadow Hand Joints** window (``omni.ui`` sliders) drives the 24 ShadowHand DoFs in radians; limits match the articulation. Use ``--no-hand-gui`` to fall back to ``--hand-joints`` (sim/zeros) only. **Printing:** arm block is still **joint_pos (rad)** for ``ArticulationCfg.init_state``. Hand block defaults to **normalized actions in ``[-1, 1]``** (same as ``env.step``), plus a length-24 list in ``shadowhand_joint_names()`` order for direct hand control. Use ``--print-hand-rad`` to also print hand ``joint_pos`` in radians. Examples (Isaac Sim python): ./python.sh scripts/data_collection/manual/record_arm_pose.py \ --task pickup --num_envs 1 --enable_cameras # Custom initial marker pose (world frame, meters / rad euler xyz) ./python.sh scripts/data_collection/manual/record_arm_pose.py \ --task pour --marker-pos 0.75 0.0 0.35 --marker-euler 0.0 0.78 0.0 # Five-finger TacSL (requires ``--enable_cameras``; for ``--task inhand`` uses tactile cfg if not ``--cfg``): ./python.sh scripts/data_collection/manual/record_arm_pose.py \ --task inhand --num_envs 1 --enable_cameras --show_rgb --show_ff # Forge peg (or ``--task Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0``): ./python.sh scripts/data_collection/full_trajectory/record_single.py \ --task forge_peg --num_envs 1 --enable_cameras # Any registered single-arm task with ``env_cfg_entry_point``: ./python.sh scripts/data_collection/full_trajectory/record_single.py --task Isaac-UR10eShadowHand-BlindGrasp-Direct-v0 --num_envs 1 --enable_cameras --show_rgb --show_ff

Run help: `python scripts/data_collection/full_trajectory/record_single.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | pickup' | - | 'Preset (pickup, pour, inhand, forge_peg, forge_gear, forge_nut), a forge Gym id alias, or any registered Gymnasium id with env_cfg_entry_point. |
| `--env` | str | ' | - | 'Env entry module:Class (overrides --task). |
| `--cfg` | str | ' | - | 'Cfg entry module:Class (overrides --task). |
| `--num_envs` | int | 1 | - | Number of envs (default: 1). |
| `--fps` | float | 30.0 | - | Simulation loop target FPS. |
| `--marker-pos` | float, nargs=3 | (0.3, 0.0, 1.0) | - | Initial marker position in world frame (m). |
| `--marker-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Initial marker orientation euler xyz in world frame (rad). |
| `--arm-base-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | VideoTeleopControl T_world_arm_base translation. |
| `--arm-base-euler` | float, nargs=3 | (0.0, 0.0, 0.0) | - | VideoTeleopControl T_world_arm_base rotation (rad). |
| `--object-init-index` | int | 0 | - | BlindGrasp object init index: 0, 1, or 2. |
| `--hold-initial-pose` | flag (store_true) | False | - | Keep the arm at its reset pose until the marker is moved or recording starts. |
| `--hand-joints` | str | sim' | ['zeros', 'sim'] | 'Initial hand vector when GUI is off, or initial slider values when --hand-gui: zeros(24) or sim hand. |
| `--hand-gui` | flag (store_true) | False | - | Show Shadow Hand joint sliders (default). |
| `--no-hand-gui` | flag (store_false) | True | - | Disable hand sliders; use --hand-joints (sim/zeros) each step as before. |
| `--print-every` | int | 30 | - | Print arm joint dict every N steps (0 = disable periodic print). |
| `--print-on-change` | flag (store_true) | False | - | Also print when IK arm joints or hand normalized actions change (thresholded). |
| `--print-hand-rad` | flag (store_true) | False | - | Also print hand joint_pos (rad) for cfg init_state (default print is normalized [-1,1] actions). |
| `--max-steps` | int | 0 | - | Stop after N steps (0 = run until close). |
| `--show_rgb` | flag (store_true) | False | - | Live matplotlib: GelSight tactile RGB (needs --enable_cameras and TacSL env). |
| `--show_ff` | flag (store_true) | False | - | Live matplotlib: TacSL shear/FF via compute_tactile_shear_image (needs FF on sensors). |
| `--env-index` | int | 0 | - | Which env index to read for tactile display (default: 0). |
| `--viewer-topmost` | flag (store_true) | False | - | Keep tactile viewer on top. Default off to avoid stealing focus. |
| `--manual-reset-only` | flag (store_true) | False | - | Disable auto-reset in env.step; only reset when pressing GUI Reset. |
| `--auto-reset` | flag (store_false) | True | - | Allow the environment to auto-reset when an episode ends. |

## `scripts/data_collection/ik/play_full_ik_single.py`

Play / eval RSL-RL policy with the same **full_ik** stack as ``train_full_ik_single.py`` (phased hand + GPU IK arm). Examples:: ./isaaclab.sh -p scripts/data_collection/ik/play_full_ik_single.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \ --num_envs 64 --checkpoint PATH/model_1000.pt \ --full-ik-config scripts/data_collection/ik/configs/full_ik/full_ik_pickup_fixed_hand.yaml IK / trajectory / ``--full-ik-config`` should match training. Optional **data recording**: ``--record_data`` saves policy observations, **hand** actions, rewards, and dones for ``--record_env_index`` to ``--record_path`` (default: ``./play_records/<task>_<timestamp>/``), one compressed ``.npz`` per completed episode (plus ``*_partial.npz`` if play stops mid-episode). Use ``--record_max_episodes N`` to stop after ``N`` saved episodes. **No trained policy**: with ``--no-checkpoint``, the script does not load ``model_*.pt``; it feeds **zero** policy actions into the wrapper so **arm + scripted/frozen hand** come entirely from full_ik (IK + phases). Use this for rollout / data collection when you did not train.

Run help: `python scripts/data_collection/ik/play_full_ik_single.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | None | - | Registered Gym task (e.g. Isaac-UR10eShadowHand-Pickup-Direct-v0). |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'RL agent config entry point (registry key). |
| `--num_envs` | int | 64 | - | Number of parallel environments. |
| `--seed` | int | None | - | Environment seed. |
| `--max_play_steps` | int | 0 | - | Stop after N env steps (0 = run until window closed or Ctrl+C). |
| `--record_data` | flag (store_true) | False | - | Save policy obs / hand actions / rewards / dones for --record_env_index to --record_path (npz per episode). |
| `--record_path` | str | None | - | Directory for recorded .npz files. Default: play_records/<task>_<timestamp>/ under CWD. |
| `--record_env_index` | int | 0 | - | Which parallel env to record when --record_data (default: 0). |
| `--record_max_episodes` | int | 0 | - | Stop after saving this many completed episodes (0 = no episode limit; still respects --max_play_steps). |
| `--no-checkpoint` | flag (store_true) | False | - | Do not load RSL-RL weights: use zero policy actions (arm/hand from full_ik expander only). For scripted or freeze_hand rollout and data collection without training. |
| `--play_success_interval` | float | 2.0 | - | Print pickup success stats every N seconds (if env exposes get_episode_success_stats). |
| `--show_rgb` | flag (store_true) | False | - | Show tactile RGB (implies --enable_cameras). Same idea as record_observations.py. |
| `--show_ff` | flag (store_true) | False | - | Show tactile force-field RGB (implies --enable_cameras). |
| `--env_index` | int | 0 | - | Which env index to visualize for tactile plots (default: 0). |
| `--fps` | float | 20.0 | - | Target display FPS when --show_rgb / --show_ff (default: 20). |
| `--trajectory` | str | object:150:0,goal:-1:0' | - | 'Comma-separated target:env_steps:use_rotation phases. target resolves to an env asset, <target>_pos/<target>_rot tensors, or legacy goal; -1 steps holds until episode end. |
| `--object-to-palm-offset` | float, nargs=3 | (0.0, 0.0, 0.05) | - | Offset in metres from the current trajectory anchor to the palm origin. |
| `--palm-in-wrist-pos` | float, nargs=3 | (0.0, 0.0, 0.35) | - | Palm origin expressed in the wrist/end-effector frame, in metres. |
| `--palm-in-wrist-euler` | float, nargs=3 | (np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0) | - | XYZ Euler rotation from the wrist frame to the palm frame, in radians. |
| `--palm-orient` | str | pickup_down' | ('fixed', 'pickup_down') | 'Palm orientation mode when the phase does not use anchor rotation. |
| `--palm-normal-local` | float, nargs=3 | (0.0, 1.0, 0.0) | - | Palm-frame axis aligned with --world-down in pickup_down mode. |
| `--palm-yaw-offset` | float | 0.0 | - | Additional world-Z yaw after pickup_down alignment, in radians. |
| `--world-down` | float, nargs=3 | (0.0, 0.0, -1.0) | - | World-space down direction used by pickup_down mode. |
| `--palm-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Fixed palm XYZ Euler orientation in world space, in radians. |
| `--palm-euler-in-anchor` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Palm XYZ Euler rotation relative to a rotating anchor, in radians. |
| `--ee-body` | str | wrist_3_link' | - | 'Robot body/link used as the differential-IK end effector. |
| `--ik-method` | str | dls' | ('pinv', 'svd', 'trans', 'dls') | 'Jacobian inverse method used by differential IK. |
| `--ik-lambda` | float | None | - | Optional DLS damping value; None uses the controller default. |
| `--ik-config` | str | None | - | Optional extra IK YAML merge. Pass 'none' to skip. |
| `--full-ik-config` | str | str(_DEFAULT_FULL_IK_YAML) | - | YAML with phase_schedule + palm/IK (see data_collection/ik/configs/full_ik/full_ik_pour.yaml). |

## `scripts/data_collection/ik/play_ik_policy.py`

Collect data with a **hand-only** RSL-RL policy and a differential-IK UR10e arm. Examples:: # Absolute path to checkpoint ./isaaclab.sh -p scripts/data_collection/ik/play_ik_policy.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \ --num_envs 64 --checkpoint logs/rsl_rl/Isaac-UR10eShadowHand-Pickup-Direct-v0/2026-03-21_19-31-05/model_1000.pt # Resolve under logs/rsl_rl/<task>/ (same as training resume) ./isaaclab.sh -p scripts/data_collection/ik/play_ik_policy.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \ --num_envs 4096 --headless --resume --load_run 2026-03-21_19-31-05 --checkpoint model_61250.pt IK / trajectory flags should match training; defaults mirror ``train_ik_rl_single.py``. Optional **data recording**: ``--record_data`` saves policy observations, **hand** actions, rewards, and dones for ``--record_env_index`` to ``--record_path`` (default: ``./play_records/<task>_<timestamp>/``), one compressed ``.npz`` per completed episode (plus ``*_partial.npz`` if play stops mid-episode). Use ``--record_max_episodes N`` to stop after ``N`` saved episodes.

Run help: `python scripts/data_collection/ik/play_ik_policy.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | None | - | Registered Gym task (e.g. Isaac-UR10eShadowHand-Pickup-Direct-v0). |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'RL agent config entry point (registry key). |
| `--num_envs` | int | 64 | - | Number of parallel environments. |
| `--seed` | int | None | - | Environment seed. |
| `--max_play_steps` | int | 0 | - | Stop after N env steps (0 = run until window closed or Ctrl+C). |
| `--record_data` | flag (store_true) | False | - | Save policy obs / hand actions / rewards / dones for --record_env_index to --record_path (npz per episode). |
| `--record_path` | str | None | - | Directory for recorded .npz files. Default: play_records/<task>_<timestamp>/ under CWD. |
| `--record_env_index` | int | 0 | - | Which parallel env to record when --record_data (default: 0). |
| `--record_max_episodes` | int | 0 | - | Stop after saving this many completed episodes (0 = no episode limit; still respects --max_play_steps). |
| `--play_success_interval` | float | 2.0 | - | Print pickup success stats every N seconds (if env exposes get_episode_success_stats). |
| `--show_rgb` | flag (store_true) | False | - | Show tactile RGB (implies --enable_cameras). Same idea as record_observations.py. |
| `--show_ff` | flag (store_true) | False | - | Show tactile force-field RGB (implies --enable_cameras). |
| `--env_index` | int | 0 | - | Which env index to visualize for tactile plots (default: 0). |
| `--fps` | float | 20.0 | - | Target display FPS when --show_rgb / --show_ff (default: 20). |
| `--trajectory` | str | object:150:0,goal:-1:0' | - | 'Comma-separated target:env_steps:use_rotation phases. target resolves to an env asset, <target>_pos/<target>_rot tensors, or legacy goal; -1 steps holds until episode end. |
| `--object-to-palm-offset` | float, nargs=3 | (0.0, 0.0, 0.05) | - | Offset in metres from the current trajectory anchor to the palm origin. |
| `--palm-in-wrist-pos` | float, nargs=3 | (0.0, 0.0, 0.35) | - | Palm origin expressed in the wrist/end-effector frame, in metres. |
| `--palm-in-wrist-euler` | float, nargs=3 | (np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0) | - | XYZ Euler rotation from the wrist frame to the palm frame, in radians. |
| `--palm-orient` | str | pickup_down' | ('fixed', 'pickup_down') | 'Palm orientation mode when the phase does not use anchor rotation. |
| `--palm-normal-local` | float, nargs=3 | (0.0, 1.0, 0.0) | - | Palm-frame axis aligned with --world-down in pickup_down mode. |
| `--palm-yaw-offset` | float | 0.0 | - | Additional world-Z yaw after pickup_down alignment, in radians. |
| `--world-down` | float, nargs=3 | (0.0, 0.0, -1.0) | - | World-space down direction used by pickup_down mode. |
| `--palm-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Fixed palm XYZ Euler orientation in world space, in radians. |
| `--palm-euler-in-anchor` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Palm XYZ Euler rotation relative to a rotating anchor, in radians. |
| `--ee-body` | str | wrist_3_link' | - | 'Robot body/link used as the differential-IK end effector. |
| `--ik-method` | str | dls' | ('pinv', 'svd', 'trans', 'dls') | 'Jacobian inverse method used by differential IK. |
| `--ik-lambda` | float | None | - | Optional DLS damping value; None uses the controller default. |
| `--hand-freeze-phase-target` | str | None | - | When set along with --hand-freeze-yaml: freeze hand joints during trajectory phases whose target matches this string (e.g. pickup: 'goal'). |
| `--hand-freeze-yaml` | str | None | - | YAML with hand_joint_pos_shadow_order (24 floats) to freeze hand joints to during grasp phase. |
| `--ik-config` | str | None | - | YAML with task + palm/IK/trajectory (see configs/ik_rl_pickup.yaml). Omitted: auto-load if present. 'none' = off. |

## `scripts/data_collection/ik/play_waypoint_ik.py`

Collect Forge/Franka trajectories with scripted Cartesian IK waypoints. The registered RSL-RL configuration is reused to construct the environment, but policy checkpoint weights are not loaded or executed. Zero normalized actions are sent while `set_franka_ik_target` drives the arm. Successful records are saved by default; `--save-outcome all` can retain timeout attempts for chain diagnostics.

Run help: `python scripts/data_collection/ik/play_waypoint_ik.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--num_envs` | int | None | - | Number of environments to simulate. |
| `--task` | str | None | - | Name of the task. |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'Name of the RL agent configuration entry point. |
| `--seed` | int | None | - | Seed used for the environment |
| `--use_pretrained_checkpoint` | flag (store_true) | False | - | Use the pre-trained checkpoint from Nucleus. |
| `--real-time` | flag (store_true) | False | - | Run in real-time, if possible. |
| `--save_data` | flag (store_true) | False | - | If set, save play trajectory data to disk. |
| `--data_path` | str | None | - | Directory to save recorded data. |
| `--num_episodes` | int | 50 | - | Number of successful episodes to record when save_data is enabled (total across all envs). |
| `--max_steps` | int | 0 | - | Maximum steps per attempt. A timeout is discarded by default or saved with `--save-outcome all`; it never increments the successful episode count. |
| `--save-outcome` | str | `success` | `success`, `all` | Save successful trajectories only, or also retain timeout attempts for diagnostics. |
| `--max-attempts` | int | 0 | - | Stop after this many success/timeout attempts; 0 keeps trying until the success target or app exit. |
| `--waypoint_max_steps` | int | 0 | - | Max steps allowed on one waypoint before skipping to the next waypoint (per-env). 0 disables skip-by-steps. |

## `scripts/data_collection/manual/record_arm_pose.py`

Tune UR10e arm joint targets from a visual marker pose + IK (no random arm actions). The canonical implementation includes: - save hand/arm yaml (`--save-yaml`, hotkey `s`) - optional fixed hand (`--fixed-hand-yaml`, hotkey `f` lock/unlock current) - optional stop-on-done (`--no-auto-reset`) to avoid continuing after env auto-reset - optional export full_ik pickup yaml (`--save-full-ik-yaml`) for play_full_ik_single

Run help: `python scripts/data_collection/manual/record_arm_pose.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | pickup' | sorted(_TASK_PRESETS.keys()) | 'Preset task. |
| `--env` | str | ' | - | 'Env entry module:Class (overrides --task). |
| `--cfg` | str | ' | - | 'Cfg entry module:Class (overrides --task). |
| `--num_envs` | int | 1 | - | Number of envs (default: 1). |
| `--fps` | float | 30.0 | - | Simulation loop target FPS. |
| `--marker-pos` | float, nargs=3 | (0.65, 0.12, 0.42) | - | Initial single-arm IK marker world position X Y Z in meters. |
| `--marker-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Initial single-arm IK marker XYZ Euler orientation in radians. |
| `--arm-base-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Robot arm-base world position as X Y Z in meters. |
| `--arm-base-euler` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Robot arm-base XYZ Euler orientation in radians. |
| `--hand-joints` | str | sim' | ['zeros', 'sim'] | 'Initial hand source: zeros or current simulator joint values. |
| `--hand-gui` | flag (store_true) | False | - | Enable the interactive Shadow Hand joint-control window. |
| `--no-hand-gui` | flag (store_false) | True | - | Disable the interactive Shadow Hand joint-control window. |
| `--print-every` | int | 30 | - | Print current arm/hand state every N steps; 0 disables periodic output. |
| `--print-on-change` | flag (store_true) | False | - | Print state whenever a commanded joint target changes. |
| `--print-hand-rad` | flag (store_true) | False | - | Include the 24 Shadow Hand joint values in radians in console output. |
| `--max-steps` | int | 0 | - | Maximum control-loop steps; 0 runs until quit or environment termination. |
| `--show_rgb` | flag (store_true) | False | - | Open the live tactile RGB viewer and enable cameras. |
| `--show_ff` | flag (store_true) | False | - | Open the live tactile force-field viewer and enable cameras. |
| `--env-index` | int | 0 | - | Parallel environment index displayed in the tactile viewer. |
| `--viewer-topmost` | flag (store_true) | False | - | Keep the tactile visualization window above other windows. |
| `--save-yaml` | str | ' | - | 'Save hand/arm YAML on hotkey [s]. |
| `--fixed-hand-yaml` | str | ' | - | 'Lock hand vector from YAML. |
| `--disable-hotkeys` | flag (store_true) | False | - | Disable keyboard shortcuts and require normal process termination. |
| `--no-auto-reset` | flag (store_true) | False | - | Stop loop when done/timeout appears. |
| `--save-full-ik-yaml` | str | str(_DEFAULT_FULL_IK_OUT) | - | Export full_ik config on [s] for pickup. |
| `--full-ik-template` | str | str(_DEFAULT_FULL_IK_TEMPLATE) | - | Template full_ik yaml. |

## `scripts/data_collection/manual/record_grasp_pose.py`

Fix arm from YAML, drive hand closure from a **cyan** visual cube (pour / pickup presets). Loads the same task presets as ``record_arm_pose.py``. Spawns ``/World/Debug/GraspClosureTarget`` (cyan). Move it along ``--closure-axis``; its world coordinate in ``[closure_min, closure_max]`` maps to grasp alpha in ``[0, 1]``. Finger flexion interpolates from open to a reference closed pose (tunable via ``--closed-hand-yaml``). By default this script allows all five fingers to move and record thumb flexion. Set ``--finger-mode four`` to lock thumb joints (THJ*) to 0 (four-finger preset). On clean exit (close Isaac window or stop the script), if ``--save-yaml`` is set, writes a file compatible with ``full_ik`` / GUI configs (``arm_joint_pos``, ``hand_joint_pos_shadow_order``). Examples (Isaac Sim python):: ./python.sh scripts/data_collection/manual/record_grasp_pose.py \ --task pour --enable_cameras \ --arm-yaml scripts/data_collection/manual/config/pour_grasp.yaml \ --save-yaml scripts/data_collection/manual/config/tuned_pour_grasp.yaml

Run help: `python scripts/data_collection/manual/record_grasp_pose.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | pour' | sorted(_TASK_PRESETS.keys()) | 'Preset task. |
| `--env` | str | ' | - | 'Env entry module:Class (overrides --task). |
| `--cfg` | str | ' | - | 'Cfg entry module:Class (overrides --task). |
| `--num_envs` | int | 1 | - | Number of envs (default: 1). |
| `--max-episode-length` | int | 200000 | - | Override episode length in env steps (large default avoids automatic timeout reset while tuning). |
| `--fps` | float | 30.0 | - | Simulation loop target FPS. |
| `--arm-yaml` | str | ' | - | 'Optional arm seed YAML (joint_pos or arm_joint_pos). If omitted, uses current env-reset arm joints. |
| `--closed-hand-yaml` | str | ' | - | 'Optional YAML with hand_joint_pos_shadow_order (24) used as alpha=1 pose; default = built-in template. |
| `--save-yaml` | str | ' | - | 'On exit, write grasp snapshot (arm + hand + closure metadata) to this path. |
| `--closure-axis` | str | x' | ('x', 'y', 'z') | 'World axis of the cyan cube position used for closure alpha. |
| `--closure-min` | float | 0.42 | - | Cube world coordinate on closure-axis → alpha=0 (open). Tune to your layout. |
| `--closure-max` | float | 0.72 | - | Cube world coordinate on closure-axis → alpha=1 (closed template). |
| `--invert-closure` | flag (store_true) | False | - | Use alpha <- 1 - alpha after mapping. |
| `--grasp-cube-pos` | float, nargs=3 | (0.55, 0.12, 0.42) | - | Initial world position of the cyan grasp cube (m). |
| `--arm-control` | str | fixed' | ('fixed', 'marker', 'ik_trajectory', 'cup_relative') | 'Arm mode: fixed=freeze from --arm-yaml, marker=magenta marker IK, ik_trajectory=follow ik_rl-style cup anchor, cup_relative=follow full_ik cup frame offsets. |
| `--arm-marker-pos` | float, nargs=3 | (0.65, 0.12, 0.42) | - | Initial world position of the magenta arm marker (marker mode). |
| `--arm-marker-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Initial world orientation (euler xyz, rad) of the magenta arm marker (marker mode). |
| `--print-arm-every` | int | 60 | - | In marker mode, print solved arm joints every N steps (0=off). |
| `--object-to-palm-offset` | float, nargs=3 | (0.0, 0.05, -0.03) | - | Desired object-to-palm XYZ offset in meters for trajectory IK. |
| `--palm-in-wrist-pos` | float, nargs=3 | (0.0, 0.0, 0.35) | - | Palm XYZ position expressed in the wrist frame, in meters. |
| `--palm-in-wrist-euler` | float, nargs=3 | (1.5707963267948966, -1.5707963267948966, 1.5707963267948966) | - | Palm XYZ Euler orientation expressed in the wrist frame, in radians. |
| `--palm-orient` | str | pickup_down' | ('fixed', 'pickup_down') | 'Palm orientation strategy: fixed Euler angles or pickup-down alignment. |
| `--palm-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Fixed palm world XYZ Euler orientation in radians. |
| `--palm-normal-local` | float, nargs=3 | (0.0, 0.0, 1.0) | - | Local palm normal vector used by pickup-down alignment. |
| `--world-down` | float, nargs=3 | (0.0, 0.0, -1.0) | - | World-frame down vector used by pickup-down alignment. |
| `--palm-yaw-offset` | float | 3.141592653589793 | - | Additional palm yaw offset in radians after alignment. |
| `--sync-full-ik-config` | str | ' | - | 'Optional full_ik YAML path. If set, sync palm/trajectory-aligned arm params from this file before start. |
| `--wrist-pos-in-cup-frame` | float, nargs=3 | (0.03, 0.0, 0.11) | - | cup_relative wrist position offset in cup frame (meters). Use same values as full_ik phase. |
| `--wrist-euler-in-cup-frame` | float, nargs=3 | (0.0, 0.0, 0.0) | - | cup_relative wrist euler offset in cup frame (rad). Use same values as full_ik phase. |
| `--disable-hotkeys` | flag (store_true) | False | - | Disable terminal hotkeys. By default: [f]=toggle finger lock, [t]=toggle thumb (4/5-finger), [r]=manual reset, [q]=quit. |
| `--finger-mode` | str | five' | ('four', 'five') | 'Four: thumb joints (THJ*) forced to 0 (cup-style preset). Five: thumb joints follow closure alpha. |
| `--grasp-profile` | str | uniform' | ('uniform', 'pickup_cube') | 'Hand closure mapping. pickup_cube uses non-uniform finger gains + opposed thumb for cube grasping. |
| `--manual-reset-only` | flag (store_true) | True | - | Disable env auto-reset on done; only reset when pressing [r]. |
| `--allow-auto-reset` | flag (store_true) | False | - | Allow environment auto-reset on done/timeout (overrides --manual-reset-only). |
| `--max-steps` | int | 0 | - | Stop after N steps (0 = run until close). |

## `scripts/data_collection/manual/record_handshape.py`

Canonical interactive wrist, handshape, and tactile-visualization recorder. Provides real-time tactile visualization while controlling handshape (the matplotlib path is aligned with ``record_arm_pose.py``; default ``--viewer matplotlib``): - --show_rgb : GelSight tactile RGB image per finger - --show_ff : tactile normal/shear rendered force-field RGB per finger

Run help: `python scripts/data_collection/manual/record_handshape.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | pickup' | sorted(TASK_PRESETS.keys()) | 'Supported task alias used to select the canonical environment/config pair. |
| `--env` | str | ' | - | 'Optional environment entry point in module:Class form; must be supplied together with --cfg. |
| `--cfg` | str | ' | - | 'Optional environment-config entry point in module:Class form; must be supplied together with --env. |
| `--num_envs` | int | 1 | - | Number of parallel simulation environments. |
| `--fps` | float | 30.0 | - | Target control and recording loop frequency in frames per second. |
| `--max-episode-length` | int | 200000 | - | Environment episode-length override in simulation steps. |
| `--arm-control` | str | marker' | ('marker', 'fixed') | 'Arm target source: interactive marker or fixed initial joints. |
| `--arm-marker-pos` | float, nargs=3 | (0.65, 0.12, 0.42) | - | Initial arm IK marker world position X Y Z in meters. |
| `--arm-marker-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Initial arm IK marker XYZ Euler orientation in radians. |
| `--pickup-ik-yaml` | str | ' | - | 'Optional pickup Full-IK YAML used to initialize palm/approach settings. |
| `--skip-auto-approach` | flag (store_true) | False | - | Skip the scripted initial approach and begin in manual tuning mode. |
| `--grasp-cube-pos` | float, nargs=3 | (0.55, 0.12, 0.42) | - | Initial grasp-control cube world position X Y Z in meters. |
| `--closure-axis` | str | x' | ('x', 'y', 'z') | 'World axis used to map grasp-cube motion to hand closure. |
| `--closure-min` | float | 0.42 | - | Cube coordinate mapped to fully open hand closure. |
| `--closure-max` | float | 0.72 | - | Cube coordinate mapped to fully closed hand closure. |
| `--closed-hand-yaml` | str | ' | - | 'Optional YAML containing the fully closed 24-joint hand template. |
| `--slider-offset-limit` | float | 1.0 | - | Per-joint offset slider half-range in radians; actual range is [-limit, +limit]. |
| `--finger-mode` | str | five' | ('four', 'five') | 'Use four-finger or five-finger closure behavior. |
| `--save-yaml` | str | ' | - | 'Output YAML path written by the save hotkey or on exit. |
| `--disable-hotkeys` | flag (store_true) | False | - | Disable keyboard shortcuts and require normal process termination. |
| `--ignore-first-key` | flag (store_true) | True | - | Ignore one initial stdin key event to avoid accidental immediate hotkey-triggered exit. |
| `--print-hand-every` | int | 0 | - | Print the current 24-joint hand vector every N steps; 0 disables it. |
| `--max-steps` | int | 0 | - | Maximum control-loop steps; 0 runs until quit or environment termination. |
| `--no-fallback-when-app-not-running` | flag (store_false) | True | - | Disable fallback stepping when simulation_app.is_running() is false. |
| `--show_rgb` | flag (store_true) | False | - | Live tactile RGB images. |
| `--show_ff` | flag (store_true) | False | - | Live tactile force-field rendering. |
| `--env-index` | int | 0 | - | Env index to display tactile for. |
| `--viewer-topmost` | flag (store_true) | False | - | Keep tactile viewer on top. Default off to avoid stealing focus (matplotlib). |
| `--viewer` | str | matplotlib' | ('cv2', 'matplotlib') | 'Tactile viewer backend. matplotlib matches record_arm_pose.py; cv2 optional. |
| `--save-tactile-video` | str | ' | - | 'Optional mp4 path to save tactile canvas when no GUI window backend is available. |

## `scripts/data_collection/manual/record_observations.py`

Record and inspect raw observations from canonical single-arm UR10e+ShadowHand tasks. The entry supports zero/random actions, an RSL-RL checkpoint, tactile viewers, and compressed per-step NPZ output while switching tasks through presets or explicit environment/config entry strings. Examples (inside Isaac Sim python): # Pour task ./python.sh scripts/data_collection/manual/record_observations.py --task pour --num_envs 1 --show_rgb --enable_cameras # Pickup task ./python.sh scripts/data_collection/manual/record_observations.py --task pickup --num_envs 1 --show_rgb --show_ff --random_actions --enable_cameras # In-hand cube reorientation (UR10e + ShadowHand, tactile) ./python.sh scripts/data_collection/manual/record_observations.py --task inhand --num_envs 1 --show_rgb --show_ff --random_actions --enable_cameras # Play trained policy (RSL-RL .pt), same enable_cameras as train/play ./python.sh scripts/data_collection/manual/record_observations.py --task inhand --num_envs 4096 --play \ --resume_path logs/rsl_rl/shadow_hand_tactile/2026-03-21_00-01-44/model_1000.pt --enable_cameras # Fully custom (module:Class) ./python.sh scripts/data_collection/manual/record_observations.py \ --env ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv \ --cfg ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg \ --num_envs 1 --show_rgb --enable_cameras

Run help: `python scripts/data_collection/manual/record_observations.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | pour' | sorted(_TASK_PRESETS.keys()) | 'Preset task. |
| `--env` | str | ' | - | 'Env entry: module:Class (overrides --task). |
| `--cfg` | str | ' | - | 'Cfg entry: module:Class (overrides --task). |
| `--play` | flag (store_true) | False | - | Use trained RSL-RL policy from --resume_path instead of random/zero actions. |
| `--resume_path` | str | ' | - | 'Path to model_*.pt checkpoint (requires --play). Same format as scripts/rsl_rl/full_rl/play.py --checkpoint. |
| `--gym_task` | str | ' | - | 'Registered Gym task id for RSL-RL agent config (default: preset mapping). Required if --env/--cfg override breaks the preset. |
| `--play_success_interval` | float | 2.0 | - | When --play: print mean success rate over all envs every N seconds (in-hand task only; default: 2). |
| `--num_envs` | int | 1 | - | Number of envs (default: 1). |
| `--env_index` | int | 0 | - | Env index to visualize (default: 0). |
| `--fps` | float | 20.0 | - | Target display FPS (default: 20). |
| `--max_steps` | int | 0 | - | If >0, stop after N steps. |
| `--random_actions` | flag (store_true) | False | - | Apply random actions instead of zeros. |
| `--show_rgb` | flag (store_true) | False | - | Show tactile RGB images. |
| `--show_ff` | flag (store_true) | False | - | Show tactile force-field images (if enabled). |
| `--record_path` | str | ' | - | Save the canonical per-step record snapshot to this directory or file prefix. The collector prefers `obs['record']` and retains compatibility with legacy/base-environment record builders. |
| `--record_format` | str | pt' | ['pt', 'npz'] | 'Record file format. |
| `--record_every` | int | 1 | - | Record every N steps (default: 1). |
| `--record_env_index` | int | -1 | - | Env index to record (default: env_index). |

## `scripts/data_collection/rl/play_record.py`

Canonical RSL-RL rollout data collector. Runs a trained RSL-RL checkpoint and, with ``--save_data``, writes successful task trajectories from ``obs['record']`` as compressed NPZ episodes. This file contains the collector implementation directly; it does not dispatch to a script under ``scripts/rsl_rl``.

Run help: `python scripts/data_collection/rl/play_record.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--video` | flag (store_true) | False | - | Record videos during training. |
| `--video_length` | int | 200 | - | Length of the recorded video (in steps). |
| `--disable_fabric` | flag (store_true) | False | - | Disable fabric and use USD I/O operations. |
| `--num_envs` | int | None | - | Number of environments to simulate. |
| `--task` | str | None | - | Name of the task. |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'Name of the RL agent configuration entry point. |
| `--seed` | int | None | - | Seed used for the environment |
| `--use_pretrained_checkpoint` | flag (store_true) | False | - | Use the pre-trained checkpoint from Nucleus. |
| `--real-time` | flag (store_true) | False | - | Run in real-time, if possible. |
| `--save_data` | flag (store_true) | False | - | If set, save play trajectory data to disk. |
| `--data_path` | str | None | - | Directory to save recorded data. |
| `--num_episodes` | int | 50 | - | Number of successful trajectories to save when --save_data is enabled (total across all envs). |
| `--max_steps` | int | 0 | - | When --save_data: max steps per rollout before reset; 0 means no explicit max-step reset. |
| `--show_rgb` | flag (store_true) | False | - | Show TacSL tactile RGB in matplotlib (implies --enable_cameras for ViTacLab tasks). |
| `--show_ff` | flag (store_true) | False | - | Show TacSL tactile force-field (compute_tactile_shear_image arrows) in matplotlib. |
| `--fps` | float | 20.0 | - | Target display FPS when using --show_rgb / --show_ff (default: 20). |
| `--env_index` | int | 0 | - | Which sub-environment to visualize for tactile panels (default: 0). |

## `scripts/data_collection/tools/count_validation_success.py`

Count True/False validation results written by policy rollout scripts. Usage: python scripts/data_collection/tools/count_validation_success.py data/validation/Isaac-UR10eShadowHand-BlindGrasp-Direct-v0/ViTacDP

Run help: `python scripts/data_collection/tools/count_validation_success.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `base_path` | str | - | - | Directory containing validation subfolders with all_success.txt. |

## `scripts/debug/export_handover_episode_mp4.py`

Export handover episode npz to mp4 without matplotlib/Tk. Supports keys produced by scripts/rsl_rl/full_rl/play.py: - third_person_camera: (T, Hc, Wc, 3), uint8 - tactile_rgb_image: (T, S, H, W, 3), uint8 - tactile_normal_force: (T, S, Hf, Wf, 1) - tactile_shear_force: (T, S, Hf, Wf, 2)

Run help: `python scripts/debug/export_handover_episode_mp4.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `npz_path` | str | - | - | Path to episode_x.npz |
| `--out` | str | ' | - | 'Output mp4 path. Default: <npz>.mp4 |
| `--fps` | float | 20.0 | - | Output FPS |
| `--ff_resolution` | int | 30 | - | Force-field cell resolution |
| `--normal_thr` | float | 8e-05 | - | Normal force threshold |
| `--shear_thr` | float | 0.0005 | - | Shear force threshold |

## `scripts/debug/inspect_blind_npz.py`

Inspect & visualize NPZ episodes produced by the blind-grasp data-collection scripts. Supports (auto-detected by file contents) all four blind-grasp tasks: * Single-arm BlindGrasp phases 1/2 and BlindGraspReplay phase 3 (``play_single_phase.py --phase N``) * Dual-arm BiBlindBinDrop phases 1-4 (``play_dual_phase.py --phase N``) The script prints every array in the npz (shape / dtype / min / max / inf-nan flags), then exports: * ``<stem>_summary.png`` - joint / action / object trajectory plots * ``<stem>_third_person_camera.mp4`` - 3rd person RGB (if present) * ``<stem>_twist_camera.mp4`` - twist RGB (if present) * ``<stem>_tactile_rgb.mp4`` - all GelSight tips tiled horizontally * ``<stem>_third_person_and_tactile.mp4`` - 3rd person on top + tactile strip below (``--combined-video``) * ``<stem>_tactile_force_field.mp4`` - normal/shear arrow field per finger * a few PNG snapshots at ``--sample-frames`` Usage:: # Inspect one npz python scripts/debug/inspect_blind_npz.py --npz Output/BlindGrasp01/episode_0000_success.npz --out-dir Output/BlindGrasp01_preview # Inspect every .npz in a folder python scripts/debug/inspect_blind_npz.py --dir Output/BlindGrasp01 --out-dir Output/BlindGrasp01_preview --max-episodes 5 # Only print stats (no video / image dump) python scripts/debug/inspect_blind_npz.py --npz episode_0001.npz --stats-only

Run help: `python scripts/debug/inspect_blind_npz.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--npz` | str | - | - | Path to a single episode .npz |
| `--dir` | str | - | - | Folder containing episode_*.npz files |
| `--out-dir` | str | None | - | Output directory (default: <input_dir>/_preview or <npz_dir>/_preview) |
| `--max-episodes` | int | 0 | - | When using --dir, process at most N files (0 = all) |
| `--fps` | float | 10.0 | - | Playback or control-loop frames per second. |
| `--sample-frames` | int, nargs='*' | [0, 1, 10, 50, 100] | - | Frames to dump as png snapshots |
| `--normal-thr` | float | 8e-05 | - | Color saturation threshold for normal force in force-field plot |
| `--shear-thr` | float | 0.0005 | - | Arrow scaling threshold for shear force in force-field plot |
| `--ff-resolution` | int | 30 | - | Per-taxel pixel size in the force-field plot |
| `--stats-only` | flag (store_true) | False | - | Only print array stats; skip mp4/png export |
| `--no-force-field` | flag (store_true) | False | - | Skip the per-finger force-field arrow video (faster) |
| `--combined-video` | flag (store_true) | False | - | Export one mp4 with third_person_camera on top and tactile_rgb strip below. |

## `scripts/debug/inspect_tactile_frame.py`

Inspect a saved tactile force frame (.npz). Usage (from repo root): python scripts/debug/inspect_tactile_frame.py --path tactile_force_frame_env0_sensor0.npz It will: - print basic statistics (shape, min, max, mean) for: - normal_force: (nrows, ncols) - shear_force: (nrows, ncols, 2) - ff_image: rendered force-field image

Run help: `python scripts/debug/inspect_tactile_frame.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--path` | str | tactile_force_frame_env0_sensor0.npz' | - | 'Path to the .npz file (default: tactile_force_frame_env0_sensor0.npz). |

## `scripts/debug/inspect_usd_structure.py`

Inspect USD file prim hierarchy. Run in Isaac Sim Python or any env with pxr (e.g. usd-core). Usage: python scripts/debug/inspect_usd_structure.py [path_to.usd] Default: source/ViTacLab/ViTacLab/assets/data/Objects/DexCube/dex_cube_sdf.usd

Run help: `python scripts/debug/inspect_usd_structure.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `usd_path` | Path, nargs='?' | DEFAULT_USD | - | USD file to inspect (default: the canonical short GelSight finger asset). |

## `scripts/debug/replay_pickup_npz_in_play_policy_env.py`

Replay pickup npz in play_policy-style env and report task success.

Run help: `python scripts/debug/replay_pickup_npz_in_play_policy_env.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--npz-path` | str | - | - | Required. 'Path to episode_xxxx.npz containing action/joint_pos.' |
| `--num-envs` | int | 1 | - | Parallel env count (match the value used when recording, e.g. 64, so RNG resets match and the object starts at the same pose as in the npz episode). |
| `--record-env-index` | int | 0 | - | Which env index the trajectory was recorded from (must match play_* --record-env-index). |
| `--target-source` | str | 'auto' | ('auto', 'joint_pos', 'action') | Which npz series to replay. 'auto' prefers joint_pos if present (full DoF, matches telemetry), else action. Note: v5/v6 npz often contain both; 'action' was previously chosen first and can combine with num_envs mismatch to look like the hand misses the cube. |
| `--env` | str | ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv' | - | 'Env symbol in module:Class format. |
| `--cfg` | str | ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg' | - | 'Cfg symbol in module:Class format. |
| `--seed` | int | None | - | Random seed. |
| `--fps` | float | 30.0 | - | Replay pacing FPS. |
| `--start-step` | int | 0 | - | Start index in npz target sequence. |
| `--max-steps` | int | 0 | - | Replay at most N target steps (0=no cap). |
| `--steps-per-target` | int | 1 | - | Minimum env.step count per target joint vector. |
| `--post-steps` | int | 0 | - | After replaying the last trajectory point, keep stepping this final target for N extra steps. |
| `--joint-pos-tol` | float | 0.0 | - | If >0, require max\|actual-target\| <= tol (rad) before advancing; <=0 disables settle gating. |
| `--settle-max-steps` | int | 0 | - | Maximum extra settle steps per target when --joint-pos-tol > 0. |
| `--allow-unsettled` | flag (store_true) | False | - | If set, do not raise error when a target fails to settle within --settle-max-steps. |
| `--break-on-success` | flag (store_true) | False | - | Stop replay immediately after success. |
| `--retry-until-success` | flag (store_true) | False | - | If set, failed replay attempts will reset env and retry until success (or max attempts). |
| `--max-reset-attempts` | int | 0 | - | Maximum replay attempts when --retry-until-success is set. 0 means unlimited. |

## `scripts/debug/replay_pickup_npz_online.py`

Online physics replay for pickup npz joint trajectories. Reads per-step joint targets from a saved ``.npz`` (key: ``joint_pos``), applies them to the robot in simulation, steps physics, and reports target-vs-actual joint tracking errors (including finger-only metrics).

Run help: `python scripts/debug/replay_pickup_npz_online.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--npz-path` | str | - | - | Required. "Path to an episode .npz containing key 'joint_pos'." |
| `--env` | str | ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv' | - | 'Environment entry point in module:Class form. |
| `--cfg` | str | ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg' | - | 'Environment-config entry point in module:Class form. |
| `--seed` | int | None | - | Random seed for deterministic resets. |
| `--fps` | float | 30.0 | - | Replay stepping frequency (for sleep pacing). |
| `--start-step` | int | 0 | - | Start index in joint_pos sequence. |
| `--max-steps` | int | 0 | - | If >0, replay at most N steps. |
| `--steps-per-target` | int | 1 | - | Repeat env.step N times for each target joint command (>=1). |
| `--settle-joint-tol` | float | 0.0 | - | If >0, keep stepping same target until max\|actual-target\| <= tol or settle limit reached. |
| `--settle-max-steps` | int | 0 | - | Extra settle steps allowed per target when --settle-joint-tol > 0. |
| `--print-interval` | int | 10 | - | Print error stats every N replay steps. |
| `--top-k-joints` | int | 5 | - | Print Top-K joints ranked by p95 tracking error (overall and finger-only). |
| `--break-on-done` | flag (store_true) | False | - | Stop replay when env returns done. |
| `--episode-steps-cap` | int | 1000000 | - | Override environment episode step cap to avoid timeout auto-reset (<=0 disables override). |
| `--save-third-mp4` | str | ' | - | 'Optional output path for third-person video. |
| `--save-error-csv` | str | ' | - | 'Optional output path for per-step per-joint absolute error CSV. |
| `--save-error-json` | str | ' | - | 'Optional output path for per-step per-joint absolute error JSON. |
| `--save-finger-detail-csv` | str | ' | - | 'Optional output CSV with per-step finger joint target/actual/error values. |
| `--save-finger-detail-json` | str | ' | - | 'Optional output JSON with per-step finger joint target/actual/error values. |

## `scripts/debug/run_forge_peg_tactile_view.py`

Run Isaac-ViTac-Forge-PegInsert-Direct-v0 and display tactile RGB images in real time. Usage (from repo root, inside Isaac Sim Python): ./python.sh scripts/debug/run_forge_peg_tactile_view.py You can also use the system Python if your PYTHONPATH already includes Isaac Lab and this project: python scripts/debug/run_forge_peg_tactile_view.py

Run help: `python scripts/debug/run_forge_peg_tactile_view.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--num-envs` | int | 1 | - | Number of parallel environments (default: 1). |
| `--fps` | float | 20.0 | - | Target display FPS (controls sleep between steps; default: 20). |
| `--random-actions` | flag (store_true) | False | - | If set, apply small random actions instead of zeros. |
| `--env-index` | int | 0 | - | Index of env to visualize when num-envs > 1 (default: 0). |

## `scripts/debug/show_ur10e_shadowhand_records.py`

Replay and visualize recorded UR10e+ShadowHand data. This script loads per-step record files created by: scripts/data_collection/manual/record_observations.py --record_path ... It can visualize: - TacSL tactile RGB images (5 fingertips) - Tactile force-field (FF) image rendered from normal/shear arrays - Third-person camera RGB Supported formats: .pt (torch.save) and .npz (flattened keys).

Run help: `python scripts/debug/show_ur10e_shadowhand_records.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `inputs` | str, nargs='+' | - | - | Record directory, glob, or file(s). Example: /tmp/pickup_rec/ or /tmp/pickup_rec/run1_step_*.pt |
| `--fps` | float | 20.0 | - | Playback FPS (default: 20). |
| `--index` | int | 0 | - | Start file index (default: 0). |
| `--max_frames` | int | 0 | - | If >0, stop after N frames. |
| `--show_rgb` | flag (store_true) | False | - | Show tactile RGB images. |
| `--show_ff` | flag (store_true) | False | - | Show tactile force-field images. |
| `--show_third` | flag (store_true) | False | - | Show third-person RGB image. |

## `scripts/debug/visualize_data.py`

从 play.py 等保存的 .npz 生成视频：相机 RGB、触觉 RGB、触觉力场（法向+切向，与 TacSL demo 一致）。

Run help: `python scripts/debug/visualize_data.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--data_dir` | str | data/rsl_rl/Isaac-UR10eShadowHand-Repose-Cube-Tactile-Direct-v0_42' | - | '含子目录（按时间戳）的根路径；默认取最新子目录 |
| `--show_dir` | str | tmp' | - | '输出 mp4 目录 |
| `--fps` | float | 5.0 | - | 每集视频帧率 |
| `--fps_all` | float | 200.0 | - | 合并长视频的帧率 |
| `--normal_thr` | float | 8e-05 | - | 法向力可视化阈值（与 TacSL 默认一致） |
| `--shear_thr` | float | 0.0005 | - | 切向力可视化阈值（与 TacSL 默认一致） |
| `--ff_resolution` | int | 30 | - | 力场栅格放大倍数（每格边长像素） |

## `scripts/debug/visualize_pickup_episode_npz.py`

Visualize pickup episode npz records. Expected npz keys (from play_full_ik_single.py): - joint_pos: (T, J) - tactile_normal_force: (T, 5, H, W, 1) - tactile_shear_force: (T, 5, H, W, 2) - tactile_rgb_image: (T, 5, H_img, W_img, 3), uint8 - third_person_camera: (T, Hc, Wc, 3), uint8 Export video examples:: python visualize_pickup_episode_npz.py episode.npz --export python visualize_pickup_episode_npz.py episode.npz --export out.mp4 python visualize_pickup_episode_npz.py episode.npz --export-third-only Combine all ``*.npz`` in a directory into one mp4 (sorted by filename):: python visualize_pickup_episode_npz.py play_records/run/ --export --no_preview python visualize_pickup_episode_npz.py play_records/run/ --export-third-only --no_preview python visualize_pickup_episode_npz.py play_records/run/ --export /tmp/all.mp4 --no_preview

Run help: `python scripts/debug/visualize_pickup_episode_npz.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `npz_path` | str | - | - | Path to episode_xxxx.npz, or a directory containing multiple *.npz (combined export only). |
| `--fps` | float | 20.0 | - | Playback FPS / export FPS |
| `--start` | int | 0 | - | Start frame index (single file only; folder export starts at 0) |
| `--max_frames` | int | 0 | - | If >0, only play N frames (single file only) |
| `--no_third` | flag (store_true) | False | - | Disable third-person view |
| `--no_tactile_rgb` | flag (store_true) | False | - | Disable tactile RGB view |
| `--no_force_plot` | flag (store_true) | False | - | Disable force curves |
| `--export, --export-video` | str, nargs='?' | None | - | Export full matplotlib layout to MP4. Optional output path; default: <npz_basename>.mp4 beside npz, or <dir>/episodes_combined.mp4 for a directory. |
| `--export-third-only` | str, nargs='?' | None | - | Export only third_person_camera to MP4 (faster, no matplotlib). For a directory: concat all episodes. Default: <npz_basename>_third.mp4 or <dir>/episodes_combined_third.mp4. |
| `--save-mp4` | str | ' | - | 'Deprecated: same as --export OUT.mp4. |
| `--no_preview` | flag (store_true) | False | - | Do not show interactive window when exporting |

## `scripts/list_envs.py`

Script to print all the available environments in Isaac Lab. The script iterates over all registered environments and stores the details in a table. It prints the name of the environment, the entry point and the config file. All the environments are registered in the `ViTacLab` extension. They start with `Isaac` in their name.

Run help: `python scripts/list_envs.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--keyword` | str | None | - | Keyword to filter environments. |

## `scripts/policy/play_policy.py`

Run and validate a trained Diffusion Policy or ViTacDP checkpoint. The script creates one canonical ViTacLab Gym task, obtains the task's ``record`` observation, executes policy action chunks, and writes one MP4 plus one success flag per parallel environment under ``data/validation``. Use ``--policy-output`` when the checkpoint output dimension is ambiguous. Run ``python scripts/policy/play_policy.py --help`` for every supported option.

Run help: `python scripts/policy/play_policy.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--num_envs` | int | 1 | - | Number of parallel environments (default: 1). |
| `--task` | str | - | - | Required. 'Registered ViTacLab Gym task ID.' |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'Gym metadata key used only to obtain wrapper/clip settings (default: rsl_rl_cfg_entry_point). |
| `--seed` | int | None | - | Environment seed; omit to use the agent config default. |
| `--data_num` | int | 200 | - | Training-dataset count embedded in the default checkpoint directory name (default: 200). |
| `--checkpoint_num` | int | 1000 | - | Checkpoint filename without .ckpt when --policy-checkpoint is omitted (default: 1000). |
| `--policy_name, --policy-name` | str | ViTacDP' | ('Diffusion_Policy', 'ViTacDP') | 'Policy family to load (default: ViTacDP). |
| `--observation-profile, --version` | str | None | - | Checkpoint observation profile such as rgb or force. --version is a deprecated compatibility alias. |
| `--checkpoint_task_name` | str | None | - | Checkpoint folder prefix. Defaults to the task ID, optionally suffixed by --observation-profile. |
| `--policy-checkpoint` | str | None | - | Exact .ckpt file. Relative paths are resolved from the project working directory; overrides folder/number lookup. |
| `--policy-output` | str | 'auto' | ('auto', 'joint_pos', 'action') | Policy output semantics: joint targets or normalized env actions. 'auto' infers by output dim. |
| `--max_steps` | int | 100 | - | Maximum policy steps per rollout; 0 waits until all envs terminate. |
| `--num-episodes` | int | 1 | - | Number of rollout batches. Each batch produces num_envs videos and success flags (default: 1). |
| `--env-max-steps` | int | None | - | Override the task's internal timeout in steps. Omit to preserve the environment definition. |
| `--warmup-steps` | int | 5 | - | Zero-action physics steps after each reset before policy inference (default: 5). |
| `--debug-observations` | flag (store_true) | False | - | Print finite-value statistics for the first policy observation of every rollout. |
| `--joint_err_print_interval` | int | 0 | - | Print joint tracking/jitter diagnostics every N steps; <=0 disables them (default: disabled). |
| `--jitter_topk` | int | 5 | - | Number of largest finger-joint jitter values shown when diagnostics are enabled (default: 5). |

## `scripts/policy/summarize_validation.py`

汇总 play_policy 写入的 validation 结果（all_success.txt）。 目录结构示例:: data/validation/<task>/<policy>_<observation_profile>/{data_num}_{ckpt}_seed{seed}/all_success.txt 每个 all_success.txt 中每行对应一个并行 env 在本轮 global episode 是否成功（True/False）。 num_envs=20 时，每 20 行通常为一轮；若重复运行同一目录会追加行。 用法（仓库根目录）:: python scripts/policy/summarize_validation.py \ data/validation/Isaac-UR10eShadowHand-Pickup-Direct-v0/ViTacDP_force python scripts/policy/summarize_validation.py \ data/validation/Isaac-UR10eShadowHand-Pickup-Direct-v0/ViTacDP_force \ --csv out.csv

Run help: `python scripts/policy/summarize_validation.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `policy_dir` | Path, nargs='?' | Path('data/validation/Isaac-UR10eShadowHand-Pickup-Direct-v0/ViTacDP_force') | - | 策略结果目录，例如 .../ViTacDP_force |
| `--csv` | Path | None | - | 可选：导出逐 run 明细 CSV |

## `scripts/random_agent.py`

Script to an environment with random action agent.

Run help: `python scripts/random_agent.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--disable_fabric` | flag (store_true) | False | - | Disable fabric and use USD I/O operations. |
| `--num_envs` | int | None | - | Number of environments to simulate. |
| `--task` | str | None | - | Name of the task. |
| `--max-steps` | int | 0 | - | Stop after N environment steps; 0 runs until the app closes. |
| `--hold-position` | flag (store_true) | False | - | Hold the robot at its post-reset joint position for static GUI inspection instead of mapping normalized zero actions to joint-range midpoints. Requires the environment to expose `robot` and `apply_joint_targets`. |

## `scripts/rsl_rl/full_ik/train_full_ik_single.py`

**full_ik**: scripted pregrasp + grasp, then UR10e **arm** via GPU IK + pour trajectory. Default YAML (``--full-ik-config``) sets ``freeze_hand_after_script: true`` so the **hand stays at the grasp YAML** after scripted phases; PPO gets a **1-d dummy action** (arm motion is IK-only). Set ``freeze_hand_after_script: false`` to learn hand joints with PPO instead. Example:: ./python.sh scripts/rsl_rl/full_ik/train_full_ik_single.py --task Isaac-UR10eShadowHand-PourDeformable-Direct-v0 \ --num_envs 16 --headless

Run help: `python scripts/rsl_rl/full_ik/train_full_ik_single.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--video` | flag (store_true) | False | - | Record videos during training. |
| `--video_length` | int | 200 | - | Length of the recorded video (in steps). |
| `--video_interval` | int | 2000 | - | Interval between video recordings (in steps). |
| `--num_envs` | int | None | - | Number of environments to simulate. |
| `--task` | str | None | - | Name of the task. |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'Name of the RL agent configuration entry point. |
| `--seed` | int | None | - | Seed used for the environment |
| `--max_iterations` | int | None | - | RL Policy training iterations. |
| `--max_episode_length` | int | None | - | Override env horizon in RL **env steps** (DirectRLEnv: sets episode_length_s = steps * sim.dt * decimation). |
| `--distributed` | flag (store_true) | False | - | Run training with multiple GPUs or nodes. |
| `--export_io_descriptors` | flag (store_true) | False | - | Export IO descriptors. |
| `--ray-proc-id, -rid` | int | None | - | Automatically configured by Ray integration, otherwise None. |
| `--trajectory` | str | object:150:0,goal:-1:0' | - | 'Comma-separated phases: name:env_steps:use_rotation (0/1). name = env asset (e.g. object, cup) or tensor prefix (e.g. goal_cup → goal_cup_pos/rot), or goal (legacy). steps=-1 = until episode end. |
| `--object-to-palm-offset` | float, nargs=3 | (0.0, 0.0, 0.05) | - | Offset from trajectory anchor to palm origin (world if use_rotation=0, anchor frame if use_rotation=1). |
| `--palm-in-wrist-pos` | float, nargs=3 | (0.0, 0.0, 0.35) | - | Palm origin in wrist_3 frame (m). |
| `--palm-in-wrist-euler` | float, nargs=3 | (np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0) | - | Palm in wrist_3 euler xyz (rad). |
| `--palm-orient` | str | pickup_down' | ('fixed', 'pickup_down') | 'When trajectory phase has use_rotation=0: fixed euler or pickup_down. |
| `--palm-normal-local` | float, nargs=3 | (0.0, 1.0, 0.0) | - | Palm-frame axis to align with --world-down (pickup_down). |
| `--palm-yaw-offset` | float | 0.0 | - | Extra yaw (rad) about world Z after pickup_down alignment. |
| `--world-down` | float, nargs=3 | (0.0, 0.0, -1.0) | - | World down direction for pickup_down. |
| `--palm-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Palm world euler xyz (rad) when --palm-orient fixed. |
| `--palm-euler-in-anchor` | float, nargs=3 | (0.0, 0.0, 0.0) | - | When use_rotation=1: euler xyz (rad) of palm relative to anchor frame (applied after anchor quat). |
| `--ee-body` | str | wrist_3_link' | - | 'End-effector link for Jacobian. |
| `--ik-method` | str | dls' | ('pinv', 'svd', 'trans', 'dls') | 'Differential IK Jacobian method. |
| `--ik-lambda` | float | None | - | dls damping lambda override; default = Isaac controller. |
| `--ik-config` | str | None | - | Optional extra IK YAML merge (pickup-style). For full_ik, prefer --full-ik-config; pass 'none' to skip. |
| `--full-ik-config` | str | str(_DEFAULT_FULL_IK_YAML) | - | YAML with phase_schedule + palm/IK/trajectory (see scripts/data_collection/ik/configs/full_ik/full_ik_pour.yaml). |

## `scripts/rsl_rl/full_rl/play.py`

Evaluation-only compatibility entry for full-joint RSL-RL checkpoints. The maintained rollout implementation is ``scripts/data_collection/rl/play_record.py``. This compatibility command allows checkpoint evaluation but deliberately rejects dataset-recording arguments so no data collector remains under ``scripts/rsl_rl``.

Run help: `python scripts/rsl_rl/full_rl/play.py --help`

No repository-owned argparse options are declared in this file; see its module description and runtime help.

## `scripts/rsl_rl/full_rl/train.py`

Script to train RL agent with RSL-RL.

Run help: `python scripts/rsl_rl/full_rl/train.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--video` | flag (store_true) | False | - | Record videos during training. |
| `--video_length` | int | 200 | - | Length of the recorded video (in steps). |
| `--video_interval` | int | 2000 | - | Interval between video recordings (in steps). |
| `--num_envs` | int | None | - | Number of environments to simulate. |
| `--task` | str | None | - | Name of the task. |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'Name of the RL agent configuration entry point. |
| `--seed` | int | None | - | Seed used for the environment |
| `--max_iterations` | int | None | - | RL Policy training iterations. |
| `--distributed` | flag (store_true) | False | - | Run training with multiple GPUs or nodes. |
| `--export_io_descriptors` | flag (store_true) | False | - | Export IO descriptors. |
| `--ray-proc-id, -rid` | int | None | - | Automatically configured by Ray integration, otherwise None. |

## `scripts/rsl_rl/ik_rl/train_ik_rl_single.py`

Train Shadow **hand** with RSL-RL; UR10e **arm** follows GPU differential IK + scripted palm trajectory. Mirrors ``train.py`` but wraps with :class:`IkHandRslRlVecEnvWrapper` (policy = hand only). **Trajectory** (task-agnostic): each segment is ``<name>:env_steps:use_rotation``. ``<name>`` resolves on the env to either an asset ``env.<name>`` (``root_pos_w`` / ``root_quat_w``), or tensors ``<name>_pos`` / ``<name>_rot`` (pos env-local), or legacy ``goal`` → ``goal_object_pos`` / ``goal_object_rot``. Example: ``cup:150:0,goal_cup:-1:0`` for pour. ``use_rotation``: ``1`` = offset in anchor frame + palm aligned with anchor rotation. Example:: ./isaaclab.sh -p scripts/rsl_rl/ik_rl/train_ik_rl_single.py --task Isaac-UR10eShadowHand-Pickup-Direct-v0 \ --num_envs 16 Palm/IK defaults load from ``scripts/data_collection/ik/configs/ik_rl/ik_rl_pickup.yaml`` when that file exists (override with ``--ik-config PATH`` or ``--ik-config none``). CLI flags such as ``--trajectory`` still override YAML.

Run help: `python scripts/rsl_rl/ik_rl/train_ik_rl_single.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--video` | flag (store_true) | False | - | Record videos during training. |
| `--video_length` | int | 200 | - | Length of the recorded video (in steps). |
| `--video_interval` | int | 2000 | - | Interval between video recordings (in steps). |
| `--num_envs` | int | None | - | Number of environments to simulate. |
| `--task` | str | None | - | Name of the task. |
| `--agent` | str | rsl_rl_cfg_entry_point' | - | 'Name of the RL agent configuration entry point. |
| `--seed` | int | None | - | Seed used for the environment |
| `--max_iterations` | int | None | - | RL Policy training iterations. |
| `--max_episode_length` | int | None | - | Override env horizon in RL **env steps** (DirectRLEnv: sets episode_length_s = steps * sim.dt * decimation). |
| `--distributed` | flag (store_true) | False | - | Run training with multiple GPUs or nodes. |
| `--export_io_descriptors` | flag (store_true) | False | - | Export IO descriptors. |
| `--ray-proc-id, -rid` | int | None | - | Automatically configured by Ray integration, otherwise None. |
| `--trajectory` | str | object:150:0,goal:-1:0' | - | 'Comma-separated phases: name:env_steps:use_rotation (0/1). name = env asset (e.g. object, cup) or tensor prefix (e.g. goal_cup → goal_cup_pos/rot), or goal (legacy). steps=-1 = until episode end. |
| `--object-to-palm-offset` | float, nargs=3 | (0.0, 0.0, 0.05) | - | Offset from trajectory anchor to palm origin (world if use_rotation=0, anchor frame if use_rotation=1). |
| `--palm-in-wrist-pos` | float, nargs=3 | (0.0, 0.0, 0.35) | - | Palm origin in wrist_3 frame (m). |
| `--palm-in-wrist-euler` | float, nargs=3 | (np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0) | - | Palm in wrist_3 euler xyz (rad). |
| `--palm-orient` | str | pickup_down' | ('fixed', 'pickup_down') | 'When trajectory phase has use_rotation=0: fixed euler or pickup_down. |
| `--palm-normal-local` | float, nargs=3 | (0.0, 1.0, 0.0) | - | Palm-frame axis to align with --world-down (pickup_down). |
| `--palm-yaw-offset` | float | 0.0 | - | Extra yaw (rad) about world Z after pickup_down alignment. |
| `--world-down` | float, nargs=3 | (0.0, 0.0, -1.0) | - | World down direction for pickup_down. |
| `--palm-euler` | float, nargs=3 | (0.0, 2.2, 0.0) | - | Palm world euler xyz (rad) when --palm-orient fixed. |
| `--palm-euler-in-anchor` | float, nargs=3 | (0.0, 0.0, 0.0) | - | When use_rotation=1: euler xyz (rad) of palm relative to anchor frame (applied after anchor quat). |
| `--ee-body` | str | wrist_3_link' | - | 'End-effector link for Jacobian. |
| `--ik-method` | str | dls' | ('pinv', 'svd', 'trans', 'dls') | 'Differential IK Jacobian method. |
| `--ik-lambda` | float | None | - | dls damping lambda override; default = Isaac controller. |
| `--hand-freeze-phase-target` | str | None | - | When set along with --hand-freeze-yaml: freeze hand joints during trajectory phases whose target matches this string (e.g. pickup: 'goal'). |
| `--hand-freeze-yaml` | str | None | - | YAML with hand_joint_pos_shadow_order (24 floats) to freeze hand joints to during grasp phase. |
| `--ik-config` | str | None | - | YAML with task (Gym id) + palm/IK/trajectory (see scripts/data_collection/ik/configs/ik_rl/ik_rl_pickup.yaml). If omitted, that file is loaded when present. Pass 'none' to disable. |

## `scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py`

Map desired **palm** pose in world frame to UR10e **arm** joint angles (ikpy via VideoTeleopControl). The IK chain ends at ``wrist_3_link``. You specify: 1. Palm pose in **world**: position + orientation (euler xyz or quaternion wxyz). 2. Fixed **wrist → palm** extrinsic ``T_wrist_palm``: pose of the palm frame **expressed in wrist_3 (tool) frame**, i.e. ``T_world_palm = T_world_wrist @ T_wrist_palm``. Then ``T_world_wrist = T_world_palm @ inv(T_wrist_palm)``, and the script solves IK for the six arm joints (same as ``video_teleop_control.VideoTeleopControl``). This matches the idea behind video teleop's ``T_tag1_hand``: composed on the right of the tracked frame to get the frame that IK actually targets. **No Isaac Sim required** — run with system Python if ``ikpy`` / ``scipy`` are installed. Examples:: # Palm at (0.65, 0.12, 0.42) m, euler xyz (rad); palm 8cm along wrist +z from wrist_3 python scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py \ --palm-pos 0.65 0.12 0.42 --palm-euler 0.0 2.2 0.0 # Same orientation as quaternion w x y z (world) python scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py \ --palm-pos 0.65 0.12 0.42 --palm-quat-wxyz 1 0 0 0 # Use the same default offset as run_video_teleop_ur10e_shadowhand_single (--tag1-hand-*) python scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py \ --palm-pos 0.65 0.12 0.42 --palm-euler 0.0 2.2 0.0 --offset-preset video-teleop-tag1

Run help: `python scripts/teleoperation/gui_teleop/ur10_palm_pose_to_arm_joints.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--palm-pos` | float, nargs=3 | - | - | Required. 'Palm origin in world (m).' |
| `--palm-euler` | float, nargs=3 | - | - | Palm orientation world euler xyz (rad). Use --degrees for degrees. |
| `--palm-quat-wxyz` | float, nargs=4 | - | - | Palm orientation quaternion (w,x,y,z) in world. |
| `--degrees` | flag (store_true) | False | - | Interpret --palm-euler as degrees. |
| `--wrist-to-palm-pos` | float, nargs=3 | (0.0, 0.0, 0.08) | - | Translation part of T_wrist_palm: palm origin in wrist_3 frame (m). Default: 8cm along wrist z. |
| `--wrist-to-palm-euler` | float, nargs=3 | (0.0, 0.0, 0.0) | - | Rotation part of T_wrist_palm (euler xyz, rad). Palm axes relative to wrist_3. |
| `--offset-degrees` | flag (store_true) | False | - | Interpret --wrist-to-palm-euler as degrees. |
| `--offset-preset` | str | '' | ['', 'video-teleop-tag1'] | Override wrist–palm offset: 'video-teleop-tag1' matches run_video_teleop default --tag1-hand-pos/euler. |
| `--arm-base-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | T_world_arm_base translation (m). |
| `--arm-base-euler` | float, nargs=3 | (0.0, 0.0, 0.0) | - | T_world_arm_base euler xyz (rad). |
| `--arm-base-degrees` | flag (store_true) | False | - | Interpret --arm-base-euler as degrees. |
| `--urdf` | str | ' | - | 'Override UR10+hand URDF path (default: VideoTeleopControl built-in left urdf). |
| `--json` | flag (store_true) | False | - | Print one JSON line: {"shoulder_pan_joint": ..., ...}. |
| `--quiet` | flag (store_true) | False | - | Only print joint block / JSON, no extra text. |

## `scripts/teleoperation/video_teleop/camera_calibration.py`

Chessboard camera calibration (delegates to ``video_teleop.tools.camera_calibration``).

Run help: `python scripts/teleoperation/video_teleop/camera_calibration.py --help`

No repository-owned argparse options are declared in this file; see its module description and runtime help.

## `scripts/teleoperation/video_teleop/list_cameras.py`

List camera indices (delegates to ``video_teleop.tools.list_cameras``).

Run help: `python scripts/teleoperation/video_teleop/list_cameras.py --help`

No repository-owned argparse options are declared in this file; see its module description and runtime help.

## `scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py`

Video teleoperation for UR10e+ShadowHand single-arm task. Receives teleop data via IPC, applies same pose transforms as run_video_teleop_receiver, solves IK for arm joints, and controls the simulation. Usage: ./python.sh scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py \ --task pour --zmq-address ipc:///tmp/shadowhand_teleop_video.ipc \ --hand-mode left # With wrist pose visualization (VisualCuboid markers): ./python.sh scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py \ --task pour --enable-visualization --hand-mode left # In-hand cube (policy is hand-only in env; teleop drives hand joints, arm stays at task default pose) ./python.sh scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py \ --task inhand --enable-visualization --hand-mode left

Run help: `python scripts/teleoperation/video_teleop/run_task/run_video_teleop_ur10e_shadowhand_single.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--task` | str | pour' | sorted(_TASK_PRESETS.keys()) | 'Task preset |
| `--env` | str | ' | - | 'Env entry: module:Class |
| `--cfg` | str | ' | - | 'Cfg entry: module:Class |
| `--num_envs` | int | 1 | - | Number of envs (left=0, right=1) |
| `--zmq-address` | str | ipc:///tmp/shadowhand_teleop_video.ipc' | - | 'ZMQ address |
| `--hand-mode` | str | left' | ['left', 'right', 'both'] | 'Hand control source used by teleoperation. |
| `--task-fps` | float | 20.0 | - | Target control FPS |
| `--enable-visualization` | flag (store_true) | False | - | Enable 3D visualization (VisualCuboid markers for wrist pose) |
| `--debug` | flag (store_true) | False | - | Print IK solution (arm joints) for debugging |
| `--tag0-world-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | AprilTag 0 reference world position X Y Z in meters. |
| `--tag0-world-euler` | float, nargs=3 | (0.0, 3.141592653589793, 1.5707963267948966) | - | AprilTag 0 reference world XYZ Euler orientation in radians. |
| `--tag1-hand-pos` | float, nargs=3 | (0.0, 0.0, 0.08) | - | AprilTag 1 to hand-target XYZ offset in meters. |
| `--tag1-hand-euler` | float, nargs=3 | (0.0, -1.5707963267948966, 1.5707963267948966) | - | AprilTag 1 to hand-target XYZ Euler offset in radians. |
| `--flip-axis` | str | none' | ['none', 'x', 'y', 'z'] | 'Axis used by the optional coordinate-frame flip. |
| `--flip-where` | str | tag1_hand' | ['tag1_hand', 'world_tag0', 'both'] | 'Apply the coordinate-frame flip before or after the tag transform. |
| `--arm-base-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | T_world_arm_base translation |
| `--arm-base-euler` | float, nargs=3 | (0.0, 0.0, 0.0) | - | T_world_arm_base rotation (rad) |
| `--pos-scale` | float, nargs=3 | (4, 1, 2) | - | Position scale factors for xyz (default: 1 1 1) |

## `scripts/teleoperation/video_teleop/run_video_teleop_receiver.py`

Command-line entry point for video teleoperation receiver (Phase 2). This script receives IPC messages and visualizes them (no robot control). Usage: # Recommended (stable): visualize using robot_frame wrist pose python scripts/teleoperation/video_teleop/run_video_teleop_receiver.py --zmq-address ipc:///tmp/shadowhand_teleop_video.ipc --enable-visualization --hand-mode left --tag0-world-euler 0 3.141592653589793 1.5707963267948966 --tag1-hand-euler 0 3.141592653589793 1.5707963267948966

Run help: `python scripts/teleoperation/video_teleop/run_video_teleop_receiver.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--zmq-address` | str | 'ipc:///tmp/shadowhand_teleop_video.ipc' | - | ZeroMQ address (IPC or TCP, e.g., 'tcp://127.0.0.1:5555') |
| `--print-rate` | float | 1.0 | - | Rate at which to print messages (Hz, 0 to disable) |
| `--disable-print` | flag (store_true) | False | - | Disable printing messages |
| `--enable-visualization` | flag (store_true) | False | - | Enable 3D visualization in Isaac Sim (VisualCuboid markers) |
| `--hand-mode` | str | 'both' | ['left', 'right', 'both'] | Which hand(s) to visualize in Isaac Sim: 'left', 'right', or 'both' |
| `--tag0-world-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | T_tag0_world translation (world -> tag0), meters |
| `--tag0-world-euler` | float, nargs=3 | (0.0, 3.141592653589793, 1.5707963267948966) | - | T_tag0_world rotation as Euler xyz (world -> tag0), radians |
| `--tag1-hand-pos` | float, nargs=3 | (0.0, 0.0, 0.0) | - | T_tag1_hand translation (tag1 -> hand), meters |
| `--tag1-hand-euler` | float, nargs=3 | (0.0, 3.141592653589793, 1.5707963267948966) | - | T_tag1_hand rotation as Euler xyz (tag1 -> hand), radians |
| `--flip-axis` | str | none' | ['none', 'x', 'y', 'z'] | 'Optional axis flip (mirror) as diag([-1,1,1]) etc. |
| `--flip-where` | str | tag1_hand' | ['tag1_hand', 'world_tag0', 'both'] | 'Where to apply T_flip (default: tag1_hand) |

## `scripts/teleoperation/video_teleop/run_video_teleop_sender.py`

Command-line entry point for video teleoperation sender Usage: python scripts/teleoperation/video_teleop/run_video_teleop_sender.py --camera 0 --hand-mode left

Run help: `python scripts/teleoperation/video_teleop/run_video_teleop_sender.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--camera` | int | 0 | - | Camera device index (default: 0) |
| `--hand-mode` | str | both' | ['left', 'right', 'both'] | 'Hand detection mode |
| `--calibration-file` | str | default_camera_calibration_yaml() | - | Path to camera calibration YAML (default: scripts/teleoperation/video_teleop/config/camera_calibration.yaml) |
| `--hand-calibration` | str | default_hand_calibration_yaml() | - | Path to hand range calibration YAML (default: scripts/teleoperation/video_teleop/config/hand_calibration.yaml) |
| `--zmq-address` | str | 'ipc:///tmp/shadowhand_teleop_video.ipc' | - | ZeroMQ address (IPC or TCP, e.g., 'tcp://127.0.0.1:5555') |
| `--send-rate` | float | 30.0 | - | Target send rate (Hz) |
| `--enable-landmarks` | flag (store_true) | False | - | Include landmarks in messages (increases message size) |
| `--disable-visualization` | flag (store_true) | False | - | Disable OpenCV visualization windows |

## `scripts/zero_agent.py`

Script to run an environment with zero action agent.

Run help: `python scripts/zero_agent.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--disable_fabric` | flag (store_true) | False | - | Disable fabric and use USD I/O operations. |
| `--num_envs` | int | None | - | Number of environments to simulate. |
| `--task` | str | None | - | Name of the task. |
| `--max-steps` | int | 0 | - | Stop after N environment steps; 0 runs until the app closes. |

## `source/video_teleop/core/video_listener.py`

Video-based hand listener (MediaPipe + AprilTag wrist pose).

Run help: `python source/video_teleop/core/video_listener.py --help`

No repository-owned argparse options are declared in this file; see its module description and runtime help.

## `source/video_teleop/tools/calibrate_hand_ranges.py`

Hand Range Calibration Script This script helps calibrate MediaPipe output ranges to match ShadowHand joint limits by recording two poses: open hand and closed fist. Usage: python source/video_teleop/tools/calibrate_hand_ranges.py --camera 6 --side right \ --output scripts/teleoperation/video_teleop/config/hand_calibration.yaml

Run help: `python source/video_teleop/tools/calibrate_hand_ranges.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--camera` | int | 0 | - | Camera device ID |
| `--side` | str | right' | ['left', 'right'] | 'Hand side to calibrate |
| `--output` | str | default_hand_calibration_yaml() | - | Output calibration YAML path (default: scripts/teleoperation/video_teleop/config/hand_calibration.yaml) |
| `--num-samples` | int | 30 | - | Number of samples to collect per pose |
| `--sample-interval` | int | 2 | - | Frames to skip between samples |

## `source/video_teleop/tools/camera_calibration.py`

Chessboard camera calibration (intrinsics + distortion), YAML output for video teleop. Usage (from ViTacLab repo root): conda activate video_teleoperator PYTHONPATH=source python source/video_teleop/tools/camera_calibration.py \ --camera 0 --rows 5 --cols 7 --square-size 0.025 \ --output scripts/teleoperation/video_teleop/config/camera_calibration.yaml Verify: PYTHONPATH=source python source/video_teleop/tools/camera_calibration.py \ --camera 0 --verify scripts/teleoperation/video_teleop/config/camera_calibration.yaml

Run help: `python source/video_teleop/tools/camera_calibration.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--camera` | int | 0 | - | OpenCV camera device index (default: 0). |
| `--rows` | int | 5 | - | Inner corner rows |
| `--cols` | int | 7 | - | Inner corner columns |
| `--square-size` | float | 0.025 | - | Square size in meters |
| `--output` | str | scripts/teleoperation/video_teleop/config/camera_calibration.yaml' | - | 'Output YAML path for the estimated camera intrinsics and distortion coefficients. |
| `--frame-width` | int | 1280 | - | Requested camera frame width in pixels. |
| `--frame-height` | int | 720 | - | Requested camera frame height in pixels. |
| `--verify` | str | ' | - | 'Path to YAML; show undistort preview |

## `source/video_teleop/tools/get_realsense_rgbd.py`

Preview aligned RGB and depth frames from an Intel RealSense L515 camera. Run this hardware utility directly with the Python environment that provides ``pyrealsense2``. Press ``q`` in the preview window to stop and release the camera cleanly.

Run help: `python source/video_teleop/tools/get_realsense_rgbd.py --help`

No repository-owned argparse options are declared in this file; see its module description and runtime help.

## `source/video_teleop/tools/list_cameras.py`

List OpenCV-accessible camera indices (quick probe). Usage (from ViTacLab repo root): conda activate video_teleoperator PYTHONPATH=source python source/video_teleop/tools/list_cameras.py --max-index 15

Run help: `python source/video_teleop/tools/list_cameras.py --help`

| Argument | Type/action | Default | Choices | Description |
|---|---|---|---|---|
| `--max-index` | int | 15 | - | Try indices 0 .. max_index-1 (default: 15) |
| `--width` | int | 640 | - | Request frame width when probing |
| `--height` | int | 480 | - | Request frame height when probing |

## `scripts/data_collection/tools/upload_records.sh`

Uploads selected `*_success.npz` files over SSH/SCP, retries failed transfers, and continues remote episode numbering. It is configured with environment variables:

| Variable | Required | Default | Meaning |
|---|---:|---|---|
| `UPLOAD_SSH_HOST` | yes | - | SSH target in `user@host` form. |
| `UPLOAD_SSH_PORT` | yes | - | SSH/SCP port. |
| `REMOTE_DIR` | yes | - | Destination directory created on the remote host. |
| `LOCAL_DIR1` | yes | - | First local recording directory. |
| `LOCAL_DIR2` | no | empty | Optional second local recording directory. |
| `MAX_FILES_PER_DIR` | no | `100` | Maximum success files selected from each local directory. |

Run: `UPLOAD_SSH_HOST=... UPLOAD_SSH_PORT=22 REMOTE_DIR=... LOCAL_DIR1=... bash scripts/data_collection/tools/upload_records.sh`.

## `scripts/policy/batch_validate_policy.sh`

Runs the canonical policy validator over checkpoint numbers, seeds, policy families, and optional ViTacDP observation profiles.

| Variable | Default | Meaning |
|---|---|---|
| `TASK` | `Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0` | Gym task ID. |
| `NUM_ENVS` | `20` | Number of parallel environments. |
| `MAX_STEPS` | `500` | Maximum policy steps per rollout. |
| `DATA_NUM` | `100` | Dataset/checkpoint directory selector used by the policy loader. |
| `CHECKPOINT_TASK_NAME` | `BottleCup` | Checkpoint task directory name. |
| `POLICY_OUTPUT` | `action` | Policy output semantics passed to the runner. |
| `RUN_DIFFUSION` | `1` | Also test camera-only Diffusion Policy when set to 1. |
| `SEED_START` | `42` | First seed, inclusive. |
| `SEED_END` | `51` | Last seed, inclusive. |
| `CKPTS` | `2000` | Space-separated checkpoint numbers. |
| `VITACDP_PROFILES` | empty | Space-separated ViTacDP observation profiles such as `rgb force`; empty uses checkpoint defaults. |
| `LOG_DIR` | `logs/batch_validate_policy` | Validation log directory. |
| `TEE_LOG` | `1` | Stream output while logging when set to 1. |

Run: `TASK=<TASK_ID> CKPTS="1000 2000" VITACDP_PROFILES="rgb force" bash scripts/policy/batch_validate_policy.sh`.

## Shared runtime option sources

- `scripts/common/rl/cli_args.py`: checkpoint, resume, run selection, logging, video, and RSL-RL runner options used by RL train/play/record entries.

- Isaac Lab `AppLauncher.add_app_launcher_args(...)`: application/device/headless/camera options. Use the remote environment's `--help` because the exact set follows the installed Isaac Lab release.

- Hydra training overrides in `policy/train_policy.py`: all tokens after the entry arguments are `key=value` overrides. Canonical examples are in `policy/README.md`.
