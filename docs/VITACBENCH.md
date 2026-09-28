# ViTacBench handoff

Branch for the paper bench: env/contract diffs plus one-command demo pipelines.
This is **not** a dump of the local worktree (no PDFs, `dist/`, logs, calibration probes, or ViTacDP training copies).

## Run (repo root, `env_isaaclab_510test`)

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
NPZ + panel mp4 land under `logs/demo_<task>/`.

## Blind Grasp blocker

Actor contract is in place: tactile + proprio only (`include_object_state_in_policy_obs=False`, `policy_record_profile=tactile_only`). Success is live multi-finger gel contact then cube `z > 0.35` (external metric, not in obs/reward).

Keyframe replay and GUI teleop do not extract. Privileged expert (`scripts/data_collection/full_trajectory/collect_blind_grasp_privileged.py`) can reach pose_008 gels (often FF, sometimes 2-finger) but **has not produced a lift** with live lift-tactile and `z>0.35`. Latest physics on the env: cube mass `0.05`, sim friction `(3.0, 2.5)`, four-finger enclosure vs opposed thumb. Still slips.

## IL contract

`scripts/debug/validate_play_record_npz.py --profile tactile_only` for blind tasks.
`policy/prepare_dataset.py` has `canonical` / `tactile_only`. Placeholder ViTacDP task yamls: `policy/ViTacDP/config/task/single_arm_tactile_only_{force,rgb}.yaml`.
