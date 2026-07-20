# Shared script infrastructure

This directory contains importable, non-executable helpers shared by training,
policy inference, and data collection. It must not contain episode loops,
success filtering, dataset buffers, or task-data writers.

- `rl/cli_args.py`: shared RSL-RL command-line options.
- `rl/rsl_rl_log_utils.py`: checkpoint/log resolution.
- `rl/ik_rl_*`: IK-RL config and action-wrapper infrastructure.
- `rl/full_ik_*`: Full-IK phase/action-wrapper infrastructure.
- `sensor_diagnostics.py`: read-only, one-shot camera/TacSL tensor summaries
  used by the zero/random environment smoke tests.

All executable data collectors live under `scripts/data_collection/`.
