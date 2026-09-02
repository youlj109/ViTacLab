#!/usr/bin/env bash
# Shadow Hand full-hand voxel tactile — body-name diagnostic (same launch style as visuotactile demo).
#
# Do NOT use --headless here: Isaac Sim headless kit can fail to import PIL/matplotlib
# while the rendering kit works (same as demo_visuotactile_sensor_v2.py).
#
#   cd ~/Code/lightwheel/IssacLab_510test/ViTacLab
#   bash bash_command/shadow_hand_tactile_test.sh
#
# Or manually:
#   python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1 --no_plot --diag_only
#   python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1
#   python scripts/demo/demo_shadow_hand_full_tactile_sensor.py --num_envs 1 --plot_2d_only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITAC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${VITAC_ROOT}"
python scripts/demo/demo_shadow_hand_full_tactile_sensor.py \
    --num_envs 1 \
    --no_plot \
    --diag_only \
    "$@"
