# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacSL VisuoTactile sensors (V1 SDF + V2 depth-based) — ViTacLab local copy."""

from .visuotactile_marker import MarkerSimulator, MarkerPattern, PATTERN_SPECS
from .visuotactile_sensor import VisuoTactileSensor
from .visuotactile_sensor_cfg import GelSightRenderCfg, VisuoTactileSensorCfg
from .visuotactile_sensor_data import VisuoTactileSensorData
from .visuotactile_sensor_v2 import VisuoTactileSensorV2, VisuoTactileSensorV2Cfg
