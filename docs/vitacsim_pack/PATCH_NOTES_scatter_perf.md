# ViTacSim V2 hotfix: force-corrected height map scatter (2026-08-09)

## Problem

`VisuoTactileSensorV2._update_force_corrected_height_map` used a Python loop over all tactile grid points (`num_tactile_points`, e.g. 100×100 = 10,000) with per-point `.item()` GPU→CPU sync. Profiling showed ~380 ms/step (~94% of tactile time) on dense grids.

## Fix

- Cache pixel indices at init: `_sample_u_idx`, `_sample_v_idx`, `_sample_flat_idx` (built in `_build_uv_sample_grid`).
- Replace the Python loop with one GPU `scatter_reduce(..., reduce="amax")` write per step.

**File:** `source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/visuotactile_sensor_v2.py`

## Apply on existing checkout

```bash
# Overlay this pack into ViTacLab, then reinstall extension:
cd source/ViTacLab && python -m pip install -e .
```

Or copy only the fixed file:

```bash
cp source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/visuotactile_sensor_v2.py \
   /path/to/ViTacLab/source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/
```

## Re-verify (optional)

```bash
bash bash_command/visuotactile_physx_alignment_check.sh
```

Expected: same PhysX alignment results; `_update_force_corrected_height_map` should drop from O(10k) Python iterations to a single vectorized scatter.
