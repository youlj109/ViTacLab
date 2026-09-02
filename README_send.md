# ViTacSim Sensor + Robot Package (for advisor/collaborators)

This package is prepared to avoid missing robot dependencies. It includes:

1) tactile sensor implementation,  
2) robot configuration code, and  
3) robot USD assets required by the included task configs.

## 1) Included Modules (Code)

### Core tactile sensor

- `source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor/`
  - `visuotactile_sensor_cfg.py`
  - `visuotactile_sensor_data.py`
  - `visuotactile_sensor.py`
  - `visuotactile_sensor_v2.py`
  - other support modules in this folder

### Robot config and base env helpers (previously missing in old pack)

- `source/ViTacLab/ViTacLab/assets/robot/`
  - `ur10e_shadowhand_direct_base_single/`
  - `ur10e_dual_shadowhand_direct_base/`
  - `__init__.py`

### Task integration entry points

- `source/ViTacLab/ViTacLab/tasks/direct/simple_gripper/forge_env.py`
- `source/ViTacLab/ViTacLab/tasks/direct/simple_gripper/forge_env_cfg.py`
- `source/ViTacLab/ViTacLab/tasks/direct/medium_dexhand/forge_dexhand/ur10e_shadowhand_forge_env_cfg.py`

### Demo / validation scripts

- `scripts/demo/demo_forge_tactile_feedback_insert.py`
- `bash_command/forge_tactile_feedback_insert_demo.sh`
- `scripts/demo/eval_visuotactile_physx_alignment.py`
- `scripts/demo/demo_visuotactile_alignment_visual.py`

## 2) Included Assets (USD)

To prevent the "assets has no robot / robot missing files" issue, include:

- `source/ViTacLab/ViTacLab/assets/data/Robots/Franka/`
- `source/ViTacLab/ViTacLab/assets/data/Robots/ShadowHand/`

These are required by current forge/simple-gripper and UR10e-shadowhand config paths.

## 3) Critical Mechanism Flags

- `use_physx_sparse_anchors=True`
- `require_physx_sparse_anchors=True` (strict runs)
- `strict_target_contact_attribution=True`

These enforce PhysX-based anchor path and prevent non-target contact attribution pollution.

## 4) Quick run

From repository root:

```bash
bash bash_command/forge_tactile_feedback_insert_demo.sh
```

Default outputs:

- Video: `logs/tactile_feedback_demo/forge_tactile_insert.mp4`
- NPZ: `logs/tactile_feedback_demo/forge_tactile_insert_tactile.npz`

## 5) Packaging notes

- This package intentionally includes robot code + robot USD assets.
- It excludes logs/checkpoints/videos unless explicitly needed.

