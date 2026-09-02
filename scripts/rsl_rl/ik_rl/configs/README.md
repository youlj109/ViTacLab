# IK+RL YAML (`--ik-config`)

These files merge into `train_ik_rl_single.py`, `play_ik_rl_single.py`, `train_ik_rl_dual.py`, and `play_ik_rl_dual.py` before Hydra parses `--task` (unless `--ik-config none`).

## Quaternion convention (Isaac Lab)

`quat` must be **wxyz** (scalar-first), matching Isaac Lab / Omniverse: `[w, x, y, z]`. Do **not** use xyzw (scalar-last).

## Keys

| Key | Meaning |
|-----|---------|
| `task` | Intended Gym id (warning if CLI `--task` differs). |
| `trajectory` | List of `{pos: [x,y,z], quat: [w,x,y,z], steps: int}` — world-frame **EE** pose (`ee_body`), `quat` **wxyz**, `steps` = env steps to hold; `-1` = until episode end. |
| `trajectory_right` / `trajectory_left` | Dual-arm only. If omitted, `trajectory` is copied to that arm when `trajectory` is set. |
| `ee_body` | Link name for IK target (default `wrist_3_link`). |
| `ik_method` | `pinv` / `svd` / `trans` / `dls`. |
| `ik_lambda` | DLS damping (smaller ⇒ faster joint steps). If omitted, ik_rl uses **0.005** (Isaac Lab default is **0.01**). Set e.g. `0.01` to match stock Isaac. |
| `ik_k_val` | `pinv` / `svd` / `trans` only: scales Jacobian step (Isaac default **1.0**). Ignored for `dls`. |
| `ik_delta_scale` | Multiplies the joint-space IK **delta** each env step (default **1.0**). Try **1.5–2.5** if motion feels slow. |
| `ik_waypoints_world_frame` | If **true**, trajectory ``pos`` is already **global** simulation world (no ``+env_origins``). Default **false**: ``pos`` is **env-local** (required for multi-env cloning). |

**Randomized object / bottle position:** if the env defines `ik_rl_trajectory_xyz_offset` (per-env, set at reset), it is **added** to every waypoint `pos` (see `source/ViTacLab/ViTacLab/tasks/direct/medium_dexhand/unscrewing_bottle_cap/unscrewing_bottle_cap_env.py`).

## Files

- `ik_rl_pickup.yaml` — auto-loaded for single-arm when present.
- `ik_rl_unscrew_dual.yaml` — auto-loaded for dual-arm scripts when present.
- `ik_rl_pour.yaml` — pour example; pass `--ik-config` explicitly.

Tune `pos` / `quat` (always **wxyz**) in simulation for your scene; values in the repo are placeholders.
