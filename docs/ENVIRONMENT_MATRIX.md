# ViTacLab Canonical Environment and Compatibility Matrix

Generated from the current Gym registration source. Static acceptance result: **31 registrations, 31 unique IDs, 31 unique environment/config pairs, and all entry-point classes present**. Runtime columns describe wiring present in source; every row still requires the remote Isaac Sim/GPU acceptance test.

Legend: `yes` means a canonical entry/schema is present; `preset` means the specialized collector has a maintained task preset; `action`/`joint_pos` is the supported offline-policy output path; `no` means the workflow is intentionally not applicable without adding a new task-specific adapter.

| Task ID | Level | Environment entry | Config entry | RSL-RL | RL NPZ collector | Full trajectory | IK / Full-IK | Diffusion Policy | ViTacDP | Offline control |
|---|---|---|---|---|---|---|---|---|---|---|
| `Isaac-ViTac-Forge-GearMesh-Breakable-Direct-v0` | level 0 | `ViTacLab.tasks.direct.medium_gripper.forge_env:ForgeBreakableEnv` | `ViTacLab.tasks.direct.medium_gripper.forge_env_cfg:ForgeTaskGearMeshBreakableCfg` | yes | yes (record or base fallback) | generic action recorder/replay | no | yes; matching camera checkpoint required | yes; matching 2-sensor checkpoint required | action |
| `Isaac-ViTac-Forge-GearMesh-Direct-v0` | level 0 | `ViTacLab.tasks.direct.simple_gripper.forge_env:ForgeEnv` | `ViTacLab.tasks.direct.simple_gripper.forge_env_cfg:ForgeTaskGearMeshCfg` | yes | yes (record or base fallback) | generic action recorder/replay | no | yes; matching camera checkpoint required | yes; matching 2-sensor checkpoint required | joint_pos or action |
| `Isaac-ViTac-Forge-NutThread-Breakable-Direct-v0` | level 0 | `ViTacLab.tasks.direct.medium_gripper.forge_env:ForgeBreakableEnv` | `ViTacLab.tasks.direct.medium_gripper.forge_env_cfg:ForgeTaskNutThreadBreakableCfg` | yes | yes (record or base fallback) | generic action recorder/replay | no | yes; matching camera checkpoint required | yes; matching 2-sensor checkpoint required | action |
| `Isaac-ViTac-Forge-NutThread-Direct-v0` | level 0 | `ViTacLab.tasks.direct.simple_gripper.forge_env:ForgeEnv` | `ViTacLab.tasks.direct.simple_gripper.forge_env_cfg:ForgeTaskNutThreadCfg` | yes | yes (record or base fallback) | generic action recorder/replay | no | yes; matching camera checkpoint required | yes; matching 2-sensor checkpoint required | joint_pos or action |
| `Isaac-ViTac-Forge-PegInsert-Breakable-Direct-v0` | level 0 | `ViTacLab.tasks.direct.medium_gripper.forge_env:ForgeBreakableEnv` | `ViTacLab.tasks.direct.medium_gripper.forge_env_cfg:ForgeTaskPegInsertBreakableCfg` | yes | yes (record or base fallback) | generic action recorder/replay | no | yes; matching camera checkpoint required | yes; matching 2-sensor checkpoint required | action |
| `Isaac-ViTac-Forge-PegInsert-Direct-v0` | level 0 | `ViTacLab.tasks.direct.simple_gripper.forge_env:ForgeEnv` | `ViTacLab.tasks.direct.simple_gripper.forge_env_cfg:ForgeTaskPegInsertCfg` | yes | yes (record or base fallback) | generic action recorder/replay | no | yes; matching camera checkpoint required | yes; matching 2-sensor checkpoint required | joint_pos or action |
| `Isaac-GelsightFinger-FrictionPretrain-Direct-v0` | pretraining | `ViTacLab.tasks.direct.pretraining.friction_pretrain.gelsight_friction_pretrain_env:GelsightFingerFrictionPretrainEnv` | `ViTacLab.tasks.direct.pretraining.friction_pretrain.gelsight_friction_pretrain_env_cfg:GelsightFingerFrictionPretrainEnvCfg` | yes | task-specific pretraining data | generic action recorder/replay | no | no | no | not applicable |
| `Isaac-GelsightFinger-MassPretrain-Direct-v0` | pretraining | `ViTacLab.tasks.direct.pretraining.mass_pretrain.gelsight_mass_pretrain_env:GelsightFingerMassPretrainEnv` | `ViTacLab.tasks.direct.pretraining.mass_pretrain.gelsight_mass_pretrain_env_cfg:GelsightFingerMassPretrainEnvCfg` | yes | task-specific pretraining data | generic action recorder/replay (action is no-op) | no | no | no | not applicable |
| `Isaac-UR10eShadowHand-Repose-Cube-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv` | `ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandEnvCfg` | yes | yes (record or base fallback) | single preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-Repose-Cube-OpenAI-FF-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv` | `ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandOpenAIEnvCfg` | yes | yes (record or base fallback) | single recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-Repose-Cube-Tactile-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv` | `ViTacLab.tasks.direct.simple_dexhand.inhand_manipulation.inhand_manipulation_env_cfg:UR10eShadowHandInHandTactileEnvCfg` | yes | yes (record or base fallback) | single recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-Repose-Cube-Vision-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.shadow_hand.shadow_hand_vision_env:ShadowHandVisionEnv` | `ViTacLab.tasks.direct.simple_dexhand.shadow_hand.shadow_hand_vision_env:ShadowHandVisionEnvCfg` | yes | yes (record or base fallback) | generic action recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-ViTac-Shadow-Hand-Over-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.shadow_hand_over.shadow_hand_over_env:ShadowHandOverEnv` | `ViTacLab.tasks.direct.simple_dexhand.shadow_hand_over.shadow_hand_over_env_cfg:ShadowHandOverEnvCfg` | yes | yes; matching RSL-RL checkpoint | generic action recorder/replay | no | yes; matching one-camera checkpoint required | yes; matching 10-sensor checkpoint required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-BiBlindBinDrop-Direct-v0` | level 3 | `ViTacLab.tasks.direct.difficult_dexhand.bi_blind_bin_drop.bi_blind_bin_drop_env:UR10eDualShadowHandBiBlindBinDropEnv` | `ViTacLab.tasks.direct.difficult_dexhand.bi_blind_bin_drop.bi_blind_bin_drop_env_cfg:UR10eDualShadowHandBiBlindBinDropEnvCfg` | yes | yes (record or base fallback) | dual preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-BiBlindGrasp-Direct-v0` | level 3 | `ViTacLab.tasks.direct.difficult_dexhand.bi_blind_grasp.bi_blind_grasp_env:UR10eDualShadowHandBiBlindGraspEnv` | `ViTacLab.tasks.direct.difficult_dexhand.bi_blind_grasp.bi_blind_grasp_env_cfg:UR10eDualShadowHandBiBlindGraspEnvCfg` | yes | yes (record or base fallback) | dual preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-BiBlindInhand-Direct-v0` | level 3 | `ViTacLab.tasks.direct.difficult_dexhand.bi_blind_inhand.bi_blind_inhand_env:UR10eDualShadowHandBiBlindInhandEnv` | `ViTacLab.tasks.direct.difficult_dexhand.bi_blind_inhand.bi_blind_inhand_env_cfg:UR10eDualShadowHandBiBlindInhandEnvCfg` | yes | yes (record or base fallback) | dual recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-BiBlindPeg-Direct-v0` | level 3 | `ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env:UR10eDualShadowHandBiPegEnv` | `ViTacLab.tasks.direct.difficult_dexhand.bi_blind_peg.bi_blind_peg_env_cfg:UR10eDualShadowHandBiBlindPegEnvCfg` | yes | yes (record or base fallback) | dual recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-BiPeg-Direct-v0` | level 2 | `ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env:UR10eDualShadowHandBiPegEnv` | `ViTacLab.tasks.direct.medium_dexhand.bi_peg.bi_peg_env_cfg:UR10eDualShadowHandBiPegEnvCfg` | yes | yes (record or base fallback) | dual preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-BiStab-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.bi_stab.bi_stab_env:UR10eDualShadowHandBiStabEnv` | `ViTacLab.tasks.direct.simple_dexhand.bi_stab.bi_stab_env_cfg:UR10eDualShadowHandBiStabEnvCfg` | yes | yes (record or base fallback) | dual recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env:UR10eDualShadowHandOverEnv` | `ViTacLab.tasks.direct.simple_dexhand.hand_over.hand_over_env_cfg:UR10eDualShadowHandOverEnvCfg` | yes | yes (record or base fallback) | dual preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-PourDeformable-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.dual_pour_water.ur10e_dual_shadowhand_pour_env:UR10eDualShadowHandPourEnv` | `ViTacLab.tasks.direct.simple_dexhand.dual_pour_water.ur10e_dual_shadowhand_pour_env_cfg:UR10eDualShadowHandPourEnvCfg` | yes | yes (record or base fallback) | dual recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0` | level 2 | `ViTacLab.tasks.direct.medium_dexhand.unscrewing_bottle_cap.unscrewing_bottle_cap_env:UR10eDualShadowHandUnscrewBottleCapEnv` | `ViTacLab.tasks.direct.medium_dexhand.unscrewing_bottle_cap.unscrewing_bottle_cap_env_cfg:UR10eDualShadowHandUnscrewBottleCapEnvCfg` | yes | yes (record or base fallback) | dual preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-BlindClassification-Direct-v0` | level 3 | `ViTacLab.tasks.direct.difficult_dexhand.blind_classification.blind_classification_env:UR10eShadowHandBlindClassificationEnv` | `ViTacLab.tasks.direct.difficult_dexhand.blind_classification.blind_classification_env_cfg:UR10eShadowHandBlindClassificationEnvCfg` | yes | yes (record or base fallback) | single recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-BlindGrasp-Direct-v0` | level 3 | `ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env:UR10eShadowHandBlindGraspEnv` | `ViTacLab.tasks.direct.difficult_dexhand.blind_grasp.blind_grasp_env_cfg:UR10eShadowHandBlindGraspEnvCfg` | yes | yes (record or base fallback) | single preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-BlindGraspReplay-Direct-v0` | level 3 | `ViTacLab.tasks.direct.difficult_dexhand.blind_grasp_replay.blind_grasp_replay_env:UR10eShadowHandBlindGraspReplayEnv` | `ViTacLab.tasks.direct.difficult_dexhand.blind_grasp_replay.blind_grasp_replay_env_cfg:UR10eShadowHandBlindGraspReplayEnvCfg` | yes | yes (record or base fallback) | single preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-BlindRetrieval-Direct-v0` | level 3 | `ViTacLab.tasks.direct.difficult_dexhand.blind_retrieval.blind_retrieval_env:UR10eShadowHandBlindRetrievalEnv` | `ViTacLab.tasks.direct.difficult_dexhand.blind_retrieval.blind_retrieval_env_cfg:UR10eShadowHandBlindRetrievalEnvCfg` | yes | yes (record or base fallback) | single recorder/replay | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-ForgeGearMesh-Direct-v0` | level 2 | `ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv` | `ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeGearMeshEnvCfg` | yes | yes (record or base fallback) | single preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10eShadowHand-ForgeNutThread-Direct-v0` | level 2 | `ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv` | `ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgeNutThreadEnvCfg` | yes | yes (record or base fallback) | single preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0` | level 2 | `ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env:UR10eShadowHandForgeEnv` | `ViTacLab.tasks.direct.medium_dexhand.forge_dexhand.ur10e_shadowhand_forge_env_cfg:UR10eShadowHandForgePegInsertEnvCfg` | yes | yes (record or base fallback) | single preset | no | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |
| `Isaac-UR10eShadowHand-Pickup-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv` | `ViTacLab.tasks.direct.simple_dexhand.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg` | yes | yes (record or base fallback) | single preset | pickup/pour preset | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | joint_pos or action |
| `Isaac-UR10eShadowHand-PourDeformable-Direct-v0` | level 1 | `ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv` | `ViTacLab.tasks.direct.simple_dexhand.pour_water.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg` | yes | yes (record or base fallback) | single preset | pickup/pour preset | yes; matching camera checkpoint required | yes; matching sensor counts/poses required | action |

## Important interpretation

- Every row is tactile-capable under `--enable_cameras` and must provide
  canonical `tactile_pos`, `tactile_normal_force`, `tactile_shear_force`, and
  `tactile_rgb_image` record tensors. Sensor-count profiles are: GelSight
  pretraining = 1, Franka Forge = 2, single Shadow-Hand/UR10e = 5, dual-hand
  (including standalone Shadow-Hand-Over) = 10.
- Tactile capability and policy input are separate contracts. Dense tactile is
  appended to policy observations only when the task configuration explicitly
  enables it; existing observation dimensions remain the default so old
  checkpoints are not silently invalidated.
- A policy family cannot be meaningfully run with an arbitrary checkpoint. Camera count/order, tactile type/count, state dimension, action dimension, and action semantics must match the checkpoint used to train that task.
- `scripts/policy/play_policy.py` merges a task-specific `record` group with the canonical single/dual robot-base fallback. This supplies missing camera/tactile pose fields when the base environment owns those sensors.
- The GelSight mass/friction pretraining tasks keep compact supervised RL
  observations but also expose the same canonical tactile record for testing
  and task-specific data tooling; they are not Diffusion Policy/ViTacDP tasks.
- `Isaac-ViTac-Shadow-Hand-Over-Direct-v0` is standalone rather than UR10e,
  but its ten tactile sensors use the same canonical record field names. It can
  be collected by the generic RSL-RL NPZ collector with a matching checkpoint.
  Its canonical record now also includes a third-person camera and pose, so a
  matching one-camera/ten-tactile Diffusion Policy or ViTacDP checkpoint can be
  deployed. Its full trajectory uses the generic normalized-action recorder.
- Full-trajectory recording/replay is now registry-driven for all 12
  single-UR10e and all 9 dual-UR10e tasks. Replay first consumes canonical
  success signals from `infos`, `obs['record']`, or the environment; Pickup-Z
  and BinDrop-XY checks are fallback criteria only for their matching tasks.
  The generic `record_action.py` / `play_action.py` pair covers Franka,
  standalone ShadowHand and GelSight pretraining action spaces without
  pretending they share UR10e marker IK. The mass-pretraining action is a
  deliberate no-op, so its action trajectory validates data flow rather than
  robot motion.
- `Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0` is fully wired for RSL-RL data
  collection and for Diffusion Policy/ViTacDP inference when the checkpoint
  matches its 60-dimensional dual-arm state/action semantics, camera order,
  and ten tactile sensors.
- "The inference entry accepts a task" does not mean one checkpoint is valid
  for every task. Each task/profile still needs a checkpoint trained with the
  same state/action dimensions and camera/tactile schema.
- For tasks listed as `action`, pass `--policy-output action`. Only use `joint_pos` where the environment implementation prevents `env.step()` from overwriting direct joint targets.

## Per-environment smoke test

Run from the repository root on the remote machine. Replace `<TASK_ID>` with every row above:

```bash
python scripts/zero_agent.py --task <TASK_ID> --num_envs 1 --max-steps 20 --enable_cameras --headless
python scripts/random_agent.py --task <TASK_ID> --num_envs 1 --max-steps 20 --enable_cameras --headless
```

Acceptance: both commands print `[SENSOR-DIAG-PASS]`; the expected 1/2/5/10
sensor keys have `missing=()`; all four tactile record fields have the matching
sensor-axis size and finite values; tactile RGB is nonzero and nonconstant;
20 steps and clean shutdown complete. A no-camera run is only a physics isolation test and does
not satisfy the project tactile requirement.

## Registration audit

```bash
python scripts/audit_project.py --verbose
```

The audit is dependency-free and validates registration IDs, env/config entry
classes, the 31-task tactile sensor-count manifest and implementation markers,
duplicate mappings, version/backup filenames, argparse help,
executable/environment module docstrings, duplicate Python modules, stale
paths, and data writers outside `scripts/data_collection`.
