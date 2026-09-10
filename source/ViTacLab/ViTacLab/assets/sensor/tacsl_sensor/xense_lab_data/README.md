# Xense lab render assets (not in Git)

This directory holds **local-only** Taxim / FOTS files for the advisor Xense sensor
(400×700). They are **not** committed to the repository.

## Required files

| File | Purpose |
|------|---------|
| `bg_clean.jpg` | Gel-only background (markers inpainted out); also used as the low-frequency illumination reference |
| `polycalib.npz` | Taxim height → RGB calibration |
| `marker_rest.npy` | Rest marker centers `(M, 2)` in pixels (220 for 11×20 grid) |

## How to populate

From repository root, with the full ball video at
`data/calibration/file-000.mp4` (gitignored):

```bash
# Import real frames + install bg_clean + marker_rest (does not copy polycalib from GelSight)
python3 scripts/calibration/import_advisor_tactile_videos.py --install-bg

# Select 50 training + 50 temporally separated validation frames, remove the
# complete marker cores/fringes, fit Taxim in RGB order, and install the result.
python3 scripts/calibration/build_xense_polycalib.py \
  --video data/calibration/file-000.mp4 \
  --num-ball 50 --num-validation 50 \
  --taxim-repo third_party/Taxim
python3 scripts/calibration/install_taxim_polycalib.py \
  --polycalib data/calibration/tactile/ball_calib_raw/polycalib.npz \
  --bg data/calibration/tactile/advisor_processed/bg_clean.jpg
```

The fit input is always marker-free. `advisor_processed/bg.jpg` is the verified
no-contact image **with** markers and `advisor_processed/bg_clean.jpg` is its
paired marker-free image. `ball_calib_raw/marker_mask_report.json` records the
post-inpaint residual-marker check, while
`ball_calib_raw_validation/` is never used to fit the polynomial table.

Joint fit output (`data/calibration/tactile/fitted_params.json`) is also local-only; scripts load it when present.

Do not use a raw no-contact frame as the illumination reference. Gaussian
low-pass filtering does not fully remove the printed marker grid, so the
renderer produces gray marker halos before adding the simulated markers.

## Force-corrected render amplitude

The advisor M2-nut profile keeps the measured gel reference stiffness at
`k_ref=66` and uses `corrected_force_render_depth_gain=0.20`. The gain is
applied only after the sparse-point robust force/depth ratio (20% low and 20%
high samples trimmed), uniformly across the complete dense height map. It does
not blend the load-invariant raw camera depth back in and therefore preserves
all relative depth relationships.

To inspect alternative gains without rerunning PhysX, render the saved
marker-free height maps with:

```bash
../IsaacLab/isaaclab.sh -p scripts/calibration/sweep_xense_force_depth_gain.py \
  --headless --enable_cameras --device cuda:0 \
  --gains 0.10 0.15 0.20 0.25 0.30 0.35
```

See [`docs/VITACSIM_CALIBRATION.md`](../../../../../../../docs/VITACSIM_CALIBRATION.md) for the full Task 2 / Task 3 pipeline.
