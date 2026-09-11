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

The advisor M2-nut profile uses the recalibrated effective point stiffness
`k_ref=1840 N/m` and `corrected_force_render_depth_gain=1.0`. Each sparse-point
indentation is `force/k_ref`; the lowest and highest 20% of force/depth ratios
are discarded, the middle 60% are averaged, and that single robust ratio scales
the complete dense height map. The global `depth_gain` remains configurable,
but its Advisor default is the neutral value `1.0`; there is no load-dependent
exponent.

The Advisor contact collider keeps the nominal 3.8 mm M2 width but uses a
2.8 mm effective contact opening: the threaded/chamfered part inside the
nominal 2.0 mm hole does not leave a full-face gel imprint.  Together with the
fitted centered depth projection scale of 1.7, this narrows the simulated ring
wall while matching the real outer footprint.  Both values remain available
as CLI overrides for geometry sweeps.

## Marker appearance

Marker geometry is fitted from the verified no-contact pair: `bg.jpg` with
markers minus `bg_clean.jpg` without markers, sampled at all 220 measured
`marker_rest.npy` positions. The Xense renderer uses a dark-blue anisotropic
Gaussian (`sigma_x=2.6 px`, `sigma_y=2.9 px`, alpha `0.60`) instead of the old
hard radius-2 disk. This is a lightly sharpened version of the direct template
fit: it keeps the real marker scale while making the center and boundary more
legible. The overlay is still applied only after marker-free Taxim optical
rendering.

To preview the new stiffness from the legacy `k_ref=66 N/m` saved maps without
rerunning PhysX, use the mathematically equivalent ratio `66/1840=0.03587`:

```bash
../IsaacLab/isaaclab.sh -p scripts/calibration/sweep_xense_force_depth_gain.py \
  --headless --enable_cameras --device cuda:0 \
  --gains 0.03587
```

See [`docs/VITACSIM_CALIBRATION.md`](../../../../../../../docs/VITACSIM_CALIBRATION.md) for the full Task 2 / Task 3 pipeline.
