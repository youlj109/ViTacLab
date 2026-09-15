# Six-load Xense tactile image comparison

Date: 2026-09-15. Selected ViTacSim settings: k_ref=1350 N/m, depth_gain=3.2.
Columns are Real, ViTacSim, the repository TacSL depth-only replay, and official
Xensim 1.0.0. All images include markers.

[Complete 400×700 images for all six loads](media/xense-six-load/all-loads-full.png) ·
[Contact crops](media/xense-six-load/all-loads-crops.png) ·
[Machine-readable MAE and RMSE](media/xense-six-load/metrics.json)

## Background-subtracted contact-region RGB RMSE

| Load | ViTacSim | TacSL depth-only replay | Official Xensim |
|---|---:|---:|---:|
| G010 | 7.902 | 11.498 | 4.609 |
| G030 | 8.970 | 11.868 | 8.297 |
| G060 | 8.993 | 10.316 | 10.770 |
| G110 | 9.608 | 10.158 | 15.040 |
| G160 | 10.029 | 10.279 | 17.893 |
| G210 | 11.331 | 11.369 | 18.555 |
| Mean | 9.472 | 10.915 | 12.528 |

## Full-frame RGB RMSE

| Load | ViTacSim | TacSL depth-only replay | Official Xensim |
|---|---:|---:|---:|
| G010 | 4.401 | 5.583 | 33.198 |
| G030 | 4.760 | 5.729 | 33.288 |
| G060 | 4.795 | 5.216 | 33.334 |
| G110 | 5.006 | 5.182 | 33.483 |
| G160 | 5.186 | 5.266 | 33.591 |
| G210 | 5.792 | 5.803 | 33.538 |
| Mean | 4.990 | 5.463 | 33.405 |

Units are RGB intensity on the 0–255 scale, not newtons or millimetres.
Mean is the arithmetic mean of six per-case RMSE values, not pooled RMSE.
G010/G030/G060/G110/G160/G210 designate nominal masses 10/30/60/110/160/210 g.
The official renderer's large full-frame error includes its different resting
background; it must not be interpreted as contact-response error alone.

## Metric protocol

For each method m, delta_m = float(contact_m) - float(no_contact_m).
RMSE = sqrt(mean((prediction - real)^2)); MAE = mean(abs(prediction - real)).
Compute over all three RGB channels, including markers. Full RGB uses all
400×700 pixels without background subtraction. Full delta RGB uses all pixels
after subtraction. ROI delta RGB uses a fixed 220×220 region, x=[90,310),
y=[240,460), identically for all loads and methods. The fixed ROI differs from
the earlier contact-centered 200×200 comparison, so those ROI numbers differ.

No per-image registration, gain fitting, histogram matching, marker removal, or
exposure normalization is performed during this export. The archive includes
all 24 source RGB images and four background images; these suffice to recompute
every reported metric independently. Individual files are under
`media/xense-six-load/G010/` through `G210/`, and `backgrounds/`.

## Baselines and limitations

- ViTacSim uses the accepted optical settings and gain=3.2 saved-depth replay;
  saved marker displacement is retained.
- TacSL is the repository's sensor_mode=tacsl depth-only implementation, not an
  independently installed upstream package. The six previously saved projected,
  uncorrected camera depth maps and marker motions are replayed through current
  shared Taxim optics and measured markers. No force correction or depth_gain
  is applied to this baseline. Its background is rendered from zero depth and
  resting markers. This controls the optical profile while retaining the older
  physical captures; it is not a fresh matched six-load physics experiment.
- Official Xensim 1.0.0 uses smooth_norm=8, rgb_gain=1.3, nstep=3 and receives
  ViTacSim's corrected depth converted from positive metres to negative mm.
  This compares optical/FEM rendering under supplied geometry, not an independent
  official reconstruction from raw PhysX contacts.
- The real background is the verified Advisor bg.jpg with markers. ViTacSim and
  TacSL share measured marker appearance derived from that real background;
  official retains its own marker/background model.
- These six real loads were used for tuning. This is an in-sample diagnostic,
  not held-out validation. Official is better on the two lightest loads under
  ROI delta RMSE. ViTacSim improves the other four, but differences from TacSL
  at G210 are small.
- The manuscript's force RMSE and parallel throughput experiments are separate.
  No new end-to-end FPS or TacSL timing is claimed here.

## Reproduction and provenance

Run from the repository root:

```bash
# Inside the configured Isaac Lab 2.3.2 container, with local calibration installed:
/isaac-sim/python.sh scripts/calibration/publish_xense_six_load_results.py --render-tacsl --headless
# NumPy/OpenCV environment; exports the committed images and metrics:
python scripts/calibration/publish_xense_six_load_results.py
# Verify every metric using only the committed images (no Isaac/SDK/local logs):
python scripts/calibration/publish_xense_six_load_results.py --verify
```

Local replay inputs:
- TacSL: `logs/vitacsim_tacsl_compare_20260914/normal_force/<load>/tacsl/`;
  `tactile_height_depth_projected.npy`, `tactile_marker_displacement.npy`.
- ViTacSim and official contact RGB: `logs/xense_batched_markers_official_20260915/<load>/`.
- ViTacSim resting RGB: `logs/xense_batched_markers_20260915/input/normal_force/no_contact/vitacsim/tactile_rgb.png`.
- Official resting RGB: `logs/xense_batched_markers_official_20260915/official_no_contact.png`.
- Verified real background: `data/calibration/tactile/advisor_processed/bg.jpg`.
- TacSL replay output: `logs/xense_tacsl_current_optics_20260915/`.

Local physical inputs, calibration assets, and vendor SDK are not included in
this snapshot. Render regeneration requires them; metric recomputation from
the committed RGB images does not.
