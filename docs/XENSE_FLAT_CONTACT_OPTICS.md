# Xense flat-contact brightness correction (2026-09-15)

## Current visual trial: fixed stiffness, increased depth gain

Selected setting: the user chose **depth_gain=3.2**; **k_ref=1350 N/m** remains
fixed. The default has been updated accordingly. Current three-way real /
ViTacSim / official Xensim comparison and per-load FPS metadata are under
`logs/xense_gain320_comparison_20260915/`. FPS labels refer to single-sensor
renderer benchmarks, not Isaac/PhysX end-to-end throughput or camera frame rate.

Earlier follow-up: default **depth_gain=2.0**, still **k_ref=1350 N/m**, after
the user requested further depth. The six-load comparison at gains
1.6/2.0/2.4/2.8/3.2 is in `logs/xense_fixed1350_gain200_preview_20260915/`.
G010/G210 default peaks are approximately 0.0545/1.1444 mm. Optical parameters
are unchanged; this is a saved-depth visual trial, not physical-depth validation.

Previous follow-up: the user requested a stronger imprint again. The default was
now **depth_gain=1.6** with **k_ref=1350 N/m** unchanged. A six-load replay
compares gains 1.2/1.4/1.6/1.8/2.0 at
`logs/xense_fixed1350_gain160_preview_20260915/`. G010/G210 peaks at gain=1.6
are approximately 0.0436/0.9155 mm. Optical settings and marker motion are
unchanged. Reproduce with `preview_xense_deeper_contact.py --headless --gains
1.2 1.4 1.6 1.8 2.0 --output logs/xense_fixed1350_gain160_preview_20260915`.

The previous trial held **k_ref=1350 N/m** and previewed depth_gain
1.0/1.1/1.2/1.3/1.4 on all six saved nut depths. Its default was **1.2**,
a moderate 20% increase from the stiffness-fit result; this is a visual trial,
not a claim of lower image error. Optics and saved marker motion are fixed.
G010/G210 peak depths at gain=1.2 are approximately 0.0327/0.6866 mm.

Run `scripts/calibration/preview_xense_deeper_contact.py --headless` in Isaac Lab.
Full and cropped real-versus-five-gain panels, individual images, height maps,
and peak-depth metadata are under `logs/xense_fixed1350_deeper_preview_20260915`.
This trial re-renders saved depths and does not rerun PhysX. Earlier optimization
results below still refer to their stated gains and remain unchanged.

## Joint stiffness / depth-gain search

A 13×13 grid of **169 parameter pairs** was evaluated against all six real nut
frames with fixed optics: k_ref in {1000,1100,1200,1250,1300,1350,1400,1450,1500,
1600,1700,1800,1840} N/m; depth_gain from 0.8 to 1.4 in steps of 0.05.
The existing **1350 N/m, 1.0** pair remains the lowest-error result
(normalized RGB RMSE 0.8779505). No default change is justified by this scan.

The two-dimensional error plot shows a broad diagonal band, not a sharply
identified stiffness/gain pair. This scan replays saved corrected depths;
equal depth_gain/k_ref ratios share predictions. It does not independently
rerun the force reconstruction for every parameter pair. Code inspection also
found k_ref-dependent force-sample validity and stiffness caps, so ratio
equivalence is conditional on unchanged active samples and caps.

Full G210 contact simulations additionally compared (1350,1.0) with the
equal-ratio pair (1485,1.1). Both had valid contact, 28 force/depth samples,
18 retained samples, and mean PhysX force 1.406962875 N. Both produced a
0.57218844 mm peak; maximum depth difference was 5.82e-8 mm and complete
corrected RGB MAE was exactly 0. This confirms equivalence for this tested
load, not for all possible contact/cap regimes. The second run is stored at
`logs/xense_joint_equivalence_physical_20260915/normal_force/G210/vitacsim/`.

Artifacts: `logs/xense_joint_kref_depth_20260915/joint_error_surface.png`,
`joint_grid.json`, `all_masses_full.png`, and `gain_fit.json` in that directory.
The complete image comparison is unchanged because the selected parameters
equal the previous defaults. This is a negative optimization result, not a
new fidelity improvement.

```bash
/isaac-sim/python.sh scripts/calibration/replay_xense_depth_gain.py --headless \
  --fit-to-real --selection all --baseline-k-ref 1350 \
  --candidate-k-refs 1000 1100 1200 1250 1300 1350 1400 1450 1500 1600 1700 1800 1840 \
  --candidate-depth-gains 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 \
  --output logs/xense_joint_kref_depth_20260915
```

## Earlier stiffness fit (before the selected depth_gain=3.2 trial)

Defaults at this stage: **k_ref=1350 N/m, depth_gain=1.0**. The configurable global
depth_gain is retained. A 19-value stiffness scan from 700 to 2200 N/m plus the
previous equivalent stiffness (1840/1.4=1314.29 N/m) selected 1350 with fixed
optics. Fitting uses all six marker-excluded real contact frames; this does not
independently measure the gel's Young modulus or validate new objects.

Mean normalized RGB RMSE changes from 0.8780866 to 0.8779505, only **0.0155%**.
The equivalent depth multiplier relative to the saved k=1840, gain=1 inputs
is 1840/1350=1.36296. Thus this refinement mostly moves the calibrated scale
into stiffness; it is not a significant new visual improvement. The image
objective is nearly flat around 1300–1350 N/m. It cannot separately identify
k_ref and depth_gain, and a more compliant gel alone cannot fix spatial shape
or optical-model mismatch.

Outputs: `logs/xense_kref_real_fit_20260915/all_masses_full.png`,
`all_masses_crops.png`, and `gain_fit.json` in the same folder. These are
saved-depth replays with the existing marker displacement arrays retained.

An additional full Isaac Lab Docker G210 contact run verified that the demo
uses k_ref=1350 and depth_gain=1.0: `contact_valid=true`, final peak corrected
depth 0.57219 mm (replay approximately 0.57219 mm), all height samples finite.
Artifacts: `logs/xense_kref1350_physical_20260915/normal_force/G210/vitacsim/`.

```bash
/isaac-sim/python.sh scripts/calibration/replay_xense_depth_gain.py --headless \
  --fit-to-real --selection all --baseline-gain 1.4 --target-depth-gain 1.0 \
  --candidate-k-refs 700 900 1100 1150 1200 1225 1250 1275 1300 1325 1350 1375 1400 1425 1450 1500 1600 1800 2200 \
  --output logs/xense_kref_real_fit_20260915
```

## Previous real-image gain fit

The previous Advisor default was **depth_gain=1.4**, selected from 54 combinations
of global depth gain (0.8–2.0) and polynomial RGB response gain (1.0–2.5).
The selected RGB gain remains **1.0**; all other optical parameters and k_ref
are unchanged. This fit uses all six real frames, with marker pixels excluded
and off-contact illumination aligned. It is an in-sample fit, not independent
validation. The earlier three-load selection of depth 1.6 did not improve its
reserved loads; the subsequent all-load fit is explicitly a different protocol.

| Load | RGB RMSE at depth_gain=1.1 | Selected depth_gain=1.4 |
|---|---:|---:|
| G010 | 2.494 | 2.503 |
| G030 | 3.711 | 3.788 |
| G060 | 4.895 | 4.834 |
| G110 | 5.689 | 5.605 |
| G160 | 5.755 | 5.713 |
| G210 | 7.030 | 6.893 |

Mean normalized RGB RMSE improves only from 0.88118 to 0.87809 (0.35%).
The medium/high-load imprint becomes stronger, but the two lightest cases get
slightly worse. A scalar depth gain cannot remove remaining spatial/color
response mismatch. Do not describe this as a large fidelity improvement or as
a measured physical indentation depth. Corrected depth peaks are approximately
0.028, 0.084, 0.168, 0.308, 0.448, and 0.588 mm.

Results: `logs/xense_joint_real_fit_20260915/all_masses_full.png`,
`all_masses_crops.png`, and `gain_fit.json` in that directory.

```bash
/isaac-sim/python.sh scripts/calibration/replay_xense_depth_gain.py --headless \
  --fit-to-real --selection all --baseline-gain 1.1 \
  --candidate-gains 0.8 1.0 1.1 1.2 1.3 1.4 1.6 1.8 2.0 \
  --rgb-response-gains 1.0 1.25 1.5 1.75 2.0 2.5 \
  --output logs/xense_joint_real_fit_20260915
```

## Earlier brightness correction and depth preview

Follow-up: the Advisor normal-force default `depth_gain` is now **1.1**, at the
user's request for a slightly deeper imprint after accepting the corrected
colors. `k_ref=1840 N/m` and the optical parameters below are unchanged. Replay
with `scripts/calibration/replay_xense_depth_gain.py --headless`; outputs are in
`logs/xense_depth_gain110_20260915`. Measurements below describe the preceding
gain=1.0 optical validation, not the deeper replay.

The Advisor nut images had a broad dark contact region. Rendering the six saved
physical height maps in Isaac Lab 2.3.2 isolated the problem before marker
compositing: even the pure polynomial table produced negative mean contact
luminance. Reversing indentation did not remove it and reversed directional
colors. Production load/chroma gains amplified the negative response.

At the nut location, the table's zero-slope response is approximately
(-5.5, -8.8, -5.6) RGB levels. The 50 training ball images themselves have a
mean center luminance response near -8 levels. Thus this is not evidence that
the calibration subtraction sign is reversed. The sphere-center response is
being transferred to a large flat nut face, where a slope-only model cannot
distinguish the different contact geometry.

## Implemented correction

For the Advisor normal-force nut profile, subtract the polynomial table's
zero-slope response at each pixel before contact masking and optical gains:

`contact_rgb = table(slope, direction, x, y) - table(0, 0, x, y)`.

This removes the flat-normal color offset and retains directional color
differences. It is an empirical flat-contact transfer correction, not a new
mechanical or universal optical model. The table, depth sign, k_ref=1840 N/m,
depth_gain=1, and saved marker displacement fields are unchanged. Other profiles
and pure ball replay retain their existing response.

The selected nut profile also uses 14 coarse-response averaging iterations and
directional tint additive 18 (previously 6 and 12). Marker footprints are
measured transmission patches from the verified resting `bg.jpg` and clean
`bg_clean.jpg`, moved by the existing marker displacement model. The raw image
is never used for Taxim calibration or low-frequency illumination.

## Measurements

Optical tuning used G010/G060/G160; G030/G110/G210 were reserved for checking
the selected parameters. Metrics exclude real marker masks. These loads share
the same nut and capture session and are not an independent object benchmark.

| Metric | Before | Corrected |
|---|---:|---:|
| Mean marker-free contact RGB RMSE, six loads | 7.002 | 4.960 |
| Held-out normalized contact RMSE | 1.213 | 0.867 |
| Complete RGB MAE, six loads | 3.808 | 2.486 |
| Background-subtracted full RGB MAE | 2.532 | 2.385 |
| Background-subtracted contact ROI MAE, including markers | 6.083 | 5.058 |

The normalized score divides each frame RMSE by the real response RMS; a blank
contact scores 1.0. The initial smoothing-only candidate scored 1.058 on the
held-out loads and was rejected because fading away the imprint was not an
adequate correction. With identical input depths, official Xensim contact ROI
MAE is still lower (4.615), so this is not a claim of overall superiority.

The zero-reference-only ablation changes G210 mean signed contact luminance
from -17.81 to -0.58 (RGB luma weights 0.2126/0.7152/0.0722). The mask is the
simulated contact support excluding real markers; it is not an independently
registered real-contact segmentation. Real images contain both highlights and
shadows, so positive luminance should not be forced at every contact pixel.

## Reproduction and artifacts

Run inside the Isaac Lab container with the repository mounted and on PYTHONPATH:

```bash
/isaac-sim/python.sh scripts/calibration/diagnose_xense_luminance.py --headless
/isaac-sim/python.sh scripts/calibration/refine_xense_appearance.py --headless \
  --flat-reference --out-dir logs/xense_flat_reference_refined_20260915
/isaac-sim/python.sh scripts/calibration/test_measured_marker_offline.py
```

The normal-force demo automatically selects this profile with `--profile advisor`.
Saved-map previews select it with `render_saved_xense_height.py --flat-contact`.
The replay comparisons retain the original saved marker displacement arrays;
they do not rerun PhysX. The full renderer smoke test computes its own FOTS
motion and is therefore not the same marker-input comparison.

Machine-local outputs:

- `logs/xense_flat_reference_refined_20260915/all_masses_full.png`
- `logs/xense_flat_reference_refined_20260915/all_masses_marker_free.png`
- `logs/xense_flat_reference_refined_20260915/fit_report.json`
- `logs/xense_flat_reference_official_20260915/all_masses_real_previous_hybrid_official.png`
- `logs/xense_flat_reference_official_20260915/metrics.json`

The measured-marker smoke test covers resting reconstruction, integer and
subpixel displacement, offscreen clipping, transmission under changed lighting,
and input immutability. Shape fidelity, high-load halo strength, pose alignment,
and displacement-field accuracy still require further work.
