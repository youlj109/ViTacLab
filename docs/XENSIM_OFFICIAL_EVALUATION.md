# Official XenseSim 1.0.0 Evaluation

## Scope

The evaluated package is the Xense-provided `xensim-1.0.0-cp311-cp311-linux_x86_64.whl`
from the local `xensim (1).zip` distribution. The wheel uses the non-commercial
Xense Simulator SDK binary runtime license, so the wheel and its bundled models
must not be committed to this repository. The local extraction is ignored at:

```text
third_party/xensim_official_v1.0.0/
```

The older `isaac_xensesim4.5_dist_v1.1.0` extension is CPython 3.10 / Isaac Sim
4.5 specific. The newer wheel is CPython 3.11 and standalone, and is therefore
the package used for all results below.

## Installation

```bash
conda create -y -n xensim-official-1.0 python=3.11 pip
conda run -n xensim-official-1.0 python -m pip install \
  third_party/xensim_official_v1.0.0/xensim-1.0.0-cp311-cp311-linux_x86_64.whl
```

Both API import and rendering were verified. The wheel also imports under the
Isaac Lab 2.3.2 image (Isaac Sim 5.1, Python 3.11). A stricter smoke test starts
Isaac Sim first and then creates and renders a `FemSensor` in the same process:

```bash
/isaac-sim/python.sh scripts/calibration/smoke_official_xensim_in_isaac.py \
  --vendor-site /path/to/xensim/site-packages
```

## Same-input nut comparison

The comparison sends exactly the same saved ViTacSim force-corrected height maps
to both renderers. For XenseSim, penetration is converted from positive metres
to negative millimetres as required by its API. The six peaks are 0.02, 0.06,
0.12, 0.22, 0.32, and 0.42 mm for G010 through G210. Official parameters were
swept over `smooth_norm={2,5,8}` and `rgb_gain={0.7,1.0,1.3}`. The selected visual
comparison uses `smooth_norm=8`, `rgb_gain=1.3`, and `nstep=3`.

```bash
conda run --no-capture-output -n xensim-official-1.0 \
  python scripts/calibration/compare_official_xensim.py \
  --smooth-norm 8 --rgb-gain 1.3 \
  --output logs/xensim_official_comparison_best
```

All MAE values below are on RGB values in the 0–255 domain and averaged over
the six mass cases.

| Measurement | ViTacSim | Official XenseSim | Interpretation |
| --- | ---: | ---: | --- |
| Full RGB MAE | **3.890** | 25.426 | ViTacSim matches this physical sensor's resting background and marker layout much better. |
| Background-subtracted full-frame MAE | 2.574 | **2.351** | Official contact response is slightly closer after removing each renderer's background. |
| Background-subtracted 200×200 contact ROI MAE | 6.382 | **4.615** | Official FEM gives more realistic gel spreading and suppresses the overly sharp hexagonal contour. |
| Top-10 marker-magnitude MAE (px) | **0.346** | 1.352 | ViTacSim's calibrated local marker motion is substantially closer to the real tracker. |
| Maximum marker-magnitude MAE (px) | **0.417** | 1.793 | Official FEM under-predicts marker motion for these shallow corrected depths. |

The full-frame metric is background dominated and must not be used alone to
judge contact fidelity. The contact ROI and marker measurements isolate two
different behaviours: official XenseSim currently wins contact deformation and
optical softness; ViTacSim wins sensor-specific appearance and marker motion.

Machine-local visual and numeric artifacts are written to:

```text
logs/xensim_official_comparison_best/all_masses_real_ours_official.png
logs/xensim_official_comparison_best/all_masses_contact_crops.png
logs/xensim_official_comparison_best/metrics.json
logs/xensim_official_comparison_best/summary.json
```

## Hybrid optical-response improvement

The official `get_depth()` result does not justify replacing the calibrated
force-to-depth reconstruction. After matching its 175×100 output resolution,
the official indentation and ViTacSim input have full-frame correlations of
0.991–0.995, and their peak-depth ratios are 1.000 for all six loads. Rendering
the official depth through the existing Taxim path also leaves the contact ROI
MAE essentially unchanged. The useful difference is therefore the official
renderer's coarse mesh/normal interpolation and optical spreading, not a new
load-to-depth curve.

ViTacSim now optionally downsamples only the Taxim RGB contact response,
performs repeated 3×3 averaging on that coarse response, and bilinearly returns
it to camera resolution. The corrected height map and the height field used by
the marker model are untouched. The Advisor defaults selected from the
marker-free sweep are:

```text
taxim_response_mesh_scale = 4
taxim_response_mesh_smooth_iterations = 6
taxim_response_mesh_blend = 1.0
```

The blend sweep (`0.50 / 0.75 / 1.00`) selected `1.00`: its marker-free mean
contact-response MAE is 4.365 and SSIM is 0.770, versus MAE 4.831 and SSIM
0.708 before mesh interpolation. In the independent complete-image comparison,
the hybrid reduces full RGB MAE from 3.890 to 3.820, background-subtracted MAE
from 2.574 to 2.497, and 200×200 contact-ROI MAE from 6.382 to 5.844. Marker
errors are exactly unchanged because marker deformation bypasses this filter.

Use `--baseline-root` to generate the four-column real / previous / hybrid /
official comparison:

```bash
conda run --no-capture-output -n xensim-official-1.0 \
  python scripts/calibration/compare_official_xensim.py \
  --input-root logs/xense_hybrid_mesh_default \
  --baseline-root logs/xense_marker_crisper \
  --smooth-norm 8 --rgb-gain 1.3 \
  --output logs/xense_hybrid_fourway_comparison
```

## Runtime observations

On the RTX 5090 workstation, official `FemSensor.step(..., nstep=3)` plus
`get_image()` took about 22.7 ms for one 400×700 sensor frame. It includes the
official FEM solve and marker rendering.

The ViTacSim renderer benchmark intentionally excludes Isaac camera capture,
PhysX, and force-to-depth reconstruction. Its Taxim-only path took about
6.1 ms/frame after extended warm-up. The current Taxim + FOTS marker path took
about 137 ms/frame in the same stable single-frame test. A short batching run
reached about 825 sensor-frames/s at batch 32 without markers, but only about
31.5 sensor-frames/s with markers. This identifies FOTS compositing—not Taxim
RGB synthesis—as the current ViTacSim rendering bottleneck.

Run the benchmark in the Isaac Lab container with:

```bash
/isaac-sim/python.sh scripts/calibration/benchmark_vitacsim_renderer.py \
  --iterations 100 --warmup 30 --batch-sizes 1
```

## Recommendation

Do not replace ViTacSim wholesale. For a single sensor and offline image
fidelity, official XenseSim's FEM contact spreading is currently better. For
ViTacLab training and reproducible research, the existing implementation remains
the better primary backend because it is integrated with batched Isaac Lab
environments, uses the measured lab background and marker map, exposes the
force-correction model, and is modifiable. The official binary is non-commercial,
opaque, sensor-model specific, and currently exposes no batched `FemSensor` API.

The official implementation is most useful as an offline optical reference.
The hybrid response filter captures part of its soft contact appearance while
retaining ViTacSim's real background, calibrated markers, force reconstruction,
and batched Isaac integration. Further fitting should target spatial optical
response and illumination; it should not change `k_ref` or introduce a
load-dependent depth law merely to imitate official RGB output.
