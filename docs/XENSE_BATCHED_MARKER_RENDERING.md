# Batched measured-marker rendering

The measured Xense marker path previously sampled and composited 220 patches
in a Python loop. It read two GPU coordinates per marker via `.item()` and
launched one `grid_sample` per patch. This made the small marker overlay more
expensive than the rest of the renderer.

The new path caches local pixel offsets and constructs all translated sampling
grids on GPU, samples the 220 templates in one call, and writes the disjoint
patches in one indexed operation. It no longer reads marker coordinates one
point at a time. A conservative footprint-overlap check still requires a scalar
device-to-host decision; the implementation is not fully synchronization-free.

Overlapping footprints use the original ordered implementation. Multiplying
all transmissions together and rounding only once would change results because
the original rounds to uint8 after every marker. Retaining the ordered fallback
also prevents conflicting indexed writes. Offscreen coordinates are masked;
noncontiguous image inputs are supported. Gaussian and disk marker modes,
FOTS force/motion calculations, optics, depth_gain=3.2, and k_ref=1350 are unchanged.

## Verification

`scripts/calibration/test_measured_marker_offline.py` checks CPU and CUDA:

- Exact equality for all six existing real-load marker-free images and saved motions.
- Resting reference reconstruction, changed illumination, input immutability.
- Fractional translations, image-edge clipping, wholly offscreen markers.
- Overlapping footprints and noncontiguous image inputs.

The full renderer benchmark additionally compared original and optimized
outputs with FOTS recomputed for every load: **all RGB pixels identical**.

## Same-process performance

RTX 5090, batch=1, 400×700, warmup=100, measured iterations=100 for each path and
load. Wall time includes a CUDA synchronization after every render. Both paths
ran sequentially in the same Isaac Lab 2.3.2 container and process.

| Load | Original renderer FPS | Batched-marker renderer FPS |
|---|---:|---:|
| G010 | 40.0 | 357.7 |
| G030 | 39.6 | 355.9 |
| G060 | 39.5 | 353.5 |
| G110 | 39.5 | 357.3 |
| G160 | 39.5 | 356.3 |
| G210 | 39.6 | 358.4 |
| Aggregate | 39.6 | 356.5 |

Aggregate mean frame time decreases from approximately **25.24 ms to 2.805 ms**,
an approximately **9.0×** renderer speedup. This includes Taxim, recomputed FOTS,
and marker compositing but excludes camera capture, PhysX, force reconstruction,
host RGB transfer, and image saving. It is not end-to-end environment FPS. The
sensor update may render both base and corrected images. Overlap-heavy inputs
use the slower fallback; batch sizes above one were not benchmarked here.

Artifacts:

These generated logs and local calibration fixtures are not committed. The
offline regression script requires the verified Advisor background pair,
marker positions, and the six saved inputs under
`logs/xense_gain320_comparison_20260915/input/normal_force/`.

- `logs/xense_batched_markers_20260915/input/vitacsim_fps.json`
- `logs/xense_batched_markers_20260915.log`
- `logs/xense_batched_markers_official_20260915/all_masses_real_ours_official.png`
- `logs/xense_batched_markers_official_20260915/metrics.json`

The fresh official Xensim 1.0.0 run measured **60.1 FPS** in aggregate with
`smooth_norm=8`, `rgb_gain=1.3`, and `nstep=3` (100 warmup and 100 measured
calls per load). Its scope includes FEM and host RGB output, unlike the
GPU-resident ViTacSim timing above. This is not a matched end-to-end simulator
speed comparison. Historical official timings in
[the previous report](XENSE_GAIN320_FPS_COMPARISON.md) are from a separate run.

Reproduce inside Isaac Lab:

```bash
/isaac-sim/python.sh scripts/calibration/test_measured_marker_offline.py
/isaac-sim/python.sh scripts/calibration/benchmark_xense_current_comparison.py \
  --headless --warmup 100 --iterations 100 --compare-serial \
  --output logs/xense_batched_markers_20260915/input
```
