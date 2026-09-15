# Selected depth_gain=3.2: real / ViTacSim / official comparison

The FPS numbers below describe the previous serial-marker implementation.
The subsequent batched-marker change preserves all checked image pixels and
measures 356.5 FPS versus 39.6 FPS for the old path in one process. See
`docs/XENSE_BATCHED_MARKER_RENDERING.md`; updated images are under
`logs/xense_batched_markers_official_20260915/`.

Settings: k_ref=1350 N/m, depth_gain=3.2, fixed flat-contact optics. Six loads:
G010/G030/G060/G110/G160/G210. Both renderers receive the same corrected height
maps; official input converts positive metres to negative millimetres. Official
Xensim 1.0.0 uses smooth_norm=8, rgb_gain=1.3, nstep=3, with markers enabled.

## FPS (RTX 5090, one 400×700 sensor)

100 warmup calls followed by 100 measured calls per load, sequential benchmarks
without concurrent GPU tests. Aggregate FPS is the inverse mean frame time.

| Load | ViTacSim | Official Xensim |
|---|---:|---:|
| G010 | 37.2 | 44.8 |
| G030 | 37.6 | 49.1 |
| G060 | 37.6 | 45.4 |
| G110 | 37.1 | 46.0 |
| G160 | 39.1 | 45.7 |
| G210 | 38.7 | 46.6 |
| Aggregate | 37.9 | 46.2 |

ViTacSim timing includes Taxim, recomputed FOTS, and measured marker compositing
inside the Isaac Lab 2.3.2 container. CUDA is synchronized after each call;
input heights are already on GPU. It excludes PhysX, camera capture, force
reconstruction, host RGB transfer, and file I/O. The benchmark drives FOTS with
the corrected height and no external shear. Comparison images retain the saved
marker displacement fields from the existing normal-force run; the benchmark
is therefore a renderer workload measurement rather than an exact full sensor
update of those images.

Official timing includes FemSensor.step(nstep=3) and get_image() in the installed
standalone Python 3.11 environment, using CPU wall time around the SDK calls.
There is no explicit SDK GPU synchronization call; get_image returns host RGB.
The comparison uses its warmed stationary-contact image at each load. This
includes official FEM but excludes Isaac/PhysX/camera/file I/O. The scopes are
different and must not be reported as end-to-end simulation throughput.

Real references are saved frames, so their FPS is labeled N/A; no sensor capture
rate was measured. The short-warmup initial trial was superseded because its
first case had large initialization outliers (median 27.8 ms, p95 562.8 ms).

## Artifacts

Under `logs/xense_gain320_comparison_20260915/`:

- `comparison/all_masses_real_ours_official.png`: complete tactile images, FPS labels.
- `comparison/all_masses_contact_crops.png`: contact crops, FPS labels.
- `comparison/metrics.json`: all image metrics and per-load benchmark records.
- `input/vitacsim_fps.json`: ViTacSim timing scope and distributions.
- `comparison/G010/` through `comparison/G210/`: individual comparison images.

Reproduction scripts: `benchmark_xense_current_comparison.py --headless
--iterations 100 --warmup 100` inside Isaac Lab, then
`compare_official_xensim.py --input-root logs/xense_gain320_comparison_20260915/input
--output logs/xense_gain320_comparison_20260915/comparison --smooth-norm 8
--rgb-gain 1.3 --vitacsim-fps logs/xense_gain320_comparison_20260915/input/vitacsim_fps.json
--benchmark-iterations 100 --benchmark-warmup 100` in the official environment.
