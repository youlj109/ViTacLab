# Xense lab render assets (not in Git)

This directory holds **local-only** Taxim / FOTS files for the advisor Xense sensor
(400×700). They are **not** committed to the repository.

## Required files

| File | Purpose |
|------|---------|
| `bg_clean.jpg` | Gel-only background (markers inpainted out) |
| `polycalib.npz` | Taxim height → RGB calibration |
| `marker_rest.npy` | Rest marker centers `(M, 2)` in pixels (220 for 11×20 grid) |

## How to populate

From repository root, with lab mp4/correct.zip under `data/calibration/tactile/` (gitignored):

```bash
# Import real frames + install bg_clean + marker_rest (does not copy polycalib from GelSight)
python3 scripts/calibration/import_advisor_tactile_videos.py --install-bg

# Ball polycalib (6 mm indent video → Taxim fit); see scripts/calibration/install_taxim_polycalib.py
python3 scripts/calibration/build_xense_polycalib.py   # or manual Taxim workflow
python3 scripts/calibration/install_taxim_polycalib.py --src data/calibration/tactile/ball_calib_raw/polycalib.npz
```

Joint fit output (`data/calibration/tactile/fitted_params.json`) is also local-only; scripts load it when present.

See [`docs/VITACSIM_CALIBRATION.md`](../../../../../../../docs/VITACSIM_CALIBRATION.md) for the full Task 2 / Task 3 pipeline.
