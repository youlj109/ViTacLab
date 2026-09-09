# Third-party dependencies

ViTacLab does **not** vendor these repos. Clone separately when needed.

| Package | Use | Install |
|---------|-----|---------|
| [Taxim](https://github.com/TacTip/Taxim) | Xense `polycalib.npz` fitting | `git clone ... ~/Taxim` then `TAXIM_REPO=~/Taxim` in `scripts/calibration/install_taxim_polycalib.py` / `build_xense_polycalib.py` |

Runtime simulation uses committed ViTacLab code + **local** `xense_lab_data/` assets (see that directory's README).
