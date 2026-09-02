# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared I/O for ViTacSim tactile RGB + marker calibration (sim vs real)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CALIB_SCHEMA = "vitacsim_tactile_calib_v1"
CALIB_SCHEMA_ADVISOR = "vitacsim_tactile_calib_advisor_v1"
WEIGHT_CASES = ("W200", "W100", "W050", "W020", "W010")
LATERAL_W100_FX = (0.0, 0.05, 0.1, 0.15, 0.2)

# Advisor real captures: M2 nut + suspended masses (mp4 from lab).
ADVISOR_WEIGHT_CASES = ("G010", "G030", "G060", "G110", "G160", "G210")
ADVISOR_MASS_G = {
    "G010": 10,
    "G030": 30,
    "G060": 60,
    "G110": 110,
    "G160": 160,
    "G210": 210,
}
# Lab real sensor per 《共谋大业》: Xense marker layout (not GelSight black-dot grid).
ADVISOR_MARKER_PATTERN = "xense"
ADVISOR_SENSOR_LABEL = "xense_lab"
# Native advisor mp4 resolution (width, height).
XENSE_LAB_HW = (400, 700)
# Sim uses same case ids as real (M2 nut + matching mass).
ADVISOR_REAL_TO_SIM_NF: dict[str, str] = {cid: cid for cid in ADVISOR_WEIGHT_CASES}
GELSIGHT_R15_HW = (240, 320)  # (width, height) for legacy cylinder validation


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p / "source" / "ViTacLab").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def default_sim_root() -> Path:
    return repo_root() / "logs" / "vitacsim_calibration" / "sweep"


def default_real_root() -> Path:
    return repo_root() / "data" / "calibration" / "tactile" / "real"


def default_manifest_path() -> Path:
    return default_real_root() / "manifest.json"


@dataclass
class TactileSample:
    case_id: str
    rgb: np.ndarray | None = None
    marker_displacement: np.ndarray | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    source_dir: Path | None = None

    @property
    def marker_disp_max_px(self) -> float:
        if self.marker_displacement is None or self.marker_displacement.size == 0:
            return 0.0
        d = np.asarray(self.marker_displacement, dtype=np.float32)
        if d.ndim != 2 or d.shape[-1] != 2:
            return 0.0
        return float(np.linalg.norm(d, axis=-1).max())

    @property
    def marker_disp_p95_px(self) -> float:
        if self.summary.get("marker_disp_p95_px") is not None:
            return float(self.summary["marker_disp_p95_px"])
        if self.marker_displacement is None or self.marker_displacement.size == 0:
            return 0.0
        d = np.asarray(self.marker_displacement, dtype=np.float32)
        if d.ndim != 2 or d.shape[-1] != 2:
            return 0.0
        return float(np.percentile(np.linalg.norm(d, axis=-1), 95))


def _load_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    try:
        from PIL import Image
    except ImportError:
        return None
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_marker(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path).astype(np.float32)


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_bg_from_path(bg_path: Path) -> np.ndarray | None:
    if not bg_path.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    return np.asarray(Image.open(bg_path).convert("RGB"), dtype=np.float32)


def sim_nf_dir(sim_root: Path, weight_id: str, *, sensor_mode: str = "vitacsim") -> Path:
    return sim_root / "normal_force" / weight_id / sensor_mode


def sim_lateral_dir(sim_root: Path, weight_id: str, fx: float, *, sensor_mode: str = "vitacsim") -> Path:
    tag = _fx_tag(fx)
    return sim_root / "shear_force" / "lateral" / tag / weight_id / sensor_mode


def real_nf_dir(real_root: Path, case_id: str) -> Path:
    return real_root / "normal_force" / case_id


def real_lateral_dir(real_root: Path, weight_id: str, fx: float) -> Path:
    return real_root / "lateral_force" / weight_id / _fx_tag(fx)


def _fx_tag(fx: float) -> str:
    s = f"{fx:.3f}".rstrip("0").rstrip(".")
    return f"Fx{s.replace('-', 'm')}_Fy0"


def _infer_sensor_mode(case_dir: Path) -> str | None:
    parts = case_dir.parts
    if "tacsl" in parts:
        return "tacsl"
    if "vitacsim" in parts:
        return "vitacsim"
    return None


def load_sample_from_dir(case_dir: Path, *, case_id: str, sensor_mode: str | None = None) -> TactileSample:
    mode = sensor_mode or _infer_sensor_mode(case_dir)
    rgb = None
    if mode == "tacsl":
        rgb = _load_rgb(case_dir / "tactile_rgb_depth.png")
    elif mode == "vitacsim":
        rgb = _load_rgb(case_dir / "tactile_rgb_corrected.png")
    if rgb is None:
        rgb = _load_rgb(case_dir / "tactile_rgb.png")
    if rgb is None:
        rgb = _load_rgb(case_dir / "tactile_rgb_corrected.png")
    if rgb is None:
        rgb = _load_rgb(case_dir / "rgb.png")
    disp = _load_marker(case_dir / "tactile_marker_displacement.npy")
    if disp is None:
        disp = _load_marker(case_dir / "marker_displacement.npy")
    return TactileSample(
        case_id=case_id,
        rgb=rgb,
        marker_displacement=disp,
        summary=_load_summary(case_dir / "summary.json"),
        source_dir=case_dir,
    )


def load_nf_cases(
    root: Path,
    *,
    prefix: str,
    weights: Iterable[str] = WEIGHT_CASES,
    include_no_contact: bool = True,
    sensor_mode: str = "vitacsim",
) -> dict[str, TactileSample]:
    out: dict[str, TactileSample] = {}
    if include_no_contact:
        if prefix == "sim":
            case_dir = root / "normal_force" / "no_contact" / sensor_mode
        else:
            case_dir = root / "normal_force" / "no_contact"
        out["no_contact"] = load_sample_from_dir(
            case_dir, case_id="no_contact", sensor_mode=sensor_mode if prefix == "sim" else None
        )
    for wid in weights:
        if prefix == "sim":
            case_dir = sim_nf_dir(root, wid, sensor_mode=sensor_mode)
        else:
            case_dir = real_nf_dir(root, wid)
        out[wid] = load_sample_from_dir(
            case_dir, case_id=wid, sensor_mode=sensor_mode if prefix == "sim" else None
        )
    return out


def load_lateral_cases(
    root: Path,
    *,
    prefix: str,
    weight_id: str = "W100",
    fx_values: Iterable[float] = LATERAL_W100_FX,
    sensor_mode: str = "vitacsim",
) -> dict[str, TactileSample]:
    out: dict[str, TactileSample] = {}
    for fx in fx_values:
        cid = f"{weight_id}_{_fx_tag(fx)}"
        if prefix == "sim":
            case_dir = sim_lateral_dir(root, weight_id, fx, sensor_mode=sensor_mode)
        else:
            case_dir = real_lateral_dir(root, weight_id, fx)
        out[cid] = load_sample_from_dir(
            case_dir, case_id=cid, sensor_mode=sensor_mode if prefix == "sim" else None
        )
    return out


def rgb_diff_magnitude(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    sim = rgb.astype(np.float32)
    bg_a = bg.astype(np.float32)
    if bg_a.shape[:2] != sim.shape[:2]:
        try:
            from PIL import Image

            bg_a = np.asarray(
                Image.fromarray(bg_a.astype(np.uint8)).resize((sim.shape[1], sim.shape[0])),
                dtype=np.float32,
            )
        except ImportError:
            pass
    return np.linalg.norm(sim - bg_a, axis=-1)


def rgb_loss_l1(
    sim_rgb: np.ndarray,
    real_rgb: np.ndarray,
    bg: np.ndarray | None,
    *,
    rgb_scale: float = 1.0,
) -> float:
    if bg is None:
        diff = sim_rgb.astype(np.float32) * rgb_scale - real_rgb.astype(np.float32)
        return float(np.abs(diff).mean())
    sim_mag = rgb_diff_magnitude(sim_rgb, bg) * float(rgb_scale)
    real_mag = rgb_diff_magnitude(real_rgb, bg)
    lo = float(min(sim_mag.min(), real_mag.min()))
    hi = float(max(sim_mag.max(), real_mag.max(), lo + 1.0))
    sim_n = (sim_mag - lo) / (hi - lo)
    real_n = (real_mag - lo) / (hi - lo)
    return float(np.abs(sim_n - real_n).mean())


def marker_curve_loss(
    sim_samples: dict[str, TactileSample],
    real_samples: dict[str, TactileSample],
    case_ids: Iterable[str],
    *,
    displacement_gain: float = 1.0,
) -> float:
    errs: list[float] = []
    for cid in case_ids:
        if cid == "no_contact":
            continue
        s = sim_samples.get(cid)
        r = real_samples.get(cid)
        if s is None or r is None:
            continue
        if s.marker_displacement is None or r.marker_displacement is None:
            continue
        sm = float(np.linalg.norm(s.marker_displacement * displacement_gain, axis=-1).max())
        rm = float(np.linalg.norm(r.marker_displacement, axis=-1).max())
        errs.append((sm - rm) ** 2)
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def case_completeness(samples: dict[str, TactileSample]) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for cid, s in samples.items():
        out[cid] = {
            "rgb": s.rgb is not None,
            "marker_displacement": s.marker_displacement is not None,
            "dir_exists": bool(s.source_dir and s.source_dir.is_dir()),
        }
    return out


def load_advisor_real_nf_cases(real_root: Path) -> dict[str, TactileSample]:
    """Load advisor M2+mass real NF cases (G010..G210 + no_contact)."""
    return load_nf_cases(real_root, prefix="real", weights=ADVISOR_WEIGHT_CASES, include_no_contact=True)


def load_advisor_sim_nf_cases(
    sim_root: Path,
    *,
    sensor_mode: str = "vitacsim",
) -> dict[str, TactileSample]:
    """Sim NF samples keyed by advisor case id (G010..G210, same as real)."""
    return load_nf_cases(
        sim_root,
        prefix="sim",
        weights=ADVISOR_WEIGHT_CASES,
        include_no_contact=True,
        sensor_mode=sensor_mode,
    )


def load_mapped_sim_nf_for_advisor(
    sim_root: Path,
    *,
    sensor_mode: str = "vitacsim",
) -> dict[str, TactileSample]:
    """Sim NF for advisor profile (direct G-case layout; legacy alias)."""
    return load_advisor_sim_nf_cases(sim_root, sensor_mode=sensor_mode)


def write_advisor_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CALIB_SCHEMA_ADVISOR,
        "sensor": ADVISOR_SENSOR_LABEL,
        "marker_pattern": ADVISOR_MARKER_PATTERN,
        "description": "Advisor real tactile calibration (lab Xense): file-000 bg + M2 nut mass sweep mp4.",
        "resolution_wh": list(XENSE_LAB_HW),
        "normal_force_cases": ["no_contact", *ADVISOR_WEIGHT_CASES],
        "real_to_sim_nf": ADVISOR_REAL_TO_SIM_NF,
        "contact_object": "M2_hex_nut",
        "mass_g": ADVISOR_MASS_G,
        "lateral_force": None,
        "files_per_case": {
            "normal_force/<case>/rgb.png": f"Tactile RGB uint8 {XENSE_LAB_HW[0]}x{XENSE_LAB_HW[1]} native",
            "normal_force/<case>/marker_displacement.npy": "(M,2) vs no_contact rest (lab Xense)",
            "normal_force/no_contact/marker_rest_detected.npy": "Rest marker centers from no_contact rgb",
        },
        "processed_bg": {
            "bg.jpg": "Raw no_contact frame (with printed markers)",
            "bg_clean.jpg": "Gel-only background (markers inpainted) for Taxim",
            "marker_rest.npy": "Detected marker rest coordinates for sim overlay",
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CALIB_SCHEMA,
        "sensor": "gelsight_r15",
        "description": "Real robot tactile calibration dataset (mirror sim sweep layout).",
        "normal_force_cases": ["no_contact", *WEIGHT_CASES],
        "lateral_force": {"weight_id": "W100", "fx_values": list(LATERAL_W100_FX)},
        "files_per_case": {
            "normal_force/<case>/rgb.png": "Tactile RGB (uint8)",
            "normal_force/<case>/marker_displacement.npy": "Shape (M,2) pixel displacement vs no_contact",
            "lateral_force/W100/<FxTag>/rgb.png": "Lateral push case RGB",
            "lateral_force/W100/<FxTag>/marker_displacement.npy": "Marker displacement under shear",
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
