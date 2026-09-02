"""Shared constants for ViTacSim NF/SF validation beta (standalone for plot/summarize)."""

from __future__ import annotations

WEIGHT_ORDER = ("W200", "W100", "W050", "W020", "W010")
WEIGHT_MASS_G = {"W200": 200, "W100": 100, "W050": 50, "W020": 20, "W010": 10}
WEIGHT_MASS_KG = {wid: g / 1000.0 for wid, g in WEIGHT_MASS_G.items()}
MODES = ("tacsl", "vitacsim")

NF_SCHEMA = "nf_v3_beta"
SF_SCHEMA = "sf_lateral_v2"

DEFAULT_FINGER_ROOT_Z = 0.444
CONTACT_VALID_FN_RATIO = 0.5

WEIGHT_FX_SWEEP: dict[str, tuple[float, ...]] = {
    "W200": (0.0, 0.1, 0.2, 0.3, 0.5),
    "W100": (0.0, 0.05, 0.1, 0.15, 0.2),
    "W050": (0.0, 0.03, 0.05, 0.08, 0.1),
    "W020": (0.0, 0.01, 0.02, 0.03, 0.05),
    "W010": (0.0, 0.005, 0.01, 0.015, 0.02),
}

SF_PRIMARY_WEIGHTS = ("W200", "W100")
SF_PANEL_WEIGHTS = ("W200", "W100", "W050")
SF_PANEL_FX = 0.15
SF_BAR_FX = 0.15

FORCE_RENDER_K_REF_W100 = 66.0


def nominal_fn_n(weight_id: str) -> float:
    return WEIGHT_MASS_KG[weight_id] * 9.81


def adaptive_force_render_k_ref(weight_id: str, *, base_k: float = FORCE_RENDER_K_REF_W100) -> float:
    ref_nom = nominal_fn_n("W100")
    nom = nominal_fn_n(weight_id)
    return base_k * ref_nom / max(nom, 1e-9)


def resolve_force_render_k_ref(weight_id: str, cli_value: float) -> float:
    if cli_value > 0.0:
        return float(cli_value)
    return adaptive_force_render_k_ref(weight_id)


def contact_valid(
    *,
    physx_fn_mean: float,
    nominal_fn: float,
    contact_count: float | int,
    ratio_threshold: float = CONTACT_VALID_FN_RATIO,
) -> bool:
    return float(physx_fn_mean) >= ratio_threshold * float(nominal_fn) and int(contact_count) > 0


def fx_list_for_weight(weight_id: str) -> tuple[float, ...]:
    return WEIGHT_FX_SWEEP.get(weight_id, (0.0, 0.1, 0.2))
