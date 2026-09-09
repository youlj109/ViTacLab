"""Standard M2 hex nut + suspended mass presets for advisor NF validation.

Geometry: GB/T 6170 M2 (nominal).
  - Width across flats s = 3.8 mm
  - Nut height m = 1.6 mm
  - Clear hole ~ 2.0 mm (M2 thread)

Origin: center of nut bottom face (hex ring contact on gel pad).
Suspended mass is modeled via rigid-body mass only (no extra pad colliders).
"""

from __future__ import annotations

from dataclasses import dataclass

# Advisor real captures (grams).
ADVISOR_CASE_MASS_G: dict[str, int] = {
    "G010": 10,
    "G030": 30,
    "G060": 60,
    "G110": 110,
    "G160": 160,
    "G210": 210,
}

ADVISOR_CASE_ORDER: tuple[str, ...] = ("G210", "G160", "G110", "G060", "G030", "G010")

# Validation scene defaults (M2 nut on horizontal GelSight pad).
ADVISOR_FINGER_ROOT_Z = 0.441
ADVISOR_WEIGHT_REST_Z = 0.442
ADVISOR_WEIGHT_CLEARANCE_Z = 0.518
ADVISOR_WEIGHT_DROP_OFFSET = 0.010
# Contact alignment is via nut pose (contact-offset-x/y), not whole-image UV shift.
ADVISOR_TACTILE_UV_SHIFT_PX = (0.0, 0.0)
ADVISOR_MARKER_LOAD_REF_FN_N = 0.72
ADVISOR_MARKER_LOAD_SCALE_EXPONENT = 0.48
ADVISOR_MARKER_DEPTH_GAMMA = 1.2
ADVISOR_MARKER_DEPTH_GAMMA_LOW_LOAD = 1.0
ADVISOR_MARKER_DEPTH_GAMMA_LOAD_T0 = 0.35
ADVISOR_MARKER_SHEAR_FROM_FORCE_FIELD = True
ADVISOR_MARKER_SHEAR_FORCE_GAIN = 3.0
ADVISOR_MARKER_SHEAR_FORCE_REF_N = 0.05


@dataclass(frozen=True)
class M2NutGeometry:
    width_across_flats: float = 0.0038
    height: float = 0.0016
    hole_diameter: float = 0.0020

    @property
    def circumradius(self) -> float:
        """Hex vertex radius from flat-to-flat width."""
        return self.width_across_flats / (3.0**0.5)

    @property
    def hole_radius(self) -> float:
        return self.hole_diameter * 0.5


M2_GEOMETRY = M2NutGeometry()

# Approximate COM offset below nut face (suspended mass on wire).
HANG_COM_OFFSET_Z = -0.025


def advisor_mass_kg(case_id: str) -> float:
    if case_id not in ADVISOR_CASE_MASS_G:
        raise KeyError(f"Unknown advisor case_id={case_id!r}; expected one of {sorted(ADVISOR_CASE_MASS_G)}")
    return float(ADVISOR_CASE_MASS_G[case_id]) / 1000.0


def nominal_fn_n(case_id: str, *, g: float = 9.81) -> float:
    return advisor_mass_kg(case_id) * g
