"""Validation weight geometry and mass presets for ViTacSim normal/shear experiments.

All weights share the same composite shape (uniform density); only mass differs.

Coordinate convention (meters, z-up):
  - Origin at the center of the large-cylinder bottom face (contact point for normal tests).
  - Weight stands vertically (+z); symmetry axis = image center for horizontal GelSight pad.

Shape (from user spec):
  - Large cylinder: D=25 mm, H=25 mm, 1 mm chamfer on top/bottom rim edges.
  - Small cylinder (on top center): D=5 mm, H=5.5 mm.
  - Sphere (on top): D=7 mm; sphere center 8 mm above large-cylinder top face.
  - Sphere overlaps the small cylinder (intentional).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationWeightGeometry:
    """Composite weight dimensions in meters."""

    large_diameter: float = 0.025
    large_height: float = 0.025
    rim_chamfer: float = 0.001

    stem_diameter: float = 0.005
    stem_height: float = 0.0055

    sphere_diameter: float = 0.007
    sphere_center_above_large_top: float = 0.008

    @property
    def large_radius(self) -> float:
        return self.large_diameter * 0.5

    @property
    def sphere_radius(self) -> float:
        return self.sphere_diameter * 0.5

    @property
    def large_top_z(self) -> float:
        return self.large_height

    @property
    def stem_bottom_z(self) -> float:
        return self.large_top_z

    @property
    def stem_top_z(self) -> float:
        return self.stem_bottom_z + self.stem_height

    @property
    def sphere_center_z(self) -> float:
        return self.large_top_z + self.sphere_center_above_large_top

    @property
    def total_height(self) -> float:
        return self.sphere_center_z + self.sphere_radius


GEOMETRY = ValidationWeightGeometry()

# 45° × 1 mm chamfer on large-cylinder top/bottom outer rims (machining spec).
# Bottom flat face center remains at z=0; stem + sphere share axis x=y=0.


@dataclass(frozen=True)
class ValidationWeightLayout:
    """Derived spawn layout (meters, z-up, origin at bottom-face center)."""

    geom: ValidationWeightGeometry = GEOMETRY

    @property
    def chamfer(self) -> float:
        return self.geom.rim_chamfer

    @property
    def large_radius(self) -> float:
        return self.geom.large_radius

    @property
    def flat_radius(self) -> float:
        """Radius of flat regions on chamfered top/bottom faces (45° chamfer)."""
        return self.large_radius - self.chamfer

    @property
    def large_height(self) -> float:
        return self.geom.large_height

    @property
    def main_cylinder_height(self) -> float:
        return self.large_height - 2.0 * self.chamfer

    @property
    def large_bottom_z(self) -> float:
        return 0.0

    @property
    def large_top_z(self) -> float:
        return self.large_height

    @property
    def stem_bottom_z(self) -> float:
        return self.large_top_z

    @property
    def stem_top_z(self) -> float:
        return self.stem_bottom_z + self.geom.stem_height

    @property
    def stem_center_z(self) -> float:
        return self.stem_bottom_z + 0.5 * self.geom.stem_height

    @property
    def sphere_center_z(self) -> float:
        return self.large_top_z + self.geom.sphere_center_above_large_top

    @property
    def total_height(self) -> float:
        return self.sphere_center_z + self.geom.sphere_radius


LAYOUT = ValidationWeightLayout()

WEIGHT_MASS_KG: dict[str, float] = {
    "W200": 0.200,
    "W100": 0.100,
    "W050": 0.050,
    "W020": 0.020,
    "W010": 0.010,
}

WEIGHT_ORDER: tuple[str, ...] = ("W200", "W100", "W050", "W020", "W010")
