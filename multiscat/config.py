from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import hbar  # type: ignore[import-untyped]
from slate_core.metadata import AxisDirections, EvenlySpacedLengthMetadata

if TYPE_CHECKING:
    from slate_core.metadata import LobattoSpacedMetadata

    from multiscat.basis import ScatteringBasisMetadata
    from multiscat.interpolate import ScatteringOperator


@dataclass(frozen=True, kw_only=True)
class OptimizationConfig:
    """Represents options for the GMRES solver."""

    precision: float = 1e-5
    max_iterations: int = 1000
    max_channel_index: int | None = None
    max_negative_energy: float | None = None


@dataclass(frozen=True, kw_only=True)
class ScatteringCondition[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedMetadata,
    E: AxisDirections,
]:
    """Represents a particular scattering condition."""

    mass: float
    incident_k: tuple[float, float, float]
    potential: ScatteringOperator[M0, M1, E]

    @property
    def metadata(self) -> ScatteringBasisMetadata[M0, M1, E]:
        """The metadata for the scattering state."""
        return self.potential.basis.metadata().children[0]

    @staticmethod
    def from_angles[
        M01: EvenlySpacedLengthMetadata,
        M11: LobattoSpacedMetadata,
        E1: AxisDirections,
    ](
        *,
        mass: float,
        theta: float = 0,
        phi: float = 0,
        energy: float,
        potential: ScatteringOperator[M01, M11, E1],
    ) -> ScatteringCondition[M01, M11, E1]:
        """
        Create a scattering condition from angles.

        All angles are in radians, and energy is in joules.
        """
        # Energy = hbar**2 k**2 / (2 * mass)
        abs_k = (2 * mass * energy) ** 0.5 / hbar
        return ScatteringCondition(
            mass=mass,
            incident_k=(
                abs_k * np.sin(theta) * np.cos(phi),
                abs_k * np.sin(theta) * np.sin(phi),
                abs_k * np.cos(theta),
            ),
            potential=potential,
        )

    @property
    def incident_energy(self) -> float:
        """The incident energy in joules."""
        kx, ky, kz = self.incident_k
        return (hbar**2 * (kx**2 + ky**2 + kz**2)) / (2 * self.mass)

    @property
    def theta(self) -> float:
        """The polar angle of the incident wavevector in radians."""
        kx, ky, kz = self.incident_k
        return np.arccos(kz / np.sqrt(kx**2 + ky**2 + kz**2))

    @property
    def phi(self) -> float:
        """The azimuthal angle of the incident wavevector in radians."""
        kx, ky, _ = self.incident_k
        return np.arctan2(ky, kx)
