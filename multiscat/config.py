from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
        _M0: EvenlySpacedLengthMetadata,
        _M1: LobattoSpacedMetadata,
        _E: AxisDirections,
    ](
        *,
        mass: float,
        theta: float = 0,
        phi: float = 0,
        energy: float,
        potential: ScatteringOperator[_M0, _M1, _E],
    ) -> ScatteringCondition[_M0, _M1, _E]:
        """Create a scattering condition from angles."""
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
