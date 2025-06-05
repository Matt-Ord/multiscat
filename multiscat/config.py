from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from slate_core.metadata import AxisDirections, SpacedLengthMetadata

if TYPE_CHECKING:
    from multiscat.basis import ScatteringBasisMetadata
    from multiscat.interpolate import ScatteringOperator
    from multiscat.lobatto import LobattoMetadata


@dataclass
class GMRESConfig:
    """Represents options for the GMRES solver."""

    precision: float
    preconditioner: int
    max_iterations: int = 1000


@dataclass
class ScatteringCondition[
    M0: SpacedLengthMetadata,
    M1: LobattoMetadata,
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
