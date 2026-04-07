from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, override

import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom as angstrom_si,
)
from scipy.constants import (  # type: ignore[import-untyped]
    atomic_mass as atomic_mass_si,
)
from scipy.constants import (  # type: ignore[import-untyped]
    hbar as hbar_si,
)
from slate_core import AsUpcast, Basis, TupleMetadata
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_delta_x
from slate_quantum import Operator, State, operator
from slate_quantum.operator import position_operator_basis

from multiscat.basis import (
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
)

if TYPE_CHECKING:
    from multiscat.basis import ScatteringBasisMetadata
    from multiscat.interpolate import ScatteringOperator


@dataclass(frozen=True, kw_only=True)
class OptimizationConfig:
    """Represents options for the GMRES solver."""

    precision: float = 1e-5
    max_iterations: int = 1000
    use_neumann_preconditioner: bool = True
    """
    As well as the default "specular" preconditioner,
    we can also apply an additional Neumann series preconditioner
    which approximately inverts the problem when the non-specular
    scattering is small.
    """
    n_channels: int | None = None
    """
    The number of channels to include in the scattering calculation.
    If None, all channels are included.
    """


@dataclass(frozen=True, kw_only=True)
class UnitSystem:
    """Defines the units used in the scattering calculation."""

    hbar: float = hbar_si
    atomic_mass: float = atomic_mass_si
    angstrom: float = angstrom_si

    @classmethod
    def si(cls) -> UnitSystem:
        """Get the SI units."""
        return cls()

    @property
    def kinetic_energy_unit(self) -> float:
        """The unit of kinetic energy in this system."""
        return (self.hbar**2) / (2 * self.atomic_mass * self.angstrom**2)


SI_UNITS = UnitSystem.si()


class ScatteringCondition[
    M0: EvenlySpacedLengthMetadata = EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata = LobattoSpacedLengthMetadata,
    E: AxisDirections = AxisDirections,
]:
    """Represents a particular scattering condition."""

    _mass: float
    _incident_k: tuple[float, float, float]
    _potential: ScatteringOperator[M0, M1, E]
    _units: UnitSystem = field(default_factory=UnitSystem.si)

    def __init__(
        self,
        *,
        mass: float,
        incident_k: tuple[float, float, float],
        potential: ScatteringOperator[M0, M1, E],
        units: UnitSystem = SI_UNITS,
    ) -> None:
        self._mass = mass
        self._incident_k = incident_k
        self._potential = potential
        self._units = units

    @property
    def initial_state(self) -> State[Basis[ScatteringBasisMetadata[M0, M1, E]]]:
        """The initial state of the scattering problem."""
        return State(
            close_coupling_basis(self.metadata).upcast(),
            np.zeros(self.metadata.shape, dtype=np.complex128).ravel(),
        )

    @property
    def mass(self) -> float:
        """The mass of the particle in atomic mass units."""
        return self._mass

    @property
    def incident_k(self) -> tuple[float, float, float]:
        """The incident wavevector in inverse angstroms."""
        return self._incident_k

    @property
    def potential(self) -> ScatteringOperator[M0, M1, E]:
        """The scattering operator."""
        return self._potential

    @property
    def units(self) -> UnitSystem:
        """The unit system used for this scattering condition."""
        return self._units

    @property
    def metadata(self) -> ScatteringBasisMetadata[M0, M1, E]:
        """The metadata for the scattering state."""
        return self._potential.basis.metadata().children[0]

    @staticmethod
    def from_angles[  # noqa: PLR0913
        M01: EvenlySpacedLengthMetadata,
        M11: LobattoSpacedLengthMetadata,
        E1: AxisDirections,
    ](
        *,
        mass: float,
        theta: float = 0,
        phi: float = 0,
        energy: float,
        potential: ScatteringOperator[M01, M11, E1],
        units: UnitSystem = SI_UNITS,
    ) -> ScatteringCondition[M01, M11, E1]:
        """
        Create a scattering condition from angles.

        All angles are in radians, and energy is in joules.
        """
        return ScatteringCondition(
            mass=mass,
            incident_k=momentum_from_angles(
                theta=theta,
                phi=phi,
                energy=energy,
                mass=mass,
                units=units,
            ),
            potential=potential,
            units=units,
        )

    @property
    def incident_energy(self) -> float:
        """The incident energy in joules."""
        kx, ky, kz = self.incident_k
        return (self.units.hbar**2 * (kx**2 + ky**2 + kz**2)) / (2 * self.mass)

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

    def with_units(self, units: UnitSystem) -> ScatteringCondition:
        """Convert this scattering condition to a different unit system."""
        return _with_units(self, units)


def momentum_from_angles(
    theta: float,
    phi: float,
    energy: float,
    mass: float,
    *,
    units: UnitSystem = SI_UNITS,
) -> tuple[float, float, float]:
    """Calculate the incident momentum from angles."""
    abs_k = (2 * mass * energy) ** 0.5 / units.hbar
    return (
        abs_k * np.sin(theta) * np.cos(phi),
        abs_k * np.sin(theta) * np.sin(phi),
        abs_k * np.cos(theta),
    )


def _metadata_with_units(
    metadata: ScatteringBasisMetadata,
    old_units: UnitSystem,
    units: UnitSystem,
) -> ScatteringBasisMetadata:
    """Convert scattering basis metadata to a different unit system."""
    vectors = fundamental_stacked_delta_x(metadata)
    length_factor = units.angstrom / old_units.angstrom
    return scattering_metadata_from_stacked_delta_x(
        vectors=tuple(v * length_factor for v in vectors),
        shape=metadata.shape,  # type: ignore[length not inferred]
    )


def _potential_with_units[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    potential: ScatteringOperator[M0, M1, E],
    old_units: UnitSystem,
    units: UnitSystem,
) -> ScatteringOperator[
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
    AxisDirections,
]:
    """Convert a scattering operator to a different unit system."""
    old_metadata = potential.basis.metadata().children[0]
    metadata = _metadata_with_units(old_metadata, old_units, units)
    out_basis = position_operator_basis(close_coupling_basis(metadata))

    converted = potential.with_basis(
        position_operator_basis(close_coupling_basis(old_metadata)),
    ).raw_data
    energy_factor = units.kinetic_energy_unit / old_units.kinetic_energy_unit
    # Note since we change the length scale, the Lobatto weights also change
    # To keep the .as_array() fixed, we therefore also need to scale the raw_data
    length_factor = units.angstrom / old_units.angstrom
    lobatto_weight_factor = length_factor
    return Operator(
        AsUpcast(out_basis, TupleMetadata((metadata, metadata))),
        converted * (energy_factor * lobatto_weight_factor),
    )


def _with_units[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    units: UnitSystem,
) -> ScatteringCondition:
    """Convert a scattering condition to a different unit system."""
    mass_factor = units.atomic_mass / condition.units.atomic_mass
    length_factor = units.angstrom / condition.units.angstrom
    return ScatteringCondition(
        mass=condition.mass * mass_factor,
        incident_k=(
            condition.incident_k[0] / length_factor,
            condition.incident_k[1] / length_factor,
            condition.incident_k[2] / length_factor,
        ),
        potential=_potential_with_units(condition.potential, condition.units, units),
        units=units,
    )


class MorseScatteringCondition[
    M0: EvenlySpacedLengthMetadata = EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata = LobattoSpacedLengthMetadata,
    E: AxisDirections = AxisDirections,
](ScatteringCondition[M0, M1, E]):
    """Represents a particular scattering condition."""

    _mass: float
    _incident_k: tuple[float, float, float]
    _units: UnitSystem = field(default_factory=UnitSystem.si)
    _morse_parameters: operator.build.CorrugatedMorseParameters
    _metadata: ScatteringBasisMetadata[M0, M1, E]

    def __init__(
        self,
        *,
        mass: float,
        incident_k: tuple[float, float, float],
        morse_parameters: operator.build.CorrugatedMorseParameters,
        metadata: ScatteringBasisMetadata[M0, M1, E],
        units: UnitSystem = SI_UNITS,
    ) -> None:
        self._mass = mass
        self._incident_k = incident_k
        self._morse_parameters = morse_parameters
        self._metadata = metadata
        self._units = units

    @property
    def morse_parameters(self) -> operator.build.CorrugatedMorseParameters:
        """The parameters of the corrugated Morse potential."""
        return self._morse_parameters

    @property
    @override
    def mass(self) -> float:
        return self._mass

    @property
    @override
    def incident_k(self) -> tuple[float, float, float]:
        return self._incident_k

    @property
    @override
    def units(self) -> UnitSystem:
        return self._units

    @property
    @override
    def metadata(self) -> ScatteringBasisMetadata[M0, M1, E]:
        return self._metadata

    @cached_property
    def _potential(self) -> ScatteringOperator[M0, M1, E]:  # type: ignore override private
        """The scattering operator."""
        return operator.build.corrugated_morse_potential(
            self.metadata,
            self.morse_parameters,
        )

    @property
    @override
    def potential(self) -> ScatteringOperator[M0, M1, E]:
        """The scattering operator."""
        return self._potential
