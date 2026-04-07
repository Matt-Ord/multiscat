from typing import TYPE_CHECKING, override

import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core import Basis, EvenlySpacedLengthMetadata, plot
from slate_core.metadata import AxisDirections, LobattoSpacedLengthMetadata
from slate_quantum import State, operator

from multiscat import OptimizationConfig, ScatteringCondition, get_scattering_matrix
from multiscat.basis import (
    ScatteringBasisMetadata,
    scattering_metadata_from_stacked_delta_x,
)
from multiscat.multiscat import (
    get_scattering_state,
)

if TYPE_CHECKING:
    from multiscat.config import UnitSystem

HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3

UNIT_CELL = 2.84 * angstrom
Z_HEIGHT = 8 * angstrom

MORSE_PARAMETERS = operator.build.CorrugatedMorseParameters(
    depth=7.63 * electron_volt * 10**-3,
    height=(1.0 / 1.1) * angstrom,
    offset=3.0 * angstrom,
    beta=0.10,
)


class CheatCondition[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](ScatteringCondition[M0, M1, E]):
    """
    A scattering condition which cheats.

    It uses the true scattering state as the initial guess state.
    """

    def __init__(self, inner: ScatteringCondition[M0, M1, E]) -> None:
        self._inner = inner
        super().__init__(
            incident_k=inner.incident_k,
            potential=inner.potential,
            mass=inner.mass,
            units=inner.units,
        )

    @property
    @override
    def initial_state(self) -> State[Basis[ScatteringBasisMetadata[M0, M1, E]]]:
        """The initial state of the scattering problem."""
        print("cheating the initial state")
        config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=80)
        return get_scattering_state(self._inner, config)

    @override
    def with_units(self, units: UnitSystem) -> CheatCondition[M0, M1, E]:
        return CheatCondition[M0, M1, E](self._inner.with_units(units))  # type: ignore[return-value]


if __name__ == "__main__":
    # If we use a cheat condition, we converge instantly to the correct solution.
    # This demonstrates that a better initial guess significantly speeds up convergence.
    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL, 0, 0]),
            np.array([0, UNIT_CELL, 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        (15, 15, 200),
    )
    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=HELIUM_ENERGY,
        theta=np.deg2rad(30),
        phi=np.deg2rad(0),
        potential=operator.build.corrugated_morse_potential(
            metadata,
            MORSE_PARAMETERS,
        ),
    )
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=80)
    s_matrix = get_scattering_matrix(CheatCondition(condition), config, backend="scipy")

    fig, ax, _mesh = plot.array_against_axes_2d_k(s_matrix, measure="abs")
    ax.set_title("The scattering matrix")
    fig.show()

    plot.wait_for_close()
