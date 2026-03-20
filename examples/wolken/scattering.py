import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core import plot
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x
from multiscat.config import ScatteringCondition
from multiscat.multiscat import get_scattered_state, get_scattering_matrix

HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3

UNIT_CELL = 2.84 * angstrom
Z_HEIGHT = 8 * angstrom

MORSE_PARAMETERS = operator.build.CorrugatedMorseParameters(
    depth=7.63 * electron_volt * 10**-3,
    height=0.91 * angstrom,
    offset=3.0 * angstrom,
    beta=0.10,
)


if __name__ == "__main__":
    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL, 0, 0]),
            np.array([0, UNIT_CELL, 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        (19, 19, 99),
    )
    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=HELIUM_ENERGY,
        theta=30,
        potential=operator.build.corrugated_morse_potential(
            metadata,
            MORSE_PARAMETERS,
        ),
    )
    state = get_scattered_state(condition)
    fig, ax, _anim0 = plot.animate_data_2d(state, axes=(0, 2, 1), measure="abs")
    ax.set_title(
        (
            "The scattered state,\n"
            f"in a lobatto basis with ({state.basis.metadata().shape}) points"
        ),
    )
    fig.show()

    s_matrix = get_scattering_matrix(state)
    fig, ax, _mesh = plot.array_against_axes_2d_k(s_matrix)
    ax.set_title("The scattering matrix")
    fig.show()

    plot.wait_for_close()
