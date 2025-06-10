import numpy as np
from _params import HELIUM_ENERGY, HELIUM_MASS, MORSE_PARAMETERS, UNIT_CELL, Z_HEIGHT
from slate_core import plot
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x
from multiscat.config import ScatteringCondition
from multiscat.multiscat import get_scattered_state, get_scattering_matrix

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
