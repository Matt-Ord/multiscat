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
from multiscat.multiscat import get_scattered_state

HELIUM_MASS = physical_constants["alpha particle mass"][0]
UNIT_CELL = 8 * angstrom
Z_HEIGHT = 3 * angstrom  # TODO: double check this value  # noqa: FIX002

MORSE_PARAMETERS = operator.build.CorrugatedMorseParameters(
    depth=7.63 * electron_volt * 10**-3,
    height=0.91 * angstrom,
    offset=1.0 * angstrom,
    beta=0.05,
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
    # This is taken from https://doi.org/10.1039/FT9908601641
    # and is a reproduction of the Wolken 4He-LiF problem in table 1,
    # originally simulated in https://doi.org/10.1063/1.1679617.
    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=20 * electron_volt * 10**-3,
        theta=np.deg2rad(30),
        phi=0,
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

    # TODO: get S matrix from result and plot it...  # noqa: FIX002
    plot.wait_for_close()
