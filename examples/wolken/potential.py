import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core import array, plot
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x

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
    lobatto_metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL, 0, 0]),
            np.array([0, UNIT_CELL, 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        (12, 12, 35),
    )
    morse_potential = operator.build.corrugated_morse_potential(
        lobatto_metadata,
        MORSE_PARAMETERS,
    )
    fig, ax, mesh = plot.array_against_axes_2d(
        array.extract_diagonal(morse_potential),
        axes=(0, 2),
    )
    # Truncate at the classical turning point
    mesh.set_clim(None, HELIUM_ENERGY)
    ax.set_title(
        (
            "A corrugated Morse potential as used by wolken for He-Li scattering,\n"
            f"in a lobatto basis with ({lobatto_metadata.shape}) points"
        ),
    )
    fig.show()
    plot.wait_for_close()
