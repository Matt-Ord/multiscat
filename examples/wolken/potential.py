import numpy as np
from _params import HELIUM_ENERGY, MORSE_PARAMETERS, UNIT_CELL, Z_HEIGHT
from slate_core import array, plot
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x

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
