import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core import plot
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.multiscat import get_scattering_matrix

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


if __name__ == "__main__":
    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL, 0, 0]),
            np.array([0, UNIT_CELL, 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        (15, 15, 550),
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
    config = OptimizationConfig(precision=1e-5, max_iterations=1000)
    s_matrix = get_scattering_matrix(condition, config)
    fig, ax, _mesh = plot.array_against_axes_2d(s_matrix, measure="abs")
    ax.set_title("The scattering matrix")
    fig.show()
    fig.savefig("scattering_matrix.png")

    plot.wait_for_close()
