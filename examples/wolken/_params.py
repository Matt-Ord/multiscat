from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_quantum import operator

# This is taken from https://doi.org/10.1039/FT9908601641
# and is a reproduction of the Wolken 4He-LiF problem in table 1,
# originally simulated in https://doi.org/10.1063/1.1679617.

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
