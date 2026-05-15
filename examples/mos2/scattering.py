import numpy as np
from model import build_mos2_potential
from scipy.constants import (  # type: ignore[import-untyped]
    electron_volt,
    physical_constants,
)
from slate_core import metadata as _metadata
from slate_core import plot
from slate_core.metadata import (
    LobattoSpacedLengthMetadata,
)

from multiscat import OptimizationConfig, ScatteringCondition, get_scattering_matrix

HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3


if __name__ == "__main__":
    repeats = (3, 3)
    potential = build_mos2_potential(
        (15 * repeats[0], 15 * repeats[1]),
        LobattoSpacedLengthMetadata(200, domain=_metadata.Domain(delta=10e-10)),
        repeats=repeats,
    )

    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=HELIUM_ENERGY,
        theta=np.deg2rad(30),
        phi=np.deg2rad(0),
        potential=potential,
    )
    config = OptimizationConfig(
        precision=1e-5,
        max_iterations=50,
        n_channels=70 * np.prod(repeats).item(),
    )
    s_matrix = get_scattering_matrix(condition, config, backend="scipy")

    fig, ax, _mesh = plot.array_against_axes_2d_k(s_matrix, measure="abs")
    ax.set_title("The scattering matrix")
    fig.show()

    plot.wait_for_close()
