from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
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
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
)
from multiscat.multiscat._multiscat import get_scattering_state

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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


def plot_scattering_state[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[Basis[ScatteringBasisMetadata[M0, M1, E]]],
) -> tuple[Figure, Axes, animation.FuncAnimation]:
    state_data = state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.reshape(condition.metadata.shape)

    abs_state = np.abs(state_data)
    fig, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]
    image = ax.imshow(  # type: ignore[reportUnknownMemberType]# cspell: disable-line
        np.fft.fftshift(abs_state[:, :, 0]),  # cspell: disable-line
        origin="lower",
        cmap="viridis",  # cspell: disable-line
    )
    fig.colorbar(image, ax=ax, label="|state|")  # type: ignore[reportUnknownMemberType]

    def _update(frame: int) -> tuple[object, ...]:
        image.set_data(np.fft.fftshift(abs_state[:, :, frame]))  # cspell: disable-line
        ax.set_title(f"|state| at z index {frame}")  # type: ignore[reportUnknownMemberType]
        return (image,)

    anim = animation.FuncAnimation(
        fig,
        _update,  # type: ignore[reportArgumentType]
        frames=abs_state.shape[2],
        interval=80,
        blit=True,  # cspell: disable-line
    )
    return fig, ax, anim


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
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=80)
    s_matrix = get_scattering_matrix(condition, config, backend="scipy")

    fig, ax, _mesh = plot.array_against_axes_2d_k(s_matrix, measure="abs")
    ax.set_title("The scattering matrix")
    fig.show()

    state = get_scattering_state(condition, config)
    fig, ax, anim = plot_scattering_state(state)
    fig.show()
    anim.save("scattering_state_animation.gif", writer="pillow")  # cspell: disable-line

    plot.wait_for_close()
