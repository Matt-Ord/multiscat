import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core import plot
from slate_quantum import operator

from multiscat import OptimizationConfig
from multiscat.basis import (
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
)
from multiscat.config import (
    MorseScatteringCondition,
    condition_in_natural_units,
    incident_k_from_angles,
)
from multiscat.multiscat import (
    get_preconditioned_state_from_state,
    get_scattering_state,
)
from multiscat.multiscat._util import get_full_a_wave, get_full_b_wave

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


def get_stable_state(
    condition: MorseScatteringCondition,
    config: OptimizationConfig,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    """Simulate the preconditioned state from the given ScatteringCondition."""
    converted_condition = condition_in_natural_units(condition)
    b = get_full_b_wave(
        converted_condition.metadata,
        converted_condition.incident_k,
    )

    # Get the scattering state
    state = get_scattering_state(condition, config)

    # Get the preconditioned state
    preconditioned_state = get_preconditioned_state_from_state(
        state,
        condition,
        n_channels=config.n_channels,
    )

    # Return the real and imaginary parts of the preconditioned state
    preconditioned_data = preconditioned_state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.reshape(condition.metadata.shape)
    return 2.0j * b * preconditioned_data


if __name__ == "__main__":
    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL / np.sqrt(2), 0, 0]),
            np.array([UNIT_CELL / np.sqrt(8), np.sqrt(3) * UNIT_CELL / np.sqrt(8), 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        (15, 15, 200),
    )

    condition = MorseScatteringCondition(
        mass=3 * HELIUM_MASS,
        incident_k=incident_k_from_angles(
            mass=3 * HELIUM_MASS,
            energy=HELIUM_ENERGY,
            theta=np.deg2rad(30),
            phi=np.deg2rad(60),
        ),
        metadata=metadata,
        morse_parameters=MORSE_PARAMETERS,
    )
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=80)

    state = get_stable_state(condition, config)
    fig, ax = plot.get_figure()
    for i in [0, 1, -1]:
        for j in [0, 1, -1]:
            ax.plot(np.abs(state[i, j, :]) ** 2)

    ax.set_ylim(0, 0.25)  # cspell: disable-line
    fig.show()
    fig.savefig("stable_state.abs.png", dpi=300)

    fig, ax = plot.get_figure()
    for i in [0, 1, -1]:
        for j in [0, 1, -1]:
            z_points = metadata.children[2].values
            ax.plot(z_points, np.real(state[i, j, :]))

    fig.show()
    fig.savefig("stable_state.real.png", dpi=300)

    metadata_1 = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL / np.sqrt(2), 0, 0]),
            np.array([UNIT_CELL / np.sqrt(8), np.sqrt(3) * UNIT_CELL / np.sqrt(8), 0]),
            np.array([0, 0, 2 * Z_HEIGHT]),
        ),
        (15, 15, 400),
    )
    condition_1 = MorseScatteringCondition(
        mass=condition.mass,
        incident_k=condition.incident_k,
        metadata=metadata_1,
        morse_parameters=condition.morse_parameters,
    )
    state_1 = get_stable_state(condition_1, config)

    for i in [0, 1, -1]:
        for j in [0, 1, -1]:
            fig, ax = plot.get_figure()
            z_points = metadata.children[2].values
            ax.plot(z_points, np.real(state[i, j, :]))
            z_points_1 = metadata_1.children[2].values
            ax.plot(z_points_1, np.real(state_1[i, j, :]))

            fig.savefig(f"stable_state.real.{i},{j}.png", dpi=300)
    for i in [0, 1, -1]:
        for j in [0, 1, -1]:
            fig, ax = plot.get_figure()
            z_points = metadata.children[2].values
            ax.plot(z_points, np.abs(state[i, j, :]) ** 2)
            z_points_1 = metadata_1.children[2].values
            ax.plot(z_points_1, np.abs(state_1[i, j, :]) ** 2)

            fig.savefig(f"stable_state.abs.{i},{j}.png", dpi=300)

    converted_condition = condition_in_natural_units(condition)
    a_full = get_full_a_wave(
        converted_condition.metadata,
        converted_condition.incident_k,
    )
    fig, ax = plot.get_figure()
    ax.plot(z_points, np.abs(a_full[0, 0, :]) ** 2)
    ax.plot(z_points, np.real(a_full[0, 0, :]))
    ax.plot(z_points, np.imag(a_full[0, 0, :]))
    fig.savefig("a_wave.abs.png", dpi=300)
    plot.wait_for_close()
