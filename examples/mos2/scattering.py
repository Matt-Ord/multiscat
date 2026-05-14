from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    electron_volt,
    physical_constants,
)
from slate_core import EvenlySpacedLengthMetadata, plot
from slate_core import metadata as _metadata
from slate_core.metadata import (
    AxisDirections,
    LengthMetadata,
    LobattoSpacedLengthMetadata,
)
from slate_quantum import operator

from multiscat import OptimizationConfig, ScatteringCondition, get_scattering_matrix
from multiscat.basis import (
    ScatteringBasisMetadata,
    as_scattering_potential,
    combine_scattering_metadata,
)
from multiscat.config import UnitSystem

if TYPE_CHECKING:
    from slate_quantum import Operator
    from slate_quantum.operator import OperatorBasis

MOS2_LATTICE_CONSTANT = 3.16


HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3


@dataclass
class AtomicParameters:
    """Parameters specifying the effect of an individual atom."""

    D: float
    a: float
    alpha: float
    b: float
    beta: float
    z0: float
    z1: float
    centre: tuple[float, float]


SULPHUR_PARAMS = AtomicParameters(
    D=19.9886,
    a=0.8122,
    alpha=1.4477,
    b=0.1958,
    beta=0.2029,
    z0=3.3719,
    z1=1.7316,
    centre=(0.0, 0.0),
)
HOLLOW_PARAMS = AtomicParameters(
    D=24.9674,
    a=0.4641,
    alpha=1.1029,
    b=0.1993,
    beta=0.6477,
    z0=3.1411,
    z1=3.8323,
    centre=(0, 1 / 3),
)
MOLYBDENUM_PARAMS = AtomicParameters(
    D=20.1000,
    a=0.9996,
    alpha=1.1500,
    b=0.0026,
    beta=1.2439,
    z0=3.2200,
    z1=4.1864,
    centre=(-1 / 2, 1 / 6),
)


def _get_vertical_factor(
    z: np.ndarray[Any, np.dtype[np.floating]],
    params: AtomicParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the Modifed Morse potential that descrives the z dependence of V(x)."""
    return params.D * (
        np.exp(2 * params.alpha * (params.z0 - z))
        - 2 * params.a * np.exp(params.alpha * (params.z0 - z))
        - 2 * params.b * np.exp(2 * params.beta * (params.z1 - z))
    )


def _get_horizontal_factor(
    x: np.ndarray[Any, np.dtype[np.floating]],
    y: np.ndarray[Any, np.dtype[np.floating]],
    params: AtomicParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the function describing the x and y dependence of V(x)."""
    relative_x = (x / MOS2_LATTICE_CONSTANT) - params.centre[0]
    relative_y = y / (MOS2_LATTICE_CONSTANT * np.sqrt(3)) - params.centre[1]
    return (
        np.cos(2 * np.pi * (relative_x - relative_y))
        + np.cos(4 * np.pi * relative_y)
        + np.cos(2 * np.pi * (relative_x + relative_y))
        + 1.5
    ) / 4.5


def _get_atomic_factor(
    positions: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    params: AtomicParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the contribution of a single atomic site to the total potential."""
    x, y, z = positions[0], positions[1], positions[2]
    return _get_vertical_factor(z, params) * _get_horizontal_factor(x, y, params)


def _mos2_potential(
    positions: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
    """Get the model MoS2 potential."""
    # Sulphur site
    v_s = _get_atomic_factor(positions, SULPHUR_PARAMS)
    # Hollow site
    v_h = _get_atomic_factor(positions, HOLLOW_PARAMS)
    # Molybdenum site
    v_m = _get_atomic_factor(positions, MOLYBDENUM_PARAMS)

    return (v_s + v_h + v_m).astype(np.complex128)


def get_mos2_metadata[MZ: LengthMetadata](
    shape: tuple[int, int],
    metadata_z: MZ,
    *,
    units: UnitSystem | None = None,
) -> ScatteringBasisMetadata[EvenlySpacedLengthMetadata, MZ, AxisDirections]:
    units = units or UnitSystem()
    lattice_constant = MOS2_LATTICE_CONSTANT * units.angstrom
    a1 = np.array([-lattice_constant, 0])
    a2 = np.array([lattice_constant / 2, lattice_constant * np.sqrt(3) / 2])

    metadata_xy = _metadata.spaced_volume_metadata_from_stacked_delta_x((a1, a2), shape)
    return combine_scattering_metadata(metadata_xy, metadata_z)


def build_mos2_potential[MZ: LengthMetadata](
    shape: tuple[int, int],
    metadata_z: MZ,
    *,
    units: UnitSystem | None = None,
) -> Operator[
    OperatorBasis[
        ScatteringBasisMetadata[EvenlySpacedLengthMetadata, MZ, AxisDirections],
    ],
    np.dtype[np.complexfloating],
]:
    units = units or UnitSystem()
    metadata = get_mos2_metadata(shape, metadata_z, units=units)

    return as_scattering_potential(
        operator.build.potential_from_function(
            metadata,
            lambda positions: (
                _mos2_potential(tuple(i / units.angstrom for i in positions))
                * (units.electron_volt * 1e-3)
            ),
        ),
        metadata,
    )


if __name__ == "__main__":
    potential = build_mos2_potential(
        (15, 15),
        LobattoSpacedLengthMetadata(200, domain=_metadata.Domain(delta=10e-10)),
    )

    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=HELIUM_ENERGY,
        theta=np.deg2rad(30),
        phi=np.deg2rad(0),
        potential=potential,
    )
    config = OptimizationConfig(precision=1e-5, max_iterations=10, n_channels=80)
    s_matrix = get_scattering_matrix(condition, config, backend="scipy")

    fig, ax, _mesh = plot.array_against_axes_2d_k(s_matrix, measure="abs")
    ax.set_title("The scattering matrix")
    fig.show()

    plot.wait_for_close()
