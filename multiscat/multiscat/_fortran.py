from typing import TYPE_CHECKING

import numpy as np

from multiscat.multiscat._util import (
    get_ab_waves,
    get_outgoing_log_derivative_wave,
    get_parallel_kinetic_energy,
    get_perpendicular_kinetic_difference,
    potential_as_array,  # type: ignore[import-untyped]
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from slate_core import (
        TupleMetadata,
    )

try:
    from multiscat_fortran import (  # type: ignore[optional]
        run_multiscat_fortran as run_multiscat_fortran_raw,  # type: ignore[optional]
    )
except ImportError:

    def run_multiscat_fortran_raw(  # noqa: PLR0913
        gmres_preconditioner_flag: int,  # noqa: ARG001
        convergence_significant_figures: int,  # noqa: ARG001
        potential_values: NDArray[np.complex128],  # noqa: ARG001
        perpendicular_kinetic_difference: NDArray[np.float64],  # noqa: ARG001
        wave_a: NDArray[np.complex128],  # noqa: ARG001
        wave_b: NDArray[np.complex128],  # noqa: ARG001
        wave_c: NDArray[np.complex128],  # noqa: ARG001
        parallel_kinetic_energy: NDArray[np.float64],  # noqa: ARG001
    ) -> NDArray[np.complex128]:
        msg = (
            "Multiscat Fortran code is not available."
            " Please ensure that the Fortran package is installed."
        )
        raise ImportError(msg)


from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
)

from multiscat.basis import (
    split_scattering_metadata,
)

if TYPE_CHECKING:
    from multiscat.config import OptimizationConfig, ScatteringCondition

    def run_multiscat_fortran_raw(  # noqa: PLR0913
        gmres_preconditioner_flag: int,
        convergence_significant_figures: int,
        potential_values: NDArray[np.complex128],
        perpendicular_kinetic_difference: NDArray[np.float64],
        wave_a: NDArray[np.complex128],
        wave_b: NDArray[np.complex128],
        wave_c: NDArray[np.complex128],
        parallel_kinetic_energy: NDArray[np.float64],
    ) -> NDArray[np.complex128]: ...


def _condition_parameters[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[
        M0,
        M1,
        E,
    ],
) -> tuple[
    tuple[float, float, float],
    np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    TupleMetadata[
        tuple[EvenlySpacedLengthMetadata, EvenlySpacedLengthMetadata],
        AxisDirections,
    ],
    LobattoSpacedLengthMetadata,
]:

    potential = potential_as_array(condition.potential)
    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)
    return (condition.incident_k, potential, metadata_x01, metadata_z)


def _optimization_parameters(config: OptimizationConfig) -> tuple[int, int]:
    if config.precision <= 0:
        msg = "Optimization precision must be greater than zero"
        raise ValueError(msg)

    preconditioner_flag = 1 if config.use_neumann_preconditioner else 0
    n_significant_figures = int(np.log10(1 / config.precision))
    return (
        preconditioner_flag,
        n_significant_figures,
    )


def run_multiscat_fortran[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[
        M0,
        M1,
        E,
    ],
    config: OptimizationConfig,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    (
        incident_k,
        potential,
        metadata_xy,
        metadata_z,
    ) = _condition_parameters(condition)
    perpendicular_kinetic_difference = get_perpendicular_kinetic_difference(
        incident_k,
        metadata_xy,
    )
    parallel_kinetic_energy = -get_parallel_kinetic_energy(metadata_z)

    a_wave, b_wave = get_ab_waves(
        metadata_z,
        perpendicular_kinetic_difference,
    )

    outgoing_log_derivative_wave = get_outgoing_log_derivative_wave(
        metadata_z,
        perpendicular_kinetic_difference.ravel(),
    )

    (
        preconditioner_flag,
        n_significant_figures,
    ) = _optimization_parameters(config)

    if config.n_channels is not None:
        msg = "Channel filtering is not yet implemented in fortran code"
        raise NotImplementedError(msg)
    return run_multiscat_fortran_raw(
        preconditioner_flag,
        n_significant_figures,
        potential_values=potential,
        perpendicular_kinetic_difference=perpendicular_kinetic_difference,
        wave_a=a_wave.ravel(),
        wave_b=b_wave.ravel(),
        wave_c=outgoing_log_derivative_wave.ravel(),
        parallel_kinetic_energy=parallel_kinetic_energy,
    )
