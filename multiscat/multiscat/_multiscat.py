from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from slate_core import (
    Array,
    Basis,
    TupleMetadata,
    basis,
)
from slate_core.basis import AsUpcast
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
)
from slate_core.util import timed
from slate_quantum import State

from multiscat.basis import (
    ScatteringBasisMetadata,
    close_coupling_basis,
    split_scattering_metadata,
)
from multiscat.config import UnitSystem
from multiscat.multiscat._fortran import run_multiscat_fortran
from multiscat.multiscat._scipy import (
    _build_scipy_operators,
    get_scattering_state_scipy,
    run_multiscat_scipy,
)
from multiscat.multiscat._scipy_von_neumann import run_multiscat_scipy_von_neumann
from multiscat.multiscat._util import (
    get_ab_wave_for_condition,  # type: ignore[import-untyped]
)

if TYPE_CHECKING:
    from multiscat.config import OptimizationConfig, ScatteringCondition


def _get_scattered_intensity_data[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    solution: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    condition: ScatteringCondition[M0, M1, E],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Recover per-channel intensities from the optimized scattered state."""
    a_wave, b_wave = get_ab_wave_for_condition(condition)

    surface_solution = solution[:, :, -1]
    # b_wave is the inverse of the outgoing wave amplitude
    # b_wave is equal to o(r)^(-1)
    # This therefore recovers the scattered state from the
    # log derivative
    surface_state = 2.0j * b_wave * surface_solution
    # a_wave[0,0] is i(r) o(r)^(-1)
    # we are subtracting the incoming component
    surface_state[0, 0] += a_wave[0, 0]
    return np.abs(surface_state) ** 2


def get_scattering_matrix_from_state[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[
        Basis[ScatteringBasisMetadata[M0, M1, E]],
        np.dtype[np.complex128],
    ],
    condition: ScatteringCondition[M0, M1, E],
    *,
    n_channels: int | None = None,
) -> Array[
    Basis[TupleMetadata[tuple[M0, M0], AxisDirections]],
    np.dtype[np.complex128],
]:
    """Recover per-channel intensities from the optimized scattered state."""
    converted_condition = _as_natural_units(condition)
    inverse_lower, _lower, _upper = _build_scipy_operators(
        converted_condition,
        n_channels=n_channels,
    )

    solution = inverse_lower.matvec(
        state.with_basis(close_coupling_basis(condition.metadata)).raw_data,
    ).reshape(condition.metadata.shape)

    channel_intensity = _get_scattered_intensity_data(
        solution,
        converted_condition,
    )

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return Array(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
        channel_intensity.astype(np.complex128),
    )


def _as_natural_units[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
) -> ScatteringCondition[
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
    AxisDirections,
]:
    # Here, we convert the scattering condition to the natural units of the problem
    # In these units, the kinetic energy is simply k^2 which simplifies the
    # later calculations significantly.
    # To do this, we set hbar = 1, and the condition.mass = 1/2. This means scaling the
    # value of the atomic mass by a factor of 1 / (2 * condition.mass).
    out = condition.with_units(
        UnitSystem(
            angstrom=1.0,
            atomic_mass=0.5 * condition.units.atomic_mass / condition.mass,
            hbar=1.0,
        ),
    )

    # This is a quick check to make sure that the mass is indeed 1/2 in the new units.
    assert np.isclose(out.mass, 1 / 2)  # noqa: S101
    return out


@timed
def get_scattering_matrix[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
    *,
    backend: Literal["fortran", "scipy"] = "fortran",
) -> Array[
    Basis[TupleMetadata[tuple[M0, M0], AxisDirections]],
    np.dtype[np.complex128],
]:
    """
    Perform a scattering calculation and return the scattering matrix.

    This provides the amplitude of the outgoing wave in each channel.

    This uses the invers lower operator as a preconditioner. It splits
    the operator into (D + L + U), and preconditions by (D+L^(-1)).

    This is more accurate than the von-neumann approach (requires less interations),
    but applying (D+L^(-1)) is less parrallelizable.
    """
    converted_condition = _as_natural_units(condition)

    if backend == "fortran":
        solution = run_multiscat_fortran(converted_condition, config)
    elif backend == "scipy":
        solution = run_multiscat_scipy(converted_condition, config)
    else:
        msg = f"Unknown backend '{backend}'. Expected 'fortran' or 'scipy'."
        raise ValueError(msg)

    channel_intensity = _get_scattered_intensity_data(
        solution,
        converted_condition,
    )

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return Array(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
        channel_intensity.astype(np.complex128),
    )


@timed
def get_scattering_matrix_von_neumann[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
    *,
    order: int = 1,
) -> Array[
    Basis[TupleMetadata[tuple[M0, M0], AxisDirections]],
    np.dtype[np.complex128],
]:
    """
    Perform a scattering calculation and return the scattering matrix.

    This provides the amplitude of the outgoing wave in each channel.

    Unlike the standard scattering matrix calculation, this uses a von Neumann
    approximation for the inverse.

    It splits the operator into (D + V_scatter), where D is the uncoupled diagonal
    block operator, and V_scatter is the inter-channel scattering potential
    (excluding specular).

    It uses the preconditioner
    (D + V_scatter)^(-1) ~ sum_{k=0}^order (-D^(-1) V_scatter)^k D^(-1))
    which may be more parallelizable than the standard (D+L)^(-1) approach.
    """
    converted_condition = _as_natural_units(condition)
    solution = run_multiscat_scipy_von_neumann(
        converted_condition,
        config,
        order=order,
    )

    channel_intensity = _get_scattered_intensity_data(
        solution,
        converted_condition,
    )

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return Array(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
        channel_intensity.astype(np.complex128),
    )


@timed
def get_scattering_state[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
) -> State[
    Basis[ScatteringBasisMetadata[M0, M1, E]],
    np.dtype[np.complex128],
]:
    """Get the full scattering state, including the interior."""
    converted_condition = _as_natural_units(condition)
    solution = get_scattering_state_scipy(converted_condition, config)

    return State(
        close_coupling_basis(condition.metadata).upcast(),
        solution,
    )
