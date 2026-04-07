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
from multiscat.config import UnitSystem, with_units
from multiscat.multiscat._fortran import run_multiscat_fortran
from multiscat.multiscat._scipy import get_scattering_state_scipy, run_multiscat_scipy
from multiscat.multiscat._scipy_von_neumann import run_multiscat_scipy_von_neumann
from multiscat.multiscat._util import (
    get_ab_wave_for_condition,  # type: ignore[import-untyped]
)

if TYPE_CHECKING:
    from multiscat.config import OptimizationConfig, ScatteringCondition


def get_scattered_intensity[
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
    out = with_units(
        condition,
        UnitSystem(
            angstrom=1.0,
            atomic_mass=0.5 * condition.units.atomic_mass / condition.mass,
            hbar=1.0,
        ),
    )

    # In these units, the kinetic energy is simply k^2
    assert out.mass == (1 / 2)  # noqa: S101
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
    """Run Multiscat through the f2py native binding."""
    converted_condition = _as_natural_units(condition)

    if backend == "fortran":
        solution = run_multiscat_fortran(converted_condition, config)
    elif backend == "scipy":
        solution = run_multiscat_scipy(converted_condition, config)
    else:
        msg = f"Unknown backend '{backend}'. Expected 'fortran' or 'scipy'."
        raise ValueError(msg)

    channel_intensity = get_scattered_intensity(
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
    """Run Multiscat with the scipy backend using a von Neumann preconditioner."""
    converted_condition = _as_natural_units(condition)
    solution = run_multiscat_scipy_von_neumann(
        converted_condition,
        config,
        order=order,
    )

    channel_intensity = get_scattered_intensity(
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
    """Run Multiscat through the f2py native binding."""
    converted_condition = _as_natural_units(condition)
    solution = get_scattering_state_scipy(converted_condition, config)

    return State(
        close_coupling_basis(condition.metadata).upcast(),
        solution,
    )
