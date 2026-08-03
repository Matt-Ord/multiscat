from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import Array
from multiscat_fortran import (
    debug_apply_upper_block_fortran,
    debug_build_preconditioner_fortran,
    debug_solve_lower_block_fortran,
    get_abc_arrays,
    get_parallel_kinetic_energy,
    get_perpendicular_kinetic_difference,
)
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core import EvenlySpacedLengthMetadata, array, basis, plot
from slate_core.metadata import LobattoSpacedLengthMetadata
from slate_quantum import operator

from multiscat.basis import (
    as_scattering_potential,
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
    split_scattering_metadata,
)
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.multiscat import (
    get_scattering_matrix,
    get_scattering_matrix_from_state,
    get_scattering_matrix_von_neumann,
    get_scattering_state,
)
from multiscat.multiscat._fortran import (
    _condition_parameters,
)
from multiscat.multiscat._scipy import (
    _build_lower_block_factors,
    _build_scipy_operators,
    _solve_specular_hamiltonian,
    get_scattering_state_scipy,
)
from multiscat.multiscat._scipy_jax import (
    apply_inverse_lower_block,
    apply_lower_op,
    apply_upper_op,
    get_scattering_state_scipy_jax,
)
from multiscat.multiscat._scipy_jax import (
    build_scipy_operator_data as build_operator_data_jax,
)
from multiscat.multiscat._util import (
    get_ab_waves as _get_ab_waves,
)
from multiscat.multiscat._util import (
    get_outgoing_log_derivative_wave as _get_outgoing_log_derivative_wave,
)
from multiscat.multiscat._util import (
    get_parallel_kinetic_energy as _get_parallel_kinetic_energy,
)
from multiscat.multiscat._util import (
    get_perpendicular_kinetic_difference as _get_perpendicular_kinetic_difference,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_core.metadata import AxisDirections

    from multiscat.interpolate import ScatteringOperator

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
HELIUM_MASS = physical_constants["alpha particle mass"][0]
UNIT_CELL = 2.84 * angstrom
Z_HEIGHT = 8 * angstrom

MORSE_PARAMETERS = operator.build.CorrugatedMorseParameters(
    depth=7.63 * electron_volt * 10**-3,
    height=(1.0 / 1.1) * angstrom,
    offset=3.0 * angstrom,
    beta=0.10,
)


def _fft_mode_to_index(mode: int, n: int) -> int:
    return mode if mode >= 0 else n + mode


def _parse_raw_intensities(
    output_file: Path,
    shape: tuple[int, int],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    # Regex for lines without the '#' prefix: two ints and one float
    pattern = re.compile(
        r"^\s*(-?\d+)\s+(-?\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*",
    )
    nx, ny = shape
    intensities = np.zeros((nx, ny), dtype=np.float64)

    with output_file.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            match = pattern.match(line)
            if not match:
                continue

            hx = int(match.group(1))
            ky = int(match.group(2))
            value = float(match.group(3))
            intensities[_fft_mode_to_index(hx, nx), _fft_mode_to_index(ky, ny)] = value

    return intensities


def _raw_potential_in_input_file_convention(
    potential: ScatteringOperator[
        EvenlySpacedLengthMetadata,
        LobattoSpacedLengthMetadata,
        AxisDirections,
    ],
) -> np.ndarray[Any, np.dtype[np.complex128]]:

    potential_diagonal = array.extract_diagonal(potential)
    nx, ny, nz = potential_diagonal.basis.metadata().shape
    basis_weights = potential_diagonal.basis.metadata().children[2].basis_weights
    basis = close_coupling_basis(potential_diagonal.basis.metadata())

    data = potential_diagonal.with_basis(basis).raw_data
    data = data.reshape((nx, ny, nz)) * (basis_weights[np.newaxis, np.newaxis, :])

    # Multiscat uses a slightly different fourier convention
    return data.ravel() / (electron_volt * 10**-3 * np.sqrt(nx * ny))


def _simple_example_condition() -> tuple[
    ScatteringCondition[
        EvenlySpacedLengthMetadata,
        LobattoSpacedLengthMetadata,
        AxisDirections,
    ],
    OptimizationConfig,
]:

    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL, 0, 0]),
            np.array([0, UNIT_CELL, 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        (10, 10, 550),
    )
    # This is taken from https://doi.org/10.1039/FT9908601641
    # and is a reproduction of the Wolken 4He-LiF problem in table 1,
    # originally simulated in https://doi.org/10.1063/1.1679617.
    potential = as_scattering_potential(
        operator.build.corrugated_morse_potential(
            metadata,
            MORSE_PARAMETERS,
        ),
        metadata,
    )
    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=20 * electron_volt * 10**-3,
        theta=np.deg2rad(30),
        phi=np.deg2rad(0),
        potential=potential,
    )
    config = OptimizationConfig(precision=1e-5, max_iterations=1000)
    return condition, config


def _fortran_backend_inputs(
    condition: ScatteringCondition[
        EvenlySpacedLengthMetadata,
        LobattoSpacedLengthMetadata,
        AxisDirections,
    ],
) -> tuple[
    np.ndarray[Any, np.dtype[np.complex128]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.complex128]],
    np.ndarray[Any, np.dtype[np.complex128]],
    np.ndarray[Any, np.dtype[np.complex128]],
]:
    (incident_k, potential_values, metadata_xy, metadata_z) = _condition_parameters(
        condition,
    )

    directions = metadata_xy.extra.vectors
    x_vector = np.asarray(directions[0]) * metadata_xy.children[0].domain.delta
    y_vector = np.asarray(directions[1]) * metadata_xy.children[1].domain.delta
    ax_angstrom = float(x_vector[0])
    ay_angstrom = float(x_vector[1])
    bx_angstrom = float(y_vector[0])
    by_angstrom = float(y_vector[1])

    perpendicular_kinetic_difference_raw = get_perpendicular_kinetic_difference(
        *incident_k,
        nx=potential_values.shape[0],
        ny=potential_values.shape[1],
        unit_cell_ax=ax_angstrom,
        unit_cell_ay=ay_angstrom,
        unit_cell_bx=bx_angstrom,
        unit_cell_by=by_angstrom,
    )
    parallel_kinetic_energy_raw = get_parallel_kinetic_energy(
        zmin=metadata_z.domain.start,
        zmax=metadata_z.domain.end,
        nz=metadata_z.fundamental_size,
    )

    wave_a_raw, wave_b_raw, wave_c_raw = get_abc_arrays(
        zmin=metadata_z.domain.start,
        zmax=metadata_z.domain.end,
        nx=potential_values.shape[0],
        ny=potential_values.shape[1],
        perpendicular_kinetic_difference=perpendicular_kinetic_difference_raw,
        n_z_points=metadata_z.fundamental_size,
    )

    return (
        np.asarray(potential_values, dtype=np.complex128),
        np.asarray(perpendicular_kinetic_difference_raw, dtype=np.float64),
        np.asarray(parallel_kinetic_energy_raw, dtype=np.float64),
        np.asarray(wave_a_raw, dtype=np.complex128),
        np.asarray(wave_b_raw, dtype=np.complex128),
        np.asarray(wave_c_raw, dtype=np.complex128),
    )


def test_simple_system() -> None:

    condition, config = _simple_example_condition()
    s_matrix = get_scattering_matrix(condition, config)
    intensities = np.real_if_close(
        s_matrix.with_basis(
            basis.transformed_from_metadata(s_matrix.basis.metadata()),
        ).raw_data.reshape(condition.metadata.shape[:2]),
    )
    nx, ny = intensities.shape

    if intensities.size == 0:
        msg = "Expected at least one diffraction intensity"
        raise AssertionError(msg)

    if not math.isclose((np.sum(intensities)), 1.0, abs_tol=1e-6):
        msg = f"Intensities sum to {(np.sum(intensities))}, expected 1.0"
        raise AssertionError(msg)

    expected = _parse_raw_intensities(
        TESTS_DIR / "data" / Path("expected_intensities.txt"),
        (nx, ny),
    )
    np.testing.assert_allclose(intensities, expected, rtol=0.0, atol=1e-5)


def _rotated_example_condition() -> tuple[
    ScatteringCondition[
        EvenlySpacedLengthMetadata,
        LobattoSpacedLengthMetadata,
        AxisDirections,
    ],
    OptimizationConfig,
]:

    rotation = np.deg2rad(20.0)
    cos_t = np.cos(rotation)
    sin_t = np.sin(rotation)
    rotation_matrix = np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    x_vector = rotation_matrix @ np.array([UNIT_CELL, 0.0, 0.0])
    y_vector = rotation_matrix @ np.array([0.0, UNIT_CELL, 0.0])

    metadata = scattering_metadata_from_stacked_delta_x(
        (
            x_vector,
            y_vector,
            np.array([0.0, 0.0, Z_HEIGHT]),
        ),
        (10, 10, 550),
    )

    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=20 * electron_volt * 10**-3,
        theta=np.deg2rad(30),
        phi=0,
        potential=as_scattering_potential(
            operator.build.corrugated_morse_potential(
                metadata,
                MORSE_PARAMETERS,
            ),
            metadata,
        ),
    )
    config = OptimizationConfig(precision=1e-5, max_iterations=1000)
    return condition, config


def test_rotated_system() -> None:

    condition, config = _rotated_example_condition()
    s_matrix = get_scattering_matrix(condition, config)
    intensities = np.real_if_close(
        s_matrix.with_basis(
            basis.transformed_from_metadata(s_matrix.basis.metadata()),
        ).raw_data.reshape(condition.metadata.shape[:2]),
    )
    nx, ny = intensities.shape

    if intensities.size == 0:
        msg = "Expected at least one diffraction intensity"
        raise AssertionError(msg)

    if not math.isclose((np.sum(intensities)), 1.0, abs_tol=1e-6):
        msg = f"Intensities sum to {(np.sum(intensities))}, expected 1.0"
        raise AssertionError(msg)

    expected = _parse_raw_intensities(
        TESTS_DIR / "data" / Path("expected_intensities.txt"),
        (nx, ny),
    )
    np.testing.assert_allclose(intensities, expected, rtol=0.0, atol=1e-5)


def test_simple_system_scipy_backend() -> None:
    condition, config = _simple_example_condition()
    s_matrix = get_scattering_matrix(condition, config, backend="scipy")
    intensities = np.real_if_close(
        s_matrix.with_basis(
            basis.transformed_from_metadata(s_matrix.basis.metadata()),
        ).raw_data.reshape(condition.metadata.shape[:2]),
    )
    nx, ny = intensities.shape

    if intensities.size == 0:
        msg = "Expected at least one diffraction intensity"
        raise AssertionError(msg)

    if not math.isclose((np.sum(intensities)), 1.0, abs_tol=1e-5):
        msg = f"Intensities sum to {(np.sum(intensities))}, expected 1.0"
        raise AssertionError(msg)

    expected = _parse_raw_intensities(
        TESTS_DIR / "data" / Path("expected_intensities.txt"),
        (nx, ny),
    )
    np.testing.assert_allclose(intensities, expected, rtol=0.0, atol=1e-5)


def test_simple_system_scattering_matrix_from_state_matches_direct() -> None:
    condition, config = _simple_example_condition()
    direct = get_scattering_matrix(condition, config, backend="scipy")
    state = get_scattering_state(condition, config)
    recovered = get_scattering_matrix_from_state(
        state,
        condition,
        n_channels=config.n_channels,
    )

    direct_intensities = np.real_if_close(
        direct.with_basis(
            basis.transformed_from_metadata(direct.basis.metadata()),
        ).raw_data.reshape(condition.metadata.shape[:2]),
    )
    recovered_intensities = np.real_if_close(
        recovered.with_basis(
            basis.transformed_from_metadata(recovered.basis.metadata()),
        ).raw_data.reshape(condition.metadata.shape[:2]),
    )

    np.testing.assert_allclose(
        recovered_intensities,
        direct_intensities,
        rtol=0.0,
        atol=1e-5,
    )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_simple_system_scipy_von_neumann_matches_scipy(order: int) -> None:
    condition, config = _simple_example_condition()
    reference = get_scattering_matrix(condition, config, backend="scipy")
    reference_intensities = np.real_if_close(
        reference.with_basis(
            basis.transformed_from_metadata(reference.basis.metadata()),
        ).raw_data.reshape(condition.metadata.shape[:2]),
    )

    s_matrix = get_scattering_matrix_von_neumann(condition, config, order=order)
    intensities = np.real_if_close(
        s_matrix.with_basis(
            basis.transformed_from_metadata(s_matrix.basis.metadata()),
        ).raw_data.reshape(condition.metadata.shape[:2]),
    )
    np.testing.assert_allclose(
        intensities,
        reference_intensities,
        rtol=0.0,
        atol=1e-5,
    )


def test_scipy_von_neumann_invalid_order_raises() -> None:
    condition, config = _simple_example_condition()
    with np.testing.assert_raises(ValueError):
        get_scattering_matrix_von_neumann(condition, config, order=-1)


def test_scipy_preconditioner_matches_fortran_debug() -> None:
    condition, _ = _simple_example_condition()
    (
        potential_values,
        perpendicular_kinetic_difference,
        parallel_kinetic_energy,
        _wave_a,
        _wave_b,
        _wave_c,
    ) = _fortran_backend_inputs(condition)
    eigenvalues_python, eigenvectors_python = _solve_specular_hamiltonian(
        potential_values[0, 0],
        parallel_kinetic_energy,
    )
    lower_block_factors_python = _build_lower_block_factors(
        perpendicular_kinetic_difference.ravel(),
        eigenvalues_python,
        eigenvectors_python,
    )

    eigenvalues_raw, preconditioner_factors_raw, eigenvectors_raw = (
        debug_build_preconditioner_fortran(
            potential_values=potential_values,
            perpendicular_kinetic_difference=perpendicular_kinetic_difference,
            parallel_kinetic_energy=parallel_kinetic_energy,
        )
    )

    eigenvalues = np.asarray(eigenvalues_raw, dtype=np.float64)
    preconditioner_factors = np.asarray(preconditioner_factors_raw, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors_raw, dtype=np.float64)

    np.testing.assert_allclose(
        eigenvalues_python,
        eigenvalues,
        rtol=3e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.abs(eigenvectors_python),
        np.abs(eigenvectors),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        lower_block_factors_python,
        preconditioner_factors,
        rtol=1e-9,
        atol=1e-10,
    )


def test_python_abc_arrays_match_fortran() -> None:
    condition, _ = _simple_example_condition()
    (
        _,
        _,
        _,
        metadata_z,
    ) = _condition_parameters(condition)
    (
        _scaled_potential_values,
        perpendicular_kinetic_difference,
        _parallel_kinetic_energy,
        wave_a_fortran,
        wave_b_fortran,
        wave_c_fortran,
    ) = _fortran_backend_inputs(condition)

    wave_c_python = _get_outgoing_log_derivative_wave(
        LobattoSpacedLengthMetadata(
            metadata_z.fundamental_size + 1,
            domain=metadata_z.domain,
        ),
        perpendicular_kinetic_difference=perpendicular_kinetic_difference.ravel(),
    )

    wave_a_python, wave_b_python = _get_ab_waves(
        LobattoSpacedLengthMetadata(
            metadata_z.fundamental_size + 1,
            domain=metadata_z.domain,
        ),
        perpendicular_kinetic_difference=perpendicular_kinetic_difference,
    )

    shape = perpendicular_kinetic_difference.shape
    np.testing.assert_allclose(
        wave_a_python,
        np.asarray(wave_a_fortran, dtype=np.complex128).reshape(shape),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        wave_b_python,
        np.asarray(wave_b_fortran, dtype=np.complex128).reshape(shape),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.array(wave_c_python).reshape(shape),
        np.asarray(wave_c_fortran, dtype=np.complex128).reshape(shape),
        rtol=1e-12,
        atol=1e-12,
    )


def test_scipy_upper_block_matches_fortran_debug() -> None:
    condition, _ = _simple_example_condition()
    (
        potential_values,
        _perpendicular_kinetic_difference,
        _parallel_kinetic_energy,
        _wave_a,
        _wave_b,
        _wave_c,
    ) = _fortran_backend_inputs(condition)

    _inverse_lower, _lower, upper = _build_scipy_operators(
        condition,
    )
    nkx, nky, nz = potential_values.shape
    rng = np.random.default_rng()
    state_in = (
        rng.standard_normal((nkx, nky, nz)) + 1j * rng.standard_normal((nkx, nky, nz))
    ).astype(np.complex128)

    state_in_raw = state_in.reshape((nkx * nky, nz)).T
    state_out_fortran_raw = debug_apply_upper_block_fortran(
        potential_values=potential_values,
        state_in=state_in_raw,
    )

    state_out_fortran = state_out_fortran_raw.reshape((nz, nkx * nky)).T.ravel()
    state_out_python = upper.matvec(state_in.ravel())  # type: ignore[unknown]
    np.testing.assert_allclose(
        state_out_python,  # type: ignore[unknown]
        state_out_fortran,
        rtol=1e-9,
        atol=1e-10,
    )


def test_scipy_lower_block_matches_fortran_debug() -> None:
    condition, _ = _simple_example_condition()
    (
        potential_values,
        perpendicular_kinetic_difference,
        parallel_kinetic_energy,
        _wave_a,
        _wave_b,
        wave_c,
    ) = _fortran_backend_inputs(condition)

    inverse_lower, _lower, _upper = _build_scipy_operators(
        condition,
    )
    rng = np.random.default_rng()
    nkx, nky, nz = potential_values.shape
    rng = np.random.default_rng()
    state_in = (
        rng.standard_normal((nkx, nky, nz)) + 1j * rng.standard_normal((nkx, nky, nz))
    ).astype(np.complex128)

    state_in_raw = state_in.reshape((nkx * nky, nz)).T
    state_out_fortran_raw = debug_solve_lower_block_fortran(
        potential_values=potential_values,
        wave_c=wave_c,
        perpendicular_kinetic_difference=perpendicular_kinetic_difference,
        parallel_kinetic_energy=parallel_kinetic_energy,
        state_in=state_in_raw,
    )

    state_out_fortran = state_out_fortran_raw.reshape((nz, nkx * nky)).T.ravel()
    state_out_python = inverse_lower.matvec(state_in.ravel())  # type: ignore[unknown]
    np.testing.assert_allclose(
        state_out_python,  # type: ignore[unknown]
        state_out_fortran,
        rtol=1e-9,
        atol=1e-10,
    )


def test_raw_potential_in_input_file_convention() -> None:
    condition, _ = _simple_example_condition()
    from_condition = _raw_potential_in_input_file_convention(condition.potential)

    reference_potential = TESTS_DIR / "data" / "expected_potential.npy"
    expected = np.load(reference_potential)

    np.testing.assert_equal(expected.shape, from_condition.shape)
    np.testing.assert_allclose((from_condition), (expected), rtol=1e-5, atol=1e-10)


def test_scipy_lower_block_and_inverse_are_inverses() -> None:
    condition, _ = _simple_example_condition()
    inverse_lower, lower, _upper = _build_scipy_operators(condition)

    nkx, nky, nz = condition.metadata.shape
    rng = np.random.default_rng(123)
    state_in = (
        rng.standard_normal((nkx, nky, nz)) + 1j * rng.standard_normal((nkx, nky, nz))
    ).astype(np.complex128)
    flat_state = state_in.ravel()

    recovered_after_inverse_then_lower = lower.matvec(inverse_lower.matvec(flat_state))  # type: ignore[unknown]
    recovered_after_lower_then_inverse = inverse_lower.matvec(lower.matvec(flat_state))  # type: ignore[unknown]

    np.testing.assert_allclose(
        recovered_after_inverse_then_lower,  # type: ignore[unknown]
        flat_state,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        recovered_after_lower_then_inverse,  # type: ignore[unknown]
        flat_state,
        rtol=1e-6,
        atol=1e-7,
    )


def test_perpendicular_kinetic_difference_matches_fortran() -> None:
    condition, _ = _simple_example_condition()
    metadata = condition.metadata
    nx, ny, _ = metadata.shape

    metadata_x01, _ = split_scattering_metadata(metadata)
    expected = _get_perpendicular_kinetic_difference(
        condition.incident_k,
        metadata_x01,
    )

    directions = metadata.extra.vectors
    x_vector = np.asarray(directions[0]) * metadata_x01.children[0].domain.delta
    y_vector = np.asarray(directions[1]) * metadata_x01.children[1].domain.delta

    incident_kx, incident_ky, incident_kz = condition.incident_k

    actual_raw = get_perpendicular_kinetic_difference(
        incident_kx=float(incident_kx),
        incident_ky=float(incident_ky),
        incident_kz=float(incident_kz),
        nx=int(nx),
        ny=int(ny),
        unit_cell_ax=float(x_vector[0]),
        unit_cell_ay=float(x_vector[1]),
        unit_cell_bx=float(y_vector[0]),
        unit_cell_by=float(y_vector[1]),
    )
    actual = np.asarray(actual_raw, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_parallel_kinetic_energy_matches_fortran() -> None:
    condition, _ = _simple_example_condition()
    metadata = condition.metadata
    _, _, nz = metadata.shape

    expected = _get_parallel_kinetic_energy(metadata.children[2])

    z_domain = metadata.children[2].domain
    zmin = float(z_domain.start)
    zmax = float(z_domain.start + z_domain.delta)

    actual_raw = get_parallel_kinetic_energy(
        zmin=zmin,
        zmax=zmax,
        nz=int(nz - 1),
    )
    actual = np.asarray(actual_raw, dtype=np.float64)

    # The Fortran assembly uses a shifted Lobatto block and n-1 size.
    # Match that block from Python and convert conventions with basis weights.
    expected = expected[1:, 1:]
    np.testing.assert_allclose(actual, -expected, rtol=2e-6)


type JaxResult = Array | np.ndarray | tuple["JaxResult", ...] | list["JaxResult"]


def _force_ready(result: JaxResult) -> None:
    """Block until any JAX arrays in `result` are actually computed."""
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()  # ty: ignore[call-non-callable]
    elif isinstance(result, (tuple, list)):
        for r in result:
            _force_ready(r)


def time_call(
    fn: Callable[..., object],
    *args: object,
    n_repeats: int = 1,
    **kwargs: object,
) -> list[float]:
    """Return wall-clock times for n_repeats calls (first entry = cold, rest = warm)."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _force_ready(cast("JaxResult", result))
        times.append(time.perf_counter() - t0)
    return times


def summarize(times: list[float]) -> str:
    arr = np.asarray(times)
    return (
        f"mean={arr.mean():.4f}s  std={arr.std():.4f}s  "
        f"min={arr.min():.4f}s  n={len(arr)}"
    )


@dataclass
class ProblemSize:
    """Describes a benchmark problem's grid dimensions and physical parameters."""

    label: str
    grid_shape: tuple[int, int, int]  # (nkx, nky, nz)
    z_height_angstrom: float
    n_channels: int | None


def build_condition_and_config(
    size: ProblemSize,
    *,
    theta_deg: float = 30.0,
    phi_deg: float = 0.0,
) -> tuple[ScatteringCondition, OptimizationConfig]:
    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL, 0, 0]),
            np.array([0, UNIT_CELL, 0]),
            np.array([0, 0, size.z_height_angstrom * angstrom]),
        ),
        size.grid_shape,
    )
    potential = as_scattering_potential(
        operator.build.corrugated_morse_potential(metadata, MORSE_PARAMETERS),
        metadata,
    )
    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=20 * electron_volt * 10**-3,
        theta=np.deg2rad(theta_deg),
        phi=np.deg2rad(phi_deg),
        potential=potential,
    )
    config = OptimizationConfig(
        precision=1e-5,
        max_iterations=1000,
        n_channels=size.n_channels,
    )
    return condition, config


def benchmark_warm_vs_cold(size: ProblemSize, *, n_repeats: int = 3) -> None:
    condition, config = build_condition_and_config(size)

    time_call(
        get_scattering_state_scipy,
        condition,
        config,
        n_repeats=n_repeats,
    )
    time_call(
        get_scattering_state_scipy_jax,
        condition,
        config,
        n_repeats=n_repeats,
    )


def benchmark_operator_overhead(size: ProblemSize, *, n_repeats: int = 20) -> None:
    condition, _ = build_condition_and_config(size)

    scipy_operator_data = _build_scipy_operators(condition, n_channels=size.n_channels)
    jax_operator_data = build_operator_data_jax(condition, n_channels=size.n_channels)

    inverse_lower_scipy, lower_scipy, upper_scipy = scipy_operator_data
    nkx, nky, nz = jax_operator_data.shape
    state_size = nkx * nky * nz

    rng = np.random.default_rng(0)
    test_vec_np = (
        rng.standard_normal(state_size) + 1j * rng.standard_normal(state_size)
    ).astype(np.complex128)
    test_vec_jax = jnp.asarray(test_vec_np)

    def time_scipy(
        fn: Callable[[np.ndarray], object],
        x: np.ndarray,
        n: int,
    ) -> list[float]:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - t0)
        return times

    def time_jax(
        fn: Callable[[Array], Array],
        x: Array,
        n: int,
    ) -> list[float]:
        fn(x).block_until_ready()  # warm-up, not timed
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(x).block_until_ready()
            times.append(time.perf_counter() - t0)
        return times

    reshaped_jax = test_vec_jax.reshape((nkx * nky, nz))
    selected_jax = reshaped_jax[jax_operator_data.channel_idx]

    operators = {
        "lower": (
            lower_scipy.matvec,
            lambda x: apply_lower_op(x, jax_operator_data),
            test_vec_np,
            test_vec_jax,
        ),
        "upper": (
            upper_scipy.matvec,
            lambda x: apply_upper_op(x, jax_operator_data),
            test_vec_np,
            test_vec_jax,
        ),
        "inverse_lower": (
            inverse_lower_scipy.matvec,
            lambda x: apply_inverse_lower_block(x, jax_operator_data),
            test_vec_np,
            selected_jax,
        ),
    }

    for scipy_fn, jax_fn, np_input, jax_input in operators.values():
        scipy_times = time_scipy(scipy_fn, np_input, n_repeats)
        jax_times = time_jax(jax_fn, jax_input, n_repeats)
        np.mean(jax_times) / np.mean(scipy_times)


def benchmark_problem_size_sweep(
    sizes: list[ProblemSize],
    *,
    n_repeats: int = 3,
) -> None:
    results = []
    for size in sizes:
        condition, config = build_condition_and_config(size)

        # warm-up jax so the timed calls reflect steady state
        for _ in range(2):
            _force_ready(get_scattering_state_scipy_jax(condition, config))

        scipy_times = time_call(
            get_scattering_state_scipy,
            condition,
            config,
            n_repeats=n_repeats,
        )
        jax_times = time_call(
            get_scattering_state_scipy_jax,
            condition,
            config,
            n_repeats=n_repeats,
        )

        scipy_mean, jax_mean = np.mean(scipy_times), np.mean(jax_times)
        ratio = jax_mean / scipy_mean
        results.append((size.label, scipy_mean, jax_mean, ratio))


def benchmark_realistic_sweep(
    size: ProblemSize,
    angles_deg: list[float],
    *,
    n_warmup: int = 1,
) -> None:
    condition0, config0 = build_condition_and_config(size, theta_deg=angles_deg[0])
    for _ in range(n_warmup):
        _force_ready(get_scattering_state_scipy_jax(condition0, config0))

    for angle in angles_deg:
        condition, config = build_condition_and_config(size, theta_deg=angle)

        t0 = time.perf_counter()
        get_scattering_state_scipy(condition, config)
        time.perf_counter() - t0

        t0 = time.perf_counter()
        _force_ready(get_scattering_state_scipy_jax(condition, config))
        time.perf_counter() - t0


def benchmark_varying_shape_sweep(sizes: list[ProblemSize]) -> None:
    for size in sizes:
        condition, config = build_condition_and_config(size)

        t0 = time.perf_counter()
        get_scattering_state_scipy(condition, config)
        time.perf_counter() - t0

        t0 = time.perf_counter()
        _force_ready(get_scattering_state_scipy_jax(condition, config))
        time.perf_counter() - t0


def plot_scattering_matrix_comparison(size: ProblemSize, output_dir: Path) -> None:
    condition, config = build_condition_and_config(size)

    s_matrix_scipy = get_scattering_matrix(condition, config, backend="scipy")
    s_matrix_jax = get_scattering_matrix(condition, config, backend="jax")

    raw_scipy = np.abs(s_matrix_scipy.raw_data)
    raw_jax = np.abs(s_matrix_jax.raw_data)
    diff = np.abs(raw_scipy - raw_jax)

    max_diff = diff.max()
    max_diff / max(raw_scipy.max(), 1e-30)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, s_matrix, title in [
        (axes[0], s_matrix_scipy, "scipy"),
        (axes[1], s_matrix_jax, "jax"),
    ]:
        _, _, _mesh = plot.array_against_axes_2d_k(s_matrix, measure="abs", ax=ax)
        ax.set_title(f"Scattering matrix ({title})")

    diff_im = axes[2].imshow(diff.reshape(int(np.sqrt(diff.size)), -1), cmap="magma")
    axes[2].set_title(f"|scipy - jax|  (max={max_diff:.2e})")
    fig.colorbar(diff_im, ax=axes[2])

    save_path = output_dir / f"comparison_{size.label.split()[0]}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
