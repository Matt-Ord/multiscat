from __future__ import annotations

import math
import re

# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
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
from slate_core import EvenlySpacedLengthMetadata, array, basis
from slate_core.metadata import LobattoSpacedLengthMetadata
from slate_quantum import operator

from multiscat.basis import (
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
    split_scattering_metadata,
)
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.multiscat._fortran import (
    _condition_parameters,
)
from multiscat.multiscat._multiscat import (
    get_scattering_matrix,
)
from multiscat.multiscat._scipy import (
    _build_lower_block_factors,
    _build_scipy_operators,
    _solve_specular_hamiltonian,
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
    condition = ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=20 * electron_volt * 10**-3,
        theta=np.deg2rad(30),
        phi=np.deg2rad(0),
        potential=operator.build.corrugated_morse_potential(
            metadata,
            MORSE_PARAMETERS,
        ),
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
        potential=operator.build.corrugated_morse_potential(
            metadata,
            MORSE_PARAMETERS,
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
        perpendicular_kinetic_difference=perpendicular_kinetic_difference,
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
        wave_c_python,
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

    _inverse_lower, upper = _build_scipy_operators(
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

    inverse_lower, _upper = _build_scipy_operators(
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
