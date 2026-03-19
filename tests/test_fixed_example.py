from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from multiscat_fortran import get_perpendicular_kinetic_difference
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core import EvenlySpacedLengthMetadata, array
from slate_quantum import operator

from multiscat.basis import (
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
    split_scattering_metadata,
)
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.multiscat import (
    _get_perpendicular_kinetic_difference,  # pyright: ignore[reportPrivateUsage]
    run_multiscat,
)

if TYPE_CHECKING:
    from slate_core.metadata import AxisDirections, LobattoSpacedLengthMetadata

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


def _parse_intensities(output_file: Path) -> dict[tuple[int, int], float]:
    pattern = re.compile(r"^#\s+(-?\d+)\s+(-?\d+)\s+([0-9.E+-]+)\s*$")
    intensities: dict[tuple[int, int], float] = {}
    for line in output_file.read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        h = int(match.group(1))
        k = int(match.group(2))
        intensities[(h, k)] = float(match.group(3))
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


def test_simple_system() -> None:

    condition, config = _simple_example_condition()
    intensities = run_multiscat(condition, config)

    if not intensities:
        msg = "Expected at least one diffraction intensity"
        raise AssertionError(msg)

    if not math.isclose(sum(intensities.values()), 1.0, abs_tol=1e-6):
        msg = f"Intensities sum to {sum(intensities.values())}, expected 1.0"
        raise AssertionError(msg)

    expected_from_file = _parse_intensities(
        TESTS_DIR / "data" / Path("expected_intensities.txt"),
    )
    for spot, expected_value in expected_from_file.items():
        if spot not in intensities:
            msg = f"Missing diffraction spot {spot}"
            raise AssertionError(msg)
        if not math.isclose(intensities[spot], expected_value, abs_tol=1e-5):
            msg = (
                f"Intensity for spot {spot} is {intensities[spot]},"
                f" expected {expected_value}"
            )
            raise AssertionError(msg)


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
        phi=np.deg2rad(0),
        potential=operator.build.corrugated_morse_potential(
            metadata,
            MORSE_PARAMETERS,
        ),
    )
    config = OptimizationConfig(precision=1e-5, max_iterations=1000)
    return condition, config


def test_rotated_system() -> None:

    condition, config = _rotated_example_condition()
    intensities = run_multiscat(condition, config)

    if not intensities:
        msg = "Expected at least one diffraction intensity"
        raise AssertionError(msg)

    if not math.isclose(sum(intensities.values()), 1.0, abs_tol=1e-6):
        msg = f"Intensities sum to {sum(intensities.values())}, expected 1.0"
        raise AssertionError(msg)

    expected_from_file = _parse_intensities(
        TESTS_DIR / "data" / Path("expected_intensities.txt"),
    )
    for spot, expected_value in expected_from_file.items():
        if spot not in intensities:
            msg = f"Missing diffraction spot {spot}"
            raise AssertionError(msg)
        if not math.isclose(intensities[spot], expected_value, abs_tol=1e-5):
            msg = (
                f"Intensity for spot {spot} is {intensities[spot]},"
                f" expected {expected_value}"
            )
            raise AssertionError(msg)


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

    expected = _get_perpendicular_kinetic_difference(
        condition.incident_k,
        metadata,
    ).reshape((nx, ny))

    metadata_x01, _ = split_scattering_metadata(metadata)
    directions = metadata.extra.vectors
    x_vector = np.asarray(directions[0]) * metadata_x01.children[0].domain.delta
    y_vector = np.asarray(directions[1]) * metadata_x01.children[1].domain.delta

    incident_kx, incident_ky, incident_kz = condition.incident_k

    actual_raw, ierr_raw = get_perpendicular_kinetic_difference(
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
    ierr = int(ierr_raw)

    if ierr != 0:
        msg = (
            "Fortran get_perpendicular_kinetic_difference "
            f"failed with error code {ierr}"
        )
        raise RuntimeError(msg)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
