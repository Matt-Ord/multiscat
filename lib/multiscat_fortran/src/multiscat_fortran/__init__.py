from __future__ import annotations

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false
import numpy as np
from numpy.typing import NDArray

from multiscat_fortran._multiscat_f2py import (
    debug_apply_upper_block_fortran as _debug_apply_upper_block_fortran,  # type: ignore [import]
)
from multiscat_fortran._multiscat_f2py import (
    debug_build_preconditioner_fortran as _debug_build_preconditioner_fortran,  # type: ignore [import]
)
from multiscat_fortran._multiscat_f2py import (
    debug_solve_lower_block_fortran as _debug_solve_lower_block_fortran,  # type: ignore [import]
)
from multiscat_fortran._multiscat_f2py import (
    get_abc_arrays as _get_abc_arrays,  # type: ignore [import]
)
from multiscat_fortran._multiscat_f2py import (
    get_lobatto_weights as _get_lobatto_weights,  # type: ignore [import]
)
from multiscat_fortran._multiscat_f2py import (
    get_parallel_kinetic_energy as _get_parallel_kinetic_energy,  # type: ignore [import]
)
from multiscat_fortran._multiscat_f2py import (
    get_perpendicular_kinetic_difference as _get_perpendicular_kinetic_difference,  # type: ignore [import]
)
from multiscat_fortran._multiscat_f2py import (
    run_multiscat_fortran as _run_multiscat_fortran,  # type: ignore [import]
)


def _raise_fortran_error(function_name: str, ierr: int) -> None:
    if ierr != 0:
        msg = f"Fortran {function_name} failed with error code {ierr}"
        raise RuntimeError(msg)


def run_multiscat_fortran(
    gmres_preconditioner_flag: int,
    convergence_significant_figures: int,
    potential_values: NDArray[np.complex128],
    perpendicular_kinetic_difference: NDArray[np.float64],
    wave_a: NDArray[np.complex128],
    wave_b: NDArray[np.complex128],
    wave_c: NDArray[np.complex128],
    parallel_kinetic_energy: NDArray[np.float64],
) -> NDArray[np.complex128]:
    scattered_state_dense, ierr = _run_multiscat_fortran(
        gmres_preconditioner_flag,
        convergence_significant_figures,
        potential_values,
        perpendicular_kinetic_difference,
        wave_a,
        wave_b,
        wave_c,
        parallel_kinetic_energy,
    )
    _raise_fortran_error("run_multiscat_fortran", int(ierr))
    return scattered_state_dense


def get_perpendicular_kinetic_difference(
    incident_kx: float,
    incident_ky: float,
    incident_kz: float,
    nx: int,
    ny: int,
    unit_cell_ax: float,
    unit_cell_ay: float,
    unit_cell_bx: float,
    unit_cell_by: float,
) -> NDArray[np.float64]:
    data, ierr = _get_perpendicular_kinetic_difference(
        incident_kx,
        incident_ky,
        incident_kz,
        nx,
        ny,
        unit_cell_ax,
        unit_cell_ay,
        unit_cell_bx,
        unit_cell_by,
    )
    _raise_fortran_error("get_perpendicular_kinetic_difference", int(ierr))
    return data


def get_parallel_kinetic_energy(
    zmin: float,
    zmax: float,
    nz: int,
) -> NDArray[np.float64]:
    data, ierr = _get_parallel_kinetic_energy(zmin, zmax, nz)
    _raise_fortran_error("get_parallel_kinetic_energy", int(ierr))
    return data


def get_lobatto_weights(
    zmin: float,
    zmax: float,
    node_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weights, points, ierr = _get_lobatto_weights(zmin, zmax, node_count)
    _raise_fortran_error("get_lobatto_weights", int(ierr))
    return weights, points


def get_abc_arrays(
    zmin: float,
    zmax: float,
    nx: int,
    ny: int,
    perpendicular_kinetic_difference: NDArray[np.float64],
    n_z_points: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    wave_a, wave_b, wave_c, ierr = _get_abc_arrays(
        zmin=zmin,
        zmax=zmax,
        nx=nx,
        ny=ny,
        perpendicular_kinetic_difference=perpendicular_kinetic_difference,
        n_z_points=n_z_points,
    )
    _raise_fortran_error("get_abc_arrays", int(ierr))
    return wave_a, wave_b, wave_c


def debug_build_preconditioner_fortran(
    potential_values: NDArray[np.complex128],
    perpendicular_kinetic_difference: NDArray[np.float64],
    parallel_kinetic_energy: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    eigenvalues, preconditioner_factors, eigenvectors, ierr = (
        _debug_build_preconditioner_fortran(
            potential_values,
            perpendicular_kinetic_difference,
            parallel_kinetic_energy,
        )
    )
    _raise_fortran_error("debug_build_preconditioner_fortran", int(ierr))
    return eigenvalues, preconditioner_factors, eigenvectors


def debug_apply_upper_block_fortran(
    potential_values: NDArray[np.complex128],
    state_in: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    state_out, ierr = _debug_apply_upper_block_fortran(potential_values, state_in)
    _raise_fortran_error("debug_apply_upper_block_fortran", int(ierr))
    return state_out


def debug_solve_lower_block_fortran(
    potential_values: NDArray[np.complex128],
    wave_c: NDArray[np.complex128],
    perpendicular_kinetic_difference: NDArray[np.float64],
    parallel_kinetic_energy: NDArray[np.float64],
    state_in: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    state_out, ierr = _debug_solve_lower_block_fortran(
        potential_values,
        wave_c,
        perpendicular_kinetic_difference,
        parallel_kinetic_energy,
        state_in,
    )
    _raise_fortran_error("debug_solve_lower_block_fortran", int(ierr))
    return state_out


__all__ = [
    "run_multiscat_fortran",
    "get_perpendicular_kinetic_difference",
    "get_parallel_kinetic_energy",
    "get_lobatto_weights",
    "get_abc_arrays",
    "debug_build_preconditioner_fortran",
    "debug_apply_upper_block_fortran",
    "debug_solve_lower_block_fortran",
]
