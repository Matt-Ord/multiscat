import numpy as np
from numpy.typing import NDArray

def run_multiscat_fortran(
    gmres_preconditioner_flag: int,
    convergence_significant_figures: int,
    potential_values: NDArray[np.complex128],
    perpendicular_kinetic_difference: NDArray[np.float64],
    wave_a: NDArray[np.complex128],
    wave_b: NDArray[np.complex128],
    wave_c: NDArray[np.complex128],
    parallel_kinetic_energy: NDArray[np.float64],
) -> NDArray[np.complex128]: ...
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
) -> NDArray[np.float64]: ...
def get_parallel_kinetic_energy(
    zmin: float,
    zmax: float,
    nz: int,
) -> NDArray[np.float64]: ...
def get_lobatto_weights(
    zmin: float,
    zmax: float,
    node_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def get_abc_arrays(
    zmin: float,
    zmax: float,
    nx: int,
    ny: int,
    perpendicular_kinetic_difference: NDArray[np.float64],
    n_z_points: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]: ...
def debug_build_preconditioner_fortran(
    potential_values: NDArray[np.complex128],
    perpendicular_kinetic_difference: NDArray[np.float64],
    parallel_kinetic_energy: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def debug_apply_upper_block_fortran(
    potential_values: NDArray[np.complex128],
    state_in: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
def debug_solve_lower_block_fortran(
    potential_values: NDArray[np.complex128],
    wave_c: NDArray[np.complex128],
    perpendicular_kinetic_difference: NDArray[np.float64],
    parallel_kinetic_energy: NDArray[np.float64],
    state_in: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...

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
