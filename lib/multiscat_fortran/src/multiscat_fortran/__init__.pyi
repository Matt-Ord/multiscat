import numpy as np
from numpy.typing import NDArray

def run_multiscat_fortran(
    helium_mass: float,
    incident_kx: float,
    incident_ky: float,
    incident_kz: float,
    gmres_preconditioner_flag: int,
    convergence_significant_figures: int,
    nkx: int,
    nky: int,
    unit_cell_ax: float,
    unit_cell_ay: float,
    unit_cell_bx: float,
    unit_cell_by: float,
    zmin: float,
    zmax: float,
    potential_values: NDArray[np.complex128],
    *,
    nz: int,
) -> tuple[
    NDArray[np.float64],
    int,
    int,
]: ...
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
) -> tuple[NDArray[np.float64], int]: ...
def get_parallel_kinetic_energy(
    zmin: float,
    zmax: float,
    nz: int,
) -> tuple[NDArray[np.float64], int]: ...
def get_lobatto_weights(
    zmin: float,
    zmax: float,
    node_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]: ...
def get_abc_arrays(
    zmin: float,
    zmax: float,
    nx: int,
    ny: int,
    perpendicular_kinetic_difference: NDArray[np.float64],
    n_z_points: int,
) -> tuple[
    NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], int
]: ...

__all__ = [
    "run_multiscat_fortran",
    "get_perpendicular_kinetic_difference",
    "get_parallel_kinetic_energy",
    "get_lobatto_weights",
    "get_abc_arrays",
]
