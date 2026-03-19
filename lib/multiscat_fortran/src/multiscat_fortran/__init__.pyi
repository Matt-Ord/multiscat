import numpy as np
from numpy.typing import NDArray

def run_multiscat_fortran(
    helium_mass: float,
    incident_energy_mev: float,
    theta_degrees: float,
    phi_degrees: float,
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
    max_channels: int,
    *,
    nz: int,
) -> tuple[
    int,
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    int,
    int,
]: ...

__all__ = ["run_multiscat_fortran"]
