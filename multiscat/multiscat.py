import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg  # type: ignore[import-untyped]
from slate_core import SimpleMetadata, TupleBasis, TupleMetadata
from slate_core.basis import AsUpcast, ContractedBasis
from slate_core.metadata import (
    PERIODIC_FEATURE,
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
    LobattoSpacedMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_k_points
from slate_quantum import Operator, State
from slate_quantum.operator import OperatorMetadata, operator_basis
from tqdm import tqdm

from multiscat.basis import (
    CloseCouplingBasis,
    ScatteringBasisMetadata,
    close_coupling_basis,
    split_scattering_metadata,
)
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.polynomial import get_barycentric_derivatives

if TYPE_CHECKING:
    from multiscat.interpolate import ScatteringOperator

type KineticDifferenceOperatorBasis[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    E: AxisDirections,
] = AsUpcast[
    ContractedBasis[
        TupleBasis[
            tuple[CloseCouplingBasis[M0, M1, E], CloseCouplingBasis[M0, M1, E]],
            None,
        ]
    ],
    OperatorMetadata[ScatteringBasisMetadata[M0, M1, E]],
]


def _get_kinetic_difference_operator_basis[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    E: AxisDirections,
](
    metadata: TupleMetadata[tuple[M0, M0, M1], E],
) -> KineticDifferenceOperatorBasis[M0, M1, E]:
    state_basis = close_coupling_basis(metadata)
    # Diagonal in the k0, k1 basis, but not in the lobatto basis.
    contracted = ContractedBasis(operator_basis(state_basis), ((0, 1, 2), (0, 1, 3)))
    return AsUpcast(contracted, TupleMetadata((metadata, metadata)))


def _get_parallel_kinetic_energy[
    M0: SimpleMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: ScatteringBasisMetadata[M0, M1, E],
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """
    Get the matrix of parallel kinetic energies T.

    Formula for this are taken from:
    "QUANTUM SCATTERING VIA THE LOG DERIVATIVE OF THE KOHN VARIATIONAL PRINCIPLE"
    D. E. Manolopoulos and R. E. Wyatt, Chem. Phys. Lett., 1988, 152,23
    """
    # We make use of the formula
    # T_ij = \sum_k=0 M+1 \omega_k u_i'(R_k) u'_j(R_k)
    # to calculate the kinetic matrix T_ij
    lobatto_metadata = metadata.children[2]
    derivatives = get_barycentric_derivatives(lobatto_metadata)
    # TODO: we should represent this data as an Operator in a sparse # noqa: FIX002
    # basis. Issue is that the array does not have and index for the perpendicular
    # direction, so we cannot use existing ContractedBasis functionality
    assert PERIODIC_FEATURE not in lobatto_metadata.features  # noqa: S101
    return np.einsum(
        "k,ik,jk->ij",
        lobatto_metadata.basis_weights,
        derivatives,
        derivatives,
    )


def _get_perpendicular_kinetic_difference[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedMetadata,
    E: AxisDirections,
](
    incident_k: tuple[float, float, float],
    metadata: ScatteringBasisMetadata[M0, M1, E],
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """
    Get the matrix of scattered energies d.

    Uses the formula
    d^2 = |k in + k scatter|**2 - |k in|**2
    """
    # TODO: we should represent this data as an Operator in a sparse # noqa: FIX002
    # basis. Issue is that the array does not have and index for the parrallel
    # direction, so we cannot use existing ContractedBasis functionality
    metadata_x01, _ = split_scattering_metadata(metadata)
    (kx, ky) = fundamental_stacked_k_points(metadata_x01, offset=incident_k[:2])
    return ((kx**2 + ky**2) - np.linalg.norm(incident_k) ** 2).ravel()


def get_kinetic_difference_operator[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    incident_k: tuple[float, float, float],
    metadata: ScatteringBasisMetadata[M0, M1, E],
) -> Operator[
    KineticDifferenceOperatorBasis[M0, M1, E],
    np.dtype[np.complexfloating],
]:
    """Get the matrix of kinetic energies minus the incident energy."""
    # The parallel kinetic energy is the same for each bloch K, but is non-diagonal
    # in the lobatto basis
    t_ij = _get_parallel_kinetic_energy(metadata)
    # The perpendicular kinetic energy difference is diagonal in bloch K, and is
    # the same for each lobatto basis function
    d_i = _get_perpendicular_kinetic_difference(incident_k, metadata)

    # The resulting difference operator is diagonal in the two bloch K indices
    # and non-diagonal in the lobatto basis
    data = t_ij[np.newaxis, :, :] + d_i[:, np.newaxis, np.newaxis]
    return Operator(
        basis=_get_kinetic_difference_operator_basis(metadata),
        data=data.astype(np.complexfloating),
    )


def _gmres[DT: np.dtype[np.number]](
    matrix: scipy.sparse.linalg.LinearOperator,
    initial_state: np.ndarray[Any, DT],
    *,
    options: OptimizationConfig,
) -> np.ndarray[Any, DT]:
    resid_bar = tqdm(total=1.0, desc="Total Convergence", position=1, leave=False)

    def _callback(pr_norm: float) -> None:
        error = round(np.log10(pr_norm / options.precision), 3)
        next_progress = round(resid_bar.total - error, 3)
        if next_progress < 0:
            resid_bar.reset(total=error)
            next_progress = 0

        resid_bar.n = next_progress
        resid_bar.refresh()

    data, _info = scipy.sparse.linalg.gmres(  # type: ignore[unknown]
        A=matrix,
        b=initial_state,
        rtol=options.precision,
        maxiter=options.max_iterations,
        callback=_callback,
        callback_type="pr_norm",
    )
    resid_bar.close()
    if _info != 0:
        warnings.warn(
            f"GMRES iteration did not converge in {options.max_iterations} "
            f"iterations to a precision of {options.precision}. "
            "This may indicate that the problem is ill-conditioned or that the "
            "initial guess is too far from the solution.",
            UserWarning,
            stacklevel=4,
        )
    return cast("np.ndarray[Any, DT]", data)


def _get_scattered_state[  # pyright: ignore[reportUnusedFunction]
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedMetadata,
    E: AxisDirections,
](
    kinetic_difference: Operator[
        KineticDifferenceOperatorBasis[M0, M1, E],
        np.dtype[np.complexfloating],
    ],
    potential: ScatteringOperator[M0, M1, E],
    *,
    options: OptimizationConfig,
) -> State[CloseCouplingBasis[M0, M1, E]]:
    """
    Get the basis for the scattered state.

    This is a diagonal basis in the lobatto basis, and a tuple basis in the
    bloch K indices.
    """
    state_metadata = potential.basis.metadata().children[0]
    state_basis = close_coupling_basis(state_metadata)
    nx, ny, nz = state_metadata.shape

    # TODO: use split operator here ...  # noqa: FIX002
    potential_raw = potential.with_basis(
        operator_basis(state_basis),
    ).raw_data.reshape(
        (nx * ny * nz, nx * ny * nz),
    )
    kinetic_raw = kinetic_difference.raw_data.reshape(
        (nx, ny, nz, nz),
    )

    initial_state = np.zeros((nx, ny, nz), dtype=np.complexfloating)
    initial_state[0, 0, -1] = 1.0

    def matmul_hamiltonian(
        state: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
    ) -> float:
        """Cost function for the GMRES solver."""
        # The cost function is the kinetic energy minus the potential energy
        cost_kinetic = np.einsum(
            "ijkl,ijl->ijk",
            kinetic_raw,
            state.reshape((nx, ny, nz)),
        ).ravel()
        # Note that the potential is likely to be sparse, since only a few
        # terms in a band along the diagonal are non-zero.
        # For performance, we should use a sparse matrix here!
        # TODO: can we do this more effiecently using fourier transforms? # noqa: FIX002
        cost_potential = np.einsum(
            "ij,j->i",
            potential_raw,
            state.ravel(),
        )
        return cost_kinetic + cost_potential

    # TODO: to make gmres work 'well', we need to build the  # noqa: FIX002
    # correct preconditioner
    # In the future, we should also probably port this to a
    # compiled language, or otherwise use some cython build
    # to speed up the loop.
    # Or we could use a more efficient approach (ML) to speed up the
    # convergence of the solver.
    # TODO: how do we ensure bcs are satisfied, we should probably  # noqa: FIX002
    # manually add the initial condition each iteration? Is this even needed?
    data = _gmres(  # type: ignore[unknown]
        matrix=scipy.sparse.linalg.LinearOperator(
            shape=potential_raw.shape,
            matvec=matmul_hamiltonian,  # type: ignore[call-arg]
            dtype=np.complexfloating,
        ),
        initial_state=initial_state.ravel(),
        options=options,
    )
    return State(state_basis, data)


def get_scattered_state[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    *,
    options: OptimizationConfig | None = None,
) -> State[CloseCouplingBasis[M0, M1, E]]:
    options = options or OptimizationConfig()
    _kinetic_difference = get_kinetic_difference_operator(
        condition.incident_k,
        condition.metadata,
    )

    msg = "This function is not implemented yet."
    raise NotImplementedError(msg)


def _condition_parameters(condition: ScatteringCondition) -> tuple[float, float, float, float]:
    scattering_vector = np.asarray(condition.incident_k)
    scattering_magnitude = float(np.linalg.norm(scattering_vector))
    if scattering_magnitude <= 0:
        msg = "Incident wavevector magnitude must be non-zero"
        raise ValueError(msg)

    mass_amu = float(condition.mass / atomic_mass)
    energy_meV = float(condition.incident_energy / (electron_volt * 10**3))
    theta_degrees = float(np.degrees(condition.theta))
    phi_degrees = float(np.degrees(condition.phi))
    return mass_amu, energy_meV, theta_degrees, phi_degrees




def _optimization_parameters(config: OptimizationConfig) -> tuple[int, int, float, int]:
    if config.precision <= 0:
        msg = "Optimization precision must be greater than zero"
        raise ValueError(msg)

    gmres_preconditioner_flag = 0
    convergence_significant_figures = int(np.log10(1 / config.precision))
    max_closed_channel_energy = (
        float(config.max_negative_energy / (electron_volt * 10**3))
        if config.max_negative_energy is not None
        else 120.0
    )
    max_channel_index = int(
        config.max_channel_index if config.max_channel_index is not None else 120
    )
    return (
        gmres_preconditioner_flag,
        convergence_significant_figures,
        max_closed_channel_energy,
        max_channel_index,
    )


def _raw_potential_in_input_file_convention(
    potential: ScatteringOperator,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    potential_diagonal = array.extract_diagonal(potential)
    nx, ny, nz = potential_diagonal.basis.metadata().shape
    basis_weights = potential_diagonal.basis.metadata().children[2].basis_weights
    basis = close_coupling_basis(potential_diagonal.basis.metadata())

    data = potential_diagonal.with_basis(basis).raw_data
    data = data.reshape((nx, ny, nz)) * basis_weights[np.newaxis, np.newaxis, :]

    # Multiscat uses a slightly different Fourier convention.
    return data.ravel() / (electron_volt * 10**-3 * np.sqrt(nx * ny))


def _potential_parameters(
    potential: ScatteringOperator,
) -> tuple[int, int, int, float, float, float, float, float, float, np.ndarray[Any, np.dtype[np.complex128]]]:
    potential_lobatto = _raw_potential_in_input_file_convention(potential)
    metadata = potential.basis.metadata().children[0]
    nx, ny, nz = metadata.shape
    z_domain = metadata.children[2].domain
    z_start_angstrom = float(z_domain.start / angstrom)
    z_end_angstrom = float((z_domain.start + z_domain.delta) / angstrom)

    metadata_x01, _ = split_scattering_metadata(metadata)
    directions = metadata.extra.vectors
    x_vector = np.asarray(directions[0]) * metadata_x01.children[0].domain.delta
    y_vector = np.asarray(directions[1]) * metadata_x01.children[1].domain.delta
    ax_angstrom = float(x_vector[0] / angstrom)
    ay_angstrom = float(x_vector[1] / angstrom)
    bx_angstrom = float(y_vector[0] / angstrom)
    by_angstrom = float(y_vector[1] / angstrom)

    nfc = nx * ny
    potential_matrix = np.asfortranarray(potential_lobatto.reshape((nfc, nz)).T)
    return (
        int(nx),
        int(ny),
        int(nz),
        ax_angstrom,
        ay_angstrom,
        bx_angstrom,
        by_angstrom,
        z_start_angstrom,
        z_end_angstrom,
        potential_matrix,
    )


def run_multiscat(
    condition: ScatteringCondition, config: OptimizationConfig
) -> dict[tuple[int, int], float]:
    """Run Multiscat through the f2py native binding and return intensities."""
    mass_amu, energy_meV, theta_degrees, phi_degrees = _condition_parameters(condition)
    (
        gmres_preconditioner_flag,
        convergence_significant_figures,
        max_closed_channel_energy,
        max_channel_index,
    ) = _optimization_parameters(config)
    (
        nkx,
        nky,
        nz,
        unit_cell_ax,
        unit_cell_ay,
        unit_cell_bx,
        unit_cell_by,
        zmin,
        zmax,
        potential_values,
    ) = _potential_parameters(condition.potential)

    max_channels = 1024
    (
        channel_count,
        channel_ix,
        channel_iy,
        channel_d,
        channel_intensity,
        _,
        _,
        _,
        ierr,
    ) = run_multiscat_fortran(
        mass_amu,
        energy_meV,
        theta_degrees,
        phi_degrees,
        gmres_preconditioner_flag,
        convergence_significant_figures,
        max_closed_channel_energy,
        max_channel_index,
        nkx,
        nky,
        unit_cell_ax,
        unit_cell_ay,
        unit_cell_bx,
        unit_cell_by,
        zmin,
        zmax,
        potential_values,
        max_channels,
        nz=nz,
    )

    if ierr != 0:
        msg = f"Fortran run_multiscat_fortran failed with error code {ierr}"
        raise RuntimeError(msg)

    intensities: dict[tuple[int, int], float] = {}
    for idx in range(int(channel_count)):
        if channel_d[idx] < 0.0:
            intensities[(int(channel_ix[idx]), int(channel_iy[idx]))] = float(
                channel_intensity[idx]
            )
    return intensities