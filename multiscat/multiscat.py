from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import scipy.sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg  # type: ignore[import-untyped]
from multiscat_fortran import (
    run_multiscat_fortran,
)
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    atomic_mass,
    electron_volt,
    hbar,
)
from slate_core import (
    Array,
    Basis,
    SimpleMetadata,
    TupleBasis,
    TupleMetadata,
    array,
    basis,
)
from slate_core.basis import AsUpcast, ContractedBasis
from slate_core.metadata import (
    AxisDirections,
    Domain,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
    LobattoSpacedMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_k_points
from slate_core.util import timed
from slate_quantum import Operator, State
from slate_quantum.operator import OperatorMetadata, operator_basis
from tqdm import tqdm

from multiscat.basis import (
    CloseCouplingBasis,
    ScatteringBasisMetadata,
    close_coupling_basis,
    split_scattering_metadata,
)
from multiscat.polynomial import (
    get_barycentric_kinetic_operator,
)

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator  # type: ignore[untyped]

    from multiscat.config import OptimizationConfig, ScatteringCondition
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


def _get_parallel_kinetic_energy(
    metadata: LobattoSpacedLengthMetadata,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the matrix of parallel kinetic energies T.

    Formula for this are taken from:
    "QUANTUM SCATTERING VIA THE LOG DERIVATIVE OF THE KOHN VARIATIONAL PRINCIPLE"
    D. E. Manolopoulos and R. E. Wyatt, Chem. Phys. Lett., 1988, 152,23
    """
    # We make use of the formula
    # T_ij = \sum_k=0 M+1 \omega_k u_i'(R_k) u'_j(R_k)
    # to calculate the kinetic matrix T_ij
    # TODO: we should represent this data as an Operator in a sparse # noqa: FIX002
    # basis. Issue is that this does not lend itself to an efficient
    # implementation when we add the parallel and perpendicular
    # kinetic energies together.
    return array.as_fundamental_basis(
        get_barycentric_kinetic_operator(metadata),
    ).raw_data.reshape((metadata.fundamental_size, metadata.fundamental_size))


def _get_perpendicular_kinetic_difference[
    M0: TupleMetadata[
        tuple[EvenlySpacedLengthMetadata, EvenlySpacedLengthMetadata],
        AxisDirections,
    ],
](
    incident_k: tuple[float, float, float],
    metadata: TupleMetadata[
        tuple[EvenlySpacedLengthMetadata, EvenlySpacedLengthMetadata],
        AxisDirections,
    ],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the matrix of scattered energies d.

    Uses the formula
    d^2 = |k in + k scatter|**2 - |k in|**2
    """
    # TODO: we should represent this data as an Operator in a sparse # noqa: FIX002
    # basis. Issue is that the array does not have and index for the parallel
    # direction, so we cannot use existing ContractedBasis functionality
    (kx, ky) = fundamental_stacked_k_points(metadata, offset=incident_k[:2])
    return ((kx**2 + ky**2) - np.linalg.norm(incident_k) ** 2).reshape(metadata.shape)  # type: ignore[no-untyped-call]


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
    metadata_xy, metadata_z = split_scattering_metadata(metadata)
    # The parallel kinetic energy is the same for each bloch K, but is non-diagonal
    # in the lobatto basis
    t_jk = _get_parallel_kinetic_energy(metadata_z)
    # The perpendicular kinetic energy difference is diagonal in both the bloch K,
    # and the lobatto basis functions. Here we scale by the lobatto weights
    d_i = _get_perpendicular_kinetic_difference(incident_k, metadata_xy).ravel()
    d_ijk = np.einsum(
        "i,jk->ijk",
        d_i,
        np.eye(t_jk.shape[0], dtype=np.complex128),
    )

    # The resulting difference operator is diagonal in the two bloch K indices
    # and non-diagonal in the lobatto basis
    data = t_jk[np.newaxis, :, :] + d_ijk
    return Operator(
        basis=_get_kinetic_difference_operator_basis(metadata),
        data=data.astype(np.complexfloating),
    )


def _condition_parameters[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[
        M0,
        M1,
        E,
    ],
) -> tuple[
    tuple[float, float, float],
    np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    TupleMetadata[
        tuple[EvenlySpacedLengthMetadata, EvenlySpacedLengthMetadata],
        AxisDirections,
    ],
    LobattoSpacedLengthMetadata,
]:
    scattering_vector = np.asarray(condition.incident_k)
    scattering_magnitude = float(np.linalg.norm(scattering_vector))
    if scattering_magnitude <= 0:
        msg = "Incident wavevector magnitude must be non-zero"
        raise ValueError(msg)

    kx, ky, kz = condition.incident_k

    potential = _raw_potential_in_input_file_convention(condition.potential)
    hbar_squared = (hbar**2 / (atomic_mass * electron_volt * angstrom**2)) * 1e3
    mass_amu = float(condition.mass / atomic_mass)
    potential = potential * ((2.0 * mass_amu) / hbar_squared)

    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)

    return (
        (kx * angstrom, ky * angstrom, kz * angstrom),
        potential,
        TupleMetadata(
            (
                EvenlySpacedLengthMetadata(
                    metadata_x01.shape[0],
                    domain=Domain(
                        start=0,
                        delta=metadata_x01.children[0].domain.delta / angstrom,
                    ),
                ),
                EvenlySpacedLengthMetadata(
                    metadata_x01.shape[1],
                    domain=Domain(
                        start=0,
                        delta=metadata_x01.children[1].domain.delta / angstrom,
                    ),
                ),
            ),
            metadata_x01.extra,
        ),
        LobattoSpacedLengthMetadata(
            fundamental_size=metadata_z.fundamental_size,
            domain=Domain(
                start=metadata_z.domain.start / angstrom,
                delta=metadata_z.domain.delta / angstrom,
            ),
        ),
    )


def _optimization_parameters(config: OptimizationConfig) -> tuple[int, int]:
    if config.precision <= 0:
        msg = "Optimization precision must be greater than zero"
        raise ValueError(msg)

    preconditioner_flag = 1 if config.use_neumann_preconditioner else 0
    n_significant_figures = int(np.log10(1 / config.precision))
    return (
        preconditioner_flag,
        n_significant_figures,
    )


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
    data = data.reshape((nx, ny, nz)) * basis_weights[np.newaxis, np.newaxis, :]

    # Multiscat uses natural units, and a different normalization convention
    return data / (electron_volt * 10**-3 * np.sqrt(nx * ny))


def get_scattering_matrix_from_state[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[CloseCouplingBasis[M0, M1, E]],
) -> Array[
    Basis[TupleMetadata[tuple[M0, M0], AxisDirections]],
    np.dtype[np.complexfloating],
]:
    """Get the scattering matrix for a given scattered state."""
    metadata_x01, _ = split_scattering_metadata(state.basis.metadata())
    return Array(
        AsUpcast(basis.from_metadata(metadata_x01), metadata_x01),
        state.as_array()[:, :, -1],
    )


def get_scattered_intensity(
    solution: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    a_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    b_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Recover per-channel intensities from the optimized scattered state."""
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


def _get_ab_waves(
    metadata: LobattoSpacedMetadata,
    perpendicular_kinetic_difference: np.ndarray[
        tuple[int, int],
        np.dtype[np.floating],
    ],
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
]:
    """Get the asymptotic initial state and final scattered state amplitude factors."""
    nkx, nky = perpendicular_kinetic_difference.shape
    channel_energy = perpendicular_kinetic_difference.ravel(order="C")
    dk = np.sqrt(np.abs(channel_energy))

    # given the incoming and outgoing waves i(r) and o(r)
    # We take the limit r-> infinity
    # a_wave is i(r) o(r)^(-1), b_wave is o(r)^(-1)
    a_wave = np.zeros(channel_energy.shape, dtype=np.complex128)
    b_wave = np.zeros(channel_energy.shape, dtype=np.complex128)

    open_channel = channel_energy < 0.0
    if np.any(open_channel):
        theta = dk[open_channel] * (metadata.delta - metadata.domain.start)
        a_wave[open_channel] = np.exp(-2.0j * theta)
        b_wave[open_channel] = np.sqrt(dk[open_channel]) * np.exp(
            -1j * theta,
        )

    b_wave = b_wave * metadata.basis_weights[-1]
    return (a_wave.reshape((nkx, nky)), b_wave.reshape((nkx, nky)))


def _get_outgoing_log_derivative_wave(
    metadata: LobattoSpacedMetadata,
    perpendicular_kinetic_difference: np.ndarray[
        tuple[int, int],
        np.dtype[np.floating],
    ],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Get the outgoing channel logarithmic derivatives."""
    nkx, nky = perpendicular_kinetic_difference.shape
    channel_energy = perpendicular_kinetic_difference.ravel()

    out = 1j * np.emath.sqrt(-channel_energy)  # cspell: disable-line

    out = out * metadata.basis_weights[-1] ** 2
    return out.reshape((nkx, nky), order="C")


def _solve_specular_hamiltonian(
    potential_values: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    parallel_kinetic_energy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    _, _, nz = potential_values.shape
    hamiltonian = parallel_kinetic_energy.copy()
    hamiltonian[np.diag_indices(nz)] += np.real(potential_values[0, 0, :])
    return np.linalg.eigh(hamiltonian)  # cspell: disable-line


def _build_lower_block_factors(
    potential_values: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    channel_energy: np.ndarray[
        tuple[int],
        np.dtype[np.float64],
    ],
    parallel_kinetic_energy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """
    Diagonalize the 1D reference Hamiltonian H_0(z) = T_z + V_0(z).

    In atom-surface scattering, V_0(z) is the laterally averaged potential
    (the specular term). By temporarily setting the surface corrugation
    coupling to zero, the matrix becomes purely diagonal in the
    diffraction channel index.

    We use this matrix to build the preconditioner for the GMRES solver.

    This function computes the eigenvalues and eigenvectors of H_0(z) once
    at the beginning of the calculation. We then use this to precondition
    our GMRES solver.
    """
    _, _, nz = potential_values.shape
    channel_count = channel_energy.shape[0]

    eigenvalues, eigenvectors = _solve_specular_hamiltonian(
        potential_values=potential_values,
        parallel_kinetic_energy=parallel_kinetic_energy,
    )

    # calculate (H_0 - E_alpha)^(-1) C_i u_i
    g = np.empty((nz, channel_count), dtype=np.float64)
    lower_block_factors = np.zeros((nz, channel_count), dtype=np.float64)
    for j in range(channel_count):
        g[:, j] = eigenvectors[nz - 1, :] / (channel_energy[j] + eigenvalues)
        lower_block_factors[:, j] = np.einsum("ki,i->k", eigenvectors, g[:, j])
    return (eigenvalues, lower_block_factors, eigenvectors)


@dataclass(frozen=True)
class _ScipyOperatorData:
    nkx: int
    nky: int
    nz: int
    specular_channel: int
    potential_pairs: np.ndarray[Any, np.dtype[np.complex128]]
    eigenvalues: np.ndarray[Any, np.dtype[np.float64]]
    eigenvectors: np.ndarray[Any, np.dtype[np.float64]]
    perpendicular_kinetic_difference: np.ndarray[Any, np.dtype[np.float64]]
    lower_block_factors: np.ndarray[Any, np.dtype[np.float64]]
    outgoing_log_derivative_wave: np.ndarray[Any, np.dtype[np.complex128]]

    @property
    def channel_count(self) -> int:
        return self.perpendicular_kinetic_difference.size


def _build_scipy_operators(
    potential_values: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    perpendicular_kinetic_difference: np.ndarray[
        tuple[int, int],
        np.dtype[np.float64],
    ],
    outgoing_log_derivative_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    parallel_kinetic_energy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> tuple[LinearOperator, LinearOperator]:
    nkx, nky, nz = potential_values.shape

    idx_x, idx_y = np.meshgrid(np.arange(nkx), np.arange(nky), indexing="ij")
    idx_x = idx_x.ravel()
    idx_y = idx_y.ravel()
    diff_x = (idx_x[:, np.newaxis] - idx_x[np.newaxis, :]) % nkx
    diff_y = (idx_y[:, np.newaxis] - idx_y[np.newaxis, :]) % nky
    potential_pairs = potential_values[diff_x, diff_y, :]

    eigenvalues, lower_block_factors, eigenvectors = _build_lower_block_factors(
        potential_values=potential_values,
        channel_energy=perpendicular_kinetic_difference.ravel(),
        parallel_kinetic_energy=parallel_kinetic_energy,
    )

    operator_data = _ScipyOperatorData(
        nkx=nkx,
        nky=nky,
        nz=nz,
        specular_channel=0,
        potential_pairs=potential_pairs,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        perpendicular_kinetic_difference=perpendicular_kinetic_difference.ravel(),
        lower_block_factors=lower_block_factors,
        outgoing_log_derivative_wave=outgoing_log_derivative_wave.ravel(),
    )
    channel_count = operator_data.channel_count

    state_size = np.prod((nkx, nky, nz))

    def _apply_inverse_lower(
        flat_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        state = flat_state.reshape((channel_count, nz))
        lower = _apply_inverse_lower_block(state, operator_data)
        return lower.ravel()

    inverse_lower = scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
        (state_size, state_size),
        _apply_inverse_lower,
        dtype=np.complex128,  # type: ignore[assignment]
    )

    def _apply_upper(
        flat_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        state = flat_state.reshape((channel_count, nz))
        upper = _apply_upper_block(state, operator_data)
        return upper.ravel()

    upper = scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
        (state_size, state_size),
        _apply_upper,
        dtype=np.complex128,  # type: ignore[assignment]
    )
    return inverse_lower, upper


def _apply_upper_block(
    state_vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    operator_data: _ScipyOperatorData,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """
    Apply the off-diagonal channel-coupling potential V_1(z)Q(x,y).

    This calculates the scattering of the wave between different (kx, ky)
    diffraction channels. Since V_1(z) is responsible for all coupling
    between diffraction channels, this step represents the physical momentum
    transfer parallel to the corrugated surface.
    """
    result = np.zeros(state_vector.shape, dtype=np.complex128)
    # Skip j=0, the specular channel, since it is included in the preconditioner
    for channel in range(1, operator_data.channel_count):
        j_minus_1 = channel - 1
        pairs = operator_data.potential_pairs[j_minus_1, channel:, :]
        result[j_minus_1, :] = np.einsum("ik,ik->k", pairs, state_vector[channel:, :])
    return result


def _apply_specular_operator(
    state_vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    operator_data: _ScipyOperatorData,
    *,
    channel_idx: int,
) -> None:
    """
    Apply the inverse specular operator (H_0 - E_i)^(-1).

    This applies the uncoupled Green's function to the state vector
    1 / (H_0 - E_i) for the channel i.
    """
    # Converts to the H_0 eigenbasis, where the uncoupled Green's function is diagonal.
    transformed_state = np.einsum(
        "lk,l->k",
        operator_data.eigenvectors,
        state_vector[channel_idx, :],
    )
    # apply the (H_0 - E_alpha)^(-1) operator
    transformed_state /= (
        operator_data.perpendicular_kinetic_difference[channel_idx]
        + operator_data.eigenvalues
    )
    # Convert back to initial basis
    state_vector[channel_idx, :] = np.einsum(
        "kl,l->k",
        operator_data.eigenvectors,
        transformed_state,
    )


def _apply_boundary_corrections(
    state_vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    operator_data: _ScipyOperatorData,
    *,
    channel_idx: int,
) -> None:
    """
    Enforce the outgoing wave boundary conditions on the state vector.

    This applies the outgoing-wave boundary correction C_i at the final
    grid point (nz - 1) using the Sherman-Morrison rule.

    We want to apply the operator
        (H_0 - E_alpha + C_i u_i v_i^T)^(-1) psi
    We have the result of (H_0 - E_alpha)^(-1) psi, and we can
    therefore calculate the inverse using the Sherman-Morrison formula.

    Here,
     u is a vector with a 1 at the boundary and 0 elsewhere
     (H_0 - E_alpha)^(-1) C_i u_i is simply lower_block_factors[:, i]
     v^T is a vector with a boundary value -C at the boundary and 0 elsewhere


    The Sherman-Morrison formula for the inverse of a rank-1 update is
    (H_0 - E_alpha + C_i u_i v_i^T)^(-1) psi
    =
    (H_0 - E_alpha)^(-1) psi -
    (lower_block * (v_i^T (H_0 - E_alpha)^(-1) psi)) / denom

    with denom = 1.0 - (v_i^T (H_0 - E_alpha)^(-1) C_i u_i)

    where v_i^T (H_0 - E_alpha)^(-1) psi is state_vector[channel_idx, -1]
          (H_0 - E_alpha)^(-1) psi  is state_vector[channel_idx, :]

    """
    denom = 1.0 - (
        operator_data.lower_block_factors[-1, channel_idx]
        * operator_data.outgoing_log_derivative_wave[channel_idx]
    )
    fac = (
        state_vector[channel_idx, -1]
        * operator_data.outgoing_log_derivative_wave[channel_idx]
        / denom
    )
    state_vector[channel_idx, :] += (
        fac * operator_data.lower_block_factors[:, channel_idx]
    )


def _apply_uncoupled_inverse_lower_block_operator(
    state_vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    operator_data: _ScipyOperatorData,
    *,
    channel_idx: int,
) -> None:
    """
    Apply the inverse operator (H_0 - E_i + C_i u_i v_i^T)^(-1).

    This applies the uncoupled Green's function to the state vector
    1 / (H_0 - E_i + C_i u_i v_i^T) for the channel i, where C_i is the
    outgoing wave boundary correction at the final grid point.
    """
    # First, apply the inverse of the specular operator (H_0 - E_i)^(-1) psi
    _apply_specular_operator(state_vector, operator_data, channel_idx=channel_idx)
    # Use the Sherman-Morrison formula to apply a boundary correction
    # The resulting state is (H_0 - E_i + C_i u_i v_i^T)^(-1) psi
    _apply_boundary_corrections(state_vector, operator_data, channel_idx=channel_idx)


def _apply_inverse_lower_block(
    state_vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    operator_data: _ScipyOperatorData,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """
    Apply the inverse of the lower block diagonal operator to the state vector.

    The lower diagonal operator
    contains (H_0 - E_i + C_i u_i v_i^T + V^lower)

    This can be inverted efficiently using a simple trick!
    For a lower diagonal operator L = (D + V^lower)
    where D is the diagonal part of the operator, and V is the
    part that is strictly lower diagonal, we can write

    (D_i out_i + V^lower_ij out_j) = state_i
    => out_i = D^(-1)_i (state_i - V^lower_ij out_j)

    but since V^lower is strictly lower diagonal, we can solve for out_i!

    """
    solved = state_vector.copy()

    for channel in range(operator_data.channel_count):
        # subtract V^lower_ij out_j
        pairs = operator_data.potential_pairs[channel, :channel, :]
        solved[channel, :] -= np.einsum("ik,ik->k", pairs, solved[:channel, :])

        # Apply D^(-1)_i
        _apply_uncoupled_inverse_lower_block_operator(
            solved,
            operator_data,
            channel_idx=channel,
        )

    return solved


def _run_gauss_seidel_gradient_decent(  # cspell: disable-line
    initial_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    inverse_lower: scipy.sparse.linalg.LinearOperator,
    upper: scipy.sparse.linalg.LinearOperator,
    *,
    config: OptimizationConfig,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """
    Use gradient decent to solve the linear system (I + L^{-1} U) psi = L^{-1} psi.

    This is equivalent to solving the original linear problem
    (L + U) psi = b, but is more efficient to solve since the operator (I + L^{-1} U)
    is closer to the identity.

    The input to this function is the initial guess at psi.
    The output of this function is L^{-1} psi, we
    can recover psi by applying the lower operator.

    Optionally, this probelm can be "double preconditioned"
    by applying a first-order Neumann series approximation to the operator
    (I - L^{-1} U) as a preconditioner to the GMRES solver. This should improve
    convergence if L^{-1} U is small.
    """
    n_states = initial_state.size

    def _apply_linear_operator(
        state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """
        Apply the preconditioned operator (I + L^{-1} * U) to the state vector.

        In the "simple" problem, we would solve (H_0 + V + C_i u_iv_i^T) psi = b.
        We split this operator into (L + U) psi = b
        where L is the lower operator and U is the upper operator.

        L contains the lower diagonal terms in the scattering
        potential, and the diagonal terms
        (H_0 and the boundary correction C_i u_i v_i^T).

        We then apply the preconditioner L^{-1} to both sides, giving
        (I + L^{-1} * U) psi = L^{-1} b

        We always use the specular preconditioner L^{-1},
        which is required for convergence.
        """
        return state + inverse_lower.matvec(upper.matvec(state))  # type: ignore[unknown]

    def _apply_neumann_preconditioner(
        state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """
        Valuates a first-order Neumann series approximation: (I - L^{-1}U).

        If L^{-1}U is small, this is an approximate solution to the scattering problem.
        """
        return state - inverse_lower.matvec(upper.matvec(state))  # type: ignore[unknown]

    linear_operator = scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
        (n_states, n_states),
        _apply_linear_operator,
    )
    preconditioner = (
        scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
            (n_states, n_states),
            _apply_neumann_preconditioner,
        )
        if config.use_neumann_preconditioner
        else None
    )

    resid_bar = tqdm(total=1.0, desc="Total Convergence", position=1, leave=False)

    def _callback(pr_norm: float) -> None:
        error = max(0, round(np.log10(pr_norm / config.precision), 3))
        next_progress = round(resid_bar.total - error, 3)
        if next_progress < 0:
            resid_bar.reset(total=error)
            next_progress = 0

        resid_bar.n = next_progress
        resid_bar.refresh()

    restart = min(config.max_iterations, initial_state.size)
    solution, gmres_info = cast(
        "tuple[np.ndarray[Any, np.dtype[np.complex128]], int]",
        scipy.sparse.linalg.gmres(  # type: ignore[unknown]
            A=linear_operator,
            b=inverse_lower(initial_state),
            rtol=config.precision,
            restart=restart,
            maxiter=config.max_iterations,
            M=preconditioner,
            callback=_callback,
            callback_type="pr_norm",
        ),
    )
    resid_bar.close()

    if gmres_info != 0:
        msg = (
            "SciPy GMRES did not converge "
            f"(info={gmres_info}, max_iterations={config.max_iterations}, "
            f"restart={restart})."
        )
        raise RuntimeError(msg)

    return solution


def _run_multiscat_scipy(  # noqa: PLR0913
    config: OptimizationConfig,
    *,
    potential_values: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    perpendicular_kinetic_difference: np.ndarray[
        tuple[int, int],
        np.dtype[np.float64],
    ],
    b_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    outgoing_log_derivative_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    parallel_kinetic_energy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    nkx, nky, nz = potential_values.shape

    inverse_lower, upper = _build_scipy_operators(
        potential_values,
        perpendicular_kinetic_difference,
        outgoing_log_derivative_wave,
        parallel_kinetic_energy,
    )

    # Prepare the initial guess
    # This is the initial state with only the incoming wave in the specular channel.
    initial_state = np.zeros((nkx, nky, nz), dtype=np.complex128)
    initial_state[0, 0, -1] = b_wave[0, 0]

    solution = _run_gauss_seidel_gradient_decent(  # cspell: disable-line
        initial_state=initial_state.ravel(),
        inverse_lower=inverse_lower,
        upper=upper,
        config=config,
    )

    return solution.reshape((nkx, nky, nz))


@timed
def get_scattering_matrix[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[
        M0,
        M1,
        E,
    ],
    config: OptimizationConfig,
    *,
    backend: Literal["fortran", "scipy"] = "fortran",
) -> Array[
    Basis[TupleMetadata[tuple[M0, M0], AxisDirections]],
    np.dtype[np.complex128],
]:
    """Run Multiscat through the f2py native binding."""
    (
        incident_k,
        potential,
        metadata_xy,
        metadata_z,
    ) = _condition_parameters(condition)
    perpendicular_kinetic_difference = _get_perpendicular_kinetic_difference(
        incident_k,
        metadata_xy,
    )
    # Note: here we differ by conventions for n_z.
    # The outer python code assumes n_z includes the boundary,
    # whereas the inner fortran code assumes n_z does not include the boundary.
    parallel_kinetic_energy = -_get_parallel_kinetic_energy(
        metadata=metadata_z,
    )

    a_wave, b_wave = _get_ab_waves(
        metadata_z,
        perpendicular_kinetic_difference,
    )

    outgoing_log_derivative_wave = _get_outgoing_log_derivative_wave(
        metadata_z,
        perpendicular_kinetic_difference,
    )

    if backend == "fortran":
        (
            preconditioner_flag,
            n_significant_figures,
        ) = _optimization_parameters(config)
        solution = run_multiscat_fortran(
            preconditioner_flag,
            n_significant_figures,
            potential_values=potential,
            perpendicular_kinetic_difference=perpendicular_kinetic_difference,
            wave_a=a_wave.ravel(),
            wave_b=b_wave.ravel(),
            wave_c=outgoing_log_derivative_wave.ravel(),
            parallel_kinetic_energy=parallel_kinetic_energy,
        )
    elif backend == "scipy":
        solution = _run_multiscat_scipy(
            config,
            potential_values=potential,
            perpendicular_kinetic_difference=perpendicular_kinetic_difference,
            b_wave=b_wave,
            outgoing_log_derivative_wave=outgoing_log_derivative_wave,
            parallel_kinetic_energy=parallel_kinetic_energy,
        )
    else:
        msg = f"Unknown backend '{backend}'. Expected 'fortran' or 'scipy'."
        raise ValueError(msg)

    channel_intensity_dense = get_scattered_intensity(
        solution,
        a_wave,
        b_wave,
    )

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return Array(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
        channel_intensity_dense.astype(np.complex128),
    )
