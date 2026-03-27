import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import scipy.sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg  # type: ignore[import-untyped]
from multiscat_fortran import (
    get_parallel_kinetic_energy,
    get_perpendicular_kinetic_difference,
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
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.polynomial import (
    get_barycentric_kinetic_operator,
)

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
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
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
    lobatto_metadata = metadata.children[2]
    return array.as_fundamental_basis(
        get_barycentric_kinetic_operator(lobatto_metadata),
    ).raw_data.reshape(
        lobatto_metadata.fundamental_size,
        lobatto_metadata.fundamental_size,
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
    # basis. Issue is that the array does not have and index for the parallel
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
    t_jk = _get_parallel_kinetic_energy(metadata)
    # The perpendicular kinetic energy difference is diagonal in both the bloch K,
    # and the lobatto basis functions. Here we scale by the lobatto weights
    d_i = _get_perpendicular_kinetic_difference(incident_k, metadata)
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
        # TODO: can we do this more efficiently using fourier transforms? # noqa: FIX002
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
) -> tuple[float, float, float, float]:
    scattering_vector = np.asarray(condition.incident_k)
    scattering_magnitude = float(np.linalg.norm(scattering_vector))
    if scattering_magnitude <= 0:
        msg = "Incident wavevector magnitude must be non-zero"
        raise ValueError(msg)

    mass_amu = float(condition.mass / atomic_mass)
    kx, ky, kz = condition.incident_k
    return mass_amu, float(kx * angstrom), float(ky * angstrom), float(kz * angstrom)


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
    return data.ravel() / (electron_volt * 10**-3 * np.sqrt(nx * ny))


def _potential_parameters(
    potential: ScatteringOperator[
        EvenlySpacedLengthMetadata,
        LobattoSpacedLengthMetadata,
        AxisDirections,
    ],
) -> tuple[
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    LobattoSpacedLengthMetadata,
]:
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

    potential_dense = np.asfortranarray(potential_lobatto.reshape((nx, ny, nz)))
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
        potential_dense,
        LobattoSpacedLengthMetadata(
            fundamental_size=nz,
            domain=Domain(
                start=z_start_angstrom,
                delta=(z_end_angstrom - z_start_angstrom),
            ),
        ),
    )


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
    scattered_log_derivative: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    a_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    b_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Recover per-channel intensities from the optimized scattered state."""
    surface_log_derivative = scattered_log_derivative[:, :, -1]
    # b_wave is the inverse of the outgoing wave amplitude
    # b_wave is equal to o(r)^(-1)
    # This therefore recovers the scattered state from the
    # log derivative
    surface_state = 2.0j * b_wave * surface_log_derivative
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

    node_count = metadata.fundamental_size + 1
    endpoint_weight = np.sqrt((metadata.delta) / (node_count * (node_count - 1)))
    b_wave = b_wave / endpoint_weight

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
    channel_energy = perpendicular_kinetic_difference.ravel(order="C")

    out = 1j * np.emath.sqrt(-channel_energy)  # cspell: disable-line

    node_count = metadata.fundamental_size + 1
    endpoint_weight = np.sqrt((metadata.delta) / (node_count * (node_count - 1)))
    out = out / (endpoint_weight**2)

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


def _build_scipy_operator_data(
    potential_values: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    perpendicular_kinetic_difference: np.ndarray[
        tuple[int, int],
        np.dtype[np.float64],
    ],
    outgoing_log_derivative_wave: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    parallel_kinetic_energy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> _ScipyOperatorData:
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

    return _ScipyOperatorData(
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
    for j in range(1, operator_data.channel_count):
        j_minus_1 = j - 1
        pairs = operator_data.potential_pairs[j_minus_1, j:, :]
        result[:, j_minus_1] = np.einsum("ik,ki->k", pairs, state_vector[:, j:])
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
        state_vector[:, channel_idx],
    )
    # apply the (H_0 - E_alpha)^(-1) operator
    transformed_state /= (
        operator_data.perpendicular_kinetic_difference[channel_idx]
        + operator_data.eigenvalues
    )
    # Convert back to initial basis
    state_vector[:, channel_idx] = np.einsum(
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

    where v_i^T (H_0 - E_alpha)^(-1) psi is state_vector[-1, channel_idx]
          (H_0 - E_alpha)^(-1) psi  is state_vector[:, channel_idx]

    """
    denom = 1.0 - (
        operator_data.lower_block_factors[-1, channel_idx]
        * operator_data.outgoing_log_derivative_wave[channel_idx]
    )
    fac = (
        state_vector[-1, channel_idx]
        * operator_data.outgoing_log_derivative_wave[channel_idx]
        / denom
    )
    state_vector[:, channel_idx] += (
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

    for j in range(operator_data.channel_count):
        # subtract V^lower_ij out_j
        pairs = operator_data.potential_pairs[j, :j, :]
        solved[:, j] -= np.einsum("ik,ki->k", pairs, solved[:, :j])

        # Apply D^(-1)_i
        _apply_uncoupled_inverse_lower_block_operator(
            solved,
            operator_data,
            channel_idx=j,
        )

    return solved


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
    channel_count = nkx * nky
    operator_data = _build_scipy_operator_data(
        potential_values,
        perpendicular_kinetic_difference,
        outgoing_log_derivative_wave,
        parallel_kinetic_energy,
    )

    def _apply_coupling_solve(
        flat_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        state = flat_state.reshape((channel_count, nz)).T
        upper = _apply_upper_block(state, operator_data)
        lower = _apply_inverse_lower_block(upper, operator_data)
        return lower.T.reshape((-1,))

    def _apply_preconditioned_operator(
        flat_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
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
        return flat_state + _apply_coupling_solve(flat_state)

    def _apply_neumann_preconditioner(
        flat_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """
        Valuates a first-order Neumann series approximation: (I - L^{-1}U).

        If L^{-1}U is small, this is an approximate solution to the scattering problem.
        """
        return flat_state - _apply_coupling_solve(flat_state)

    # Prepare the rhs vector L^{-1} b, where b
    # is the initial state with only the incoming wave in the specular channel.
    rhs = np.zeros((nz, channel_count), dtype=np.complex128)
    rhs[-1, operator_data.specular_channel] = b_wave[0, 0]
    rhs = _apply_inverse_lower_block(rhs, operator_data)

    linear_operator = scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
        (nz * channel_count, nz * channel_count),
        _apply_preconditioned_operator,
    )
    preconditioner_operator = (
        scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
            (nz * channel_count, nz * channel_count),
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

    # restart should not exceed system dimension.
    krylov_dim = min(config.max_iterations, nz * channel_count)  # cspell: disable-line
    solution, gmres_info = cast(
        "tuple[np.ndarray[Any, np.dtype[np.complex128]], int]",
        scipy.sparse.linalg.gmres(  # type: ignore[unknown]
            A=linear_operator,
            b=rhs.T.reshape((-1,)),
            rtol=config.precision,
            restart=krylov_dim,  # cspell: disable-line
            maxiter=config.max_iterations,
            M=preconditioner_operator,
            callback=_callback,
            callback_type="pr_norm",
        ),
    )
    resid_bar.close()

    if gmres_info != 0:
        msg = (
            "SciPy GMRES did not converge "
            f"(info={gmres_info}, max_iterations={config.max_iterations}, "
            f"restart={krylov_dim})."  # cspell: disable-line
        )
        raise RuntimeError(msg)

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
    mass_amu, incident_kx, incident_ky, incident_kz = _condition_parameters(
        condition,
    )

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
        metadata_z,
    ) = _potential_parameters(condition.potential)
    perpendicular_kinetic_difference = get_perpendicular_kinetic_difference(
        incident_kx,
        incident_ky,
        incident_kz,
        nx=nkx,
        ny=nky,
        unit_cell_ax=unit_cell_ax,
        unit_cell_ay=unit_cell_ay,
        unit_cell_bx=unit_cell_bx,
        unit_cell_by=unit_cell_by,
    )
    parallel_kinetic_energy = get_parallel_kinetic_energy(
        zmin=zmin,
        zmax=zmax,
        nz=nz,
    )

    a_wave, b_wave = _get_ab_waves(
        metadata_z,
        perpendicular_kinetic_difference,
    )

    outgoing_log_derivative_wave = _get_outgoing_log_derivative_wave(
        metadata_z,
        perpendicular_kinetic_difference,
    )

    hbar_squared = (hbar**2 / (atomic_mass * electron_volt * angstrom**2)) * 1e3
    scaled_potential_values = np.asfortranarray(
        potential_values * ((2.0 * mass_amu) / hbar_squared),
    )

    if backend == "fortran":
        (
            preconditioner_flag,
            n_significant_figures,
        ) = _optimization_parameters(config)
        scattered_state_dense = run_multiscat_fortran(
            preconditioner_flag,
            n_significant_figures,
            potential_values=scaled_potential_values,
            perpendicular_kinetic_difference=perpendicular_kinetic_difference,
            wave_a=a_wave.ravel(),
            wave_b=b_wave.ravel(),
            wave_c=outgoing_log_derivative_wave.ravel(),
            parallel_kinetic_energy=parallel_kinetic_energy,
        )
    elif backend == "scipy":
        scattered_state_dense = _run_multiscat_scipy(
            config,
            potential_values=scaled_potential_values,
            perpendicular_kinetic_difference=perpendicular_kinetic_difference,
            b_wave=b_wave,
            outgoing_log_derivative_wave=outgoing_log_derivative_wave,
            parallel_kinetic_energy=parallel_kinetic_energy,
        )
    else:
        msg = f"Unknown backend '{backend}'. Expected 'fortran' or 'scipy'."
        raise ValueError(msg)

    channel_intensity_dense = get_scattered_intensity(
        scattered_state_dense,
        a_wave,
        b_wave,
    )

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return Array(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
        channel_intensity_dense.astype(np.complex128),
    )
