import warnings
from typing import Any, cast

import numpy as np
import scipy.sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg  # type: ignore[import-untyped]
from slate_core import (
    Array,
    Basis,
    SimpleMetadata,
    TupleBasis,
    TupleMetadata,
    array,
    basis,
)
from slate_core.basis import AsUpcast, ContractedBasis, DiagonalBasis
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
    LobattoSpacedMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_k_points
from slate_quantum import Operator, State
from slate_quantum.operator import (
    OperatorMetadata,
    operator_basis,
)
from tqdm import tqdm

from multiscat.basis import (
    CloseCouplingBasis,
    ScatteringBasisMetadata,
    close_coupling_basis,
    split_scattering_metadata,
)
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.interpolate import ScatteringOperator
from multiscat.polynomial import (
    get_barycentric_kinetic_operator,
)

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
    # TODO: we should represent this data as an Operator in a sparse # noqa: FIX002
    # basis. Issue is that this does not lend itself to an efficient
    # implementation when we add the parrallel and perpendicular
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
    # The parallel kinetic energy is diagonal in bloch K, but is non-diagonal
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
        data=data.astype(np.complex128),
    )


def _get_scattered_state_log_derivative[
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
    # TODO: build the necessary operators for the log derivative  # noqa: FIX002
    # functional.
    msg = "This function is not implemented yet. "
    raise NotImplementedError(msg)


def _gmres[DT: np.dtype[np.number]](
    matrix: scipy.sparse.linalg.LinearOperator,
    b: np.ndarray[Any, DT],
    *,
    initial_state: np.ndarray[Any, DT] | None = None,
    inverse_preconditioner: scipy.sparse.linalg.LinearOperator | None = None,
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
        b=b,
        x0=initial_state,
        rtol=options.precision,
        maxiter=options.max_iterations,
        M=inverse_preconditioner,
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


def _get_scattered_state[
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

    fundamental_state_basis = AsUpcast(
        basis.from_metadata(state_metadata),
        state_metadata,
    )
    potential_basis = DiagonalBasis(operator_basis(fundamental_state_basis)).upcast()
    # We ensure we are storing V(x) as a diagonal operator for efficiency
    potential = potential.with_basis(potential_basis)
    kinetic_basis = _get_kinetic_difference_operator_basis(
        state_metadata,
    )
    kinetic_difference = kinetic_difference.with_basis(kinetic_basis)
    kinetic_raw = kinetic_difference.raw_data.reshape(
        (nx, ny, nz, nz),
    )

    def matmul_hamiltonian(
        state: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
        """Cost function for the GMRES solver."""
        state_array = Array(state_basis, state)
        # The cost function is the kinetic energy minus the potential energy
        # TODO: we want to make einsum be smarter about  # noqa: FIX002
        # contracted basis, so we dont need to do this manually.
        # Once this is done we can repalce with the commented code below.
        # ! cost_kinetic = (
        # !     linalg.einsum("(i j'),j -> i", kinetic_difference, state_array)
        # !     .with_basis(state_basis)
        # !     .raw_data
        # ! )
        cost_kinetic = np.einsum(
            "ijkl,ijl->ijk",
            kinetic_difference.with_basis(kinetic_basis).raw_data.reshape(
                (nx, ny, nz, nz),
            ),
            state.reshape((nx, ny, nz)),
        ).ravel()
        # TODO: we want to make einsum be smarter about # noqa: FIX002
        # contracted basis, so we dont need to do this manually.
        # Once this is done we can repalce with the commented code below.
        # TODO: we should make use of the fact that V(k) is sparse! # noqa: FIX002
        # ! cost_potential = (
        # !     linalg.einsum(
        # !         "(i j'),j -> i",
        # !         potential,
        # !         state_array.with_basis(fundamental_state_basis),
        # !     )
        # !     .with_basis(state_basis)
        # !     .raw_data
        # ! )
        cost_potential = (
            Array(
                fundamental_state_basis,
                np.einsum(
                    "i,i -> i",
                    potential.with_basis(potential_basis).raw_data,
                    state_array.with_basis(fundamental_state_basis).raw_data,
                ),
            )
            .with_basis(state_basis)
            .raw_data
        )
        return cost_kinetic + cost_potential

    # Build the inverse_preconditioner. this is a matrix which
    # approximates the inverse of the Hamiltonian, but is easier to
    # invert.
    # We use a bloch diagonal Hamiltonian here, which is the kinetic energy
    # plus the potential energy at V_{G=0}
    # TODO: better einsum support for this, we should be able to use  # noqa: FIX002
    # einsum to contract the potential operator with the state basis
    # This gives the components of the diagonal V(x)
    diagonal_potential = np.fft.fftn(
        potential.with_basis(potential_basis).raw_data.reshape(
            (nx, ny, nz),
        ),
        axes=(0, 1),
        s=(1, 1),
        norm="ortho",
    )
    bloch_diagonal_potential = np.einsum(
        "ijk,kl->ijkl",
        diagonal_potential,
        np.eye(nz),
    )
    bloch_diagonal_hamiltonian = kinetic_raw + bloch_diagonal_potential
    inverse_preconditioner = cast(
        "np.ndarray[Any, np.dtype[np.complexfloating]]",
        np.linalg.inv(bloch_diagonal_hamiltonian),
    )

    def matmul_preconditioner(
        state: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
        """Cost function for the GMRES solver."""
        # The cost function is the kinetic energy minus the potential energy
        return np.einsum(
            "ijkl,ijl->ijk",
            inverse_preconditioner,
            state.reshape((nx, ny, nz)),
        ).ravel()

    incoming = np.zeros((nx, ny, nz), dtype=np.complex128)
    incoming[0, 0, -1] = 1 / (state_metadata.children[2].basis_weights[-1])

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
            shape=(state_metadata.fundamental_size, state_metadata.fundamental_size),
            matvec=matmul_hamiltonian,  # type: ignore[call-arg]
            dtype=np.complex128,
        ),
        b=-matmul_hamiltonian(incoming.ravel()),
        inverse_preconditioner=scipy.sparse.linalg.LinearOperator(
            shape=(state_metadata.fundamental_size, state_metadata.fundamental_size),
            matvec=matmul_preconditioner,  # type: ignore[call-arg]
            dtype=np.complex128,
        ),
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
    kinetic_difference = get_kinetic_difference_operator(
        condition.incident_k,
        condition.metadata,
    )
    if options.method == "log_derivative":
        return _get_scattered_state_log_derivative(
            kinetic_difference,
            condition.potential,
            options=options,
        )

    return _get_scattered_state(
        kinetic_difference,
        condition.potential,
        options=options,
    )


def get_scattering_matrix[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[CloseCouplingBasis[M0, M1, E]],
) -> Array[
    Basis[TupleMetadata[tuple[M0, M0], AxisDirections]],
    np.dtype[np.complexfloating],
]:
    metadata_x01, _ = split_scattering_metadata(state.basis.metadata())
    return Array(
        AsUpcast(basis.from_metadata(metadata_x01), metadata_x01),
        state.as_array()[:, :, -1],
    )
