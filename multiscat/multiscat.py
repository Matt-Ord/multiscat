from typing import Any, cast

import numpy as np
import scipy.sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg  # type: ignore[import-untyped]
from slate_core import SimpleMetadata, TupleBasis, TupleMetadata
from slate_core.basis import AsUpcast, ContractedBasis
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_k_points
from slate_quantum import Operator, State
from slate_quantum.operator import OperatorMetadata, operator_basis

from multiscat.basis import (
    CloseCouplingBasis,
    ScatteringBasisMetadata,
    close_coupling_basis,
    split_scattering_metadata,
)
from multiscat.config import OptimizationConfig, ScatteringCondition
from multiscat.interpolate import ScatteringOperator
from multiscat.lobatto import (
    LobattoSpacedLengthMetadata,
    LobattoSpacedMetadata,
    get_lobatto_derivative_matrix,
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
    derivatives = get_lobatto_derivative_matrix(lobatto_metadata)
    # TODO: we should represent this data as an Operator in a sparse # noqa: FIX002
    # basis. Issue is that the array does not have and index for the perpendicular
    # direction, so we cannot use existing ContractedBasis functionality
    assert not lobatto_metadata.is_periodic  # noqa: S101
    return np.einsum(
        "k,ik,jk->ij",
        lobatto_metadata.quadrature_weights,
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

    potential_raw = potential.with_basis(
        operator_basis(state_basis),
    ).raw_data.reshape(
        (nx * ny * nz, nx * ny * nz),
    )
    kinetic_raw = kinetic_difference.raw_data.reshape(
        (nx, ny, nz, nz),
    )

    initial_state = np.zeros((nx, ny, nz), dtype=np.complexfloating)
    initial_state[0, 0, 0] = 1.0

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
    # TODO: how do we ensure bcs are satisfied, we chould probably  # noqa: FIX002
    # manually add the initial condition each iteration? Is this evven needed?
    data, _info = scipy.sparse.linalg.gmres(  # type: ignore[unknown]
        A=scipy.sparse.linalg.LinearOperator(
            shape=potential_raw.shape,
            matvec=matmul_hamiltonian,  # type: ignore[call-arg]
            dtype=np.complexfloating,
        ),
        b=initial_state.ravel(),
        rtol=options.precision,
        maxiter=options.max_iterations,
    )
    return State(
        state_basis,
        cast("np.ndarray[Any, np.dtype[np.complexfloating]]", data),
    )


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
