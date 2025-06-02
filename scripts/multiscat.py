from dataclasses import dataclass

import numpy as np
from slate_core import SimpleMetadata, TupleBasis, TupleMetadata
from slate_core.basis import AsUpcast, ContractedBasis
from slate_core.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_k_points
from slate_quantum import Operator, State
from slate_quantum.operator import OperatorMetadata, operator_basis

from multiscat.config import GMRESConfig
from multiscat.lobatto import (
    CloseCouplingBasis,
    LobattoMetadata,
    ScatteringBasisMetadata,
    ScatteringPotentialWithMetadata,
    close_coupling_basis,
    get_lobatto_derivative_matrix,
    get_split_scattering_metadata,
)


@dataclass
class ScatteringCondition:
    """Represents a particular scattering condition."""

    mass: float
    incident_k: tuple[float, float, float]
    potential: ScatteringPotentialWithMetadata

    @property
    def metadata(self) -> ScatteringBasisMetadata:
        """The metadata for the scattering state."""
        return self.potential.basis.metadata().children[0]


def _get_parallel_kinetic_energy(
    metadata: LobattoMetadata,
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
    derivatives = get_lobatto_derivative_matrix(metadata)
    return np.einsum("k,ik,jk->ij", metadata.weights, derivatives, derivatives)


def _get_perpendicular_kinetic_difference(
    incident_k: tuple[float, float, float],
    metadata: TupleMetadata[
        tuple[SpacedLengthMetadata, SpacedLengthMetadata],
        AxisDirections,
    ],
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """
    Get the matrix of scattered energies d.

    Uses the formula
    d^2 = |k in + k scatter|**2 - |k in|**2
    """
    # TODO: previously we worked in a sparse basis in d # noqa: FIX002
    # such that di < config.dmax
    # These channels will only have a small scattering contribution
    # It might be good to do this too!
    (kx, ky) = fundamental_stacked_k_points(metadata, offset=incident_k[:2])

    # d is the difference between the initial energy
    # and the final energy
    return ((kx**2 + ky**2) - np.linalg.norm(incident_k) ** 2).ravel()


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
    OperatorMetadata[TupleMetadata[tuple[M0, M0, M1], E]],
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


def get_kinetic_difference_operator[
    M0: SpacedLengthMetadata,
    M1: LobattoMetadata,
    E: AxisDirections,
](
    incident_k: tuple[float, float, float],
    metadata: ScatteringBasisMetadata[M0, M1, E],
) -> Operator[
    KineticDifferenceOperatorBasis[M0, M1, E],
    np.dtype[np.complexfloating],
]:
    """Get the matrix of kinetic energies minus the incident energy."""
    metadata_x01, lobatto_metadata = get_split_scattering_metadata(metadata)

    # The parallel kinetic energy is the same for each bloch K, but is non-diagonal
    # in the lobatto basis
    # TODO: slate should be clever enough to be ablt to efficiently # noqa: FIX002
    # add two contracted operators with the same underlying basis
    # If this was supported we could return "proper" Operator[] from this function
    t_ij = _get_parallel_kinetic_energy(lobatto_metadata)
    # The perpendicular kinetic energy difference is diagonal in bloch K, and is
    # the same for each lobatto basis function
    d_i = _get_perpendicular_kinetic_difference(incident_k, metadata_x01)

    # The resulting difference operator is diagonal in the two bloch K indices
    # and non-diagonal in the lobatto basis
    data = t_ij[np.newaxis, :, :] + d_i[:, np.newaxis, np.newaxis]
    return Operator(
        basis=_get_kinetic_difference_operator_basis(metadata),
        data=data.astype(np.complexfloating),
    )


def _get_scattered_state[
    M0: SpacedLengthMetadata,
    M1: LobattoMetadata,
    E: AxisDirections,
](
    kinetic_difference: Operator[
        KineticDifferenceOperatorBasis[M0, M1, E],
        np.dtype[np.complexfloating],
    ],
    potential: ScatteringPotentialWithMetadata[
        M0,
        M1,
        E,
    ],
    *,
    options: GMRESConfig,
) -> State[CloseCouplingBasis[M0, M1, E]]:
    """
    Get the basis for the scattered state.

    This is a diagonal basis in the lobatto basis, and a tuple basis in the
    bloch K indices.
    """
    msg = (
        "This function is not implemented yet..."
        "Please implement the GMRES solver to compute the scattered state."
    )
    raise NotImplementedError(
        msg,
    )


def get_scattered_state(
    condition: ScatteringCondition,
    *,
    options: GMRESConfig,
) -> State[CloseCouplingBasis]:
    kinetic_difference = get_kinetic_difference_operator(
        condition.incident_k,
        condition.metadata,
    )
    return _get_scattered_state(
        kinetic_difference,
        condition.potential,
        options=options,
    )
