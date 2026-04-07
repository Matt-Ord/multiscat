from typing import TYPE_CHECKING

import numpy as np
from slate_quantum import State

if TYPE_CHECKING:
    from multiscat.config import (  # type: ignore[import-untyped]
        ScatteringCondition,
    )
    from multiscat.interpolate import ScatteringOperator


from slate_core import (
    Basis,
    TupleMetadata,
    array,
)
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
    LobattoSpacedMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_k_points

from multiscat.basis import (
    ScatteringBasisMetadata,
    close_coupling_basis,
    split_scattering_metadata,
)
from multiscat.polynomial import (
    get_barycentric_kinetic_operator,
)


def get_parallel_kinetic_energy(
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


def get_perpendicular_kinetic_difference(
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


def get_ab_waves(
    metadata: LobattoSpacedMetadata,
    perpendicular_kinetic_difference: np.ndarray[
        tuple[int, int],
        np.dtype[np.floating],
    ],
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
]:
    """
    Get the asymptotic initial state and final scattered state amplitude factors.

    Here a_wave is the amplitude of the incoming wave, i(r)
    and b_wave is the inverse of the amplitude of the outgoing wave, o(r)^(-1).
    """
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


def get_ab_wave_for_condition[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
]:
    """Get the asymptotic initial state and final scattered state amplitude factors."""
    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)
    perpendicular_kinetic_difference = get_perpendicular_kinetic_difference(
        condition.incident_k,
        metadata_x01,
    )
    return get_ab_waves(metadata_z, perpendicular_kinetic_difference)


def get_outgoing_log_derivative_wave(
    metadata: LobattoSpacedMetadata,
    perpendicular_kinetic_difference: np.ndarray[
        tuple[int],
        np.dtype[np.floating],
    ],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """Get the outgoing channel logarithmic derivatives."""
    channel_energy = perpendicular_kinetic_difference

    out = 1j * np.emath.sqrt(-channel_energy)  # cspell: disable-line

    return out * metadata.basis_weights[-1] ** 2


def potential_as_array(
    potential: ScatteringOperator[
        EvenlySpacedLengthMetadata,
        LobattoSpacedLengthMetadata,
        AxisDirections,
    ],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    potential_diagonal = array.extract_diagonal(potential)
    nx, ny, nz = potential_diagonal.basis.metadata().shape
    basis_weights = potential_diagonal.basis.metadata().children[2].basis_weights
    basis = close_coupling_basis(potential_diagonal.basis.metadata())

    data = potential_diagonal.with_basis(basis).raw_data.reshape((nx, ny, nz))
    return data * basis_weights[np.newaxis, np.newaxis, :] / np.sqrt(nx * ny)


def get_b_wave(
    metadata: LobattoSpacedMetadata,
    energy: float,
) -> complex:
    """Get the inverse of outgoing wave amplitude, for a channel with a given energy."""
    open_channel = energy < 0.0
    if not open_channel:
        return 0
    dk = np.sqrt(np.abs(energy))
    theta = dk * (metadata.delta - metadata.domain.start)
    return np.sqrt(dk) * np.exp(-1j * theta) * metadata.basis_weights[-1]


def get_target_state[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
) -> State[Basis[ScatteringBasisMetadata[M0, M1, E]]]:
    """Get the target state of the scattering problem."""
    _, metadata_z = split_scattering_metadata(condition.metadata)
    initial_state = np.zeros(condition.metadata.shape, dtype=np.complex128)
    specular_perpendicular_kinetic_difference = -(condition.incident_k[2] ** 2)
    initial_state[0, 0, -1] = get_b_wave(
        metadata_z,
        specular_perpendicular_kinetic_difference,
    )
    return State(
        close_coupling_basis(condition.metadata).upcast(),
        initial_state.ravel(),
    )
