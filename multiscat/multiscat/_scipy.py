from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg  # type: ignore[import-untyped]
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
)

from multiscat.basis import (
    split_scattering_metadata,
)
from multiscat.multiscat._gmres import run_gauss_seidel_gradient_decent
from multiscat.multiscat._util import (
    get_b_wave,
    get_outgoing_log_derivative_wave,
    get_parallel_kinetic_energy,
    get_perpendicular_kinetic_difference,
    potential_as_array,  # type: ignore[import-untyped]
)

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator  # type: ignore[untyped]

    from multiscat.config import OptimizationConfig, ScatteringCondition


def _solve_specular_hamiltonian(
    specular_potential: np.ndarray[tuple[int], np.dtype[np.complex128]],
    parallel_kinetic_energy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
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
    nz = specular_potential.size
    hamiltonian = parallel_kinetic_energy.copy()
    hamiltonian[np.diag_indices(nz)] += np.real(specular_potential)
    return np.linalg.eigh(hamiltonian)  # cspell: disable-line


def _build_lower_block_factors(
    channel_energy: np.ndarray[tuple[int], np.dtype[np.float64]],
    eigenvalues: np.ndarray[tuple[int], np.dtype[np.float64]],
    eigenvectors: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Calculate (H_0 + E_i C_i u_i)^(-1)."""
    channel_count = channel_energy.shape[0]
    nz = eigenvectors.shape[-1]

    # calculate (H_0 - E_alpha)^(-1) C_i u_i
    g = np.empty((nz, channel_count), dtype=np.float64)
    lower_block_factors = np.zeros((nz, channel_count), dtype=np.float64)
    for j in range(channel_count):
        g[:, j] = eigenvectors[-1, :] / (channel_energy[j] + eigenvalues)
        # Convert g back to the original basis
        lower_block_factors[:, j] = np.einsum("ki,i->k", eigenvectors, g[:, j])
    return lower_block_factors


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


@dataclass(frozen=True)
class _ScipyOperatorData:
    potential_pairs: np.ndarray[Any, np.dtype[np.complex128]]
    eigenvalues: np.ndarray[Any, np.dtype[np.float64]]
    eigenvectors: np.ndarray[Any, np.dtype[np.float64]]
    perpendicular_kinetic_difference: np.ndarray[Any, np.dtype[np.float64]]
    lower_block_factors: np.ndarray[Any, np.dtype[np.float64]]
    outgoing_log_derivative_wave: np.ndarray[Any, np.dtype[np.complex128]]

    @property
    def channel_count(self) -> int:
        return self.perpendicular_kinetic_difference.size


def _build_scipy_operators[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
) -> tuple[LinearOperator, LinearOperator]:
    # This is where most of the bad code lives.
    # Operator Data is legacy (from before using a LinearOperator)
    # And this should be removed / simplified in the future.
    # Ideally, we should clean this up, and add channel
    # filtering (ie discarding channels with very high kinetic energy,
    # maybe specify n_channels)
    # We should also build lower block factors when we define inverse_lower
    potential_values = potential_as_array(condition.potential)
    nkx, nky, nz = potential_values.shape

    idx_x, idx_y = np.meshgrid(np.arange(nkx), np.arange(nky), indexing="ij")
    idx_x = idx_x.ravel()
    idx_y = idx_y.ravel()
    diff_x = (idx_x[:, np.newaxis] - idx_x[np.newaxis, :]) % nkx
    diff_y = (idx_y[:, np.newaxis] - idx_y[np.newaxis, :]) % nky
    potential_pairs = potential_values[diff_x, diff_y, :]

    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)
    perpendicular_kinetic_difference = get_perpendicular_kinetic_difference(
        condition.incident_k,
        metadata_x01,
    )
    parallel_kinetic_energy = -get_parallel_kinetic_energy(metadata_z)

    eigenvalues, eigenvectors = _solve_specular_hamiltonian(
        specular_potential=potential_values[0, 0],
        parallel_kinetic_energy=parallel_kinetic_energy,
    )

    lower_block_factors = _build_lower_block_factors(
        channel_energy=perpendicular_kinetic_difference.ravel(),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )

    outgoing_log_derivative_wave = get_outgoing_log_derivative_wave(
        metadata_z,
        perpendicular_kinetic_difference,
    )

    operator_data = _ScipyOperatorData(
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


def _get_initial_state[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """
    Get the initial state for the scattering problem.

    For now, our guess at the solution is the state with only the incoming wave in
    the specular channel.
    """
    _, metadata_z = split_scattering_metadata(condition.metadata)
    initial_state = np.zeros(condition.metadata.shape, dtype=np.complex128)
    specular_perpendicular_kinetic_difference = -(condition.incident_k[2] ** 2)
    initial_state[0, 0, -1] = get_b_wave(
        metadata_z,
        specular_perpendicular_kinetic_difference,
    )
    return initial_state.ravel()


def run_multiscat_scipy[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    # We split our hamiltonian into a lower block diagonal part L,
    # and an upper part U which contains the off-diagonal coupling between channels.
    inverse_lower, upper = _build_scipy_operators(condition)
    initial_state = _get_initial_state(condition)

    solution = run_gauss_seidel_gradient_decent(  # cspell: disable-line
        initial_state=initial_state,
        inverse_lower=inverse_lower,
        upper=upper,
        config=config,
    )

    return solution.reshape(condition.metadata.shape)  # type: ignore[cant infer shape]
