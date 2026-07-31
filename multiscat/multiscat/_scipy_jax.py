from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
)

from multiscat.basis import (
    close_coupling_basis,
    split_scattering_metadata,
)
from multiscat.multiscat._gmres_jax import (
    run_gauss_seidel_gradient_decent,  # cspell: disable-line
)
from multiscat.multiscat._util import (
    get_outgoing_log_derivative_wave,
    get_parallel_kinetic_energy,
    get_perpendicular_kinetic_difference,
    get_target_state,
    potential_as_array,  # type: ignore[import-untyped]
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from multiscat.config import OptimizationConfig, ScatteringCondition


@jax.jit
def _solve_specular_hamiltonian(
    specular_potential: jnp.ndarray[tuple[int], jnp.dtype[jnp.complex128]],
    parallel_kinetic_energy: jnp.ndarray[tuple[int, int], jnp.dtype[jnp.float64]],
) -> tuple[
    jnp.ndarray[tuple[int], jnp.dtype[jnp.float64]],
    jnp.ndarray[Any, jnp.dtype[jnp.float64]],
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
    hamiltonian = parallel_kinetic_energy.at[jnp.diag_indices(nz)].add(
        jnp.real(specular_potential),
    )
    return jnp.linalg.eigh(hamiltonian)  # cspell: disable-line


@jax.jit
def _build_lower_block_factors(
    channel_energy: Array,
    eigenvalues: Array,
    eigenvectors: Array,
) -> Array:
    """Calculate (H_0 + E_alpha)^(-1) C_i u_i."""
    denominator = eigenvalues[:, None] + channel_energy[None, :]

    g = eigenvectors[-1, :, None] / denominator

    return eigenvectors @ g


@jax.jit
def _apply_upper_block(
    state_vector: jnp.ndarray[tuple[int, int], jnp.dtype[np.complex128]],
    operator_data: ScipyOperatorData,
):
    """
    Apply the off-diagonal channel-coupling potential V_1(z)Q(x,y).

    This calculates the scattering of the wave between different (kx, ky)
    diffraction channels. Since V_1(z) is responsible for all coupling
    between diffraction channels, this step represents the physical momentum
    transfer parallel to the corrugated surface.
    """
    N = state_vector.shape[0]

    upper = jnp.where(
        jnp.triu(jnp.ones((N, N), dtype=bool), k=1)[:, :, None],
        operator_data.potential_pairs,
        0,
    )
    return jnp.einsum("ijk,jk->ik", upper, state_vector)


@jax.jit
def apply_inverse_diagonal_to_channel(
    channel_state: Array,
    operator_data: DiagonalOperatorData,
    channel: int,
) -> Array:
    """Apply (H_0 - E_i + C_i u_i v_i^T)^(-1) to a single channel state vector."""
    # 1. Transform to H_0 eigenbasis
    transformed = operator_data.eigenvectors.T @ channel_state

    # 2. Apply uncoupled diagonal Green's function
    denom_diag = (
        operator_data.perpendicular_kinetic_difference[channel]
        + operator_data.eigenvalues
    )
    transformed = transformed / denom_diag

    # 3. Transform back to real-space grid basis
    channel_state = operator_data.eigenvectors @ transformed

    # 4. Sherman-Morrison boundary condition update
    denom_sm = 1.0 - (
        operator_data.lower_block_factors[-1, channel]
        * operator_data.outgoing_log_derivative_wave[channel]
    )

    fac = (
        channel_state[-1]
        * operator_data.outgoing_log_derivative_wave[channel]
        / denom_sm
    )

    return channel_state + fac * operator_data.lower_block_factors[:, channel]


@jax.jit
def apply_inverse_lower_block(
    state_vector: Array,
    operator_data: ScipyOperatorData,
) -> Array:
    """Apply the inverse of the lower block diagonal operator to the state vector."""
    n = state_vector.shape[0]
    channel_indices = jnp.arange(n)

    def body(channel, solved):
        # 1. Mask out channels j >= channel so array shapes remain strictly static
        mask = (channel_indices < channel)[:, None]  # Shape: (n, 1)
        masked_solved = solved * mask  # Zeroes out rows >= channel

        # 2. Compute channel coupling term
        coupling = jnp.einsum(
            "ik,ik->k",
            operator_data.potential_pairs[channel],
            masked_solved,
        )

        rhs = solved[channel] - coupling

        # 3. Invert diagonal operator for current channel
        updated = apply_inverse_diagonal_to_channel(
            rhs,
            operator_data,
            channel,
        )

        return solved.at[channel].set(updated)

    return lax.fori_loop(0, n, body, state_vector)


@jax.jit
def _apply_inverse_specular_operator(
    state_vector: Array,
    operator_data: DiagonalOperatorData,
) -> Array:
    """
    Apply the inverse specular operator (H_0 - E_i)^(-1) to all channels.

    This applies the uncoupled Green's function to the state vector
    1 / (H_0 - E_i) for all channels at once.
    """
    # 1. Transform to H_0 eigenbasis
    transformed_state = state_vector @ operator_data.eigenvectors

    # 2. Apply diagonal energy denominator across channels and z-grid
    denominator = (
        operator_data.perpendicular_kinetic_difference[:, None]
        + operator_data.eigenvalues[None, :]
    )
    transformed_state = transformed_state / denominator

    # 3. Transform back to original grid basis
    return transformed_state @ operator_data.eigenvectors.T


@jax.jit
def _add_boundary_corrections(
    state_vector: Array,
    operator_data: DiagonalOperatorData,
) -> Array:
    """
    Enforce outgoing wave boundary conditions across all channels simultaneously.

    Applies the Sherman-Morrison boundary correction at the final grid point (nz - 1).
    """
    denom = 1.0 - (
        operator_data.lower_block_factors[-1, :]
        * operator_data.outgoing_log_derivative_wave
    )
    fac = state_vector[:, -1] * operator_data.outgoing_log_derivative_wave / denom

    # Functional rank-1 correction broadcast across all channels
    return state_vector + fac[:, None] * operator_data.lower_block_factors.T


@jax.jit
def apply_inverse_diagonal(
    state_vector: Array,
    operator_data: DiagonalOperatorData,
) -> Array:
    """Apply D^(-1) to all channels in parallel."""
    out = _apply_inverse_specular_operator(state_vector, operator_data)
    return _add_boundary_corrections(out, operator_data)


@jax.jit
def apply_diagonal(
    state_vector: Array,
    operator_data: DiagonalOperatorData,
) -> Array:
    """
    Apply uncoupled diagonal block operator D_i = (H_0 - E_i + C_i u_i v_i^T)
    to all channels in parallel.
    """
    # 1. Capture input boundary values at z = -1 before applying (H_0 - E_i)
    boundary_values = state_vector[:, -1]

    # 2. Transform to H_0 eigenbasis (n_channels, nz) @ (nz, nz)
    transformed = state_vector @ operator_data.eigenvectors

    # 3. Multiply by diagonal spectrum (H_0 - E_i)
    spectrum = (
        operator_data.perpendicular_kinetic_difference[:, None]
        + operator_data.eigenvalues[None, :]
    )
    transformed = transformed * spectrum

    # 4. Transform back to original grid basis
    out = transformed @ operator_data.eigenvectors.T

    # 5. Apply rank-1 boundary correction -C_i e_n e_n^T at final grid point
    boundary_update = operator_data.outgoing_log_derivative_wave * boundary_values
    return out.at[:, -1].add(-boundary_update)


@jax.jit
def apply_lower_block(
    state_vector: Array,
    operator_data: ScipyOperatorData,
) -> Array:
    """Apply the lower block diagonal operator (D + V^lower) to the state vector."""
    n = state_vector.shape[0]

    # 1. Mask upper triangular and diagonal elements (j >= i) to keep only j < i
    mask = jnp.tril(jnp.ones((n, n), dtype=bool), k=-1)[:, :, None]
    lower_potentials = jnp.where(mask, operator_data.potential_pairs, 0)

    # 2. Compute channel-coupling term V^lower @ state_vector in parallel
    coupling = jnp.einsum("ijk,jk->ik", lower_potentials, state_vector)

    # 3. Combine diagonal operation D @ state_vector and lower coupling
    return apply_diagonal(state_vector, operator_data) + coupling


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiagonalOperatorData:
    eigenvalues: Any
    eigenvectors: Any
    perpendicular_kinetic_difference: Any
    outgoing_log_derivative_wave: Any
    lower_block_factors: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScipyOperatorData(DiagonalOperatorData):
    potential_pairs: Any
    channel_idx: Any
    shape: tuple[int, int, int] = field(metadata={"static": True})


def build_scipy_operator_data[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    *,
    n_channels: int | None = None,
) -> ScipyOperatorData:
    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)
    perpendicular_kinetic_difference = get_perpendicular_kinetic_difference(
        condition.incident_k,
        metadata_x01,
    ).ravel()

    channel_idx = np.argsort(perpendicular_kinetic_difference)
    if n_channels is not None:
        channel_idx = channel_idx[:n_channels]

    if perpendicular_kinetic_difference[channel_idx[-1]] < condition.incident_energy:
        first_non_propagating = 1 + np.searchsorted(
            np.sort(perpendicular_kinetic_difference),
            condition.incident_energy,
        )
        warnings.warn(
            "The highest kinetic energy channel is below the incident energy. "
            f"This may lead to poor convergence."
            f"Consider increasing n_channels to {first_non_propagating}.",
            UserWarning,
            stacklevel=5,
        )
    perpendicular_kinetic_difference = perpendicular_kinetic_difference[channel_idx]

    potential_values = potential_as_array(condition.potential)
    specular_potential = potential_values[0, 0].copy()
    # We will treat the specular potential separately in the preconditioner
    potential_values[0, 0] = 0.0

    nkx, nky, _nz = potential_values.shape
    # The (ikx, iky) of each channel
    cx, cy = np.unravel_index(channel_idx, (nkx, nky))
    # Pairwise differences of channel indices
    diff_x = (cx[:, np.newaxis] - cx[np.newaxis, :]) % nkx
    diff_y = (cy[:, np.newaxis] - cy[np.newaxis, :]) % nky
    potential_pairs = potential_values[diff_x, diff_y, :]

    parallel_kinetic_energy = -get_parallel_kinetic_energy(metadata_z)
    eigenvalues, eigenvectors = _solve_specular_hamiltonian(
        specular_potential=specular_potential,
        parallel_kinetic_energy=parallel_kinetic_energy,
    )

    lower_block_factors = _build_lower_block_factors(
        channel_energy=jnp.asarray(perpendicular_kinetic_difference),
        eigenvalues=jnp.asarray(eigenvalues),
        eigenvectors=jnp.asarray(eigenvectors),
    )

    outgoing_log_derivative_wave = get_outgoing_log_derivative_wave(
        metadata_z,
        perpendicular_kinetic_difference,
    )

    return ScipyOperatorData(
        potential_pairs=jnp.asarray(potential_pairs),
        eigenvalues=jnp.asarray(eigenvalues),
        eigenvectors=jnp.asarray(eigenvectors),
        perpendicular_kinetic_difference=jnp.asarray(perpendicular_kinetic_difference),
        lower_block_factors=jnp.asarray(lower_block_factors),
        outgoing_log_derivative_wave=jnp.asarray(outgoing_log_derivative_wave),
        channel_idx=jnp.asarray(channel_idx),
        shape=condition.metadata.shape,  # ty: ignore[invalid-argument-type]
    )


@jax.jit
def apply_inverse_lower_op(
    flat_state: Array,
    operator_data: ScipyOperatorData,
) -> Array:
    nkx, nky, nz = operator_data.shape
    out = flat_state.reshape((nkx * nky, nz))

    lower = apply_inverse_lower_block(
        out[operator_data.channel_idx],
        operator_data,
    )
    return out.at[operator_data.channel_idx].set(lower, unique_indices=True).ravel()


@jax.jit
def apply_lower_op(
    flat_state: Array,
    operator_data: ScipyOperatorData,
) -> Array:
    nkx, nky, nz = operator_data.shape
    out = flat_state.reshape((nkx * nky, nz))

    lower = apply_lower_block(
        out[operator_data.channel_idx],
        operator_data,
    )
    return out.at[operator_data.channel_idx].set(lower, unique_indices=True).ravel()


@jax.jit
def apply_upper_op(
    flat_state: Array,
    operator_data: ScipyOperatorData,
) -> Array:
    nkx, nky, nz = operator_data.shape
    out = flat_state.reshape((nkx * nky, nz))

    active_state = flat_state.reshape((nkx * nky, nz))[operator_data.channel_idx]
    upper = _apply_upper_block(active_state, operator_data)

    return out.at[operator_data.channel_idx].set(upper, unique_indices=True).ravel()


def build_jax_operators[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    *,
    n_channels: int | None = None,
) -> tuple[
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
]:
    """Build pure JAX linear operator functions for GPU-accelerated solvers."""
    operator_data = build_scipy_operator_data(
        condition,
        n_channels=n_channels,
    )

    # Bind operator_data via simple lambdas
    def inverse_lower(state):
        return apply_inverse_lower_op(state, operator_data)

    def lower(state):
        return apply_lower_op(state, operator_data)

    def upper(state):
        return apply_upper_op(state, operator_data)

    return inverse_lower, lower, upper


def run_multiscat_scipy_jax[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    # We split our hamiltonian into a lower block diagonal part L,
    # and an upper part U which contains the off-diagonal coupling between channels.
    inverse_lower, _lower, upper = build_jax_operators(
        condition,
        n_channels=config.n_channels,
    )
    target_state = (
        get_target_state(condition.metadata, condition.incident_k)
        .with_basis(
            close_coupling_basis(condition.metadata),
        )
        .raw_data.ravel()
    )
    initial_state = condition.initial_state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.ravel()

    solution = run_gauss_seidel_gradient_decent(  # cspell: disable-line
        target_state=target_state,
        initial_state=initial_state,
        inverse_lower=inverse_lower,
        upper=upper,
        config=config,
    )

    return solution.reshape(condition.metadata.shape)  # ty:ignore[invalid-return-type]


def get_scattering_state_scipy_jax[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    """Run Multiscat through the scipy GMRES solver."""
    inverse_lower, lower, upper = build_jax_operators(
        condition,
        n_channels=config.n_channels,
    )
    target_state = (
        get_target_state(condition.metadata, condition.incident_k)
        .with_basis(
            close_coupling_basis(condition.metadata),
        )
        .raw_data.ravel()
    )
    initial_state = condition.initial_state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.ravel()

    solution = run_gauss_seidel_gradient_decent(  # cspell: disable-line
        target_state=target_state,
        initial_state=initial_state,
        inverse_lower=inverse_lower,
        upper=upper,
        config=config,
        lower=lower,
    )

    return solution.reshape(condition.metadata.shape)  # ty:ignore[invalid-return-type]


def get_preconditioned_scattering_state_scipy_jax[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    """Run Multiscat through the scipy GMRES solver."""
    inverse_lower, _lower, upper = build_jax_operators(
        condition,
        n_channels=config.n_channels,
    )
    target_state = (
        get_target_state(condition.metadata, condition.incident_k)
        .with_basis(
            close_coupling_basis(condition.metadata),
        )
        .raw_data.ravel()
    )
    initial_state = condition.initial_state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.ravel()

    solution = run_gauss_seidel_gradient_decent(  # cspell: disable-line
        target_state=target_state,
        initial_state=initial_state,
        inverse_lower=inverse_lower,
        upper=upper,
        config=config,
    )

    return solution.reshape(condition.metadata.shape)  # ty:ignore[invalid-return-type]
