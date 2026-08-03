from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
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

    Jax vs numpy: adapted for JAX's immutable
    arrays: the diagonal update uses `.at[...].add(...)` (functional,
    returns a new array) instead of an in-place `copy()` + mutation.
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
    """
    Calculate (H_0 + E_alpha)^(-1) C_i u_i.

    Adaptions for JAX:
    vectorized across all channels at once via broadcasting
    instead of looping over each channel
    individually. `denominator[:, j]` corresponds to the original loop's
    `channel_energy[j] + eigenvalues`, and `eigenvectors @ g` computes
    what the original did per-column via
    `einsum("ki,i->k", eigenvectors, g[:, j])`.
    """
    denominator = eigenvalues[:, None] + channel_energy[None, :]

    g = eigenvectors[-1, :, None] / denominator

    return eigenvectors @ g


@jax.jit
def _apply_upper_block(
    state_vector: jnp.ndarray[tuple[int, int], jnp.dtype[np.complex128]],
    operator_data: ScipyOperatorData,
) -> jnp.ndarray[tuple[int, int], jnp.dtype[np.complex128]]:
    """
    Apply the off-diagonal channel-coupling potential V_1(z)Q(x,y).

    This calculates the scattering of the wave between different (kx, ky)
    diffraction channels. Since V_1(z) is responsible for all coupling
    between diffraction channels, this step represents the physical momentum
    transfer parallel to the corrugated surface.

    Adaptions for JAX:
    vectorized across all channels at once instead of looping.
    The strictly-upper-triangular mask
    (`triu(..., k=1)`) selects exactly the same j > channel coupling
    terms the original loop restricts to via `channel + 1 :` slicing;
    the einsum then sums over those masked entries for every channel
    simultaneously.
    """
    n = state_vector.shape[0]

    upper = jnp.where(
        jnp.triu(jnp.ones((n, n), dtype=bool), k=1)[:, :, None],
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
    """
    Apply the inverse operator (H_0 - E_i + C_i u_i v_i^T)^(-1).

    This applies the uncoupled Green's function to the state vector
    1 / (H_0 - E_i + C_i u_i v_i^T) for the channel i, where C_i is the
    outgoing wave boundary correction at the final grid point.

    Step A (previously _apply_inverse_specular_operator_to_channel)

    Apply the inverse specular operator (H_0 - E_i)^(-1).

    This applies the uncoupled Green's function to the state vector
    1 / (H_0 - E_i) for the channel i.

    Step B (previously _apply_boundary_corrections_to_channel)

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

    where v_i^T (H_0 - E_alpha)^(-1) psi is channel_state[-1]
          (H_0 - E_alpha)^(-1) psi  is channel_state

    """
    # A.1 Transform to H_0 eigenbasis, where the uncoupled Green's function is diagonal.
    transformed = operator_data.eigenvectors.T @ channel_state

    # A.2 Apply uncoupled diagonal Green's function (H_0 - E_alpha)^(-1)
    denom_diag = (
        operator_data.perpendicular_kinetic_difference[channel]
        + operator_data.eigenvalues
    )
    transformed = transformed / denom_diag

    # A.3 Convert back to initial basis
    channel_state = operator_data.eigenvectors @ transformed

    # B Sherman-Morrison boundary condition update
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

    Adaptions for JAX:
      - The numpy version slices `potential_pairs[channel, :channel, :]`
        and `solved[:channel, :]` to restrict the coupling sum to
        already-solved channels; the JAX version instead masks the full
        array (`channel_indices < channel`), since `lax.fori_loop`
        requires statically-shaped arrays and can't dynamically slice a
        shrinking range.
      - `solved[channel, :] -=` (in-place) becomes `rhs = solved[channel]
        - coupling` (functional), since JAX arrays are immutable.
      - The in-place call to `_apply_inverse_diagonal_to_channel` becomes
        a functional call whose result is written back via
        `solved.at[channel].set(updated)`.
      - The Python `for channel in range(...)` loop becomes
        `lax.fori_loop`, required for this to be jit-compiled.

    """
    n = state_vector.shape[0]
    channel_indices = jnp.arange(n)

    def body(channel: int, solved: jnp.ndarray) -> jnp.ndarray:
        # Mask out channels j >= channel so array shapes remain strictly static
        mask = (channel_indices < channel)[:, None]  # Shape: (n, 1)
        masked_solved = solved * mask  # Zeroes out rows >= channel

        # Compute channel coupling term (V^lower_ij out_j)
        coupling = jnp.einsum(
            "ik,ik->k",
            operator_data.potential_pairs[channel],
            masked_solved,
        )
        # subtract V^lower_ij out_j
        rhs = solved[channel] - coupling

        # Apply D^(-1)_i
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

    Adaptions for JAX:
      - `einsum("cl,lk->ck", state_vector, eigenvectors)` is
        `state_vector @ eigenvectors`.
      - `einsum("cl,kl->ck", transformed_state, eigenvectors)` is
        `transformed_state @ eigenvectors.T` (contracting eigenvectors'
        second index, not its first — note the transpose here, unlike
        the per-channel version of this operator).

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

    This applies the outgoing-wave boundary correction C_i at the final
    grid point (nz - 1) using the Sherman-Morrison rule for all channels at once.

    Adaptions for JAX:
    `denom` and `fac` are computed identically.
    Only difference: returns the corrected state as a new
    array (`state_vector + ...`) instead of mutating `state_vector` in
    place (`state_vector += ...`), since JAX arrays are immutable.
    Confirm call sites use the returned value rather than assuming an
    in-place update.

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
    # Apply (H_0 - E_i)^(-1) to all channels at once in the H_0 eigenbasis.
    out = _apply_inverse_specular_operator(state_vector, operator_data)
    # Apply the Sherman-Morrison boundary correction for each channel to the out.
    return _add_boundary_corrections(out, operator_data)


@jax.jit
def apply_diagonal(
    state_vector: Array,
    operator_data: DiagonalOperatorData,
) -> Array:
    """
    Apply the uncoupled diagonal block operator (H_0 - E_i + C_i u_i v_i^T).

    This is the inverse operation of apply_inverse_diagonal

    Adaptions for JAX:
      - `state_vector @ eigenvectors` matches numpy's
        `einsum("cl,lk->ck", out, eigenvectors)`.
      - `transformed @ eigenvectors.T` matches numpy's
        `einsum("cl,kl->ck", transformed_state, eigenvectors)`.
      - Boundary values are captured before the transform (same as numpy's
        `boundary_values = out[:, -1].copy()`), then subtracted from index
        -1 afterward, functionally (`.at[:, -1].add(-boundary_update)`)
        instead of in-place.

    """
    # A. Capture input boundary values at z = -1 before applying (H_0 - E_i)
    # The rank-1 boundary update depends on the input state, not A @ state.
    boundary_values = state_vector[:, -1]

    # B. Apply A = (H_0 - E_i) in the H_0 eigenbasis.
    # B.1 Transform to H_0 eigenbasis (n_channels, nz) @ (nz, nz)
    transformed = state_vector @ operator_data.eigenvectors

    # B.2 Multiply by diagonal spectrum (H_0 - E_i)
    spectrum = (
        operator_data.perpendicular_kinetic_difference[:, None]
        + operator_data.eigenvalues[None, :]
    )
    transformed = transformed * spectrum

    # B.3 Transform back to original grid basis
    out = transformed @ operator_data.eigenvectors.T

    # C. Apply rank-1 boundary correction -C_i e_n e_n^T at final grid point
    boundary_update = operator_data.outgoing_log_derivative_wave * boundary_values
    return out.at[:, -1].add(-boundary_update)


@jax.jit
def apply_lower_block(
    state_vector: Array,
    operator_data: ScipyOperatorData,
) -> Array:
    """
    Apply the lower block diagonal operator (D + V^lower) to the state vector.

    This is the inverse operation of apply_inverse_lower_block.

    """
    n = state_vector.shape[0]

    # 1. Mask upper triangular and diagonal elements (j >= i) to keep only j < i
    lower_potentials = (
        jnp.tril(
            jnp.ones((n, n), dtype=bool),
            k=-1,
        )[:, :, None]
        * operator_data.potential_pairs
    )

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
    """
    Build the operator data required for the scattering matrix solve.

    Same as the original numpy implementation: selects and sorts channels
    by perpendicular kinetic energy, warns if the highest-energy channel
    is below the incident energy, separates the specular potential from
    the coupling potential, computes channel-pair potential couplings via
    circular index differences, solves the specular Hamiltonian for its
    eigenbasis, and builds the lower-block Sherman-Morrison factors and
    outgoing-wave log-derivative data.

    Adaptions for JAX:
      - `channel_energy`, `eigenvalues`, and `eigenvectors` are converted
        to JAX arrays (`jnp.asarray`) *before* being passed into
        `_build_lower_block_factors`, so that function operates on JAX
        arrays internally rather than numpy arrays.
      - All fields of the returned `ScipyOperatorData` are wrapped in
        `jnp.asarray(...)`, so the returned object holds JAX arrays
        throughout, rather than the numpy arrays the original version
        returns. This is required for downstream JAX-jitted consumers.
    """
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

    active_state = out[operator_data.channel_idx]
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
    def inverse_lower(state: jnp.ndarray) -> jnp.ndarray:
        return apply_inverse_lower_op(state, operator_data)

    def lower(state: jnp.ndarray) -> jnp.ndarray:
        return apply_lower_op(state, operator_data)

    def upper(state: jnp.ndarray) -> jnp.ndarray:
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
