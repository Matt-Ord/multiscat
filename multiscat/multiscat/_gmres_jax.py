from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
from jax import Array
from slate_core.util import timed
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from multiscat.config import OptimizationConfig


@timed
def run_gauss_seidel_gradient_decent(  # noqa: PLR0913
    target_state: Array | np.ndarray,
    inverse_lower: Callable[[Array], Array],
    upper: Callable[[Array], Array],
    lower: Callable[[Array], Array] | None = None,
    initial_state: Array | np.ndarray | None = None,
    *,
    config: OptimizationConfig,
) -> Array:
    """
    Use gradient decent to solve the linear system (I + L^{-1} U) psi = L^{-1} psi.

    This is equivalent to solving the original linear problem
    (L + U) psi = b, but is more efficient to solve since the operator (I + L^{-1} U)
    is closer to the identity.

    The input to this function is the target state vector b.
    The output of this function is L^{-1} psi, or simply psi if the
    lower operator is provided.

    Optionally, this probelm can be "double preconditioned"
    by applying a first-order Neumann series approximation to the operator
    (I - L^{-1} U) as a preconditioner to the GMRES solver. This should improve
    convergence if L^{-1} U is small.
    """
    target_state_jax = jnp.asarray(target_state)

    # 1. Linear operator: (I + L^(-1) U)
    def _apply_linear_operator(state: Array) -> Array:
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
        return state + inverse_lower(upper(state))

    # 2. Optional Neumann preconditioner: (I - L^(-1) U)
    def _apply_neumann_preconditioner(state: Array) -> Array:
        """
        Valuates a first-order Neumann series approximation: (I - L^{-1}U).

        If L^{-1}U is small, this is an approximate solution to the scattering problem.
        """
        return state - inverse_lower(upper(state))

    # 3. Handle initial state and preconditioning correctly
    b = inverse_lower(target_state_jax)

    if initial_state is None:
        x0 = None
    else:
        init_jax = jnp.asarray(initial_state)
        x0 = None if jnp.all(init_jax == 0) else inverse_lower(init_jax)

    restart = 50
    preconditioner = (
        _apply_neumann_preconditioner if config.use_neumann_preconditioner else None
    )

    pbar = tqdm(
        total=config.max_iterations,
        desc="JAX GMRES Iterations",
        leave=False,
    )

    try:
        solution, gmres_info = jax.scipy.sparse.linalg.gmres(
            A=_apply_linear_operator,
            b=b,
            x0=x0,
            tol=config.precision,
            restart=restart,
            maxiter=config.max_iterations,
            M=preconditioner,  # Pass the Neumann preconditioner
        )
    finally:
        pbar.close()

    if int(gmres_info) != 0:
        msg = f"JAX GMRES did not converge (info={gmres_info})."
        raise RuntimeError(msg)

    if lower is None:
        return solution

    return lower(solution)
