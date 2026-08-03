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
    target_state_jax = jnp.asarray(target_state)

    # 1. Linear operator: (I + L^(-1) U)
    def _apply_linear_operator(state: Array) -> Array:
        return state + inverse_lower(upper(state))

    # 2. Optional Neumann preconditioner: (I - L^(-1) U)
    def _apply_neumann_preconditioner(state: Array) -> Array:
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
