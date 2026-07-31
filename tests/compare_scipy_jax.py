import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import functools
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_quantum import operator

from multiscat import OptimizationConfig, ScatteringCondition
from multiscat.basis import (
    as_scattering_potential,
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
)
from multiscat.multiscat._gmres_jax import run_gauss_seidel_gradient_decent
from multiscat.multiscat._scipy_jax import (
    apply_inverse_lower_op,
    apply_upper_op,
    build_scipy_operator_data,
)
from multiscat.multiscat._util import get_target_state

HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3
UNIT_CELL = 2.84 * angstrom
Z_HEIGHT = 8 * angstrom

MORSE_PARAMETERS = operator.build.CorrugatedMorseParameters(
    depth=7.63 * electron_volt * 10**-3,
    height=(1.0 / 1.1) * angstrom,
    offset=3.0 * angstrom,
    beta=0.10,
)

GRID_SHAPE = (15, 15, 200)
N_CHANNELS = 80
ANGLES_DEG = [10.0, 20.0, 30.0, 40.0, 50.0]


def build_condition(theta_deg: float) -> ScatteringCondition:
    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL, 0, 0]),
            np.array([0, UNIT_CELL, 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        GRID_SHAPE,
    )
    potential = as_scattering_potential(
        operator.build.corrugated_morse_potential(metadata, MORSE_PARAMETERS),
        metadata,
    )
    return ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=HELIUM_ENERGY,
        theta=np.deg2rad(theta_deg),
        phi=np.deg2rad(0),
        potential=potential,
    )


def get_target_and_initial(
    condition: ScatteringCondition,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    target_state = (
        get_target_state(condition.metadata, condition.incident_k)
        .with_basis(close_coupling_basis(condition.metadata))
        .raw_data.ravel()
    )
    initial_state = condition.initial_state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.ravel()
    return jnp.asarray(target_state), jnp.asarray(initial_state)


def _force_ready(result) -> None:
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()


def time_solve(target_state, initial_state, inverse_lower, upper, config) -> float:
    t0 = time.perf_counter()
    solution = run_gauss_seidel_gradient_decent(
        target_state=target_state,
        initial_state=initial_state,
        inverse_lower=inverse_lower,
        upper=upper,
        config=config,
    )
    _force_ready(solution)
    return time.perf_counter() - t0


if __name__ == "__main__":
    config = OptimizationConfig(
        precision=1e-5, max_iterations=1000, n_channels=N_CHANNELS
    )

    # -----------------------------------------------------------------
    # Build ONE condition/operator_data just to construct STABLE
    # inverse_lower/upper closures. We'll reuse these exact objects
    # across every angle below, even though in reality each angle has
    # its own condition (and thus, strictly, its own channel selection/
    # potential coupling). This is intentionally an artificial test:
    # it isolates "does reusing the SAME A/M object change GMRES's
    # per-call cost", independent of whether the physics differs.
    # -----------------------------------------------------------------
    base_condition = build_condition(ANGLES_DEG[0])
    operator_data = build_scipy_operator_data(base_condition, n_channels=N_CHANNELS)

    stable_inverse_lower = functools.partial(
        apply_inverse_lower_op, operator_data=operator_data
    )
    stable_upper = functools.partial(apply_upper_op, operator_data=operator_data)

    print(
        "=== Test A: SAME inverse_lower/upper objects, only target/initial state vary ==="
    )
    print(
        f"id(stable_inverse_lower)={id(stable_inverse_lower)}  id(stable_upper)={id(stable_upper)}"
    )

    # warm up once
    warm_target, warm_initial = get_target_and_initial(base_condition)
    _force_ready(
        run_gauss_seidel_gradient_decent(
            target_state=warm_target,
            initial_state=warm_initial,
            inverse_lower=stable_inverse_lower,
            upper=stable_upper,
            config=config,
        ),
    )

    same_object_times = []
    for angle in ANGLES_DEG:
        condition = build_condition(angle)
        target_state, initial_state = get_target_and_initial(condition)

        # NOTE: target_state/initial_state now correspond to `condition` at
        # this angle, but inverse_lower/upper are STILL BUILT FROM
        # base_condition (theta=ANGLES_DEG[0]). This is deliberately
        # "physically wrong" — it answers a narrower question: does
        # object identity of A/M alone change GMRES's dispatch/retrace
        # cost, holding operator identity fixed? Correctness of the
        # resulting `solution` is NOT meaningful here and shouldn't be
        # compared against another backend.
        elapsed = time_solve(
            target_state, initial_state, stable_inverse_lower, stable_upper, config
        )
        same_object_times.append(elapsed)
        print(
            f"  theta={angle:5.1f}deg  time={elapsed:.4f}s  "
            f"id(inverse_lower)={id(stable_inverse_lower)} (unchanged)"
        )

    print(
        f"  mean={np.mean(same_object_times):.4f}s  std={np.std(same_object_times):.4f}s  "
        f"max/min ratio={max(same_object_times) / min(same_object_times):.2f}x"
    )

    print(
        "\n=== Test B: FRESH inverse_lower/upper objects per angle (current real behavior) ==="
    )
    fresh_object_times = []
    for angle in ANGLES_DEG:
        condition = build_condition(angle)
        target_state, initial_state = get_target_and_initial(condition)

        # Rebuild operator_data AND the closures fresh every time,
        # matching what get_scattering_state_scipy_jax actually does today.
        fresh_operator_data = build_scipy_operator_data(
            condition, n_channels=N_CHANNELS
        )
        fresh_inverse_lower = functools.partial(
            apply_inverse_lower_op, operator_data=fresh_operator_data
        )
        fresh_upper = functools.partial(
            apply_upper_op, operator_data=fresh_operator_data
        )

        elapsed = time_solve(
            target_state, initial_state, fresh_inverse_lower, fresh_upper, config
        )
        fresh_object_times.append(elapsed)
        print(
            f"  theta={angle:5.1f}deg  time={elapsed:.4f}s  "
            f"id(inverse_lower)={id(fresh_inverse_lower)} (new every time)"
        )

    print(
        f"  mean={np.mean(fresh_object_times):.4f}s  std={np.std(fresh_object_times):.4f}s  "
        f"max/min ratio={max(fresh_object_times) / min(fresh_object_times):.2f}x"
    )

    print("\n=== Verdict ===")
    same_cv = np.std(same_object_times) / np.mean(same_object_times)
    fresh_cv = np.std(fresh_object_times) / np.mean(fresh_object_times)
    print(f"  coefficient of variation, SAME objects:  {same_cv:.3f}")
    print(f"  coefficient of variation, FRESH objects: {fresh_cv:.3f}")
    if fresh_cv > same_cv * 1.5:
        print(
            "  => Supports closure-identity hypothesis: reusing A/M reduces call-to-call variance."
        )
    else:
        print(
            "  => Does NOT support closure-identity hypothesis: variance similar either way,"
        )
        print(
            "     likely a genuine convergence/iteration-count difference across angles instead."
        )
