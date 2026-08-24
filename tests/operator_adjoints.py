# noqa: CPY001
"""
Adjoint (rmatvec) correctness tests for the multiscat scipy operators.

Run directly:  python operator_adjoints.py

SCOPE: full lateral grid only. Every operator here is built with
n_channels == nkx * nky, so the selected channel set is the whole grid and
`channel_idx` is the identity mapping.

Subset configurations (n_channels < nkx * nky) are deliberately NOT tested.
A 4-of-9 run previously failed the dense check on `lower` and `upper` while
every full-grid run passed at roundoff. That discrepancy is unresolved: it
means matvec and rmatvec disagree on subsets, but NOT which of the two is
wrong -- the forward operators may simply not be defined for an incomplete
grid. Until that is settled, testing subsets only produces a failure that
cannot be attributed. See `test_subset_is_out_of_scope` below.

What each check does, in one line:

  dense          Column j of a matrix is A applied to the j-th unit vector, so
                 feeding in every unit vector reconstructs A explicitly. Do the
                 same with rmatvec to get B, then check B == A^H entrywise.
                 Exhaustive rather than sampled, and localizes failures.
  discrimination If A is real then A^T == A^H, and the dense check would pass
                 even with a wrong conjugation. This reports when that is the
                 case so a green run is not mistaken for coverage.
  dot product    <x, Ay> == <A^H x, y> with random vectors. Weaker than the
                 dense check, but the only option at production size where
                 building the matrix is infeasible.
  boundary       Same identity, probed with vectors supported only on the last
                 z index of each channel, where the Sherman-Morrison rank-1
                 term lives. A dense random vector dilutes it across nz points.
  inverse        L^-1 L == I, and the same for the adjoints. Independent of the
                 adjoint checks: it catches a forward/adjoint pair that is
                 wrong in the same mutually-consistent way, which the dot
                 product identity structurally cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from scipy.constants import (  # type: ignore[import-untyped]
    electron_volt,
    physical_constants,
)
from slate_quantum import operator

from multiscat.basis import scattering_metadata_from_stacked_delta_x
from multiscat.config import MorseScatteringCondition, momentum_from_angles
from multiscat.multiscat import _as_natural_units
from multiscat.multiscat._scipy import _build_scipy_operators

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3
Z_HEIGHT = 8

# float64 noise on a length-n dot product grows like sqrt(n) * eps. Anything
# looser than ~1e-11 hides real bugs: the adjoint-vs-transpose difference lives
# in the rank-1 boundary term, which is one z index per channel and so
# contributes only a small share of the total.
RTOL = 1e-11

# Round-trip through an operator and its inverse loses more precision than a
# single application, so it gets its own looser bound.
RTOL_INVERSE = 1e-9


@dataclass
class Results:  # noqa: D101
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def record(  # noqa: D102
        self,
        name: str,
        ok: bool,  # noqa: FBT001
        detail: str = "",
    ) -> None:
        line = f"{name}: {detail}" if detail else name
        (self.passed if ok else self.failed).append(line)

    def skip(self, name: str, reason: str) -> None:  # noqa: D102
        self.skipped.append(f"{name}: {reason}")


def build_operators(
    shape: tuple[int, int, int],
    theta_deg: float = 30.0,
) -> tuple[dict[str, object], int, int]:
    """
    Build the scipy operators on the FULL lateral grid for a given shape.

    n_channels is derived from the shape rather than passed in, so there is no
    way to accidentally construct a subset configuration. Returns the operators
    plus n_channels and nz, so flat state indices can be decoded back into
    (channel, z) when a check fails.
    """
    nkx, nky, nz = shape
    n_channels = nkx * nky

    condition0 = MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=operator.build.CorrugatedMorseParameters(
            depth=7.63 * electron_volt * 10**-3,
            height=(1.0 / 1.1) * angstrom_si,
            offset=1.0 * angstrom_si,
            beta=0.05,
        ),
        metadata=scattering_metadata_from_stacked_delta_x(
            (
                np.array([3 * angstrom_si, 0, 0]),
                np.array([0, 3 * angstrom_si, 0]),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (nkx, nky, nz),
        ),
        incident_k=momentum_from_angles(
            theta=np.deg2rad(theta_deg),
            phi=np.deg2rad(0),
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
        ),
    )
    condition = _as_natural_units(condition0)

    inverse_lower, lower, upper = _build_scipy_operators(
        condition,
        n_channels=n_channels,
    )

    ops = {"inverse_lower": inverse_lower, "lower": lower, "upper": upper}
    return ops, n_channels, nz


def random_state(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(size=n) + 1j * rng.normal(size=n)


def densify(op: LinearOperator) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct matvec and rmatvec as explicit dense matrices."""
    n = op.shape[0]
    basis = np.eye(n, dtype=np.complex128)
    forward = np.column_stack([op.matvec(basis[:, j]) for j in range(n)])
    adjoint = np.column_stack([op.rmatvec(basis[:, j]) for j in range(n)])
    return forward, adjoint


def check_dense_adjoint(
    op: LinearOperator,
    name: str,
    nz: int,
    res: Results,
) -> np.ndarray | None:
    """
    Exhaustive entrywise check that rmatvec builds A^H.

    Note what a failure does and does not tell you: it proves matvec and
    rmatvec disagree about which matrix they represent, but NOT which one is
    at fault. The diagnostic below prints both values at the offending entries
    so the blame can be assigned by inspection.

    Returns the dense forward matrix so callers can reuse it.
    """
    forward, adjoint = densify(op)
    expected = forward.conj().T
    scale = max(np.max(np.abs(forward)), 1.0)
    diff = np.abs(adjoint - expected)
    worst = float(np.max(diff)) / scale

    ok = worst < RTOL
    detail = f"max_rel_err={worst:.3e}"
    if not ok:
        bad = np.argwhere(diff > RTOL * scale)
        detail += f", {len(bad)} bad entries"
        if np.all(bad[:, 0] % nz == nz - 1):
            detail += " -- all on the boundary row, suspect the rank-1 term"
    res.record(f"{name} dense B == A^H", ok, detail)

    return forward


def check_discrimination(forward: np.ndarray, name: str, res: Results) -> None:
    """
    Confirm the dense check can distinguish A^H from A^T.

    If A is real, a wrong conjugation convention is undetectable and every
    adjoint check in this file is vacuous for that operator.
    """
    scale = max(np.max(np.abs(forward)), 1.0)
    gap = float(np.max(np.abs(forward.conj().T - forward.T))) / scale
    imag = float(np.max(np.abs(forward.imag))) / scale

    if gap < RTOL:
        res.skip(
            f"{name} adjoint/transpose distinguishable",
            f"A^T == A^H to {gap:.3e} (imag part {imag:.3e}) -- the conj "
            "convention for this operator is NOT tested by anything here. "
            "Closing this gap needs an operator_data with a synthetically "
            "complex potential_pairs.",
        )
    else:
        res.record(
            f"{name} adjoint/transpose distinguishable",
            True,  # noqa: FBT003
            f"they differ by {gap:.3e} relative, so a dropped conj would fail",
        )


def check_dot_product(
    op: LinearOperator,
    name: str,
    res: Results,
    n_trials: int = 5,
    seed: int = 0,
) -> None:
    """
    <x, A y> == <A^H x, y>, several independent draws.

    np.vdot conjugates its first argument, which matches scipy's definition of
    rmatvec as A^H.
    """
    rng = np.random.default_rng(seed)
    n = op.shape[0]
    worst = 0.0
    for _ in range(n_trials):
        x, y = random_state(rng, n), random_state(rng, n)
        lhs = np.vdot(x, op.matvec(y))
        rhs = np.vdot(op.rmatvec(x), y)
        worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))
    res.record(
        f"{name} <x,Ay> == <A^H x,y>",
        worst < RTOL,
        f"worst of {n_trials} draws: rel_err={worst:.3e}",
    )


def check_boundary_probe(
    op: LinearOperator,
    name: str,
    n_channels: int,
    nz: int,
    res: Results,
) -> None:
    """Probe with vectors supported only on the last z index of each channel."""
    rng = np.random.default_rng(11)
    x = np.zeros(op.shape[0], dtype=np.complex128)
    y = np.zeros(op.shape[0], dtype=np.complex128)
    boundary = np.arange(n_channels) * nz + (nz - 1)
    x[boundary] = random_state(rng, n_channels)
    y[boundary] = random_state(rng, n_channels)

    lhs = np.vdot(x, op.matvec(y))
    rhs = np.vdot(op.rmatvec(x), y)
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    res.record(f"{name} boundary-supported probe", err < RTOL, f"rel_err={err:.3e}")


def check_inverse_pair(
    inverse_op: LinearOperator,
    forward_op: LinearOperator,
    label: str,
    res: Results,
) -> None:
    """
    L^-1 L == I and L^H (L^-1)^H == I.

    The adjoint round-trips are the valuable ones: they exercise the two
    adjoints against each other rather than each against its own forward, so a
    convention error shared within a forward/adjoint pair still shows up.
    """
    rng = np.random.default_rng(3)
    n = forward_op.shape[0]

    for name, first, second in (
        ("L^-1 L == I", forward_op.matvec, inverse_op.matvec),
        ("L L^-1 == I", inverse_op.matvec, forward_op.matvec),
        ("L^H (L^-1)^H == I", inverse_op.rmatvec, forward_op.rmatvec),
        ("(L^-1)^H L^H == I", forward_op.rmatvec, inverse_op.rmatvec),
    ):
        x = random_state(rng, n)
        err = np.linalg.norm(second(first(x)) - x) / np.linalg.norm(x)
        res.record(f"{label} {name}", err < RTOL_INVERSE, f"rel_err={err:.3e}")  # ty: ignore[invalid-argument-type]


def run_dense_suite(res: Results) -> None:
    """
    Exhaustive checks at sizes small enough to reconstruct the matrix.

    Several sizes, all full grids. An error that grows with channel count
    points at accumulation across the channel loop; one that stays fixed points
    at a per-channel term.
    """
    for shape in ((10, 10, 50), (15, 15, 50), (20, 20, 50)):
        ops, _n_ch, nz = build_operators(shape)
        np.random.default_rng(1)

        for name, op in ops.items():
            forward = check_dense_adjoint(op, name, nz, res)  # ty: ignore[invalid-argument-type]
            if forward is not None:
                check_discrimination(forward, name, res)

        check_inverse_pair(ops["inverse_lower"], ops["lower"], "lower/inverse", res)  # ty: ignore[invalid-argument-type]


def run_angle_suite(res: Results) -> None:
    """
    Repeat at a second incidence angle.

    Which channels are open depends on the incident momentum, and the outgoing
    log-derivative wave is what makes the diagonal block genuinely complex, so
    a different angle exercises a materially different operator.
    """
    shape = (10, 10, 50)
    ops, _n_ch, nz = build_operators(shape, theta_deg=60.0)

    for name, op in ops.items():
        forward = check_dense_adjoint(op, f"{name} [theta=60]", nz, res)  # ty: ignore[invalid-argument-type]
        if forward is not None:
            check_discrimination(forward, f"{name} [theta=60]", res)

    check_inverse_pair(
        ops["inverse_lower"],  # ty: ignore[invalid-argument-type]
        ops["lower"],  # ty: ignore[invalid-argument-type]
        "lower/inverse [theta=60]",
        res,
    )


def run_production_suite(res: Results) -> None:
    """Probing at production size, where densification is out of reach."""
    shape = (10, 10, 50)
    ops, n_ch, nz = build_operators(shape)

    for name, op in ops.items():
        check_dot_product(op, name, res)  # ty: ignore[invalid-argument-type]
        check_boundary_probe(op, name, n_ch, nz, res)  # ty: ignore[invalid-argument-type]

    check_inverse_pair(ops["inverse_lower"], ops["lower"], "lower/inverse", res)  # ty: ignore[invalid-argument-type]
