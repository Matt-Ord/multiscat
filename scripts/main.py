from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from multiscat.config import (
    N_MAX,
    N_VFC_FIXED_MAX,
    N_Z_FIXED_MAX,
    Config,
    GMRESConfig,
    read_config,
)
from multiscat.fixed_potential import load_fixed_potential

if TYPE_CHECKING:
    from multiscat.basis import LobattoBasis, XYBasis
    from multiscat.fixed_potential import FixedPotential
    from multiscat.scattering_condition import ScatteringCondition


def get_lobatto_derivative_matrix(
    basis: LobattoBasis,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    return np.array(
        [p(basis.points) for p in basis.lobatto_points.derivative_polynomials],
    )


def get_t_matrix_discreet_variable_representation(
    basis: LobattoBasis,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Calculate the kinetic energy matrix, T, in a normalized Lobatto basis.

    Formula for this are taken from:
    "QUANTUM SCATTERING VIA THE LOG DERIVATIVE OF THE KOHN VARIATIONAL PRINCIPLE"
    D. E. Manolopoulos and R. E. Wyatt, Chem. Phys. Lett., 1988, 152,23
    """
    derivatives = get_lobatto_derivative_matrix(basis)

    # We make use of the formula
    # T_ij = \sum_k=0 M+1 \omega_k u_i'(R_k) u'_j(R_k)
    # to calculate the kinetic matrix T_ij
    return np.einsum("k,ik,jk->ij", basis.weights, derivatives, derivatives)  # type: ignore einsum


def get_k_matrix(
    potential: FixedPotential,
    condition: ScatteringCondition,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    r"""
    Calculate the K matrix.

    this is done according to the formula defined in.

    K_{ij} = \int_0^s dr u'(r)_i U'(r)_j + U_i(r)[V(r)-k^2]u_j(r)

    Returns
    -------
    np.ndarray[Any, np.dtype[np.complex128]]
        _description_

    """
    # T_ij = \int_0^s dr u'(r)_i U'(r)_j
    t_matrix = get_t_matrix_discreet_variable_representation(potential.z_basis)
    # TODO: scale by omega?  # noqa: FIX002
    # V_ij = \int_0^s dr u(r)_i U(r)_j V(r)
    v_matrix = np.diag(potential.data[:, 0])
    incoming_energy = condition.energy * potential.z_basis.weights
    return t_matrix + (v_matrix - incoming_energy)


def process_scattering_condition_non_reactive(
    potential: FixedPotential,
    condition: ScatteringCondition,
) -> None:
    k_matrix = get_k_matrix(potential, condition)
    k_00 = k_matrix[0, 0]
    k_01 = k_matrix[0, 1:]
    k_11 = k_matrix[1:, 1:]
    k_11_inv = np.linalg.inv(k_11)
    k_10 = k_matrix[1:, 0]

    y = k_00 - np.einsum("ij,jk,kl->il", k_01, k_11_inv, k_10)

    i_s = condition.momentum**-0.5 * np.exp(
        -1j * condition.momentum * potential.z_basis.delta_x,
    )
    o_s = condition.momentum**-0.5 * np.exp(
        1j * condition.momentum * potential.z_basis.delta_x,
    )
    i_s_deriv = -1j * condition.momentum * i_s
    o_s_deriv = 1j * condition.momentum * o_s
    return np.einsum(
        "ij,jk->ik",
        np.linalg.inv(y * o_s - o_s_deriv),
        (y * i_s - i_s_deriv),
    )


def get_scattering_energy(
    basis: XYBasis,
    condition: ScatteringCondition,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    Get the matrix of scattered energies d.

    Uses the formula
    d = |k in + k scatter|**2 - |k in|**2
    """
    pkx, pky, _pkz = condition.momentum
    abs_squared_k = np.linalg.norm(condition.momentum) ** 2
    # TODO: previously we worked in a sparse basis in d # noqa: FIX002
    # such that di < config.dmax
    # These channels will only have a small scattering contribution
    # It might be good to do this too!

    (kx, ky) = basis.k_points_stacked

    e_int = (pkx + kx) ** 2 + (pky + ky) ** 2
    # d is the difference between the initial energy
    # and the final energy
    return e_int - abs_squared_k


def _get_a(
    scattered_energy_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
    zmax: float,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    # a(r) = i(r)o(r)^-1
    # where o(r) = k^-1/2 h_0(1)(kr)
    # where i(r) = k^-1/2 h_0(2)(kr)
    # h_l(1/2)(x) = e^(+-ix) for l = 0
    # a(r) = e^(-2ik.x)

    return np.exp(-1j * 2.0 * zmax * np.lib.scimath.sqrt(scattered_energy_difference))


def _get_b(
    scattered_energy_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
    zmax: float,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    # b(r) = o(r)^-1
    # where o(r) = k^-1/2 h_0(1)(kr)
    # h_l(1/2)(x) = e^(+-ix) for l = 0
    dk = np.lib.scimath.sqrt(scattered_energy_difference)
    theta = dk * zmax
    return (np.abs(dk) ** 0.5) * np.exp(-1j * theta)


def _get_c(
    scattered_energy_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    # c(r) = o'(r)o(r)^-1
    # where o(r) = k^-1/2 h_0(1)(kr)
    # h_l(1/2)(x) = x(J_l(x) +/- iY_l(x))
    # c(r) = -i*k

    return -1j * np.lib.scimath.sqrt(scattered_energy_difference)


def get_abc_in_basis(
    xy_basis: XYBasis,
    lobatto_basis: LobattoBasis,
    condition: ScatteringCondition,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.complex128]],
    np.ndarray[tuple[int], np.dtype[np.complex128]],
    np.ndarray[tuple[int], np.dtype[np.complex128]],
]:
    d = get_scattering_energy(xy_basis, condition)

    a = np.zeros(xy_basis.n, dtype=np.complex128)
    b = np.zeros(xy_basis.n, dtype=np.complex128)

    # TODO: is is sqrt omega here, and is it n-1 elements... # noqa: FIX002
    # why omega mz - i think it is omega[-1], and we should have discarded z=0
    # state as the wavefunction is zero here!
    w = lobatto_basis.weights
    z_max = lobatto_basis.points[-1]

    mz = lobatto_basis.points.size - 1

    b = _get_a(d, z_max)
    b = _get_b(d, z_max) / w[mz]
    c = _get_c(d) / (w[mz] ** 2)
    return a, b, c


def precon(
    m: int,
    n: int,
    vfc: np.ndarray[Any, Any],
    d: np.ndarray[Any, Any],
    t: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Construct the matrix factors required for the preconditioner.

    This is required for use in GMRES.

    Parameters
    ----------
    m (int): Size parameter.
    n (int): Size parameter.
    vfc (np.ndarray): Fourier component matrix (complex128).
    nfc (int): Number of Fourier components.
    nfc00 (int): Zero Fourier component.
    d (np.ndarray): Array for d values.
    e (np.ndarray): Array for e values (output).
    f (np.ndarray): Array for f values (output).
    t (np.ndarray): Matrix from tshapes (modified in-place).

    Returns
    -------
    None

    """
    nfc00 = 0
    if m > t.shape[0]:
        msg = "precon 1"
        raise ValueError(msg)

    # Modify t to be H0 by adding the real part of vfc(k, nfc00)
    # to its diagonal elements
    h0 = t.copy()
    for k in range(m):
        h0[k, k] += np.real(vfc[k, nfc00])

    # Get eigenvalues and eigenvectors of h0 (symmetric real matrix)
    e, v = np.linalg.eigh(h0)

    f = np.zeros_like(v)

    for j in range(n):
        g = np.zeros(m, dtype=np.complex128)

        for k in range(m):
            g[k] = v[m - 1, k] / (d[j] + e[k])
            f[k, j] = 0.0

        for i in range(m):
            for k in range(m):
                f[k, j] += v[k, i] * g[i]
    return e, v, f


ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]
DoubleArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]


def zrotg(a: complex, b: complex) -> tuple[complex, complex, complex, complex]:
    """
    Construct and perform a complex Givens rotation.

    Parameters
    ----------
    a : complex number
        First input value.
    b : complex number
        Second input value.

    Returns
    -------
    r : complex number
        Value of r after rotation.
    z : complex number
        Value of z after rotation.
    c : complex number
        c after rotation.
    s : complex number
        s after rotation.

    """
    # Initialize variables
    scale = abs(a) + abs(b)
    if scale == 0.0:
        c = 1.0
        s = 0.0
        r = 0.0
        z = 0.0
    else:
        r = np.sqrt((abs(a) / scale) ** 2 + (abs(b) / scale) ** 2) * scale
        r = -r if a.real * b.real < 0 else r
        c = a / r
        s = b / r
        z = 1.0 if abs(a) >= abs(b) and c != 0.0 else (1.0 / c)

    return r, z, c, s


def gmres(  # noqa: C901, PLR0912, PLR0913, PLR0915
    m: int,
    ix: IntArray,
    iy: IntArray,
    n: int,
    n00: int,
    vfc: ComplexArray,
    nfc: int,
    a: ComplexArray,
    b: ComplexArray,
    c: ComplexArray,
    d: DoubleArray,
    e: DoubleArray,
    f: ComplexArray,
    t: ComplexArray,
    eps: float,
    ipc: int,
) -> ComplexArray:
    """Complex Generalized Minimal Residual Algorithm (GMRES) subroutine."""
    # Setup constants
    l = 2000  # parameter (l = 2000)  # noqa: E741
    h = np.zeros((l + 1, l + 1), dtype=np.complex128)
    g = np.zeros(l + 1, dtype=np.complex128)
    z = np.zeros(l + 1, dtype=np.complex128)
    co = np.zeros(l + 1, dtype=np.complex128)
    si = np.zeros(l + 1, dtype=np.complex128)
    temp = np.zeros(1, dtype=np.complex128)

    # Store x matrices in xx rather than write to disk
    xx = np.zeros((m * n, l + 1), dtype=np.complex128)

    # Setup for GMRES(l)
    mn = m * n
    x = np.zeros(mn, dtype=np.complex128)
    s = np.zeros(N_MAX, dtype=np.complex128)

    # Initial step
    kount = 0
    xx[:, 0] = x[:]
    y = x.copy()
    ivx, ivy = ix, iy
    upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
    x[:] = -x[:]
    x[m * n00] = b[n00] + x[m * n00]
    lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
    x[:] = x[:] - y[:]
    if ipc == 1:
        y[:] = x[:]
        upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
        lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
        x[:] = y[:] - x[:]

    xnorm = np.linalg.norm(x)
    g[0] = xnorm

    # Generic recursion (iteration)
    kconv = 0
    p = np.zeros(n, dtype=np.complex128)
    kk = 1
    for k in range(1, l + 1):
        kount += 1  # noqa: SIM113
        x /= xnorm
        xx[:, k] = x[:]
        y[:] = x[:]
        upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
        lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
        x[:] = y[:] + x[:]

        if ipc == 1:
            y[:] = x[:]
            upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
            lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
            x[:] = y[:] - x[:]

        y[:] = xx[:, 0]
        s[:] = y[m * np.arange(1, n + 1)]

        for j in range(1, k):
            y[:] = xx[:, j]
            h[j, k - 1] = np.sum(np.conj(y[:]) * x[:])
            x[:] -= y[:] * h[j, k - 1]

            if j < k:
                for i in range(1, n + 1):
                    s[i - 1] += y[m * i - 1] * z[j - 1]

        for i in range(1, n + 1):
            s[i - 1] = (0.0 + 2.0j) * b[i - 1] * s[i - 1]

        s[n00] = a[n00] + s[n00]
        xnorm = np.linalg.norm(x)
        h[k, k - 1] = xnorm

        for j in range(1, k):
            temp[:] = co[j - 1] * h[j - 1, k - 1] + np.conj(si[j - 1]) * h[j, k - 1]
            h[j, k - 1] = np.conj(co[j - 1]) * h[j, k - 1] - si[j - 1] * h[j - 1, k - 1]
            h[j - 1, k - 1] = temp[:]

        _, _, co[k - 1], si[k - 1] = zrotg(h[k - 1, k - 1], h[k, k - 1])
        g[k] = -si[k - 1] * g[k - 1]
        g[k - 1] = co[k - 1] * g[k - 1]

        z[:] = g[:]

        for j in range(k, 0, -1):
            z[j - 1] /= h[j - 1, j - 1]

            for i in range(1, j):
                z[i - 1] -= h[i - 1, j - 1] * z[j - 1]

        # Convergence test
        unit = 0.0
        diff = 0.0
        for j in range(1, n + 1):
            pj = np.conj(s[j - 1]) * s[j - 1]
            unit += pj
            diff = max(diff, abs(pj - p[j - 1]))
            p[j - 1] = pj

        diff = max(diff, abs(unit - 1.0))

        if diff < eps:
            kconv += 1
        else:
            kconv = 0

        kk = k
        if kconv == 3 or xnorm == 0.0:  # noqa: PLR2004
            break

    # Back substitution for x
    x[:] = xx[:, 0]
    for j in range(1, kk + 1):
        y[:] = xx[:, j]
        x += y * z[j - 1]

    # All done?
    if kconv < 3 and xnorm > 0.0:  # noqa: PLR2004
        pass

    # Yes!
    return p


def upper(  # noqa: PLR0913
    x: ComplexArray,
    m: int,  # noqa: ARG001
    ix: IntArray,
    iy: IntArray,
    n: int,
    vfc: ComplexArray,
    ivx: IntArray,
    ivy: IntArray,
    nfc: int,
) -> None:
    """
    Perform the block upper triangular matrix multiplication y = U*x.

    A = L+U.
    The result y is overwritten on x on return.
    """
    for j in range(1, n + 1):
        x[:, j - 1] = 0.0j
        for i in range(j + 1, n + 1):
            for l in range(1, nfc + 1):  # noqa: E741
                if ix[i - 1] + ivx[l - 1] != ix[j - 1]:
                    continue
                if iy[i - 1] + ivy[l - 1] != iy[j - 1]:
                    continue
                x[:, j - 1] += vfc[:, l - 1] * x[:, i - 1]


def lower(  # noqa: PLR0913
    x: ComplexArray,
    m: int,
    ix: IntArray,
    iy: IntArray,
    n: int,
    vfc: ComplexArray,
    ivx: IntArray,
    ivy: IntArray,
    nfc: int,
    c: ComplexArray,  # noqa: ARG001
    d: DoubleArray,
    e: DoubleArray,
    f: ComplexArray,  # noqa: ARG001
    t: ComplexArray,
) -> None:
    """
    Solves the block lower triangular linear equation L*y = x.

    A = L+U.
    The result y is overwritten on x on return.
    """
    mmax = 200
    y = np.zeros(mmax, dtype=np.complex128)
    if m > mmax:
        msg = "m exceeds mmax in lower subroutine"
        raise ValueError(msg)

    for j in range(1, n + 1):
        for i in range(1, j):
            for l in range(1, nfc + 1):  # noqa: E741
                if ix[i - 1] + ivx[l - 1] != ix[j - 1]:
                    continue
                if iy[i - 1] + ivy[l - 1] != iy[j - 1]:
                    continue
                x[:, j - 1] -= vfc[:, l - 1] * x[:, i - 1]

        for k in range(1, m + 1):
            y[k - 1] = 0.0j
            for l in range(1, m + 1):  # noqa: E741
                y[k - 1] += x[l - 1, j - 1] * t[l - 1, k - 1]
            y[k - 1] /= d[j - 1] + e[k - 1]

        for k in range(1, m + 1):
            x[k - 1, j - 1]


def process_scattering_condition(
    potential: FixedPotential,
    condition: ScatteringCondition,
    config: GMRESConfig,
) -> None:
    lobatto_basis = potential.z_basis
    mz = lobatto_basis.points.size - 1
    t_matrix = get_t_matrix_discreet_variable_representation(lobatto_basis)

    # get reciprocal lattice points
    # (also calculate how many channels are required for the calculation)
    d = get_scattering_energy(potential.xy_basis, condition)
    # TODO: write n diffaction channels output  # noqa: FIX002

    a, b, c = get_abc_in_basis(potential.xy_basis, lobatto_basis, condition)
    e, v, f = precon(
        mz,
        d.size,
        potential.data,
        d,
        t_matrix,
    )

    gmres(
        mz,
        potential.xy_basis.nk_points_stacked[0],
        potential.xy_basis.nk_points_stacked[1],
        d.size,
        0,
        potential.data,
        np.prod(potential.xy_basis.shape).item(),
        a,
        b,
        c,
        d,
        e,
        f,
        v,
        config.precision,
        config.preconditioner,
    )

    # TODO: write outputs  # noqa: FIX002


def process_potentials(
    config: Config,
) -> None:
    np.zeros((N_Z_FIXED_MAX, N_VFC_FIXED_MAX), dtype=np.complex128)

    for index in range(config.startindex, config.endindex + 1):
        fourier_file = Path(f"pot{index:05d}.in")
        out_file = Path(f"diffrac{index:05d}.out") if config.itest == 1 else None

        if out_file is not None:
            with out_file.open("a") as f:
                f.write(f"Diffraction intensities for potential: {fourier_file}\n")

        # TODO: Calculate scattering over the incident conditions  # noqa: FIX002
        print()
        print(f"Calculating scattering for potential: {fourier_file}")
        print("Energy / meV    Theta / deg    Phi / deg        I00         Sum")

        # Initialize the potential
        fixed_potential = load_fixed_potential(config, fourier_file)
        for condition in config.scattering_conditions:
            process_scattering_condition(
                fixed_potential,
                condition,
                config.gmres_config,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiscat: Close Coupled Scattering Program",
    )
    parser.add_argument("inputfile", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    inputfile = args.inputfile
    if not inputfile:
        print("Error: you must supply a configuration file to run Multiscat.")
        return

    print("\nMultiscat: Close Coupled Scattering Program")
    print("=============================================\n")
    print(f"Reading parameters from input file: {inputfile}\n")

    # Read parameters from config file
    config = read_config(Path(inputfile))

    # Print parameters
    print(config)

    process_potentials(config)


if __name__ == "__main__":
    main()
