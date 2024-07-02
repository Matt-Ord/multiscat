from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Hard limits on Multiscat parameters included everywhere at compile time
NZFIXED_MAX = 550  # max no of z points in fixed pot.
NVFCFIXED_MAX = 4096  # max no of fourier cmpts (fixed, from file)
NMAX = 1024  # diffraction channels
MMAX = 550  # z grid points
NFCX = 4096  # max number of potential fourier components per z slice


@dataclass
class Config:
    fourier_labels_file: str
    scatt_cond_file: str
    itest: int
    ipc: int
    eps: float
    nsf: int
    nfc: int
    zmin: float
    zmax: float
    vmin: float
    dmax: float
    imax: int
    a1: float
    a2: float
    b2: float
    nzfixed: int
    stepzmin: float
    stepzmax: float
    startindex: int
    endindex: int
    hemass: float

    def __post_init__(self):
        if self.nfc > NFCX:
            msg = (
                "ERROR: the .conf file needs more fourier components than"
                "allowed by the .inc file (nfc > nfcx)"
            )
            raise ValueError(msg)
        if self.nzfixed > NZFIXED_MAX:
            msg = (
                "ERROR: the .conf file needs more z points than"
                "allowed by the .inc file (nzfixed > NZFIXED_MAX)"
            )
            raise ValueError(msg)
        if self.nfc > NVFCFIXED_MAX:
            msg = (
                "ERROR: the .conf file needs more fourier components than"
                "allowed by the .inc file (nfc > NVFCFIXED_MAX)"
            )
            raise ValueError(msg)


def _parse_value(line: str) -> str:
    return line.split("!")[0].strip()


def read_config(file_path: Path) -> Config:
    with file_path.open("r") as file:
        lines = file.readlines()

    fourier_labels_file = _parse_value(lines[0])
    scatt_cond_file = _parse_value(lines[1])
    itest = int(_parse_value(lines[2]))
    ipc = int(_parse_value(lines[3]))
    ipc = min(max(ipc, 0), 1)
    nsf = int(_parse_value(lines[4]))
    nsf = min(max(nsf, 2), 5)
    eps = 0.5 * (10 ** (-nsf))
    nfc = int(_parse_value(lines[5]))
    zmin, zmax = map(float, _parse_value(lines[6]).split(","))
    vmin = float(_parse_value(lines[7]))
    dmax = float(_parse_value(lines[8]))
    imax = int(_parse_value(lines[9]))
    a1 = float(_parse_value(lines[10]))
    a2 = float(_parse_value(lines[11]))
    b2 = float(_parse_value(lines[12]))
    nzfixed = int(_parse_value(lines[13]))
    stepzmin = float(_parse_value(lines[14]))
    stepzmax = float(_parse_value(lines[15]))
    startindex = int(_parse_value(lines[16]))
    endindex = int(_parse_value(lines[17]))
    hemass = float(_parse_value(lines[18]))

    return Config(
        fourier_labels_file=fourier_labels_file,
        scatt_cond_file=scatt_cond_file,
        itest=itest,
        ipc=ipc,
        eps=eps,
        nsf=nsf,
        nfc=nfc,
        zmin=zmin,
        zmax=zmax,
        vmin=vmin,
        dmax=dmax,
        imax=imax,
        a1=a1,
        a2=a2,
        b2=b2,
        nzfixed=nzfixed,
        stepzmin=stepzmin,
        stepzmax=stepzmax,
        startindex=startindex,
        endindex=endindex,
        hemass=hemass,
    )


def print_config(config: Config) -> None:
    print(f"Fourier labels file = {config.fourier_labels_file}")
    print(f"Loading scattering conditions from {config.scatt_cond_file}")
    print(f"Output mode = {config.itest}")
    print(f"GMRES preconditioner flag = {config.ipc}")
    print(f"Convergence sig. figures = {config.nsf}")
    print(f"Total number of Fourier components to use = {config.nfc}")
    print(f"z integration range = ({config.zmin}, {config.zmax})")
    print(f"Potential well depth = {config.vmin}")
    print(f"Max energy of closed channels = {config.dmax}")
    print(f"Max index of channels = {config.imax}")
    print(f"Unit cell (A) = {config.a1} x {config.b2}")
    print(f"Number of z points in Fourier components (nzfixed) = {config.nzfixed}")
    print(
        f"Calculating for potential input files between {config.startindex}.in and {config.endindex}.in",
    )
    print(f"hemass = {config.hemass}")


def load_fixed_pot(config: Config, fourier_file: Path, rmlmda: float):
    # Initialize vfcfixed to zeros
    vfcfixed = np.zeros((config.nzfixed, config.nfc), dtype=np.complex128)

    # Open the data file and read in the fourier components
    with fourier_file.open("r") as file:
        # Discard the first 5 lines
        for _ in range(5):
            next(file)

        # Loop over fourier components
        for i in range(config.nfc):
            # Loop over z values in fourier components
            for j in range(config.nzfixed):
                line = file.readline().strip()
                real, imag = map(float, line.strip("()").split(","))
                vfcfixed[j, i] = complex(real, imag)

    # Scale to the program units
    vfcfixed *= rmlmda

    return vfcfixed


def get_mz(_config: Config, /) -> int:
    return 550


def lobatto(
    a: float,
    b: float,
    n: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Calculate an n-point Gauss-Lobatto quadrature rule in the interval a < x < b.

    Function localizes zeros of the derivative of the (n-1)th Legendre polynomial
    and calculates associated weights.
    """
    w = np.zeros(n)
    x = np.zeros(n)

    l = (n + 1) // 2
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)
    weight = (b - a) / (n * (n - 1))

    # Specific to Lobatto quadrature, first point is a
    x[0] = a
    w[0] = weight

    for k in range(1, l + 1):
        # As zeros are symmetric, there is only need to find positive ones
        # z is approximated zero of P[n-1] using Francesco Tricomi approximation
        # then accuracy of the zero is improved using Newton-Raphson
        z = np.cos(np.pi * (4 * k - 3) / (4 * n - 2))
        p1 = 1.0
        for _i in range(7):
            p2 = 0.0
            p1 = 1.0
            for j in range(1, n):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j

            # p2 gets overwritten to be P[n-1]'
            p2 = (n - 1) * (p2 - z * p1) / (1.0 - z * z)
            # p3 gets overwritten to be P[n-1]''
            p3 = (2.0 * z * p2 - n * (n - 1) * p1) / (1.0 - z * z)
            # Actual Newton-Raphson step
            z = z - p2 / p3

        # Write in shifted and scaled zeros and weights
        x[k - 1] = shift - scale * z
        x[n - k] = shift + scale * z

        # Write in weights (they are always positive)
        w[k - 1] = weight / (p1 * p1)
        w[n - k] = w[k - 1]

    # Specific to Lobatto quadrature, last point is b
    x[n - 1] = b
    w[n - 1] = weight

    return w, x


def tshape(
    a: float,
    b: float,
    m: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Calculate the kinetic energy matrix, T, in a normalized Lobatto shape function basis.

    Formula for this are taken from:
    "QUANTUM SCATTERING VIA THE LOG DERIVATIVE VERSION OF THE KOHN VARIATIONAL PRINCIPLE"
    D. E. Manolopoulos and R. E. Wyatt, Chem. Phys. Lett., 1988, 152,23
    """
    if m > MMAX:
        msg = "tshape 1"
        raise ValueError(msg)

    n = m + 1
    ww, xx = lobatto(a, b, n)

    # Scale weights
    ww = np.sqrt(ww)

    tt = np.zeros((n, n))
    for i in range(n):
        ff = 0.0
        for j in range(n):
            if j == i:
                continue

            gg = 1.0 / (xx[i] - xx[j])
            ff += gg

            for k in range(n):
                if k in (j, i):
                    continue
                gg = gg * (xx[j] - xx[k]) / (xx[i] - xx[k])

            tt[j, i] = ww[j] * gg / ww[i]

        tt[i, i] = ff

    w = np.zeros(m)
    x = np.zeros(m)
    t = np.zeros((m, m))

    for i in range(m):
        w[i] = ww[i + 1]
        x[i] = xx[i + 1]

        for j in range(i + 1):
            hh = 0.0
            for k in range(n):
                hh += tt[k, i + 1] * tt[k, j + 1]

            t[i, j] = hh
            t[j, i] = hh

    return w, x, t


def potent(
    stepzmin: float,
    stepzmax: float,
    nzfixed: int,
    vfcfixed: np.ndarray[Any, Any],
    nfc: int,
    m: int,
    z: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Interpolate the data from Matlab to the data points requested by the call to tshape/findmz.

    The whole of the requested vfc matrix is generated here. Also, have to set which is the zero Fourier component (nfc00).
    """
    # Initialize the vfc array with complex zeros
    vfc = np.zeros((m, nfc), dtype=np.complex128)

    # Generate vfc matrix by interpolation of vfcfixed
    for i in range(nfc):  # loop over Fourier components
        for j in range(m):  # loop over requested points
            # Locate what would be the index in the list of z points
            zindex = (z[j] - stepzmin) / (stepzmax - stepzmin) * (nzfixed - 1) + 1
            indexlow = int(zindex)  # truncate to integer

            # Pick out the value we are interested in
            if zindex == indexlow:
                # Have got exact value - no need to interpolate
                vfc[j, i] = vfcfixed[
                    indexlow - 1,
                    i,
                ]  # Adjusted for 0-based indexing in Python
            else:
                # Need to interpolate - interpolate real and imaginary parts separately
                atmp = vfcfixed[
                    indexlow - 1,
                    i,
                ]  # Adjusted for 0-based indexing in Python
                btmp = vfcfixed[indexlow, i]  # Adjusted for 0-based indexing in Python
                vrealtmp = atmp + (btmp - atmp) * (zindex - indexlow)

                # Store for use
                vfc[j, i] = vrealtmp

    return vfc


def basis(
    config: Config,
    a1: float,
    a2: float,
    b2: float,
    ei: float,
    theta: float,
    phi: float,
    rmlmda: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    """Calculate reciprocal lattice and the z-component of energy of outgoing wave for each channel.

    Parameters
    ----------
    a1 (float): Length of real space lattice vector along axis.
    a2 (float): X coordinate of other real space lattice vector.
    b2 (float): Y coordinate of other real space lattice vector.
    ei (float): Incident energy.
    theta (float): Theta angle in degrees.
    phi (float): Phi angle in degrees.
    rmlmda (float): Constant 2m/h^2.
    dmax (float): Maximum value of d.
    imax (int): Maximum index value.
    nmax (int): Maximum number of channels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        - d (np.ndarray): Array of d values.
        - ix (np.ndarray): Array of ix indices.
        - iy (np.ndarray): Array of iy indices.
        - n00 (int): Index of the zero Fourier component.

    """
    # Reciprocal lattice vectors
    ax = a1
    ay = 0.0
    bx = a2
    by = b2

    Auc = abs(ax * by)
    RecUnit = 2 * np.pi / Auc
    gax = by * RecUnit
    gay = -bx * RecUnit
    gbx = -ay * RecUnit
    gby = ax * RecUnit

    ered = rmlmda * ei  # ered is just k_i^2
    thetad = theta * np.pi / 180.0  # Convert theta to radians
    phid = phi * np.pi / 180.0  # Convert phi to radians

    pkx = np.sqrt(ered) * np.sin(thetad) * np.cos(phid)
    pky = np.sqrt(ered) * np.sin(thetad) * np.sin(phid)

    # Initialize arrays
    d = np.zeros(NMAX, dtype=np.float64)
    ix = np.zeros(NMAX, dtype=np.int32)
    iy = np.zeros(NMAX, dtype=np.int32)

    n = 0
    n00 = -1

    for i1 in range(-config.imax, config.imax + 1):
        for i2 in range(-config.imax, config.imax + 1):
            gx = gax * i1 + gbx * i2
            gy = gay * i1 + gby * i2
            eint = (pkx + gx) ** 2 + (pky + gy) ** 2
            di = eint - ered
            if di < config.dmax:
                n += 1
                if n <= NMAX:
                    ix[n - 1] = i1  # Adjust for 0-based indexing in Python
                    iy[n - 1] = i2
                    d[n - 1] = di
                    if i1 == 0 and i2 == 0:
                        n00 = n
                else:
                    msg = "ERROR: n too big! (basis)"
                    raise ValueError(msg)

    return d[:n], ix[:n], iy[:n], n00


def waves(
    config: Config,
    w0: float,
) -> tuple[complex, complex, complex]:
    """Construct and store the diagonal matrices a, b, and c that enter the log derivative Kohn expression for the S-matrix.

    Parameters
    ----------
    w0 (float): Input value w0.
    zmax (float): Maximum value of z.

    Returns
    -------
    Tuple[np.complex128, np.complex128, np.complex128]:
        - a (np.complex128): Diagonal matrix a.
        - b (np.complex128): Diagonal matrix b.
        - c (np.complex128): Diagonal matrix c.

    """
    dk = np.sqrt(abs(w0))

    if w0 < 0.0:
        theta = dk * config.zmax
        bcc = np.cos(2.0 * theta)
        bcs = np.sin(2.0 * theta)
        cc = np.cos(theta)
        cs = np.sin(theta)

        a = complex(bcc, -bcs)
        b = (dk**0.5) * complex(cc, -cs)
        c = complex(0.0, dk)
        return a, b, c

    a = complex(0.0, 0.0)
    b = complex(0.0, 0.0)
    c = complex(-dk, 0.0)

    return a, b, c


def precon(
    m: int,
    n: int,
    vfc: np.ndarray[Any, Any],
    nfc00: int,
    d: np.ndarray[Any, Any],
    t: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Construct the matrix factors required for the block lower triangular preconditioner used in GMRES.

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
    if m > t.shape[0]:
        msg = "precon 1"
        raise ValueError(msg)

    # Modify t to be H0 by adding the real part of vfc(k, nfc00) to its diagonal elements
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


def label_fourier_components(
    config: Config,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    """Label the Fourier components based on the provided file.

    The components are listed in 'fourier_labels_file'
    and appear in the same order as in the potential file.

    Parameters
    ----------
    fourier_labels_file (str): Path to the file containing Fourier labels.
    nfc (int): Number of Fourier components.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]:
        Tuple containing arrays of ivx and ivy components and the index of the zero Fourier component (nfc00).

    """
    ivx = np.zeros(config.nfc, dtype=int)
    ivy = np.zeros(config.nfc, dtype=int)
    nfc00 = -1

    with open(config.fourier_labels_file) as file:
        for i in range(config.nfc):
            line = file.readline()
            ivx[i], ivy[i] = map(int, line.split())
            if ivx[i] == 0 and ivy[i] == 0:
                nfc00 = i

    return ivx, ivy, nfc00


ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]
DoubleArray = np.ndarray[Any, np.dtype[np.complex128]]


def zrotg(a: complex, b: complex) -> tuple[complex, complex, complex, complex]:
    """Construct and perform a complex Givens rotation.

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


def gmres(
    m: int,
    ix: DoubleArray,
    iy: DoubleArray,
    n: int,
    n00: int,
    vfc: ComplexArray,
    ivx: DoubleArray,
    ivy: DoubleArray,
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
    """Complex Generalised Minimal Residual Algorithm (GMRES) subroutine."""
    # Setup constants
    l = 2000  # parameter (l = 2000)
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
    s = np.zeros(NMAX, dtype=np.complex128)

    # Initial step
    kount = 0
    xx[:, 0] = x[:]
    y = x.copy()
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
        kount += 1
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
        if kconv == 3 or xnorm == 0.0:
            break

    # Back substitution for x
    x[:] = xx[:, 0]
    for j in range(1, kk + 1):
        y[:] = xx[:, j]
        x += y * z[j - 1]

    # All done?
    if kconv < 3 and xnorm > 0.0:
        pass

    # Yes!
    return p


def upper(
    x: ComplexArray,
    m: int,
    ix: DoubleArray,
    iy: DoubleArray,
    n: int,
    vfc: ComplexArray,
    ivx: DoubleArray,
    ivy: DoubleArray,
    nfc: int,
) -> None:
    """Performs the block upper triangular matrix multiplication y = U*x, where A = L+U.
    The result y is overwritten on x on return.
    """
    for j in range(1, n + 1):
        x[:, j - 1] = 0.0j
        for i in range(j + 1, n + 1):
            for l in range(1, nfc + 1):
                if ix[i - 1] + ivx[l - 1] != ix[j - 1]:
                    continue
                if iy[i - 1] + ivy[l - 1] != iy[j - 1]:
                    continue
                x[:, j - 1] += vfc[:, l - 1] * x[:, i - 1]


def lower(
    x: ComplexArray,
    m: int,
    ix: DoubleArray,
    iy: DoubleArray,
    n: int,
    vfc: ComplexArray,
    ivx: DoubleArray,
    ivy: DoubleArray,
    nfc: int,
    c: ComplexArray,
    d: DoubleArray,
    e: DoubleArray,
    f: ComplexArray,
    t: ComplexArray,
) -> None:
    """Solves the block lower triangular linear equation L*y = x, where A = L+U.
    The result y is overwritten on x on return.
    """
    mmax = 200
    y = np.zeros(mmax, dtype=np.complex128)
    if m > mmax:
        msg = "m exceeds mmax in lower subroutine"
        raise ValueError(msg)

    for j in range(1, n + 1):
        for i in range(1, j):
            for l in range(1, nfc + 1):
                if ix[i - 1] + ivx[l - 1] != ix[j - 1]:
                    continue
                if iy[i - 1] + ivy[l - 1] != iy[j - 1]:
                    continue
                x[:, j - 1] -= vfc[:, l - 1] * x[:, i - 1]

        for k in range(1, m + 1):
            y[k - 1] = 0.0j
            for l in range(1, m + 1):
                y[k - 1] += x[l - 1, j - 1] * t[l - 1, k - 1]
            y[k - 1] /= d[j - 1] + e[k - 1]

        for k in range(1, m + 1):
            x[k - 1, j - 1]


def process_potentials(
    config: Config,
    rmlmda: float,
) -> None:
    np.zeros((NZFIXED_MAX, NVFCFIXED_MAX), dtype=np.complex128)

    ivx, ivy, nfc00 = label_fourier_components(config)

    for index in range(config.startindex, config.endindex + 1):
        fourierfile = Path(f"pot{index:05d}.in")
        outfile = Path(f"diffrac{index:05d}.out") if config.itest == 1 else None

        if outfile is not None:
            with outfile.open("a") as f:
                f.write(f"Diffraction intensities for potential: {fourierfile}\n")

        # Initialize the potential
        vfcfixed = load_fixed_pot(config, fourierfile, rmlmda)

        # TODO: Calculate scattering over the incident conditions required
        print("")
        print(f"Calculating scattering for potential: {fourierfile}")
        print("Energy / meV    Theta / deg    Phi / deg        I00         Sum")

        with Path(config.scatt_cond_file).open() as file:
            while True:
                line = file.readline()
                if not line:
                    print("-- End of scattering conditions file --")

                    break

                try:
                    ei, theta, phi = map(float, line.split())
                except ValueError:
                    print("#### ERROR: Invalid line found in input file  ####")
                    print(
                        "#### (Make sure scatCond does not contain empty lines) ####",
                    )
                    break

                m = get_mz(config)
                if outfile is not None:
                    with outfile.open("a") as f:
                        f.write(
                            f"Required number of z grid points, m = {m}\n",
                        )

                if m > MMAX:
                    msg = "Mz Too Big!"
                    raise ValueError(msg)

                w, z, t = tshape(config.zmin, config.zmax, m)

                # TODO: interpolate vfcs to required z positions
                vfc = potent(
                    config.stepzmin,
                    config.stepzmax,
                    config.nzfixed,
                    vfcfixed,
                    config.nfc,
                    m,
                    z,
                )

                # TODO: get reciprocal lattice points (also calculate how many channels are required for the calculation)
                d, ix, iy, n00 = basis(
                    config,
                    config.a1,
                    config.a2,
                    config.b2,
                    ei,
                    theta,
                    phi,
                    rmlmda,
                )
                if outfile is not None:
                    with outfile.open("a") as f:
                        f.write(
                            f"Number of diffraction channels, n ={d.size}\n",
                        )

                a = np.zeros(d.size, dtype=np.complex128)
                b = np.zeros(d.size, dtype=np.complex128)
                c = np.zeros(d.size, dtype=np.complex128)
                for i in range(d.size):
                    a[i], b[i], c[i] = waves(config, d[i])
                    b[i] = b[i] / w[m]
                    c[i] = c[i] / (w[m] ** 2)

                e, v, f = precon(m, d.size, vfc, nfc00, d, t)

                gmres(
                    m,
                    ix,
                    iy,
                    d.size,
                    n00,
                    vfc,
                    ivx,
                    ivy,
                    config.nfc,
                    a,
                    b,
                    c,
                    d,
                    e,
                    f,
                    v,
                    config.eps,
                    config.ipc,
                )

                # TODO: write outputs
                # output(ei, theta, phi, ix, iy, d.size, n00, d, p, config.itest)


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
    print_config(config)

    rmlmda = 2 * config.hemass / 4.18020
    process_potentials(config, rmlmda)


if __name__ == "__main__":
    main()
