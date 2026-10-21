"""
Tables of the universal function of Omega_L with a twist φ of the boundary condition:

    s_φ(x) = -∫_0^∞ Re ln(1 - e^{-sqrt(t² + x²) + 2πiφ}) dt = Σ_{m≥1} cos(2πmφ) (x/m) K1(m x).

φ = 0 is the table of njl3.full (s(x) with the x² ln x term split off). For φ ≠ 0 the function is
finite at x = 0, s_φ(0) = π² (φ² - φ + 1/6), but has a series of x^{2j} ln x terms and branch points at
x = ±2πiφ, so near 0 the pieces are graded geometrically, [2^{-k-1}, 2^{-k}] down to X_MIN (on each
piece s_φ is analytic and the Chebyshev series converges geometrically); below X_MIN s_φ = s_φ(0)
(error ~X_MIN² ln X_MIN < 1e-15).

    python -m njl3.inh.make_tables        → njl3/inh/data/s_table_phi<φ>.json for φ = 0.1 … 0.5
"""

import json
import os
import sys
from multiprocessing import Pool

import mpmath as mp
import numpy as np

PHIS = (0.1, 0.2, 0.3, 0.4, 0.5)
X_MIN = 2.0**-27
EDGES = [0.0] + [2.0**-k for k in range(27, 0, -1)] + [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
DEG = 24
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')


def table_path(phi):
    return os.path.join(OUT_DIR, f's_table_phi{phi:g}.json')


def s_mp(x, phi):
    """Reference value (mpmath, 25 digits): -Re ln(1 - e^{-r+iθ}) = -(1/2) ln(1 - 2 e^{-r} cos θ + e^{-2r}),
    smooth for θ ≠ 0."""
    mp.mp.dps = 25
    x = mp.mpf(x)
    c = mp.cos(2 * mp.pi * mp.mpf(phi))
    if x == 0 and phi == 0:
        return mp.pi**2 / 6

    def f(t):
        e = mp.exp(-mp.sqrt(t * t + x * x))
        return -mp.log1p(-2 * e * c + e * e) / 2
    return mp.quad(f, [0, 0.5, 2, 6, 15, 40, 100])


def piece(args):
    a, b, phi = args
    k = np.arange(DEG + 1)
    nodes = np.cos(np.pi * (k + 0.5) / (DEG + 1))
    xs = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    vals = np.array([float(s_mp(x, phi)) for x in xs])
    coef = np.polynomial.chebyshev.chebfit(nodes, vals, DEG)
    return [float(c) for c in coef]


def make(phi, pool):
    segs = [(a, b) for a, b in zip(EDGES[1:-1], EDGES[2:])]       # [X_MIN, 64]
    coefs = pool.map(piece, [(a, b, phi) for a, b in segs])
    table = dict(description=f's_phi(x) = -∫_0^∞ Re ln(1 - exp(-sqrt(t²+x²) + 2πiφ)) dt, φ = {phi}; '
                             f'x < x_min: s_phi(0)',
                 phi=phi, s0=float(s_mp(0, phi)), x_min=X_MIN, x_max=EDGES[-1],
                 pieces=[dict(a=a, b=b, coef=c) for (a, b), c in zip(segs, coefs)])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(table_path(phi), 'w') as f:
        json.dump(table, f)
    print(f'{table_path(phi)}: {len(segs)} pieces, degree {DEG}', flush=True)


def main():
    phis = [float(a) for a in sys.argv[1:]] or PHIS
    with Pool() as pool:
        for phi in phis:
            make(phi, pool)


if __name__ == '__main__':
    main()
