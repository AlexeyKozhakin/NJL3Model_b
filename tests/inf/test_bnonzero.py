"""L -> infinity, b ≠ 0: the 1D reduction of W against the 2D integral, special values, and njl3.full at large L."""

import math

import pytest
from scipy.integrate import dblquad

from njl3.inf.formulas import W, W_b0, omega_rel

G_ = -1.0


def W_2d(M, b, mu):
    """W with the p2 integral done numerically (independent of the closed form J)."""
    tot = 0.0
    for s in (1.0, -1.0):
        A = lambda p: math.sqrt(M * M + p * p) + s * b
        qb = lambda p: math.sqrt(max(mu * mu - A(p)**2, 0.0))
        f = lambda q, p: max(mu - math.hypot(A(p), q), 0.0)
        tot += 2.0 * dblquad(f, 0.0, b + mu + 1.0, lambda p: -qb(p), qb, epsabs=1e-10)[0]
    return -tot / (2.0 * math.pi)**2


def test_special_values():
    for mu in (2.0, 4.0):
        assert W(0.0, 0.0, mu) == pytest.approx(-mu**3 / (6 * math.pi), abs=1e-13)
        assert W(0.0, 1.3, mu) == pytest.approx(W(0.0, 0.0, mu), abs=1e-13)      # b only shifts the momentum at M = 0
        assert W(1.0, 0.0, mu) == pytest.approx(W_b0(1.0, mu), abs=1e-13)


@pytest.mark.slow
@pytest.mark.parametrize('M,b,mu', [(0.5, 1.0, 3.0), (1.2, 2.5, 3.0), (0.1, 3.0, 3.2)])
def test_1d_against_2d(M, b, mu):
    assert W(M, b, mu) == pytest.approx(W_2d(M, b, mu), abs=1e-9)


@pytest.mark.slow
def test_limit_of_compact_problem():
    from njl3.full.formulas import omega_rel as om_full
    for M, b, mu in [(0.5, 1.0, 3.0), (1.2, 2.5, 3.0), (2.0, 0.5, 4.0)]:
        assert om_full(M, b, mu, 160.0, G_) == pytest.approx(omega_rel(M, b, mu, G_), abs=3e-6)


def test_b_independence_below_fermi_edge():
    """Omega(M, b) = Omega(M, 0) for b ≤ mu - M (to machine precision; analytic proof open)."""
    import numpy as np
    from njl3.inf.formulas import omega_rel_b0
    rng = np.random.default_rng(1)
    for _ in range(100):
        mu = rng.uniform(0.5, 6.0); M = rng.uniform(0.0, mu); b = rng.uniform(0.0, mu - M)
        assert omega_rel(M, b, mu, G_) == pytest.approx(omega_rel_b0(M, mu, G_), abs=1e-12)


def test_b_never_lowers_omega():
    """Omega(M, b) ≥ Omega(M, 0) on a grid: the homogeneous minimum is global at L = infinity."""
    import numpy as np
    from njl3.inf.formulas import omega_rel_b0
    for mu in (2.5, 3.2, 4.5):
        for M in np.linspace(0.05, mu + 1.0, 15):
            o0 = omega_rel_b0(M, mu, G_)
            for b in np.linspace(0.0, mu + 2.0, 25):
                assert omega_rel(M, b, mu, G_) >= o0 - 1e-12
