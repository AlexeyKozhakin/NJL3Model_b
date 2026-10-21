"""njl3.hom: homogeneous potential for arbitrary twist φ and coupling g."""

import math

import mpmath
import numpy as np
import pytest

from njl3 import b0
from njl3.hom import formulas as F
from njl3.hom import M_vacuum, find_minimum, isolate_roots, phase_at

PHIS = [0.07, 0.2, 0.25, 0.4, 0.5]


def test_phi_zero_is_b0():
    for mu, L in ((2.0, 1.0), (3.15, 3.5), (5.0, 8.0)):
        assert phase_at(mu, L, -1.0, 0.0) == b0.phase_at(mu, L, -1.0)
        assert F.gap_G(0.7, mu, L, -1.0, 0.0) == b0.gap_G(0.7, mu, L, -1.0)


@pytest.mark.parametrize('mu,L', [(2.0, 1.0), (3.15, 3.5), (3.5, 6.0)])
def test_small_phi_continuous_to_b0(mu, L):
    for M in (0.05, 0.8, 2.5):
        assert F.gap_G(M, mu, L, -1.0, 1e-9) == pytest.approx(b0.gap_G(M, mu, L, -1.0), abs=1e-7)
    assert F.gap_G(0.8, mu, L, -1.0, 1e-3) == pytest.approx(b0.gap_G(0.8, mu, L, -1.0), abs=1e-3)


def test_phi_symmetries():
    for phi in (0.1, 0.3):
        for M in (0.2, 1.5):
            a = F.gap_G(M, 3.0, 2.5, -1.0, phi)
            assert F.gap_G(M, 3.0, 2.5, -1.0, 1.0 - phi) == pytest.approx(a, rel=1e-14)
            assert F.gap_G(M, 3.0, 2.5, -1.0, phi + 2.0) == pytest.approx(a, rel=1e-12)


@pytest.mark.parametrize('phi', PHIS)
def test_lnD_against_mode_sum(phi):
    """d/dx ln D(x) = -1 + Σ_n 2x / ((2π(n+φ))² + x²) — the vacuum log is the twisted mode sum."""
    for x in (0.3, 2.0, 7.0):
        h = 1e-5
        lhs = (float(F._lnD(x + h, phi)) - float(F._lnD(x - h, phi))) / (2 * h)
        s = mpmath.nsum(lambda n: 2 * x / ((2 * mpmath.pi * (n + phi)) ** 2 + x * x), [-mpmath.inf, mpmath.inf])
        assert lhs == pytest.approx(-1.0 + float(s), abs=1e-8)


@pytest.mark.parametrize('phi', PHIS)
def test_gap_is_derivative_of_closed_potential(phi):
    """M G(M) = dOmega/dM with Omega from the complex polylogarithms."""
    for mu, L, g in ((2.5, 1.5, -1.0), (3.2, 4.0, -1.0), (3.0, 2.0, 1.0)):
        for M in (0.3, 1.1, 2.9, 3.6):
            h = 1e-5
            d = (F.omega_rel_closed(M + h, mu, L, g, phi) - F.omega_rel_closed(M - h, mu, L, g, phi)) / (2 * h)
            assert d == pytest.approx(M * F.gap_G(M, mu, L, g, phi), abs=2e-7)


@pytest.mark.parametrize('phi', [0.1, 0.3, 0.5])
def test_quadrature_matches_closed_form(phi):
    for mu, L in ((3.0, 2.0), (3.3, 6.0)):
        for M in (0.1, 0.45):
            assert F.omega_rel(M, mu, L, -1.0, phi) == pytest.approx(
                F.omega_rel_closed(M, mu, L, -1.0, phi), abs=1e-12)


def test_large_L_limit_independent_of_phi():
    """L -> ∞: G -> 1/g + max(M, mu)/π for any φ."""
    for phi in (0.0, 0.2, 0.5):
        for M in (1.0, 2.0, 4.0):
            assert F.gap_G(M, 3.0, 400.0, -1.0, phi) == pytest.approx(-1.0 + max(M, 3.0) / math.pi, abs=1e-2)


@pytest.mark.parametrize('phi', PHIS)
def test_splitting_monotone(phi):
    L, mu = 2.0, 3.5
    M = np.linspace(0.0, F.S_max(mu, L, phi), 400)
    A = [F.part_A(x, L, phi) for x in M]
    B = [F.part_B(x, mu, L, phi) for x in M]
    assert np.all(np.diff(A) > 0) and np.all(np.diff(B) <= 1e-15)
    for x, a, b in zip(M[1:], A[1:], B[1:]):
        assert -1.0 + a + b == pytest.approx(F.gap_G(x, mu, L, -1.0, phi), abs=1e-12)


@pytest.mark.parametrize('phi', PHIS)
@pytest.mark.parametrize('g', [-1.0, 1.0])
def test_certified_minimum_vs_dense_scan(phi, g):
    rng = np.random.default_rng(int(phi * 100) + (7 if g > 0 else 0))
    for _ in range(12):
        mu, L = rng.uniform(0.2, 6.0), rng.uniform(0.3, 8.0)
        M_star, Om, _ = find_minimum(mu, L, g, phi)
        Ms = np.concatenate([np.linspace(1e-4, 8.0, 3000), np.geomspace(1e-6, 1e-1, 200)])
        Om_scan = min(min(F.omega_rel(m, mu, L, g, phi) for m in Ms[::10]), 0.0)
        assert Om <= Om_scan + 1e-9
        for r, kind in isolate_roots(mu, L, g, phi):
            if kind != 'touch':
                assert abs(F.gap_G(r, mu, L, g, phi)) < 1e-9


def test_vacuum_restoration_threshold():
    """mu = 0: G(0) = 1/g + ln(4 sin² πφ)/(πL) > 0 at small L iff 4 sin² πφ > 1, i.e. φ > 1/6."""
    assert M_vacuum(0.2, -1.0, 0.3) == 0.0
    assert M_vacuum(0.2, -1.0, 0.1) > 0.0
    assert M_vacuum(50.0, -1.0, 0.3) == pytest.approx(math.pi, rel=1e-6)
