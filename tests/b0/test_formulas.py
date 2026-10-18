"""Formulas at b = 0: mpmath reference, cancellation of logarithms, derivatives, monotone splitting."""

import math

import numpy as np
import pytest

from njl3.b0 import formulas as F
from njl3.b0 import reference as R

G_ = -1.0
RNG = np.random.default_rng(2026)


def _random_points(n):
    for _ in range(n):
        mu = float(RNG.uniform(0.2, 7.0))
        L = float(np.exp(RNG.uniform(math.log(0.05), math.log(10.0))))
        yield mu, L


def test_G_matches_mpmath():
    worst = 0.0
    for mu, L in _random_points(60):
        for M in [1e-10, 1e-4, 0.3 * mu, 0.9 * mu, mu + 1.0]:
            worst = max(worst, abs(F.gap_G(M, mu, L, G_) - float(R.G(M, mu, L, G_))))
    assert worst < 1e-12


def test_G0_matches_mpmath():
    for mu, L in _random_points(60):
        assert F.gap_G0(mu, L, G_) == pytest.approx(float(R.G0(mu, L, G_)), abs=1e-13)


def test_log_cancellation_G_tends_to_G0():
    """G(M) = G0 + O(M²): the logarithms of Omega_L and of mode n = 0 cancel."""
    for mu, L in _random_points(30):
        assert F.gap_G(1e-9, mu, L, G_) == pytest.approx(F.gap_G0(mu, L, G_), abs=1e-12)


def test_dG_matches_finite_differences():
    for mu, L in _random_points(30):
        for M in [0.2 * mu, 0.6 * mu, mu + 0.5]:
            if any(abs(M - S) < 1e-3 for S in F.thresholds(mu, L)):
                continue
            h = 1e-6 * max(M, 1.0)
            fd = (F.gap_G(M + h, mu, L, G_) - F.gap_G(M - h, mu, L, G_)) / (2 * h)
            assert F.gap_dG_vec(np.array([M]), mu, L)[0] == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_dG_vanishes_at_zero():
    for mu, L in _random_points(10):
        assert abs(F.gap_dG_vec(np.array([1e-9]), mu, L)[0]) < 1e-7


def test_G2_is_second_order_coefficient():
    for mu, L in _random_points(30):
        if min(abs(mu - k) for k, _ in F.modes(mu, L)[1:] or [(1e9, 0)]) < 1e-2:
            continue
        M = 1e-3
        est = (F.gap_G(M, mu, L, G_) - F.gap_G0(mu, L, G_)) / M**2
        assert est == pytest.approx(F.G2(mu, L), rel=1e-3, abs=1e-6)


def test_monotone_split():
    """G = 1/g + A + B with A increasing and B decreasing on [0, mu] (basis of the certified search)."""
    for mu, L in _random_points(30):
        M = np.linspace(1e-6, mu, 1500)
        A = np.array([F.part_A(m, L) for m in M])
        B = np.array([F.part_B(m, mu, L) for m in M])
        assert np.all(np.diff(A) >= -1e-15)
        assert np.all(np.diff(B) <= 1e-15)
        assert np.max(np.abs(F.gap_G_vec(M, mu, L, G_) - (1.0 / G_ + A + B))) < 1e-12


@pytest.mark.parametrize('M', [1e-12, 1e-6, 1e-3, 0.1, 0.45])
def test_omega_rel_small_M_matches_mpmath(M):
    for mu, L in _random_points(15):
        ref = float(R.omega_rel(M, mu, L, G_))
        assert F.omega_rel(M, mu, L, G_) == pytest.approx(ref, rel=1e-10, abs=1e-15)


def test_omega_rel_closed_matches_mpmath_for_large_M():
    for mu, L in _random_points(20):
        for M in [0.6, 0.9 * mu + 0.6, mu + 1.5]:
            ref = float(R.omega_rel(M, mu, L, G_))
            assert F.omega_rel(M, mu, L, G_) == pytest.approx(ref, abs=1e-11)


def test_closed_omega_fails_at_small_M():
    """Documents why the stable form is needed: the closed form loses precision."""
    mu, L, M = 3.2, 1.0, 1e-8
    ref = float(R.omega_rel(M, mu, L, G_))
    assert abs(F.omega_rel_closed(M, mu, L, G_) - ref) > 1e3 * abs(F.omega_rel(M, mu, L, G_) - ref)


def test_vacuum_term_closed_form():
    for M in [0.1, 1.0, 3.0]:
        assert F.dU(M) == pytest.approx(M**3 / (3 * math.pi))


def test_omega_L_at_zero_is_zeta3():
    for L in [0.3, 1.0, 4.0]:
        assert F.omega_L(0.0, L) == pytest.approx(2 * 1.2020569031595942 / (math.pi * L**3), rel=1e-14)
