"""L -> infinity, b = 0: closed forms against direct integration and against the L -> infinity limit of njl3.b0."""

import math

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

from njl3.inf import M_vacuum, find_minimum_b0, gap_G_b0, omega_rel_b0
from njl3.inf.formulas import W_b0

G_ = -1.0


def W_direct(M, mu):
    """Defining 2D integral (polar coordinates), both branches."""
    if M >= mu:
        return 0.0
    P = math.sqrt(mu * mu - M * M)
    return -2.0 * quad(lambda p: p * (mu - math.sqrt(M * M + p * p)), 0.0, P)[0] / (2.0 * math.pi)


def W_direct_cartesian(M, mu):
    """The same in the original variables (p1, p2), as written in the model."""
    if M >= mu:
        return 0.0
    f = lambda p2, p1: max(mu - math.sqrt(M * M + p1 * p1 + p2 * p2), 0.0)
    P = math.sqrt(mu * mu - M * M)
    val = dblquad(f, -P, P, lambda p1: -math.sqrt(max(P * P - p1 * p1, 0.0)),
                  lambda p1: math.sqrt(max(P * P - p1 * p1, 0.0)), epsabs=1e-11)[0]
    return -2.0 * val / (2.0 * math.pi)**2


@pytest.mark.parametrize('M,mu', [(0.0, 2.0), (0.5, 2.0), (1.9, 2.0), (0.3, 4.0), (2.5, 4.0), (3.0, 2.0)])
def test_W_closed_form(M, mu):
    assert W_b0(M, mu) == pytest.approx(W_direct(M, mu), abs=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize('M,mu', [(0.5, 2.0), (1.2, 3.0)])
def test_W_cartesian(M, mu):
    assert W_b0(M, mu) == pytest.approx(W_direct_cartesian(M, mu), abs=1e-8)


def test_quadratic_below_mu_and_gap():
    for mu in (1.0, 2.5, math.pi, 4.0):
        for M in np.linspace(0.01, min(mu, 5.0), 9):
            assert omega_rel_b0(M, mu, G_) == pytest.approx(M * M * (1 / (2 * G_) + mu / (2 * math.pi)), abs=1e-13)
            h = 1e-6
            d = (omega_rel_b0(M + h, mu, G_) - omega_rel_b0(M - h, mu, G_)) / (2 * h)
            if abs(M - mu) > 2 * h:
                assert d / M == pytest.approx(gap_G_b0(M, mu, G_), abs=1e-7)


def test_phases():
    Mv = M_vacuum(G_)
    assert Mv == pytest.approx(math.pi)
    r = find_minimum_b0(2.0, G_)
    assert (r.phase, r.M_star) == ('V', pytest.approx(math.pi)) and r.omega_star < 0
    r = find_minimum_b0(3.5, G_)
    assert (r.phase, r.M_star, r.omega_star) == ('T', 0.0, 0.0)
    assert find_minimum_b0(math.pi, G_).degenerate
    # at mu = π the potential is flat on [0, π]
    for M in np.linspace(0, math.pi, 7):
        assert omega_rel_b0(M, math.pi, G_) == pytest.approx(0.0, abs=1e-12)


def test_limit_of_compact_problem():
    """njl3.b0 at large L tends to the L = infinity result."""
    from njl3.b0.formulas import omega_rel
    from njl3.b0.minimize import M_vacuum as M_vac_L
    assert M_vac_L(40.0, G_) == pytest.approx(math.pi, abs=1e-12)
    for L, tol in ((10.0, 2e-2), (40.0, 2e-3)):
        for mu, M in ((2.0, 1.0), (4.0, 2.0), (4.0, 3.5)):
            assert omega_rel(M, mu, L, G_) == pytest.approx(omega_rel_b0(M, mu, G_), abs=tol)
