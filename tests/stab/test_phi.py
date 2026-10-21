"""Second variation Phi(k, q1): closed forms vs the original formula, symmetries, sign in vacuum."""

import math
import warnings

import pytest

from njl3.b0 import phase_at
from njl3.stab import channels, phi, phi_direct_mat, phi_direct_vac, phi_mat, phi_vac
from njl3.stab.formulas import singular_q1

G = -1.0
POINTS = [(3.03, 2.0), (3.15, 10.0)]          # phases I0 and I4 (M* < mu, filled channels)


@pytest.fixture(scope='module', params=POINTS)
def point(request):
    mu, L = request.param
    ph, M = phase_at(mu, L, G)
    assert ph.startswith('I') and channels(M, mu, L)
    return mu, L, M


@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('q1', [0.3, 2.7, 5.0])
def test_matter_closed_form_vs_original(point, k, q1):
    mu, L, M = point
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = phi_direct_mat(k, q1, M, mu, L)
    assert phi_mat(k, q1, M, mu, L) == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize('k,q1', [(0, 0.3), (0, 8.0), (1, 2.7), (3, 0.3)])
def test_vacuum_feynman_vs_direct_sum(k, q1):
    M, L = 1.3, 3.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a, b = phi_direct_vac(k, q1, M, L, 300), phi_direct_vac(k, q1, M, L, 600)
    ref = 2.0 * b - a                          # the unsymmetrized sum converges as 1/n_cut
    assert phi_vac(k, q1, M, L) == pytest.approx(ref, rel=2e-6)


def test_vacuum_large_L_limit():
    """L -> ∞: Phi_vac -> N/(4π) ∫dx/a = N/(2π q) · arctan(q/(2M))."""
    M, q1 = 1.0, 2.0
    N = 2.0 * M * M + q1 * q1 / 2.0
    exact = N / (2.0 * math.pi * q1) * math.atan(q1 / (2.0 * M))
    assert phi_vac(0, q1, M, 200.0) == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize('k', [1, 2, 5])
def test_symmetric_in_k(point, k):
    mu, L, M = point
    assert phi(-k, 1.7, M, mu, L) == pytest.approx(phi(k, 1.7, M, mu, L), rel=1e-12)


def test_vacuum_positive():
    for M, L in ((3.14, 2.0), (1.0, 10.0)):
        for k in (0, 1, 4):
            for q1 in (0.01, 1.0, 10.0):
                assert phi_vac(k, q1, M, L) > 0.0


def test_log_singularity_at_2kF(point):
    """k = 0: Phi_mat -> -∞ logarithmically as q1 -> 2 P_n."""
    mu, L, M = point
    q_s = max(singular_q1(M, mu, L, 0))
    vals = [phi(0, q_s * (1 - d), M, mu, L) for d in (1e-3, 1e-6, 1e-9)]
    assert vals[0] > vals[1] > vals[2]
    steps = [vals[0] - vals[1], vals[1] - vals[2]]
    assert steps[1] == pytest.approx(steps[0], rel=0.05)   # equal steps per decade³: log
