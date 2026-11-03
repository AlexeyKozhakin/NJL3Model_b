"""Stability at T > 0 (njl3.stab.thermal): G_T by smearing, the smeared loop on a grid, the scan of the minimum."""

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from njl3.hom import formulas as H
from njl3.stab import formulas as S
from njl3.stab import thermal as TH

warnings.simplefilter('ignore')
PI = math.pi


def _lnf(x):
    return math.log1p(math.exp(-x)) if x > 0 else -x + math.log1p(math.exp(x))


def _omega_T_direct(M, mu, L, T, g=-1.0):
    """Omega_T(M, b = 0) by direct integration of the Fermi–Dirac logarithms (φ = 0) — independent of formula (A)."""
    vac = M * M / (2 * g) + M ** 3 / (3 * PI) + H.omega_L(M, L, 0.0)
    th = 0.0
    for n in range(-60, 61):
        m = math.hypot(M, 2 * PI * n / L)
        if m - mu > 45 * T:
            continue
        f = lambda p: -T * (_lnf((math.hypot(m, p) - mu) / T) + _lnf((math.hypot(m, p) + mu) / T))
        th += 2 * quad(f, 0, math.sqrt(max((mu + 45 * T) ** 2 - m * m, 0)), limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    return vac + th / (PI * L)


@pytest.mark.parametrize('mu,L,T', [(3.0, 1.0, 0.3), (0.5, 1.0, 0.2)])
def test_G_T_zero_is_curvature(mu, L, T):
    """G_T(0⁺) = Omega_T''(0) (2 Omega/M², linear extrapolation to M = 0)."""
    o0 = _omega_T_direct(0.0, mu, L, T)
    Ms = np.array([0.002, 0.001])
    r = 2 * (np.array([_omega_T_direct(m, mu, L, T) for m in Ms]) - o0) / Ms ** 2
    assert TH.gap_G_T(0.0, mu, L, -1.0, T) == pytest.approx(2 * r[1] - r[0], abs=2e-6)


def test_G_T_to_zero_T():
    assert TH.gap_G_T(0.0, 3.0, 1.0, -1.0, 1e-4) == pytest.approx(H.gap_G0(3.0, 1.0, -1.0, 0.0), abs=1e-6)
    assert TH.gap_G_T(1.0, 3.0, 2.0, -1.0, 1e-4) == pytest.approx(H.gap_G(1.0, 3.0, 2.0, -1.0, 0.0), abs=1e-6)


def test_G_T_vanishes_at_minimum():
    M, _, _ = TH.hom_T(3.3, 2.0, -1.0, 0.05)
    assert abs(TH.gap_G_T(M, 3.3, 2.0, -1.0, 0.05)) < 1e-5


def test_normalization_q_to_zero():
    """Phi_T(0, q1 -> 0) = ½ Omega_T''(M) = ½ M G_T'(M) at the minimum (direct Omega_T)."""
    mu, L, T = 2.8, 2.0, 0.2
    M, _, _ = TH.hom_T(mu, L, -1.0, T)
    h = 2e-3
    d2 = (_omega_T_direct(M + h, mu, L, T) - 2 * _omega_T_direct(M, mu, L, T) + _omega_T_direct(M - h, mu, L, T)) / h ** 2
    assert S.phi(0, 1e-3, M, mu, L, T) == pytest.approx(0.5 * d2, abs=2e-5)


@pytest.mark.parametrize('k,q1', [(0, 1.0), (1, 2.5), (0, 3.0)])
def test_smeared_grid_matches_exact(k, q1):
    """Smeared T = 0 closed form = exact Phi_T matter part (away from the dips). The fixed nodes do not resolve the
    channel-opening cusps in nu: ~1e-5, enough for the grid (the minimum is refined with the exact Phi_T)."""
    M, mu, L, T = 2.181715, 3.3, 2.0, 0.05
    a = TH.phi_mat_T_vec(k, np.array([q1]), M, mu, L, T)[0]
    assert a == pytest.approx(S.phi_mat_T(k, q1, M, mu, L, T), abs=1e-4)


def test_phi_zero_uses_G_T():
    assert S.phi_zero(0, 1e-4, 3.0, 1.0, -1.0, T=0.3) == pytest.approx(
        0.5 * TH.gap_G_T(0.0, 3.0, 1.0, -1.0, 0.3), abs=1e-5)


@pytest.mark.parametrize('mu,L,T,ref', [(3.0, 1.0, 0.05, -0.273168), (3.0, 1.0, 0.3, -0.005529),
                                        (4.0, 1.0, 1.0, 0.151138)])
def test_scan_min_T_vs_brute_force(mu, L, T, ref):
    """References: exact Phi_T on a 1500-point q1 grid for every k (brute force)."""
    r = TH.scan_min_T(mu, L, -1.0, T)
    assert r['phi_min'] <= ref + 1e-5
    assert r['phi_min'] == pytest.approx(ref, abs=2e-5)
