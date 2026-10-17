"""njl3.full: Omega(M, b), логарифмическая неустойчивость у порогов, асимптотика, минимум по (M, b)."""

import math
import os
import sys

import numpy as np
import pytest

from njl3.b0 import formulas as B0
from njl3.full import formulas as F
from njl3.full.minimize import channel_asymptotics, channel_minimum, find_minimum, h_universal, thresholds

G_ = -1.0
RNG = np.random.default_rng(11)


def test_s_small_x_expansion():
    """s(x) = π²/6 − πx/2 − (x²/4)[ln(x/4π) + γ − 1/2] + ζ(3)x⁴/(64π²) + O(x⁶)."""
    for x in [1e-3, 1e-2, 0.05]:
        approx = (math.pi**2 / 6 - math.pi * x / 2 - x * x / 4 * (math.log(x / (4 * math.pi)) + 0.5772156649015329 - 0.5)
                  + 1.2020569031595942 * x**4 / (64 * math.pi**2))
        assert F.s_func(np.array([x]))[0] == pytest.approx(approx, abs=x**6 + 1e-15)


def test_b0_limit_matches_closed_forms():
    for _ in range(25):
        M = float(RNG.uniform(0, 4)); L = float(np.exp(RNG.uniform(math.log(0.1), math.log(10)))); mu = float(RNG.uniform(0.2, 7))
        assert F.omega_L(M, 0.0, L) == pytest.approx(B0.omega_L(M, L), abs=1e-12 * max(1.0, abs(B0.omega_L(M, L))))
        assert F.omega_muL(M, 0.0, mu, L) == pytest.approx(B0.omega_muL(M, mu, L), abs=1e-13)
        assert F.dU(M, 0.0) == pytest.approx(B0.dU(M), rel=1e-14)


def test_dU_closed_form_continuity_and_limits():
    for M in [0.3, 1.0, 2.5]:
        assert F.dU(M, M * (1 - 1e-12)) == pytest.approx(F.dU(M, M), rel=1e-9)
        assert F.dU(M, M) == pytest.approx(M**3 / 4)
        assert F.dU(M, 3 * M) == pytest.approx(M * M * 3 * M / 4)


def test_vectorised_equals_scalar():
    M = RNG.uniform(0, 4, 50); b = RNG.uniform(0, 6, 50)
    v = F.omega_rel_vec(M, b, 3.7, 2.3, G_)
    s = np.array([F.omega_rel(m, bb, 3.7, 2.3, G_) for m, bb in zip(M, b)])
    assert np.max(np.abs(v - s)) < 1e-12


@pytest.mark.parametrize('mu, L', [(5.0, 3.0), (6.0, 1.0)])
def test_log_slope_at_thresholds(mu, L):
    """[Ω(M, S_n) − Ω(0)]/M² = A_n − c_n ln(1/M): наклон по ln M равен c_n."""
    for n, S, C in thresholds(mu, L):
        c = C * S / (2 * math.pi * L * mu)
        y = [F.omega_rel(M, S, mu, L, G_) / M**2 for M in (1e-3, 1e-5)]
        assert (y[0] - y[1]) / math.log(100) == pytest.approx(c, rel=2e-3)


def test_universal_crossover_function():
    """Ω(M, S_n + tM) − Ω(0) ≈ M²[A_n − c_n ln(1/M) + c_n h(t)] с универсальной h."""
    mu, L, M = 5.0, 3.0, 1e-5
    for n, S, C in thresholds(mu, L)[:2]:
        c = C * S / (2 * math.pi * L * mu)
        y0 = F.omega_rel(M, S, mu, L, G_) / M**2
        for t in [-3.0, -1.5, -0.5, 0.7, 2.0]:
            h = (F.omega_rel(M, S + t * M, mu, L, G_) / M**2 - y0) / c
            assert h == pytest.approx(h_universal(t), abs=2e-3)


def test_asymptotics_agree_with_numerics():
    for mu, L in [(6.0, 1.0), (5.0, 3.0)]:
        for n, S, C in thresholds(mu, L):
            a = channel_asymptotics(S, C, mu, L, G_)
            ch = channel_minimum(S, C, mu, L, G_)
            if 1e-5 < a['M'] < 1e-2:
                assert ch['lnM'] == pytest.approx(a['lnM'], abs=2e-3)
                assert ch['b'] == pytest.approx(S, abs=0.05 * ch['M'])


@pytest.mark.parametrize('mu, L, phase', [(1.0, 1.0, 'V'), (2.8, 0.5, 'B0'), (3.5, 1.0, 'B0'), (5.0, 3.0, 'B1'), (2.0, 9.0, 'V')])
def test_known_phases(mu, L, phase):
    r = find_minimum(mu, L, G_)
    assert r.phase == phase


def test_no_trivial_phase_with_free_b():
    """Логарифмическая неустойчивость: при μ > 0 минимум по (M, b) всегда ниже Ω(0, 0)."""
    for mu, L in [(6.0, 1.0), (6.5, 8.0), (4.4, 2.0)]:
        r = find_minimum(mu, L, G_)
        assert r.omega_star < 0.0 and r.phase != 'T'


@pytest.mark.slow
def test_dU_closed_form_matches_2d_integral():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from functions.dU_fast import dU_scalar
    for M, b in [(0.5, 0.3), (2.0, 1.0), (0.8, 2.5), (3.0, 3.5)]:
        assert dU_scalar(b, M, 1600, 1600) == pytest.approx(F.dU(M, b), abs=2e-6)


def test_channels_compared_without_noise_margin():
    """Каналы сравниваются между собой без запаса шума (раньше B1 «защищался» от более глубокого B2)."""
    r = find_minimum(6.647, 5.134, -1.0)
    om = {ch['n']: ch['omega'] for ch in r.channels}
    assert om[2] < om[1] < 0
    assert r.phase == 'B2'


def test_temperature_cutoff():
    """T > 0: щель канала n выживает при T < (e^γ/π)(S_n/mu) M*_n (сверено notebooks/57)."""
    mu, L = 6.647, 5.134
    r0 = find_minimum(mu, L, -1.0)
    n = int(r0.phase[1:])
    S = dict((q[0], q[1]) for q in thresholds(mu, L))[n]
    Tc = math.exp(np.euler_gamma) / math.pi * S / mu * r0.M_star
    assert find_minimum(mu, L, -1.0, T=0.9 * Tc).phase == r0.phase
    assert find_minimum(mu, L, -1.0, T=1.1 * Tc).phase != r0.phase
    assert find_minimum(mu, L, -1.0, T=1e-3).phase == 'T'
    # сильная щель (V) не затрагивается
    assert find_minimum(3.0, 1.0, -1.0, T=1e-3).M_star > 0.1
