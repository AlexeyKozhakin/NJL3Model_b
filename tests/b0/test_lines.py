"""Линии переходов, пики, асимптотики; сверка с численными интеграторами общего кода."""

import math
import os
import sys

import numpy as np
import pytest

from njl3.b0 import formulas as F
from njl3.b0.lines import mu_stability, mu_VT, peak, peaks, trace_L, trace_lines

G_ = -1.0


@pytest.mark.parametrize('L, mu_s', [(0.5, 2.1932800507), (1.0, 2.4052386905), (2.0, 3.2702434800)])
def test_stability_line(L, mu_s):
    assert mu_stability(L, G_) == pytest.approx(mu_s, abs=1e-9)
    assert F.gap_G0(mu_stability(L, G_), L, G_) == pytest.approx(0.0, abs=1e-14)


def test_stability_line_explicit_single_mode():
    """При n_max = 0: mu_s = e^{pi L / (2|g|)} / (2L)."""
    for L in [0.3, 0.5, 1.0]:
        assert mu_stability(L, G_) == pytest.approx(math.exp(math.pi * L / 2) / (2 * L), rel=1e-13)


@pytest.mark.parametrize('L, mu', [(0.5, 2.8684035017), (1.0, 2.6873373974)])
def test_vacuum_trivial_line(L, mu):
    assert mu_VT(L, G_) == pytest.approx(mu, abs=1e-9)


PEAKS = [(1, 1.6113000800, 3.8994507512), (2, 3.7293741534, 3.3695655350),
         (3, 5.7804936446, 3.2608903461), (4, 7.8105857856, 3.2177792958),
         (5, 9.8309644908, 3.1956098067)]


@pytest.mark.parametrize('n, L, mu', PEAKS)
def test_peaks_explicit(n, L, mu):
    Ln, mun = peak(n, G_)
    assert (Ln, mun) == pytest.approx((L, mu), abs=1e-9)
    assert mun * Ln == pytest.approx(2 * math.pi * n)
    assert mu_stability(Ln, G_) == pytest.approx(mun, abs=1e-6)


def test_peaks_list():
    assert [n for n, _, _ in peaks(G_, 10.0)] == [1, 2, 3, 4, 5]


def test_peak_asymptotics_zeta():
    """mu_n - pi = 2√2 |zeta(-1/2)| n^{-3/2} + O(n^{-5/2})."""
    c = 2 * math.sqrt(2) * 0.2078862250773080
    for n in [200, 2000]:
        _, mu = peak(n, G_)
        assert (mu - math.pi) * n**1.5 == pytest.approx(c, rel=2.0 / n)


def test_trace_single_first_order_transition():
    tr = trace_L(1.0, G_, n_mu=200)
    assert [(r['frm'], r['to'], r['order']) for r in tr] == [('V', 'T', 1)]
    assert tr[0]['mu'] == pytest.approx(mu_VT(1.0, G_), abs=1e-10)


def test_trace_cascade_and_second_order():
    tr = trace_L(3.5, G_, n_mu=200)
    assert [(r['frm'], r['to'], r['order']) for r in tr] == [('V', 'I0', 1), ('I0', 'I1', 1), ('I1', 'T', 2)]
    assert [r['mu'] for r in tr] == pytest.approx([3.1014684751, 3.1208059008, 3.2248655571], abs=1e-9)
    assert tr[2]['mu'] == pytest.approx(mu_stability(3.5, G_), abs=1e-10)


@pytest.mark.slow
def test_critical_end_point():
    out = trace_lines(np.linspace(1.20, 1.25, 4), G_, mu_lo=2.5, mu_hi=3.2, n_mu=60, jobs=2)
    ce = [s for s in out['special'] if s['kind'] == 'critical_end']
    assert len(ce) == 1
    assert ce[0]['L'] == pytest.approx(1.2226473, abs=1e-6)
    assert ce[0]['mu'] == pytest.approx(2.7909025, abs=1e-5)


@pytest.mark.slow
def test_closed_forms_match_numerical_integrators():
    """Мост к статье 2: замкнутые формулы = прямое численное интегрирование (functions/*_fast)."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from functions.dU_fast import dU_scalar
    from functions.Omega_L_fast import Omega_L_scalar
    from functions.Omega_mu_L import fun_Omega_L_mu_int

    for L in [0.5, 1.0, 2.0]:
        for mu in [1.5, 3.2, 5.0]:
            for M in [0.3, 1.0, 2.5]:
                assert dU_scalar(0.0, M, 400, 400) - dU_scalar(0.0, 0.0, 400, 400) == pytest.approx(F.dU(M), abs=1e-5)
                assert Omega_L_scalar(L, 0.0, M, 0.0, 400, 400) == pytest.approx(F.omega_L(M, L), abs=1e-5)
                assert fun_Omega_L_mu_int(mu, L, 0.0, M, 0, 6400) == pytest.approx(F.omega_muL(M, mu, L), abs=1e-5)
