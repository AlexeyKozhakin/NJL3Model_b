"""NJL3 on R² at T > 0 (njl3.inf.thermal)."""
import math

import pytest
from scipy.integrate import quad

from njl3.inf import formulas as F0
from njl3.inf import thermal as TH

G = -1.0


@pytest.mark.parametrize('M,mu,T', [(1.0, 2.0, 0.3), (2.5, 3.0, 0.1), (0.5, 1.0, 1.0)])
def test_closed_form_b0(M, mu, T):
    assert TH.W_T_b0(M, mu, T) == pytest.approx(TH.W_T(M, 0.0, mu, T), abs=1e-10)


@pytest.mark.parametrize('M,b,mu', [(1.0, 0.0, 2.0), (1.0, 0.5, 2.0), (2.0, 1.5, 3.0), (3.0, 0.5, 3.3)])
def test_zero_temperature_limit(M, b, mu):
    assert TH.omega_rel_T(M, b, mu, 1e-3, G) == pytest.approx(F0.omega_rel(M, b, mu, G), abs=2e-6)


def test_critical_temperature_mu0():
    """T_c(mu = 0) = M_vac / (2 ln 2) — GN in 2+1 at large N."""
    Tc = math.pi / (2 * math.log(2))
    assert TH.gap_G_T_b0(0.0, 0.0, Tc, G) == pytest.approx(0.0, abs=1e-12)
    assert TH.boundary_mu(Tc * (1 - 1e-9), G) == pytest.approx(0.0, abs=1e-3)


def test_smearing_identity():
    """W_T(mu) = ∫_0^∞ [K(nu - mu) + K(nu + mu)] W_0(nu) dnu, K = -f', ∫ weight = 1."""
    M, b, mu, T = 1.5, 0.8, 3.0, 0.2
    K = lambda x: 1 / (4 * T * math.cosh(min(abs(x) / (2 * T), 350)) ** 2)
    rhs = quad(lambda nu: (K(nu - mu) + K(nu + mu)) * F0.W(M, b, nu), 0, mu + 40 * T, points=[mu], limit=200)[0]
    assert TH.W_T(M, b, mu, T) == pytest.approx(rhs, abs=1e-8)


@pytest.mark.parametrize('T', [0.1, 0.5])
@pytest.mark.parametrize('mu', [2.8, math.pi, 3.5])
def test_spiral_not_favoured(mu, T):
    for M in (1.0, 2.0, 2.5):
        om0 = TH.omega_rel_T(M, 0.0, mu, T, G)
        for b in (0.3, 1.0):
            assert TH.omega_rel_T(M, b, mu, T, G) - om0 > -1e-12
