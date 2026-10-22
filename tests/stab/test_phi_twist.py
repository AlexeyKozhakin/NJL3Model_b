"""njl3.stab with a twist φ of the boundary condition (and g = ±1 through M* of njl3.hom)."""

import math
import warnings

import pytest

from njl3.hom import phase_at
from njl3.stab import formulas as S

PHIS = [0.1, 0.25, 0.5]


def _I_point(phi, g=-1.0):
    """A point of an intermediate phase (0 < M* < mu) for this φ, g."""
    for mu in (3.05, 3.1, 3.13, 3.15, 3.2, 3.3, 3.5):
        for L in (2.0, 3.0, 4.0, 5.0, 7.0, 10.0):
            ph, M = phase_at(mu, L, g, phi)
            if ph.startswith('I'):
                return mu, L, M
    pytest.skip('no intermediate phase found')


@pytest.mark.parametrize('phi', PHIS)
@pytest.mark.parametrize('k,q1', [(0, 0.7), (1, 2.3), (2, 4.1)])
def test_matter_closed_form_vs_original(phi, k, q1):
    warnings.simplefilter('ignore')
    mu, L, M = _I_point(phi)
    assert S.phi_mat(k, q1, M, mu, L, phi) == pytest.approx(S.phi_direct_mat(k, q1, M, mu, L, phi=phi), abs=1e-11)


@pytest.mark.parametrize('phi', PHIS)
@pytest.mark.parametrize('k,q1', [(0, 0.5), (1, 2.7), (3, 0.3)])
def test_vacuum_vs_direct_sum(phi, k, q1):
    warnings.simplefilter('ignore')
    M, L = 1.3, 3.0
    a, b = S.phi_direct_vac(k, q1, M, L, 300, phi=phi), S.phi_direct_vac(k, q1, M, L, 600, phi=phi)
    assert S.phi_vac(k, q1, M, L, phi) == pytest.approx(2 * b - a, rel=5e-6)


@pytest.mark.parametrize('phi', PHIS)
def test_symmetries(phi):
    mu, L, M = _I_point(phi)
    for k in (1, 2):
        assert S.phi(-k, 1.7, M, mu, L, 0.0, phi) == pytest.approx(S.phi(k, 1.7, M, mu, L, 0.0, phi), rel=1e-11)
    # φ and 1 − φ: the same set of |p₂ₙ|
    assert S.phi(1, 1.7, M, mu, L, 0.0, 1 - phi) == pytest.approx(S.phi(1, 1.7, M, mu, L, 0.0, phi), rel=1e-11)


def test_phi_zero_default_unchanged():
    mu, L, M = 3.15, 10.0, 1.0459477758243578
    assert S.phi(1, 2.7, M, mu, L) == S.phi(1, 2.7, M, mu, L, 0.0, 0.0)


def test_large_L_independent_of_phi():
    v = [S.phi_vac(0, 1.5, 1.0, 200.0, p) for p in (0.0, 0.2, 0.5)]
    assert max(v) - min(v) < 1e-12


@pytest.mark.parametrize('phi', PHIS)
def test_finite_T_and_scan(phi):
    warnings.simplefilter('ignore')
    mu, L, M = _I_point(phi)
    q = 1.3
    assert S.phi(0, q, M, mu, L, 1e-5, phi) == pytest.approx(S.phi(0, q, M, mu, L, 0.0, phi), abs=1e-5)
    r = S.scan_min(M, mu, L, T=1e-3, phi=phi)
    assert math.isfinite(r['phi_min'])
