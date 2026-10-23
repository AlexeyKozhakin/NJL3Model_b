"""Second variation around the symmetric solution M = 0."""

import math
import warnings

import pytest

from njl3.stab import formulas as S

PTS = [(3.6, 5.0, -1.0), (4.5, 3.0, -1.0), (5.0, 8.0, 1.0), (2.0, 1.5, -1.0)]


@pytest.mark.parametrize('mu,L,g', PTS)
@pytest.mark.parametrize('phi', [0.0, 0.3, 0.5])
def test_regulator_independent(mu, L, g, phi):
    for k, q1 in ((0, 1.0), (1, 0.4), (2, 3.1)):
        a = S.phi_zero(k, q1, mu, L, g, phi=phi, m_reg=1e-3)
        b = S.phi_zero(k, q1, mu, L, g, phi=phi, m_reg=1e-4)
        assert a == pytest.approx(b, abs=1e-6)
        # the vectorized (map) version agrees with the scalar one at the regulator mass
        import numpy as np
        v = S.half_G0(mu, L, g, phi) + S.phi_vec(k, np.array([q1]), S.M_ZERO, mu, L, phi)[0]
        assert v == pytest.approx(b, abs=1e-7)


@pytest.mark.parametrize('mu,L,g', PTS)
@pytest.mark.parametrize('phi', [0.0, 0.2])
def test_q_to_zero_gives_half_G0(mu, L, g, phi):
    assert S.phi_zero(0, 1e-4, mu, L, g, phi=phi) == pytest.approx(S.half_G0(mu, L, g, phi), abs=1e-6)


@pytest.mark.parametrize('mu,L,g', PTS[:3])
@pytest.mark.parametrize('k,q1', [(0, 1.0), (1, 1.0), (0, 0.1), (2, 1.7)])
def test_matches_zhokhov_formula_with_corrected_matter_factor(mu, L, g, k, q1):
    warnings.simplefilter('ignore')
    # the literal formula has an O(m) regulator term: (1/πL) ln(1 − e^{−Lm}) = (1/πL) ln(Lm) − m/(2π) + …
    m = S.M_ZERO
    assert S.phi_zero(k, q1, mu, L, g) == pytest.approx(S.phi_zero_direct(q1, k, mu, L, g, m) + m / (2 * math.pi), abs=2e-6)


def test_formula_as_written_fails_the_q_to_zero_check():
    """With the matter prefactor 2/L the limit q -> 0 is not ½ G(0⁺) (documents the correction)."""
    warnings.simplefilter('ignore')
    mu, L, g = 3.6, 5.0, -1.0
    v = S.phi_zero_direct(0.01, 0, mu, L, g, 1e-5, matter_factor=1.0)
    assert abs(v - S.half_G0(mu, L, g)) > 0.1


def test_scan_min_zero_finite():
    warnings.simplefilter('ignore')
    r = S.scan_min_zero(4.0, 5.0, -1.0, T=1e-3)
    assert r['phi_min'] < S.half_G0(4.0, 5.0, -1.0) + 1e-12


@pytest.mark.parametrize('M,mu,L', [(1e-4, 2.145, 0.5969849246231156), (1e-4, 2.46, 0.5969849246231156),
                                    (0.4, 3.3, 2.0), (1e-4, 3.6, 5.0)])
def test_finite_T_at_the_singular_point_is_continuous(M, mu, L):
    """At q1 = q_s the pole p0 = -q1/2 meets the Fermi point -P (to ~1e-13): QUADPACK must not break there,
    and the 1/E peak of a light mode at p = 0 must be resolved."""
    warnings.simplefilter('ignore')
    for k in (0, 1):
        for x in [x for x in S.singular_q1(M, mu, L, k) if x > 1e-6]:
            a = S.phi_mat_T(k, x, M, mu, L, 1e-3)
            for d in (1e-12, -1e-12, 1e-9):
                assert a == pytest.approx(S.phi_mat_T(k, x + d, M, mu, L, 1e-3), abs=1e-6)
