"""Closed-form Phi''(k = 0, q1 = 0) (phi_dd0) against a finite difference of the closed-form Phi."""
import warnings

import numpy as np
import pytest

from njl3.hom import phase_at
from njl3.stab import formulas as S

H = 0.01


def fd(M, mu, L, phi):
    m = M if M > 0 else S.M_ZERO
    v = S.phi_vec(0, np.array([H, 2 * H]), m, mu, L, phi)
    return 2.0 * (v[1] - v[0]) / (3.0 * H * H)


@pytest.mark.parametrize('phi', [0.0, 0.1, 0.25, 0.5])
@pytest.mark.parametrize('mu,L,g', [(2.8, 2.0, -1.0), (3.3, 2.0, -1.0), (3.6, 5.0, -1.0), (5.0, 1.0, -1.0),
                                    (3.15, 6.0, -1.0), (4.0, 3.0, 1.0), (1.0, 3.0, -1.0)])
def test_closed_form_matches_finite_difference(mu, L, g, phi):
    warnings.simplefilter('ignore')
    _, M = phase_at(mu, L, g, phi)
    assert S.phi_dd0(M, mu, L, phi) == pytest.approx(fd(M, mu, L, phi), abs=2e-5)


def test_even_in_q1():
    for k in (0, 1, 2):
        assert S.phi(k, -0.3, 1.2, 3.3, 2.0) == pytest.approx(S.phi(k, 0.3, 1.2, 3.3, 2.0), abs=1e-12)


def test_phase_T_zero_mode_formula():
    """φ = 0, M* = 0: R(0) = L/(24π) - (1/(πL)) [1/(4mu²) + Σ_{n≠0} mu/(2 P_n k_n²)] (checked by hand at (3.6, 5))."""
    assert S.phi_dd0(0.0, 3.6, 5.0) == pytest.approx(0.00798, abs=2e-5)
