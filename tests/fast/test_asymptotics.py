"""M2: analytic channel coefficient A vs the numerical limit of Omega(M, S)/M² + c ln(1/M)."""

import math
import warnings

import pytest

from njl3.fast.asymptotics import channel_A
from njl3.inh import formulas as F


@pytest.mark.parametrize('phi', [0.0, 0.1, 0.3, 0.5])
@pytest.mark.parametrize('mu,L,g', [(3.6, 5.0, -1.0), (4.5, 8.0, -1.0), (2.0, 1.0, 1.0), (3.2, 2.5, -1.0)])
def test_A_matches_numerical_limit(phi, mu, L, g):
    warnings.simplefilter('ignore')
    for j, (k, C) in enumerate(F.levels(mu, L, phi)):
        S = math.sqrt(mu * mu - k * k)
        A, c = channel_A(j, mu, L, g, phi)
        assert c == pytest.approx(C * S / (2 * math.pi * L * mu), rel=1e-14)
        M = 1e-4
        num = F.omega_rel(M, S, mu, L, g, phi) / M**2 + c * math.log(1 / M)
        assert A == pytest.approx(num, abs=5e-7)          # O(M² ln M) ≈ 1e-7 at M = 1e-4
