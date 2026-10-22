"""Compiled core njl3.fast.nb and minimize_nb vs njl3.inh / njl3.fast.minimize."""

import json
import os
import warnings

import numpy as np
import pytest

from njl3.fast import minimize_nb
from njl3.fast.nb import Ctx
from njl3.inh import formulas as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.parametrize('phi', [0.0, 0.1, 0.3, 0.5])
def test_omega_matches_inh(phi):
    rng = np.random.default_rng(int(10 * phi))
    for _ in range(25):
        mu, L, M, b = rng.uniform(0.5, 6.5), rng.uniform(0.3, 10), rng.uniform(1e-4, 3.5), rng.uniform(0, 6)
        g = float(rng.choice([-1.0, 1.0]))
        assert Ctx(mu, L, g, phi).omega(M, b) == pytest.approx(F.omega_rel(M, b, mu, L, g, phi), abs=1e-12, rel=1e-12)
    assert Ctx(3.6, 5.0, -1.0, phi).omega(1e-6, 3.3) == pytest.approx(F.omega_rel(1e-6, 3.3, 3.6, 5.0, -1.0, phi), abs=5e-14)  # Omega ~ 1e-13 here: a difference of O(1) terms, rounding ~1e-14


def test_find_minimum_matches_reference_points():
    warnings.simplefilter('ignore')
    rows = json.load(open(os.path.join(ROOT, 'web', 'fast', 'test', 'expected_fast.json')))[::3]
    for r in rows:
        a = minimize_nb.find_minimum(r['mu'], r['L'], r['g'], r['phi'], r['T'])
        assert a.phase == r['phase']
        if r['lnM'] is not None:
            assert abs(a.ln_M_star - r['lnM']) < 1e-4
