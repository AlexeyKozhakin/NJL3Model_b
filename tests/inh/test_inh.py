"""njl3.inh: Omega(M, b) and its global minimum for a twist φ."""

import numpy as np
import pytest

from njl3.full import formulas as FF
from njl3.full.minimize import find_minimum as full_find_minimum
from njl3.hom import formulas as H
from njl3.inh import find_minimum, omega_rel, s_func
from njl3.inh import formulas as F
from njl3.inh.make_tables import s_mp

PHIS = [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.parametrize('phi', PHIS)
def test_table_vs_mpmath(phi):
    rng = np.random.default_rng(int(phi * 10))
    xs = np.concatenate([rng.uniform(0, 64, 20), np.geomspace(1e-9, 2, 20), [0.0]])
    ref = np.array([float(s_mp(x, phi)) for x in xs])
    assert np.max(np.abs(s_func(xs, phi) - ref)) < 2e-15


@pytest.mark.parametrize('phi', PHIS)
def test_b0_equals_hom(phi):
    """At b = 0 the potential is the homogeneous one (independent check with complex polylogs)."""
    for mu, L, g in ((3.0, 2.0, -1.0), (3.3, 6.0, -1.0), (2.0, 1.0, 1.0)):
        for M in (0.3, 1.5, 3.0):
            assert omega_rel(M, 0.0, mu, L, g, phi) == pytest.approx(H.omega_rel_closed(M, mu, L, g, phi), abs=5e-13)


def test_phi_zero_is_full():
    for M, b in ((0.4, 1.0), (1.2, 2.9)):
        assert omega_rel(M, b, 3.2, 4.0, -1.0, 0.0) == FF.omega_rel(M, b, 3.2, 4.0, -1.0)
    a, c = find_minimum(3.6, 5.0, -1.0, 0.0, T=1e-3), full_find_minimum(3.6, 5.0, -1.0, T=1e-3)
    assert (a.phase, a.M_star, a.b_star) == (c.phase, c.M_star, c.b_star)


def test_one_minus_phi_equivalent():
    assert find_minimum(3.6, 5.0, -1.0, 0.7, T=1e-3).phase == find_minimum(3.6, 5.0, -1.0, 0.3, T=1e-3).phase
    with pytest.raises(ValueError):
        find_minimum(3.0, 2.0, -1.0, 0.15)


def test_large_L_independent_of_phi():
    """L -> ∞: Omega(M, b) does not depend on φ (theorem of njl3.inf: equals Omega(M, 0) for b + M ≤ mu)."""
    v = [omega_rel(1.0, 1.5, 3.0, 300.0, -1.0, phi) for phi in (0.0, 0.2, 0.5)]
    assert max(v) - min(v) < 1e-3
    assert v[1] == pytest.approx(omega_rel(1.0, 0.0, 3.0, 300.0, -1.0, 0.2), abs=1e-3)


@pytest.mark.parametrize('phi', [0.2, 0.5])
def test_minimum_not_above_grid(phi):
    """The global minimum is not worse than a coarse (M, b) grid scan."""
    for mu, L, g in ((3.6, 5.0, -1.0), (3.15, 3.5, -1.0), (2.0, 1.0, 1.0)):
        r = find_minimum(mu, L, g, phi)
        Mg, bg = np.meshgrid(np.linspace(0.02, 4.0, 40), np.linspace(0.0, 4.0, 41))
        scan = min(0.0, float(np.min(F.omega_rel_vec(Mg.ravel(), bg.ravel(), mu, L, g, phi))))
        assert r.omega_star <= scan + 1e-10
