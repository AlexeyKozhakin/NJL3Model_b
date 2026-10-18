"""Certified search for the roots of G and the global minimum, phases, known values."""

import math

import numpy as np
import pytest

from njl3.b0 import formulas as F
from njl3.b0 import reference as R
from njl3.b0.minimize import (
    M_vacuum, analyze_point, find_minimum, find_minimum_scan, isolate_roots, phase_at,
)

G_ = -1.0
RNG = np.random.default_rng(7)


def _random_points(n):
    for _ in range(n):
        yield float(RNG.uniform(0.5, 7.0)), float(np.exp(RNG.uniform(math.log(0.1), math.log(10.0))))


def test_roots_match_mpmath():
    count = 0
    for mu, L in _random_points(40):
        for r, kind in isolate_roots(mu, L, G_):
            if kind == 'touch':
                continue
            assert r == pytest.approx(float(R.root_near(r, mu, L, G_)), rel=1e-13)
            count += 1
    assert count > 20


def test_certified_finds_everything_a_dense_scan_finds():
    """On a dense grid (20 000 points) the scan finds no root that the certified method misses."""
    for mu, L in _random_points(25):
        M = np.concatenate([np.geomspace(1e-8, 1e-2, 2000), np.linspace(1e-2, 2.0 * mu + 4.0, 18000)])
        Gs = F.gap_G_vec(M, mu, L, G_)
        dense = int(np.count_nonzero(np.sign(Gs[:-1]) * np.sign(Gs[1:]) < 0))
        roots = [r for r, k in isolate_roots(mu, L, G_) if k != 'touch']
        assert len(roots) >= dense


def test_scan_misses_narrow_pair():
    """Example from the paper: a pair of roots 0.003 apart with a scan step of 0.013."""
    mu, L = 3.3, 3.73
    roots = isolate_roots(mu, L, G_)
    assert [k for _, k in roots] == ['min', 'max', 'min']
    assert roots[2][0] - roots[1][0] < 0.005
    assert len(find_minimum(mu, L, G_)[2]) == 2


def test_global_minimum_agrees_with_scan_away_from_narrow_pairs():
    for mu, L in _random_points(40):
        Mc, Oc, _ = find_minimum(mu, L, G_)
        Ms, Os, _ = find_minimum_scan(mu, L, G_)
        assert Oc <= Os + 1e-12


@pytest.mark.parametrize('L, M_vac', [(0.5, 3.7923395005), (1.0, 3.2229018436), (2.0, 3.1434548869)])
def test_vacuum_mass(L, M_vac):
    assert M_vacuum(L, G_) == pytest.approx(M_vac, abs=1e-9)


def test_vacuum_mass_large_L_is_pi():
    assert M_vacuum(30.0, G_) == pytest.approx(math.pi, abs=1e-12)


def test_vacuum_mass_golden_ratio_at_small_L():
    phi = (1 + math.sqrt(5)) / 2
    for L in [0.02, 0.01]:
        asym = 2 * math.log(phi) / L + math.pi / math.sqrt(5)
        assert abs(M_vacuum(L, G_) - asym) < 1.0 * L


@pytest.mark.parametrize('mu, L, phase', [
    (1.0, 1.0, 'V'), (6.0, 1.0, 'T'), (3.11, 3.5, 'I0'), (3.15, 3.5, 'I1'), (3.3, 3.5, 'T'),
])
def test_phase_labels(mu, L, phase):
    assert phase_at(mu, L, G_)[0] == phase


def test_vacuum_phase_is_mu_independent():
    for mu in [0.5, 1.5, 2.5]:
        assert phase_at(mu, 1.0, G_)[1] == pytest.approx(M_vacuum(1.0, G_), abs=1e-12)


def test_analyze_point_fields():
    r = analyze_point(3.5, 2.0, G_)
    assert r.phase == 'T' and r.M_star == 0.0
    assert r.G0 == pytest.approx(0.1413133299, abs=1e-9)
    assert [k for _, k in r.roots] == ['max', 'min']
    d = r.to_dict()
    assert d['phase'] == 'T' and len(d['local_minima']) == 1
