"""JS-ядро web/full/js/njl3full.js совпадает с Python-пакетом njl3.full."""

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest

from njl3.full import formulas as F
from njl3.full.minimize import channel_asymptotics, find_minimum, thresholds

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
G_ = -1.0


def _golden(seed=3):
    rng = np.random.default_rng(seed)
    s = [dict(x=float(x), s=float(F.s_func(np.array([x]))[0]))
         for x in np.concatenate([np.geomspace(1e-9, 0.5, 30), rng.uniform(0.5, 70, 60)])]
    om = []
    for _ in range(150):
        mu = float(rng.uniform(0.3, 7)); L = float(np.exp(rng.uniform(math.log(0.1), math.log(10))))
        M = float(rng.choice([rng.uniform(0, 4), np.exp(rng.uniform(math.log(1e-6), 0))]))
        b = float(rng.uniform(0, mu + 2))
        om.append(dict(M=M, b=b, mu=mu, L=L, omega=F.omega_rel(M, b, mu, L, G_)))
    ch = []
    for _ in range(25):
        mu = float(rng.uniform(1, 7)); L = float(np.exp(rng.uniform(math.log(0.3), math.log(8))))
        for n, S, C in thresholds(mu, L)[:3]:
            a = channel_asymptotics(S, C, mu, L, G_)
            ch.append(dict(mu=mu, L=L, S=S, C=C, A=a['A'], lnM=a['lnM']))
    pts = []
    for mu, L in [(1.0, 1.0), (2.8, 0.5), (3.0, 1.0), (3.5, 1.0), (4.0, 2.0), (3.8, 4.0), (6.0, 1.0),
                  (5.0, 3.0), (2.0, 9.0), (4.5, 1.5), (3.2, 2.5)]:
        r = find_minimum(mu, L, G_)
        pts.append(dict(mu=mu, L=L, phase=r.phase, lnM=r.ln_M_star, b=r.b_star, omega=r.omega_star))
    return dict(g=G_, s=s, omega=om, channels=ch, points=pts)


@pytest.mark.skipif(shutil.which('node') is None, reason='нужен Node.js')
def test_js_port_full_matches_python(tmp_path):
    path = tmp_path / 'golden.json'
    path.write_text(json.dumps(_golden()))
    res = subprocess.run(['node', os.path.join(ROOT, 'web', 'full', 'test', 'compare.js'), str(path)],
                         capture_output=True, text=True, check=True)
    err = json.loads(res.stdout)
    # omega: |Δ| / max(1, |Ω|). При малых L Ω_L ~ 1e2–1e3, а Ω — разность Ω_L(M, b) − Ω_L(0, b) ~ 1:
    # разный порядок суммирования в Python и JS даёт ~1e-12 (предел точности самой разности)
    assert err['s'] < 1e-14 and err['omega'] < 1e-11
    # A_n = Ω(M_PROBE)/M_PROBE² + c ln(1/M_PROBE): расхождение Ω ~1e-14 делится на 1e-8 -> ~1e-6
    assert err['A'] < 1e-5 and err['lnM_asym'] < 1e-4
    assert err['phase_mismatch'] == 0
    assert err['lnM_star'] < 1e-4 and err['b_star'] < 1e-6 and err['omega_star_rel'] < 1e-6
