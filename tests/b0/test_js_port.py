"""The JS core of the web app (web/b0/js/njl3b0.js) agrees with the Python package njl3.b0."""

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest

from njl3.b0 import formulas as F
from njl3.b0.lines import peak
from njl3.b0.minimize import M_vacuum, find_minimum, isolate_roots, phase_label

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
G_ = -1.0


def _golden(n_fun=400, n_pts=200, seed=11):
    rng = np.random.default_rng(seed)
    functions, points = [], []
    while len(functions) < n_fun:
        mu = float(rng.uniform(0.2, 7.0)); L = float(np.exp(rng.uniform(math.log(0.05), math.log(10.0))))
        kind = rng.integers(4)
        M = {0: float(np.exp(rng.uniform(math.log(1e-10), math.log(1e-2)))),
             1: float(rng.uniform(0.0, mu)),
             2: float(rng.uniform(mu, mu + 5.0)),
             3: float(rng.uniform(0.0, 0.5))}[int(kind)]
        if M <= 0:
            continue
        functions.append(dict(mu=mu, L=L, M=M, G=F.gap_G(M, mu, L, G_), G0=F.gap_G0(mu, L, G_),
                              G2=F.G2(mu, L), omega=F.omega_rel(M, mu, L, G_)))
    for _ in range(n_pts):
        mu = float(rng.uniform(0.5, 7.0)); L = float(np.exp(rng.uniform(math.log(0.1), math.log(10.0))))
        roots = isolate_roots(mu, L, G_)
        M, _, _ = find_minimum(mu, L, G_, roots=roots)
        points.append(dict(mu=mu, L=L, roots=[r for r, _ in roots], M_star=M, phase=phase_label(M, mu, L)))
    vacuum = [dict(L=L, M_vac=M_vacuum(L, G_)) for L in (0.05, 0.3, 1.0, 3.0, 10.0)]
    peaks = [dict(n=n, L=peak(n, G_)[0], mu=peak(n, G_)[1]) for n in range(1, 8)]
    return dict(g=G_, functions=functions, points=points, vacuum=vacuum, peaks=peaks)


@pytest.mark.skipif(shutil.which('node') is None, reason='needs Node.js')
def test_js_port_matches_python(tmp_path):
    path = tmp_path / 'golden.json'
    path.write_text(json.dumps(_golden()))
    res = subprocess.run(['node', os.path.join(ROOT, 'web', 'b0', 'test', 'compare.js'), str(path)],
                         capture_output=True, text=True, check=True)
    err = json.loads(res.stdout)
    assert err['G'] < 1e-12 and err['G0'] < 1e-12 and err['G2'] < 1e-12
    assert err['omega_abs'] < 1e-11 and err['omega_rel'] < 1e-9
    assert err['root_count_mismatch'] == 0 and err['roots'] < 1e-12
    assert err['phase_mismatch'] == 0 and err['M_star'] < 1e-12
    assert err['M_vac'] < 1e-12 and err['peaks'] < 1e-12
