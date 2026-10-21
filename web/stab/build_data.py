"""
Data for the web app web/stab: stability maps of the homogeneous GN solution (from
notebooks/82_stab_minphi.py) and reference values of Phi for the JS test.

    python notebooks/82_stab_minphi.py                                   # full map
    python notebooks/82_stab_minphi.py --region 3.0 3.4 2 10 peaks --n 160 200
    python web/stab/build_data.py
→ web/stab/data/maps.json, web/stab/test/expected.json
"""
import json
import math
import os
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'notebooks', 'output')


def r4(a):
    return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in a]


def maps():
    out = {}
    for name, title in (('full', 'Вся область'), ('peaks', 'Пики')):
        d = np.load(os.path.join(OUT, f'82_stab_minphi_{name}.npz'))
        phase = d['phase'].astype(str)
        isI = np.char.startswith(phase, 'I')
        out[name] = dict(
            title=title, mu=[round(float(x), 6) for x in d['mu']], L=[round(float(x), 6) for x in d['L']],
            phase=[[p for p in row] for row in phase],
            M=r4(d['M']), phi_min=r4(d['phi_min']), phi_min_k0=r4(d['phi_min_k0']),
            k=[[int(v) for v in row] for row in d['k']],
            T0_inf=[[bool(v) for v in row] for row in isI],     # T = 0: min Phi = −∞ (2k_F singularity)
        )
    path = os.path.join(HERE, 'data', 'maps.json')
    with open(path, 'w') as f:
        json.dump(dict(T=1e-3, g=-1.0, q_max=10.0, maps=out), f, separators=(',', ':'))
    print(path, f'{os.path.getsize(path) / 1e6:.2f} MB')


def expected():
    warnings.simplefilter('ignore')
    from njl3.b0 import phase_at
    from njl3.stab import phi, phi_vac, phi_mat
    rows = []
    for mu, L in ((2.8, 2.0), (3.03, 2.0), (3.3, 2.0), (3.15, 10.0)):
        _, M = phase_at(mu, L, -1.0)
        for k in (0, 1, 3):
            for q1 in (0.05, 1.7, 4.0, 9.0):
                rows.append(dict(mu=mu, L=L, M=M, k=k, q1=q1, vac=phi_vac(k, q1, M, L),
                                 mat=phi_mat(k, q1, M, mu, L), T3=phi(k, q1, M, mu, L, 1e-3)))
    path = os.path.join(HERE, 'test', 'expected.json')
    with open(path, 'w') as f:
        json.dump(rows, f, indent=0)
    print(path, len(rows), 'rows')


if __name__ == '__main__':
    maps()
    expected()
