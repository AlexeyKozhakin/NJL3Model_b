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


PHIS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
GS = (-1.0, 1.0)


def maps():
    """One file per (φ, g, region): data/maps_phi<φ>_g<g>_<region>.json."""
    n = 0
    for phi in PHIS:
        for g in GS:
            for name, title in (('full', 'Вся область'), ('peaks', 'Пики')):
                tag = '' if (phi == 0.0 and g == -1.0) else f'_phi{phi:g}_g{g:g}'
                src = os.path.join(OUT, f'82_stab_minphi_{name}{tag}.npz')
                if not os.path.exists(src):
                    print('missing', src)
                    continue
                d = np.load(src)
                phase = d['phase'].astype(str)
                isI = np.char.startswith(phase, 'I')
                out = dict(title=title, phi=phi, g=g, T=1e-3, q_max=10.0,
                           mu=[round(float(x), 6) for x in d['mu']], L=[round(float(x), 6) for x in d['L']],
                           phase=[[p for p in row] for row in phase],
                           phi_min=r4(d['phi_min']), phi_min_k0=r4(d['phi_min_k0']),
                           k=[[int(v) for v in row] for row in d['k']],
                           T0_inf=[[bool(v) for v in row] for row in isI])
                path = os.path.join(HERE, 'data', f'maps_phi{phi:g}_g{g:g}_{name}.json')
                with open(path, 'w') as f:
                    json.dump(out, f, separators=(',', ':'))
                n += 1
    print(n, 'map files')


def expected():
    warnings.simplefilter('ignore')
    from njl3.hom import phase_at
    from njl3.stab import phi, phi_vac, phi_mat
    rows = []
    for ph_ in (0.0, 0.1, 0.25, 0.5):
        for mu, L in ((2.8, 2.0), (3.03, 2.0), (3.3, 2.0), (3.15, 10.0)):
            _, M = phase_at(mu, L, -1.0, ph_)
            if M == 0.0:
                continue
            for k in (0, 1, 3):
                for q1 in (0.05, 1.7, 4.0, 9.0):
                    rows.append(dict(phi=ph_, mu=mu, L=L, M=M, k=k, q1=q1, vac=phi_vac(k, q1, M, L, ph_),
                                     mat=phi_mat(k, q1, M, mu, L, ph_), T3=phi(k, q1, M, mu, L, 1e-3, ph_)))
    path = os.path.join(HERE, 'test', 'expected.json')
    with open(path, 'w') as f:
        json.dump(rows, f, indent=0)
    print(path, len(rows), 'rows')


if __name__ == '__main__':
    maps()
    expected()
