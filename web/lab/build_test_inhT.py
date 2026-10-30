"""Reference values for the JS port web/lab/js/njl3inhT.js (Omega at T > 0, global minimum).

    python web/lab/build_test_inhT.py   → web/lab/test/expected_inhT.json
"""
import json
import math
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
warnings.simplefilter('ignore')

from njl3.fast.minimize_T import find_minimum
from njl3.fast.thermal import CtxT

omega = []
for M, b, mu, L, phi, T in [(1.0, 0.0, 2.0, 2.0, 0.0, 0.01), (0.5, 1.2, 3.3, 5.0, 0.0, 0.01), (0.05, 3.1, 3.2, 6.0, 0.0, 0.001),
                            (0.3, 2.0, 3.0, 4.0, 0.3, 0.01), (0.02, 2.9, 3.15, 8.0, 0.1, 0.001), (2.0, 0.5, 3.0, 1.5, 0.5, 0.01),
                            (3.14, 0.0, 3.04, 2.4, 0.0, 0.01), (0.08, 2.58, 3.6, 5.0, 0.0, 0.001)]:
    for g in (-1.0, 1.0):
        omega.append(dict(M=M, b=b, mu=mu, L=L, phi=phi, T=T, g=g, omega=CtxT(mu, L, g, phi, T).omega(M, b)))
points = []
for mu, L in [(2.0, 2.0), (3.3, 2.0), (3.6, 5.0), (3.15, 6.0), (3.04, 2.4), (5.0, 1.0), (1.0, 0.5), (4.0, 2.0), (6.0, 8.0)]:
    for T in (0.001, 0.01):
        r = find_minimum(mu, L, -1.0, 0.0, T)
        points.append(dict(mu=mu, L=L, g=-1.0, phi=0.0, T=T, phase=r.phase, M=r.M_star, b=r.b_star if math.isfinite(r.b_star) else None,
                           omega=r.omega_star))
with open(os.path.join(HERE, 'test', 'expected_inhT.json'), 'w') as f:
    json.dump(dict(omega=omega, points=points), f, indent=0)
print(len(omega), 'omega,', len(points), 'points')
