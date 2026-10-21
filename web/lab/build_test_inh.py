"""Reference values of njl3.inh for the JS test: python web/lab/build_test_inh.py → web/lab/test/expected_inh.json"""
import json
import math
import os
import warnings

import numpy as np

from njl3.inh import find_minimum, omega_rel, s_func

warnings.simplefilter('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)
fin = lambda x: None if not math.isfinite(x) else x
out = dict(s=[], omega=[], points=[])
for phi in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    for x in list(rng.uniform(0, 64, 10)) + list(np.geomspace(1e-9, 1, 10)):
        out['s'].append(dict(phi=phi, x=float(x), s=float(s_func(np.array([x]), phi)[0])))
    for _ in range(8):
        mu, L, M, b = rng.uniform(0.5, 6), rng.uniform(0.5, 8), rng.uniform(0.01, 3), rng.uniform(0, 5)
        g = float(rng.choice([-1.0, 1.0]))
        out['omega'].append(dict(phi=phi, g=g, mu=mu, L=L, M=M, b=b, omega=omega_rel(M, b, mu, L, g, phi)))
    for mu, L, g in ((3.6, 5.0, -1.0), (4.5, 8.0, -1.0), (2.0, 1.0, -1.0), (3.15, 3.5, -1.0), (2.0, 1.0, 1.0),
                     (3.2, 2.0, -1.0)):
        r = find_minimum(mu, L, g, phi, T=1e-3)
        out['points'].append(dict(phi=phi, g=g, mu=mu, L=L, T=1e-3, phase=r.phase, lnM=fin(r.ln_M_star),
                                  b=fin(r.b_star), omega=r.omega_star))
with open(os.path.join(HERE, 'test', 'expected_inh.json'), 'w') as f:
    json.dump(out, f)
print({k: len(v) for k, v in out.items()})
