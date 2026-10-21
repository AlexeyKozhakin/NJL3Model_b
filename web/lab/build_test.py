"""Reference values of njl3.hom for the JS port test: python web/lab/build_test.py → web/lab/test/expected_hom.json"""
import json
import os

import numpy as np

from njl3.hom import formulas as F
from njl3.hom import find_minimum, phase_at

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
rows = []
for phi in (0.0, 0.03, 0.1, 1 / 6, 0.25, 0.4, 0.5):
    for g in (-1.0, 1.0):
        for _ in range(6):
            mu, L = float(rng.uniform(0.2, 6.0)), float(rng.uniform(0.3, 8.0))
            M, Om, _ = find_minimum(mu, L, g, phi)
            ph, _ = phase_at(mu, L, g, phi)
            Ms = [0.05, 0.7, 2.2, 4.5]
            rows.append(dict(mu=mu, L=L, g=g, phi=phi, M_star=M, omega_star=Om, phase=ph,
                             G=[F.gap_G(m, mu, L, g, phi) for m in Ms],
                             omega=[F.omega_rel(m, mu, L, g, phi) for m in Ms], Ms=Ms))
with open(os.path.join(HERE, 'test', 'expected_hom.json'), 'w') as f:
    json.dump(rows, f)
print(len(rows), 'rows')
