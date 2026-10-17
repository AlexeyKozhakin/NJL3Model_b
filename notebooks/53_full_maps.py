"""
Статья 2: карты глобального минимума по (M, b) на сетке (mu, L), g = -1.

Для каждой точки — njl3.full.minimize.find_minimum: фаза ('V', 'I<j>' при b = 0
или 'B<n>' у порога S_n), M*, ln M* (экспоненциально малые щели), b*, Omega*,
флаг асимптотики.

Выход: notebooks/output/53_full_maps.npz и (компактно) web/full/data/maps.json
Запуск:  python notebooks/53_full_maps.py [--n N_MU N_L]
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full.minimize import find_minimum

G = -1.0
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
args = sys.argv[1:]
N_MU, N_L = (int(args[1]), int(args[2])) if args and args[0] == '--n' else (60, 60)
MU = np.linspace(0.05, 7.0, N_MU)
L = np.linspace(0.1, 10.0, N_L)


def row(i):
    out = []
    for mu in MU:
        r = find_minimum(float(mu), float(L[i]), G)
        out.append((r.phase, r.M_star, r.ln_M_star, r.b_star, r.omega_star, r.asymptotic))
    return i, out


def main():
    t0 = time.time()
    res = [None] * N_L
    with Pool() as pool:
        for k, (i, out) in enumerate(pool.imap_unordered(row, range(N_L))):
            res[i] = out
            print(f"  строка {k + 1}/{N_L} (L = {L[i]:.3f}), {time.time() - t0:.0f} с", flush=True)
    phase = np.array([[c[0] for c in r] for r in res])
    M = np.array([[c[1] for c in r] for r in res])
    lnM = np.array([[c[2] for c in r] for r in res])
    b = np.array([[c[3] for c in r] for r in res])
    Om = np.array([[c[4] for c in r] for r in res])
    asym = np.array([[c[5] for c in r] for r in res])
    np.savez(os.path.join(ROOT, 'notebooks', 'output', '53_full_maps.npz'), mu=MU, L=L, phase=phase, M=M,
             lnM=lnM, b=b, omega=Om, asym=asym)
    os.makedirs(os.path.join(ROOT, 'web', 'full', 'data'), exist_ok=True)
    r9 = lambda x: None if not np.isfinite(x) else float(f'{x:.9g}')
    with open(os.path.join(ROOT, 'web', 'full', 'data', 'maps.json'), 'w') as f:
        json.dump(dict(g=G, mu=[r9(x) for x in MU], L=[r9(x) for x in L],
                       phase=phase.tolist(), lnM=[[r9(x) for x in r] for r in lnM],
                       b=[[r9(x) for x in r] for r in b], omega=[[r9(x) for x in r] for r in Om],
                       asym=asym.astype(int).tolist()), f, separators=(',', ':'))
    print(f"Готово за {time.time() - t0:.0f} с")


if __name__ == '__main__':
    main()
