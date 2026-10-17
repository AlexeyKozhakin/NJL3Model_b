"""
Статья 2: карты глобального минимума по (M, b) на сетке (mu, L), g = -1.

Для каждой точки — njl3.full.minimize.find_minimum: фаза ('V', 'I<j>' при b = 0
или 'B<n>' у порога S_n), M*, ln M* (экспоненциально малые щели), b*, Omega*,
флаг асимптотики.

--T T: порог по температуре — щели, не переживающие T, считаются нулевыми
(njl3.full.minimize: T_c,n = (e^γ/π)(S_n/mu) M*_n); каналы ниже порога не обсчитываются.

Выход: notebooks/output/53_full_maps.npz и (компактно) web/full/data/maps.json;
при T > 0 — notebooks/output/53_full_maps_T<T>.npz и .json (данные приложения не трогаются).
Запуск:  python notebooks/53_full_maps.py [--n N_MU N_L] [--T T]
"""

import json
import os
import sys
import time
from multiprocessing import Pool

# один поток BLAS на процесс: параллельность — по процессам (иначе потоки numpy дерутся за ядра)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full.minimize import find_minimum

G = -1.0
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
args = sys.argv[1:]
N_MU, N_L = (int(args[args.index('--n') + 1]), int(args[args.index('--n') + 2])) if '--n' in args else (60, 60)
T = float(args[args.index('--T') + 1]) if '--T' in args else 0.0
MU = np.linspace(0.05, 7.0, N_MU)
L = np.linspace(0.1, 10.0, N_L)


def point(ij):
    i, j = ij
    r = find_minimum(float(MU[j]), float(L[i]), G, T=T)
    return i, j, (r.phase, r.M_star, r.ln_M_star, r.b_star, r.omega_star, r.asymptotic)


def main():
    t0 = time.time()
    res = [[None] * N_MU for _ in range(N_L)]
    # по точкам, тяжёлые (большие L·mu — много порогов) первыми: равномерная загрузка процессов
    jobs = sorted(((i, j) for i in range(N_L) for j in range(N_MU)), key=lambda q: -L[q[0]] * MU[q[1]])
    with Pool() as pool:
        for k, (i, j, out) in enumerate(pool.imap_unordered(point, jobs, chunksize=4)):
            res[i][j] = out
            if (k + 1) % (N_MU * 6) == 0:
                print(f"  {k + 1}/{len(jobs)} точек, {time.time() - t0:.0f} с", flush=True)
    phase = np.array([[c[0] for c in r] for r in res])
    M = np.array([[c[1] for c in r] for r in res])
    lnM = np.array([[c[2] for c in r] for r in res])
    b = np.array([[c[3] for c in r] for r in res])
    Om = np.array([[c[4] for c in r] for r in res])
    asym = np.array([[c[5] for c in r] for r in res])
    tag = f'_T{T:g}' if T > 0 else ''
    np.savez(os.path.join(ROOT, 'notebooks', 'output', f'53_full_maps{tag}.npz'), mu=MU, L=L, phase=phase, M=M,
             lnM=lnM, b=b, omega=Om, asym=asym, T=T)
    os.makedirs(os.path.join(ROOT, 'web', 'full', 'data'), exist_ok=True)
    r9 = lambda x: None if not np.isfinite(x) else float(f'{x:.9g}')
    out_json = (os.path.join(ROOT, 'web', 'full', 'data', 'maps.json') if T == 0
                else os.path.join(ROOT, 'notebooks', 'output', f'53_full_maps{tag}.json'))
    with open(out_json, 'w') as f:
        json.dump(dict(g=G, T=T, mu=[r9(x) for x in MU], L=[r9(x) for x in L],
                       phase=phase.tolist(), lnM=[[r9(x) for x in r] for r in lnM],
                       b=[[r9(x) for x in r] for r in b], omega=[[r9(x) for x in r] for r in Om],
                       asym=asym.astype(int).tolist()), f, separators=(',', ':'))
    print(f"Готово за {time.time() - t0:.0f} с")


if __name__ == '__main__':
    main()
