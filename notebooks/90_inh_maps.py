"""
Карты глобального минимума по (M, b) на сетке (mu, L) для граничного условия φ и знака g
(пакет njl3.inh; копия notebooks/53_full_maps.py с параметрами --phi, --g).

Для каждой точки — njl3.full.minimize.find_minimum: фаза ('V', 'I<j>' при b = 0
или 'B<n>' у порога S_n), M*, ln M* (экспоненциально малые щели), b*, Omega*,
флаг асимптотики.

--refine K: адаптивное уточнение — K раз делить пополам ячейки, где соседние точки в разных
фазах или ln M* меняется больше чем на REFINE_DLNM; итоговая сетка (2^K (N − 1) + 1)², внутри
однородных ячеек — билинейная интерполяция (флаг interp).

--T T: порог по температуре — щели, не переживающие T, считаются нулевыми
(njl3.full.minimize: T_c,n = (e^γ/π)(S_n/mu) M*_n); каналы ниже порога не обсчитываются.

Выход: notebooks/output/53_full_maps<tag>.npz и .json; с --web json пишется в данные
приложения web/full/data/maps.json.
--region MU0 MU1 L0 L1 NAME: только эта область (подробная карта для приложения),
выход notebooks/output/53_full_maps_<NAME>.json.

Запуск:  python notebooks/90_inh_maps.py --phi PHI --g G [--n N_MU N_L] [--T T] [--refine K] [--region MU0 MU1 L0 L1 NAME]
→ web/lab/data/inh/maps_phi<PHI>_g<G>_<NAME|full>.json
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

from njl3.inh import find_minimum

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
args = sys.argv[1:]
PHI = float(args[args.index('--phi') + 1]) if '--phi' in args else 0.0
G = float(args[args.index('--g') + 1]) if '--g' in args else -1.0
N_MU, N_L = (int(args[args.index('--n') + 1]), int(args[args.index('--n') + 2])) if '--n' in args else (60, 60)
T = float(args[args.index('--T') + 1]) if '--T' in args else 0.0
K = int(args[args.index('--refine') + 1]) if '--refine' in args else 0
REFINE_DLNM = 1.0
if '--region' in args:
    _r = args[args.index('--region') + 1:args.index('--region') + 6]
    MU_RANGE, L_RANGE, REGION = (float(_r[0]), float(_r[1])), (float(_r[2]), float(_r[3])), _r[4]
else:
    MU_RANGE, L_RANGE, REGION = (0.05, 7.0), (0.1, 10.0), None
R = 2 ** K
NF_MU, NF_L = R * (N_MU - 1) + 1, R * (N_L - 1) + 1          # итоговая (тонкая) сетка
MU = np.linspace(*MU_RANGE, NF_MU)
L = np.linspace(*L_RANGE, NF_L)


def point(ij):
    i, j = ij
    r = find_minimum(float(MU[j]), float(L[i]), G, PHI, T=T)
    return i, j, (r.phase, r.M_star, r.ln_M_star, r.b_star, r.omega_star, r.asymptotic)


def compute(pool, pts, res, t0, label):
    # тяжёлые (большие L·mu — много порогов) первыми: равномерная загрузка процессов
    jobs = sorted(pts, key=lambda q: -L[q[0]] * MU[q[1]])
    for k, (i, j, out) in enumerate(pool.imap_unordered(point, jobs, chunksize=4)):
        res[(i, j)] = out
    print(f"  {label}: {len(jobs)} точек, всего {len(res)}, {time.time() - t0:.0f} с", flush=True)


def needs_refine(res, corners):
    ph = {res[c][0] for c in corners}
    if len(ph) > 1:
        return True
    lm = [res[c][2] for c in corners if np.isfinite(res[c][2])]
    return len(lm) > 1 and max(lm) - min(lm) > REFINE_DLNM


def main():
    t0 = time.time()
    res = {}
    with Pool() as pool:
        compute(pool, [(i, j) for i in range(0, NF_L, R) for j in range(0, NF_MU, R)], res, t0, 'базовая сетка')
        cells = [(i, j) for i in range(0, NF_L - 1, R) for j in range(0, NF_MU - 1, R)]
        s = R
        unrefined = []                                       # (i, j, s) — ячейки без уточнения
        while s > 1:
            ref = []
            for i, j in cells:
                corners = [(i, j), (i + s, j), (i, j + s), (i + s, j + s)]
                (ref if needs_refine(res, corners) else unrefined).append((i, j, s))
            h = s // 2
            new = {(i + a, j + c) for i, j, _ in ref for a in (0, h, s) for c in (0, h, s)} - res.keys()
            compute(pool, sorted(new), res, t0, f'уточнение шаг {h}: {len(ref)} ячеек')
            cells = [(i + a, j + c) for i, j, _ in ref for a in (0, h) for c in (0, h)]
            s = h
    # остальное — билинейная интерполяция внутри однородных ячеек (фаза одна)
    phase = np.empty((NF_L, NF_MU), dtype=object)
    M, lnM, b, Om = (np.full((NF_L, NF_MU), np.nan) for _ in range(4))
    asym = np.zeros((NF_L, NF_MU), dtype=bool)
    interp = np.ones((NF_L, NF_MU), dtype=bool)
    for (i, j), c in res.items():
        phase[i, j], M[i, j], lnM[i, j], b[i, j], Om[i, j], asym[i, j] = c
        interp[i, j] = False
    for i, j, s in unrefined:
        C = [res[(i, j)], res[(i + s, j)], res[(i, j + s)], res[(i + s, j + s)]]
        for a in range(s + 1):
            for c in range(s + 1):
                if not interp[i + a, j + c]:
                    continue
                u, v = a / s, c / s
                w = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]
                bil = lambda k: sum(wk * Ck[k] for wk, Ck in zip(w, C)) if all(np.isfinite(Ck[k]) for Ck in C) else (
                    C[0][k] if not np.isfinite(C[0][k]) else np.nan)
                phase[i + a, j + c] = C[0][0]
                lnM[i + a, j + c] = bil(2)
                M[i + a, j + c] = np.exp(lnM[i + a, j + c]) if np.isfinite(lnM[i + a, j + c]) else 0.0
                b[i + a, j + c], Om[i + a, j + c] = bil(3), bil(4)
                asym[i + a, j + c] = all(Ck[5] for Ck in C)
    phase = phase.astype(str)
    print(f"сетка {NF_L}×{NF_MU}: посчитано {len(res)} точек ({100 * len(res) / interp.size:.0f}%), "
          f"остальное — интерполяция", flush=True)
    name = REGION or 'full'
    out_dir = os.path.join(ROOT, 'web', 'lab', 'data', 'inh')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f'maps_phi{PHI:g}_g{G:g}_{name}.json')
    r6 = lambda x: None if not np.isfinite(x) else float(f'{x:.6g}')
    r9 = lambda x: None if not np.isfinite(x) else float(f'{x:.9g}')
    with open(out_json, 'w') as f:
        json.dump(dict(phi=PHI, g=G, T=T, refine=K, region=name, mu=[r9(x) for x in MU], L=[r9(x) for x in L],
                       phase=phase.tolist(), lnM=[[r6(x) for x in r] for r in lnM],
                       b=[[r6(x) for x in r] for r in b], omega=[[r6(x) for x in r] for r in Om],
                       asym=asym.astype(int).tolist(), interp=interp.astype(int).tolist()), f, separators=(',', ':'))
    print(f"Готово за {time.time() - t0:.0f} с ({out_json})", flush=True)


if __name__ == '__main__':
    main()
