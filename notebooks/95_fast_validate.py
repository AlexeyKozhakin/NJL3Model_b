"""
Validation of njl3.fast against the reference maps of njl3.inh (web/lab/data/inh/maps_phi*_g*_*.json):
recompute the explicitly computed (non-interpolated) grid points, compare phase, ln M*, b*, time.

    python notebooks/95_fast_validate.py [--every K] [maps ...]
→ notebooks/output/95_fast_validate.json
"""
import glob
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def point(a):
    warnings.simplefilter('ignore')
    from njl3.fast.minimize import find_minimum
    mu, L, g, phi, T = a
    t = time.perf_counter()
    r = find_minimum(mu, L, g, phi, T)
    return r.phase, r.ln_M_star, r.b_star, time.perf_counter() - t, r.omega_star


def main():
    args = sys.argv[1:]
    every = int(args[args.index('--every') + 1]) if '--every' in args else 1
    files = [a for a in args if a.endswith('.json')] or sorted(glob.glob(os.path.join(ROOT, 'web/lab/data/inh/maps_*.json')))
    report = {}
    with Pool() as pool:
        for fn in files:
            d = json.load(open(fn))
            pts = [(i, j) for i in range(len(d['L'])) for j in range(len(d['mu'])) if not d['interp'][i][j]][::every]
            jobs = [(d['mu'][j], d['L'][i], d['g'], d['phi'], d['T']) for i, j in pts]
            t0 = time.time()
            res = pool.map(point, jobs, chunksize=8)
            wall = time.time() - t0
            ph_bad, dl, db, bad = 0, [], [], []
            ref_better = fast_better = 0
            ref_list = []
            for (i, j), (ph, lnM, b, _, om) in zip(pts, res):
                rph, rl, rb = d['phase'][i][j], d['lnM'][i][j], d['b'][i][j]
                rom = d['omega'][i][j] or 0.0
                tol = max(1e-10, 1e-3 * abs(rom))           # 6 stored digits; asymptotic wells: |Δ ln M| ~ 1e-3
                ref_better += rom < om - tol
                if rom < om - tol:
                    ref_list.append(dict(mu=d['mu'][j], L=d['L'][i], ref=[rph, rl, rb, rom, d['asym'][i][j]],
                                         fast=[ph, lnM if math.isfinite(lnM) else None, b if math.isfinite(b) else None, om]))
                fast_better += om < rom - tol
                if ph != rph:
                    ph_bad += 1
                    bad.append((d['mu'][j], d['L'][i], rph, ph))
                    continue
                if rl is not None and math.isfinite(lnM):
                    dl.append(abs(lnM - rl))
                if rb is not None and math.isfinite(b):
                    db.append(abs(b - rb))
            name = os.path.basename(fn)
            report[name] = dict(points=len(pts), phase_mismatch=ph_bad, ref_better=int(ref_better), fast_better=int(fast_better), wall_s=round(wall, 1),
                                cpu_ms_per_point=round(1e3 * float(np.mean([r[3] for r in res])), 2),
                                dlnM_max=float(max(dl, default=0)), dlnM_p99=float(np.percentile(dl, 99)) if dl else 0,
                                db_max=float(max(db, default=0)), db_p99=float(np.percentile(db, 99)) if db else 0,
                                mismatches=bad[:40], ref_deeper=ref_list[:40])
            r = report[name]
            print(f"{name}: {r['points']} pts, phase mismatch {ph_bad} ({100 * ph_bad / len(pts):.2f}%), "
                  f"Ω: reference deeper {r['ref_better']}, fast deeper {r['fast_better']}; "
                  f"|Δln M| max {r['dlnM_max']:.2e} p99 {r['dlnM_p99']:.2e}, |Δb| max {r['db_max']:.2e} p99 {r['db_p99']:.2e}, "
                  f"{r['cpu_ms_per_point']} ms/pt, wall {r['wall_s']} s", flush=True)
    json.dump(report, open(os.path.join(ROOT, 'notebooks', 'output', '95_fast_validate.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
