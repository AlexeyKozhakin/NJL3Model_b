"""Python API of njl3.b0: one point, the phase sequence along mu at fixed L, peaks, stability edge."""

from njl3.b0 import M_vacuum, analyze_point, mu_stability, peaks, trace_L

g = -1.0

r = analyze_point(mu=3.15, L=3.5, g=g)
print(f'point mu = 3.15, L = 3.5: phase {r.phase}, M* = {r.M_star:.12f}, Omega* - Omega(0) = {r.omega_star:.10e}')
print('  local minima:', ', '.join(f'M = {m:.10f} (dOmega = {o:.6e})' for m, o in r.local_minima))

print(f'vacuum mass M_vac(L = 3.5) = {M_vacuum(3.5, g):.12f}')
print(f'stability edge of M = 0 at L = 1: mu_s = {mu_stability(1.0, g):.12f}')

print('transitions in mu at L = 3.5:')
for t in trace_L(3.5, g):
    print(f"  {t['frm']:>3} -> {t['to']:<3} at mu = {t['mu']:.10f}, order {t['order']}")

print('peaks with L_n <= 6:')
for n, Ln, mun in peaks(g, 6.0):
    print(f'  n = {n}: L_n = {Ln:.10f}, mu_n = {mun:.10f}')
