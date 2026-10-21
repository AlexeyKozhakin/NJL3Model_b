# NJL3Model_b

Phase structure of the Nambu–Jona-Lasinio model in 2+1 dimensions on $\mathbb{R}^2\times S^1$
at chemical potential $\mu$ and compactification length $L$.

The package **`njl3.b0`** solves the case $b=0$ ($\varphi=0$, $g<0$):

- closed-form expressions for the renormalised potential $\Omega(M)$ and the gap function $G(M)$;
- a **certified** search for the global minimum in $M$: all roots of the gap equation are
  provably found;
- phase-transition lines (first and second order) as solutions of equations, to $\sim10^{-12}$;
- special points: peaks (explicit formula) and critical end points.

The general case $(M,b)$ is developed in `njl3.full` (work in progress).

## Installation

Python ≥ 3.10.

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[plot,test]"
```

## Quick start: command line

```bash
njl3-b0 point --mu 3.15 --L 3.5          # one point
njl3-b0 peaks --L-max 10                 # peaks mu_n = 2πn/L_n from the explicit formula
njl3-b0 diagram --mu 2.5 4 --L 0.1 10 --n 300 300 --out map.npz --plot map.png --vmax 4
njl3-b0 lines --mu 0.05 7 --L 0.1 10 --nL 400 --out lines.json --plot lines.png
```

The same interface is available as `python -m njl3.b0 ...`. The default is $g=-1$ (option `--g`).

Example output of `point`:

```
μ = 3.15, L = 3.5, g = -1.0
  phase       : I1
  M*          : 0.893548965789433
  Ω(M*) − Ω(0): -2.848978e-03
  G(0)        : -0.0146140160   (M = 0 unstable)
  G2          : +0.0195714647
  roots of G  : 0.893548965789 (min), 2.51079672677 (max), 2.9122649966 (min)
  minima      : M=0.8935489658, ΔΩ=-2.8490e-03, M=2.912264997, ΔΩ=3.2166e-02
```

## Python API

```python
from njl3.b0 import analyze_point, find_minimum, trace_L, peak, mu_stability

r = analyze_point(mu=3.15, L=3.5, g=-1.0)
r.phase, r.M_star, r.roots          # 'I1', 0.8935..., [(M, 'min'|'max'|'touch'), ...]

M_star, dOmega, minima = find_minimum(3.15, 3.5, -1.0)
trace_L(3.5, -1.0)                  # transitions in mu at L = 3.5: V→I0, I0→I1, I1→T (second order)
peak(2, -1.0)                       # (L_2, mu_2) = (3.7293741534, 3.3695655350)
mu_stability(1.0, -1.0)             # stability edge of M = 0: 2.4052386905
```

## Phases

| Label | Condition | Meaning |
|---|---|---|
| `V` | $M^*=M_{\rm vac}(L)\ge\mu$ | vacuum: the Fermi sea is empty; independent of $\mu$ |
| `I<j>` | $0<M^*<\mu$ | intermediate: Kaluza–Klein modes $n=0,\dots,j$ are filled |
| `T` | $M^*=0$ | trivial (chirally symmetric) |

## Data formats

**`diagram`** (`.npz`, `.json` or `.csv`). Arrays of shape `(N_L, N_mu)`:

| Field | Content |
|---|---|
| `mu`, `L` | grids |
| `M_star` | global minimum $M^*$ |
| `omega_star` | $\Omega(M^*)-\Omega(0)$ |
| `phase` | phase label |
| `n_minima` | number of local minima with $M>0$ |

The CSV has one row per point: `L, mu, M_star, omega_star, phase, n_minima`.

**`lines`** (`.json`):

```json
{
  "g": -1.0, "mu_range": [0.05, 7.0], "L_range": [0.1, 10.0],
  "lines": {
    "V-T":   [{"L": ..., "mu": ..., "order": 1, "M_left": ..., "M_right": ...}, ...],
    "I1-T":  [...], "I0-I1": [...], ...
  },
  "special": [
    {"kind": "peak", "n": 1, "L": 1.6113000800, "mu": 3.8994507512},
    {"kind": "critical_end", "L": 1.2226473, "mu": 2.7909025},
    {"kind": "tricritical", "L": ..., "mu": ..., "on_phase_boundary": false}
  ]
}
```

`order`: 1 — first-order transition (jump in $M^*$), 2 — second order.

## Method (summary)

1. At $b=0$ every term of $\Omega$ is expressed through elementary functions and
   polylogarithms. $d\Omega/dM=M\,G(M)$, and the logarithmic singularities at $M=0$ cancel.
2. On $[0,\mu]$, $G=1/g+A(M)+B(M)$ with $A$ increasing and $B$ decreasing. This gives the
   bounds $1/g+A(p)+B(q)\le G\le 1/g+A(q)+B(p)$ on $[p,q]$, and bisection with pruning finds
   **all** roots. On $[\mu,\infty)$, $G$ is monotone: at most one root.
3. Near $M=0$ the stable forms of $G$, $G'$ and $\Omega(M)-\Omega(0)=\int_0^M m\,G\,dm$ are used.

Accuracy against mpmath (50 digits): $G$ — $10^{-14}$, roots — $7\cdot10^{-15}$,
lines — $\le3\cdot10^{-13}$. The method is described in detail in the accompanying paper.

## Web application

`web/b0/` is an interactive phase diagram in the browser (no build step, no compute server).
The map $M^*(L,\mu)$ is computed in background workers with the same formulas (JS port
`web/b0/js/njl3b0.js`, agrees with Python to $\sim10^{-13}$), with the exact transition lines
from `web/b0/data/phase_lines.json` drawn on top. Clicking a point shows $\Omega(M)$, $G(M)$,
the roots and the minima. The view is stored in the URL; the map (CSV), the lines and the point
(JSON) can be exported.

Online: <https://alexeykozhakin.github.io/NJL3Model_b/web/b0/>

Locally (any HTTP server; background workers do not run from `file://`):

```bash
python -m http.server 8000      # then open http://localhost:8000/web/b0/
python web/b0/build_data.py     # recompute the transition lines (a few minutes)
```

## Tests

```bash
pytest tests/b0 -m "not slow"     # fast, ~1 s
pytest tests/b0                   # all, incl. direct numerical integration and the JS port (needs Node.js)
```

## Repository layout

| Directory | Content |
|---|---|
| `njl3/b0/` | package for $b=0$: `formulas`, `minimize`, `lines`, `reference` (mpmath), `cli` |
| `tests/b0/` | tests of the package |
| `web/b0/` | web application: phase diagram at $b=0$ |
| `njl3/stab/`, `tests/stab/` | stability of the homogeneous GN solution: second variation $\Phi(k, q_1)$ |
| `web/stab/` | web application: stability map and $\Phi(q_1)$ at a point (<https://alexeykozhakin.github.io/NJL3Model_b/web/stab/>) |
| `njl3/hom/`, `tests/hom/` | homogeneous case for an arbitrary twist $\varphi$ and coupling $g$ (certified minimum; $\varphi=0$ coincides with `njl3.b0`) |
| `web/lab/` | web application with tabs: homogeneous phase diagram for any $\varphi$, $g$; stability (<https://alexeykozhakin.github.io/NJL3Model_b/web/lab/>) |
| `njl3/full/`, `web/full/`, `tests/full/` | general case $(M,b)$ (work in progress) |
| `functions/`, `gpu/` | earlier numerical integrators for $(M,b)$ (CPU/numba, PyTorch) |
| `web/` | earlier web calculator for the general case (JavaScript) |
| `notebooks/` | research scripts |
| `docs/` | theory notes and reports |
