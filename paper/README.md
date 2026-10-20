# Reproducing the paper

Every figure and table of the paper on the $b=0$ phase diagram is produced by one script that
uses only the `njl3.b0` package (plus scipy, mpmath and matplotlib):

```bash
pip install -e ".[plot]"
python paper/reproduce.py              # everything, ~15 min on 8 cores (mostly the transition lines)
python paper/reproduce.py --quick      # smaller grids and samples, ~1-2 min
python paper/reproduce.py fig2 tab-scan   # selected targets
python paper/reproduce.py --ru --print     # journal version: Russian labels, colour readable in greyscale print
```

Output goes to `paper/output/` (figures as PNG and PDF, tables as Markdown and CSV).

| Target | Content |
|---|---|
| `fig1` | transition lines on the whole range and in the strip $\mu\in[2.5,4]$, with peaks, critical end points and mode thresholds |
| `fig2` | global minimum $M^*(L,\mu)$ in the peak region $\mu\in[3,3.4]$, $L\in[2,10]$ |
| `fig3` | $G(M)$ from the closed form against finite differences of direct numerical integration |
| `fig4` | slices $M^*(\mu)$ at $L=3.5$ and $M^*(L)$ at $\mu=3.15$: sequence of phases and order of every transition |
| `tab-special` | special points: critical end points and peaks |
| `tab-accuracy` | double precision against mpmath (50 digits): $G$, $\Omega$, roots, analytic and traced lines |
| `tab-scan` | certified search against a grid scan: missed minima, wrong global minima, time per point |
| `tab-perf` | performance: single operations (Python and the JS core in Node.js), a map, the transition lines |

`fig1`, `tab-special` and `tab-accuracy` share the traced lines, cached in `paper/output/lines.json`.
