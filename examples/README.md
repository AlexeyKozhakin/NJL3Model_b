# Test-run examples

Inputs with their expected output, for checking an installation of `njl3.b0`.

```bash
pip install -e ".[plot,test]"
python examples/run_examples.py            # all examples (~1.5 min), output in examples/output/
python examples/run_examples.py --fast     # without the slow one (~5 s)
```

Every output is compared with `examples/expected/` number by number (relative tolerance
`1e-9`); the script prints `OK` or the mismatches. The same check runs in the test suite
(`pytest tests/b0`, the slow example with `-m slow`).

| # | Command | What it shows | Time |
|---|---|---|---|
| 01 | `njl3-b0 point --mu 3.15 --L 3.5` | phase I1: global minimum, all roots of G, local minima | < 1 s |
| 02 | `njl3-b0 point --mu 3.5 --L 2 --json` | trivial phase next to a metastable vacuum minimum; JSON output | < 1 s |
| 03 | `njl3-b0 peaks --L-max 10` | peaks of the phase boundary from the explicit formula | < 1 s |
| 04 | `njl3-b0 diagram --mu 2.5 4 --L 0.1 4 --n 60 60 --out 04_diagram.csv` | map of M* and phases on a 60×60 grid | ~1 s |
| 05 | `python examples/api_example.py` | Python API: point, vacuum mass, stability edge, transitions along mu, peaks | < 1 s |
| 06 | `njl3-b0 lines --mu 0.05 7 --L 0.1 10 --nL 40 --out 06_lines.json` | transition lines and special points (peaks, critical end points) | ~1–2 min |

`--update` regenerates `examples/expected/` (for maintainers, after a deliberate change).
