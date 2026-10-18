"""
njl3.b0 — phase diagram of the NJL3 model on R² × S¹ at b = 0, phi = 0.

Closed-form potential, certified search for the global minimum in M,
phase-transition lines and special points.

    >>> from njl3.b0 import analyze_point
    >>> r = analyze_point(mu=3.5, L=2.0, g=-1.0)
    >>> r.phase, r.M_star

Command line: ``njl3-b0 --help`` (or ``python -m njl3.b0 --help``).
"""

from njl3.b0.formulas import (
    G2, dU, gap_G, gap_G0, gap_G_vec, gap_dG_vec, modes, omega_L, omega_muL,
    omega_rel, omega_rel_closed, thresholds,
)
from njl3.b0.minimize import (
    PointResult, M_vacuum, analyze_point, find_minimum, find_minimum_scan,
    isolate_roots, phase_at, phase_label,
)
from njl3.b0.lines import (
    compute_grid, mu_stability, mu_VT, peak, peaks, trace_L, trace_lines,
    tricritical_points,
)

__all__ = [
    'G2', 'dU', 'gap_G', 'gap_G0', 'gap_G_vec', 'gap_dG_vec', 'modes', 'omega_L',
    'omega_muL', 'omega_rel', 'omega_rel_closed', 'thresholds',
    'PointResult', 'M_vacuum', 'analyze_point', 'find_minimum', 'find_minimum_scan',
    'isolate_roots', 'phase_at', 'phase_label',
    'compute_grid', 'mu_stability', 'mu_VT', 'peak', 'peaks', 'trace_L',
    'trace_lines', 'tricritical_points',
]
