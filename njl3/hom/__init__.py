"""
njl3.hom — homogeneous NJL3/GN potential on R² × S¹ (b = 0) for an arbitrary twist φ of the boundary
condition and an arbitrary coupling g; certified global minimum. At φ = 0 coincides with njl3.b0.

    >>> from njl3.hom import analyze_point
    >>> r = analyze_point(mu=3.0, L=2.0, g=-1.0, phi=0.25)
"""

from njl3.hom.formulas import (
    gap_G, gap_G0, gap_G_vec, levels, modes, omega_rel, omega_rel_closed, reduce_phi, thresholds,
)
from njl3.hom.minimize import (
    PointResult, M_vacuum, analyze_point, find_minimum, isolate_roots, phase_at, phase_label,
)

__all__ = ['gap_G', 'gap_G0', 'gap_G_vec', 'levels', 'modes', 'omega_rel', 'omega_rel_closed',
           'reduce_phi', 'thresholds', 'PointResult', 'M_vacuum', 'analyze_point', 'find_minimum',
           'isolate_roots', 'phase_at', 'phase_label']
