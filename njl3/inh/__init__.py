"""
njl3.inh — Omega(M, b) on R² × S¹ (chiral spiral) for a twist φ ∈ {0, 0.1, …, 0.5} of the boundary
condition and any g; global minimum with log asymptotics and temperature cut-off. At φ = 0 coincides
with njl3.full.
"""

from njl3.inh.formulas import PHIS, levels, omega_rel, omega_rel_vec, s_func
from njl3.inh.minimize import find_minimum, thresholds

__all__ = ['PHIS', 'levels', 'omega_rel', 'omega_rel_vec', 's_func', 'find_minimum', 'thresholds']
