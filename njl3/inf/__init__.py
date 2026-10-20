"""
njl3.inf — NJL3 in 2+1 dimensions without compactification (L -> infinity), phi = 0.

    Omega(M, b) = M²/(2g) + ΔV_phys(M, b) + W_mu(M, b),
    W_mu = -∫dp1/2π ∫dp2/2π Σ± (mu - sqrt((E1 ± b)² + p2²))₊,  E1 = sqrt(M² + p1²).

This is the L -> infinity limit of the problem on R² × S¹ (njl3.b0, njl3.full).
"""

from njl3.inf.formulas import M_vacuum, W_b0, gap_G_b0, omega_rel_b0
from njl3.inf.minimize import ResultB0, find_minimum_b0

__all__ = ['M_vacuum', 'W_b0', 'gap_G_b0', 'omega_rel_b0', 'ResultB0', 'find_minimum_b0']
