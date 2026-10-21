"""
njl3.stab — stability of the homogeneous GN solution on R² × S¹ (phi = 0) against inhomogeneous
fluctuations: the second variation Phi(k, q1) at M = M*(mu, L) from njl3.b0.

Phi < 0 for some (k, q1) means that the homogeneous phase is unstable.
"""

from njl3.stab.formulas import (
    channels, numerator, phi, phi_direct_mat, phi_direct_vac, phi_mat, phi_vac, singular_q1,
)

__all__ = ['channels', 'numerator', 'phi', 'phi_direct_mat', 'phi_direct_vac', 'phi_mat', 'phi_vac', 'singular_q1']
