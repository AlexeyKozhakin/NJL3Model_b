import numpy as np
from functions.Omega_mu_L import fun_Omega_L_mu_int


def fun_Omega_mu_L_phys(mu, L, b, M, phi=0):
     return (fun_Omega_L_mu_int(mu, L, b, M, phi=0)
                  - fun_Omega_L_mu_int(mu, L, b, 0, phi=0)
                  + fun_Omega_L_mu_int(mu, L, 0, 0, phi=0))