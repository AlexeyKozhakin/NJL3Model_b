import numpy as np
from functions.Omega_L import fun_Omega_L


def fun_Omega_L_phys(L_vals, b_vals, M_vals, N_h_p1=100, N_h_p2=100, DTYPE=np.float32):
     return (fun_Omega_L(L_vals, b_vals, M_vals, N_h_p1=N_h_p1, N_h_p2=N_h_p2).astype(DTYPE) -
            fun_Omega_L(L_vals, b_vals, 0, N_h_p1=N_h_p1, N_h_p2=N_h_p2).astype(DTYPE) +
            fun_Omega_L(L_vals, 0, 0, N_h_p1=N_h_p1, N_h_p2=N_h_p2).astype(DTYPE))
