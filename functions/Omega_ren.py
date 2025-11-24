from functions.Omega_mu_L_phys import fun_Omega_mu_L_phys
from functions.Omega_L_phys import fun_Omega_L_phys as Omega_L_phys
from functions.dU import fun_dU_phys as dU_phys
import time


# def Omega_ren(M,b,L,mu,g=1.0):
#     start = time.perf_counter()  # старт замера времени
#     dU = dU_phys(b, M)
#     end = time.perf_counter()    # конец замера времени
#     print(f"Время выполнения: {end - start:.6f} секунд")
    
#     start = time.perf_counter()  # старт замера времени
#     Omega_L = Omega_L_phys(L, b, M)
#     end = time.perf_counter()    # конец замера времени
#     print(f"Время выполнения: {end - start:.6f} секунд")

#     start = time.perf_counter()  # старт замера времени
#     Omega_mu_L = fun_Omega_mu_L_phys(mu, L, b, M)
#     end = time.perf_counter()    # конец замера времени
#     print(f"Время выполнения Omega_mu_L_phys: {end - start:.6f} секунд")
    
#     return (M**2 / (2 * g) +
#             dU +
#             Omega_L +
#             Omega_mu_L)

def Omega_ren(M,b,L,mu,g=1.0):

    dU = dU_phys(b, M)
    
    Omega_L = Omega_L_phys(L, b, M)

    Omega_mu_L = fun_Omega_mu_L_phys(mu, L, b, M)
    
    return (M**2 / (2 * g) +
            dU +
            Omega_L +
            Omega_mu_L)
