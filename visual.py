import numpy as np
import matplotlib.pyplot as plt
from functions.Omega_ren import Omega_ren


def plot_Omega_vs_b(Omega_ren, 
                    M, L, mu, g, 
                    bmin, bmax, 
                    points=200):
    """
    Строит график зависимости Omega_ren(M, b, L, mu, g) от b
    в диапазоне [bmin, bmax].

    Omega_ren — твоя функция.
    points — количество точек дискретизации.
    """

    b_vals = np.linspace(bmin, bmax, points)
    omega_vals = [Omega_ren(M, b, L, mu, g)[0][0][0] for b in b_vals]

    #print(omega_vals)

    plt.figure(figsize=(8,5))
    plt.plot(b_vals, omega_vals)
    plt.xlabel("b")
    plt.ylabel("Omega_ren")
    plt.title("Omega_ren(b) при фиксированных M, L, mu, g")
    plt.grid(True)
    plt.show()


mu=6
L=5
g=-1.0
M=0

plot_Omega_vs_b(Omega_ren,
                M=M,
                L=L,
                mu=mu,
                g=g,
                bmin=0.0,
                bmax=10.0,
                points=300)
