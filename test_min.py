from functions.minimum_Omega_b_M import find_minimum_2d
from functions.Omega_ren import Omega_ren


mu=5
L=0.5
g=-1.0

bm, Mm = find_minimum_2d(
    Omega_ren,
    bmin=0.0, bmax=10.0,
    Mmin=0.0, Mmax=10.0,
    mu=mu, L=L, g=g,
    eps=1e-3
)

print("Минимум в точке:")
print("b =", bm)
print("M =", Mm)


# Пример использования функции Omega_ren
M = Mm  # примерное значение массы
b = bm  # примерное значение магнитного поля
L = L  # примерное значение длины
mu = mu  # примерное значение химического потенциала
g = g  # примерное значение константы взаимодействия


omega_ren_value = Omega_ren(M, b, L, mu, g)

print(f"Omega_ren(b=bm, M=Mm): {omega_ren_value}")



# Пример использования функции Omega_ren
M = Mm  # примерное значение массы
b = 0  # примерное значение магнитного поля
L = L  # примерное значение длины
mu = mu  # примерное значение химического потенциала
g = g  # примерное значение константы взаимодействия


omega_ren_value = Omega_ren(M, b, L, mu, g)


print(f"Omega_ren(b=0, M=Mm): {omega_ren_value}")
