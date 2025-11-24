from functions.Omega_ren import Omega_ren
import time

if __name__ == "__main__":
    # Пример использования функции Omega_ren
    M = 0.3  # примерное значение массы
    b = 0.1  # примерное значение магнитного поля
    L = 1.0  # примерное значение длины
    mu = 2  # примерное значение химического потенциала
    g = 1.0  # примерное значение константы взаимодействия

    start = time.perf_counter()  # старт замера времени
    omega_ren_value = Omega_ren(M, b, L, mu, g)
    end = time.perf_counter()    # конец замера времени

    print(f"Omega_ren: {omega_ren_value}")
    print(f"Время выполнения: {end - start:.6f} секунд")
