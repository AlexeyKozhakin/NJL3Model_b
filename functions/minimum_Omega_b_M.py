import numpy as np

def find_minimum_2d(Omega_ren, 
                    bmin, bmax, 
                    Mmin, Mmax, 
                    mu, L, g, 
                    eps=1e-3):
    """
    Поиск минимума функции Omega_ren(M, b, L, mu, g)
    в прямоугольной области двумя этапами:
      1) Грубое сканирование на сетке 10×10
      2) Уточнение минимизации через сужение области (аналог золотого сечения)
    
    Возвращает:
        bm, Mm — координаты точки минимума
    """

    # ---------------------------------------------------------
    # 1) ГРУБЫЙ ПОИСК НА 10×10 СЕТКЕ
    # ---------------------------------------------------------
    Ns = 2
    b_vals = np.linspace(bmin, bmax, Ns)
    M_vals = np.linspace(Mmin, Mmax, Ns)

    best_val = float("inf")
    best_b = None
    best_M = None

    for b in b_vals:
        for M in M_vals:
            val = Omega_ren(M, b, L, mu, g)
            if val < best_val:
                best_val = val
                best_b = b
                best_M = M

    # ---------------------------------------------------------
    # 2) УТОЧНЕНИЕ ЧЕРЕЗ СУЖЕНИЕ ОБЛАСТИ (2D зол. сечение)
    # ---------------------------------------------------------

    # Начальная малая область вокруг найденной точки
    # Берём соседние квадраты, но учитываем границы
    def neighbor_interval(x, xmin, xmax, step):
        left  = max(x - step, xmin)
        right = min(x + step, xmax)
        return left, right

    # "Шаг" по сетке был таким:
    b_step = (bmax - bmin) / (Ns - 1)
    M_step = (Mmax - Mmin) / (Ns - 1)

    bL, bR = neighbor_interval(best_b, bmin, bmax, b_step)
    ML, MR = neighbor_interval(best_M, Mmin, Mmax, M_step)

    # коэффициент золотого сечения
    gr = (np.sqrt(5) - 1) / 2  # ≈ 0.618

    while (bR - bL > eps) or (MR - ML > eps):

        # Внутренние точки по правилу золотого сечения
        b1 = bL + (1 - gr) * (bR - bL)
        b2 = bL + gr * (bR - bL)

        M1 = ML + (1 - gr) * (MR - ML)
        M2 = ML + gr * (MR - ML)

        # 4 точки (b1, M1), (b1, M2), (b2, M1), (b2, M2)
        points = [
            (b1, M1),
            (b1, M2),
            (b2, M1),
            (b2, M2)
        ]

        values = []
        for b, M in points:
            values.append(Omega_ren(M, b, L, mu, g))

        # выбираем наименьшее значение
        idx = np.argmin(values)
        best_b, best_M = points[idx]

        # Сужаем область в сторону минимального квадрата
        if idx == 0:
            # минимальна левая-нижняя область
            bR = b2
            MR = M2
        elif idx == 1:
            # левая-верхняя
            bR = b2
            ML = M1
        elif idx == 2:
            # правая-нижняя
            bL = b1
            MR = M2
        else:
            # правая-верхняя
            bL = b1
            ML = M1

        # цикл продолжается пока область не станет меньше eps

    return best_b, best_M
