import numpy as np
from math import floor
from numba import jit
from numba import njit



def Fpn_plus_numpy(p1, n, M, b, L, phi, mu):
    """
    Векторизованная версия функции Fpn_plus.

    Параметры:
    - p1: тензор размерности (n_values, p1_values), содержит значения p1.
    - n: массив размерности (n_values,), содержит значения n.
    - M, b, L, phi, mu: константы.

    Возвращает:
    - Тензор размерности (n_values, p1_values) со значениями функции.
    """
    # Вычисление E1 для каждого p1
    E1 = np.sqrt(M**2 + p1**2)

    # Преобразование n к размерности (n_values, 1) для последующего вещания
    n = n[:, np.newaxis]

    # Вычисление значений функции
    result = (mu - np.sqrt((E1 + b)**2 + (2 * np.pi / L * (n + phi))**2)) / (2 * np.pi)
    result = result*np.heaviside(result,0)
    return result


def Fpn_minus_numpy(p1, n, M, b, L, phi, mu):
    """
    Векторизованная версия функции Fpn_plus.

    Параметры:
    - p1: тензор размерности (n_values, p1_values), содержит значения p1.
    - n: массив размерности (n_values,), содержит значения n.
    - M, b, L, phi, mu: константы.

    Возвращает:
    - Тензор размерности (n_values, p1_values) со значениями функции.
    """
    # Вычисление E1 для каждого p1
    E1 = np.sqrt(M**2 + p1**2)

    # Преобразование n к размерности (n_values, 1) для последующего вещания
    n = n[:, np.newaxis]

    # Вычисление значений функции
    result = (mu - np.sqrt((E1 - b)**2 + (2 * np.pi / L * (n + phi))**2)) / (2 * np.pi)
    result = result*np.heaviside(result,0)
    return result


def divide_intervals(bounds, N_h):
    """
    Делит каждый отрезок в bounds на N_h отрезков и возвращает узлы,
    соответствующие серединам этих отрезков, а также шаги каждого отрезка.

    :param bounds: np.ndarray, размерностью (B, 2), начальные и конечные границы отрезков
    :param N_h: int, количество отрезков, на которые разбивается каждый отрезок
    :return:
        - midpoints: np.ndarray, размерностью (B, N_h), узлы - середины отрезков
        - steps: np.ndarray, размерностью (B,), шаг каждого отрезка
    """
    # Разделяем начальные и конечные границы
    start, end = bounds[:, 0], bounds[:, 1]

    # Вычисляем шаг каждого отрезка
    steps = (end - start) / N_h  # (B,)

    # Создаём массив с равномерным распределением концов отрезков (включая границы) (1, N_h + 1)
    t = np.linspace(0, 1, N_h + 1)  # (N_h + 1,)

    # Вычисляем узлы: линейная интерполяция между start и end для концов каждого подотрезка
    edges = start[:, np.newaxis] + t * (end[:, np.newaxis] - start[:, np.newaxis])  # (B, N_h + 1)

    # Вычисляем середины отрезков: среднее значение соседних концов
    midpoints = (edges[:, :-1] + edges[:, 1:]) / 2  # (B, N_h)

    return midpoints, steps






def integration_limits_plus(M, b, mu, L, Nmax, phi):
    n = np.linspace(0, Nmax, Nmax + 1)
    SQ = (mu ** 2 - 4 * (np.pi / L) ** 2 * (n + phi) ** 2) ** 0.5
    p_left = np.zeros_like(n)
    p_right = np.zeros_like(n)
    if b > 0:
        p_right = np.sqrt((-b - SQ) ** 2 - M ** 2)
    elif b < 0:
        p_right = np.sqrt((-b + SQ) ** 2 - M ** 2)
    else:
        p_right = (SQ ** 2 - M ** 2) ** 0.5
    return p_left, p_right


def integration_limits_minus(M, b, mu, L, Nmax, phi):
    n = np.linspace(0, Nmax, Nmax + 1)
    SQ = (mu ** 2 - 4 * (np.pi / L) ** 2 * (n + phi) ** 2) ** 0.5
    p_left = np.zeros_like(n)
    p_right = np.zeros_like(n)
    if b > 0:
        p_right = np.sqrt((b + SQ) ** 2 - M ** 2)
    elif b < 0:
        p_right = np.sqrt((b - SQ) ** 2 - M ** 2)
    else:
        p_right = (SQ ** 2 - M ** 2) ** 0.5
    return p_left, p_right


def Fpn_plus(p1, n, M, b, L, phi, mu):
    E1 = np.sqrt(M**2 + p1**2)
    return (mu - np.sqrt((E1 + b)**2 + (2 * np.pi / L * (n + phi))**2))/(2 * np.pi)


def Fpn_minus(p1, n, M, b, L, phi, mu):
    E1 = np.sqrt(M**2 + p1**2)
    return (mu - np.sqrt((E1 - b)**2 + (2 * np.pi / L * (n + phi))**2))/(2 * np.pi)


def n_max_plus(M,b,L,mu,phi):
    if abs(b)>M and b<0:
      if Fpn_plus(np.sqrt(b**2-M**2),0, M, b, L, phi, mu)<=0:
        n = -1
      else:
        n = floor(L*mu/(2*np.pi)-phi)
    elif mu**2<= (M+b)**2:
      n=-1
    else:
      n = floor(L/(2*np.pi)*np.sqrt(mu**2-(M+b)**2)-phi)
    return n


def n_max_minus(M,b,L,mu,phi):
    if abs(b)>M and b>0:
      if Fpn_minus(np.sqrt(b**2-M**2),0, M, b, L, phi, mu)<=0:
        n = -1
      else:
        n = floor(L*mu/(2*np.pi)-phi)
    elif mu**2<= (M-b)**2:
      n=-1
    else:
      n = floor(L/(2*np.pi)*np.sqrt(mu**2-(M-b)**2)-phi)
    return n



# Оптимизированная версия с jit и минимизацией пересозданий массивов

def fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=100):
    Nmax_plus = n_max_plus(M, b, L, mu, phi)
    Unp = 0.0
    if Nmax_plus >= 0:
        n_plus = np.linspace(0, Nmax_plus, Nmax_plus + 1)
        p_left, p_right = integration_limits_plus(M, b, mu, L, Nmax_plus, phi)
        steps = (p_right - p_left) / N_h
        midpoints = np.empty((len(n_plus), N_h))
        for i in range(len(n_plus)):
            for j in range(N_h):
                midpoints[i, j] = p_left[i] + (j + 0.5) * steps[i]
        # Векторизованное вычисление
        E1 = np.sqrt(M**2 + midpoints**2)
        F_tp = (mu - np.sqrt((E1 + b)**2 + (2 * np.pi / L * (n_plus[:, None] + phi))**2)) / (2 * np.pi)
        F_tp = F_tp * (F_tp > 0)
        F_tp = F_tp * steps[:, None]
        Unp = 2 * np.sum(F_tp[1:, :]) + np.sum(F_tp[0, :])

    Nmax_minus = n_max_minus(M, b, L, mu, phi)
    Unm = 0.0
    if Nmax_minus >= 0:
        n_minus = np.linspace(0, Nmax_minus, Nmax_minus + 1)
        p_left, p_right = integration_limits_minus(M, b, mu, L, Nmax_minus, phi)
        steps = (p_right - p_left) / N_h
        midpoints = np.empty((len(n_minus), N_h))
        for i in range(len(n_minus)):
            for j in range(N_h):
                midpoints[i, j] = p_left[i] + (j + 0.5) * steps[i]
        # Векторизованное вычисление
        E1 = np.sqrt(M**2 + midpoints**2)
        F_tm = (mu - np.sqrt((E1 - b)**2 + (2 * np.pi / L * (n_minus[:, None] + phi))**2)) / (2 * np.pi)
        F_tm = F_tm * (F_tm > 0)
        F_tm = F_tm * steps[:, None]
        Unm = 2 * np.sum(F_tm[1:, :]) + np.sum(F_tm[0, :])

    return -(2 / L) * (Unp + Unm)