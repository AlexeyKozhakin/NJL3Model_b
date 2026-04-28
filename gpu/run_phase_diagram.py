"""
Главный скрипт расчёта фазовой диаграммы на GPU.

Запуск:
    python gpu/run_phase_diagram.py

Или из корня проекта в Google Colab:
    !python gpu/run_phase_diagram.py

Последовательность:
    1. Загрузить конфигурацию (gpu/config.py)
    2. Вычислить Omega_ren(mu, L, b, M) — тензор (N_MU, N_L, N_B, N_M)
    3. Найти минимум по (b, M) для каждой точки (mu, L)
    4. Сохранить web/data/phase_diagram.json
"""

import sys
import os
import time

# Чтобы импорты работали из корня проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

import gpu.config as cfg
from gpu.Omega_ren_torch import compute as compute_omega_ren
from gpu.minimizer import find_minimum, save_phase_diagram


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


def run(region: dict | None = None) -> None:
    """
    Запускает расчёт для одного региона (b, M).

    region : один элемент cfg.REGIONS; если None — берётся первый.
    """
    region = region or cfg.REGIONS[0]

    device = cfg.DEVICE
    dtype  = cfg.DTYPE

    print(f"Device : {device}")
    print(f"dtype  : {dtype}")

    # --- Сетки ---
    mu_grid = torch.linspace(cfg.MU_MIN, cfg.MU_MAX, cfg.N_MU, dtype=dtype)
    L_grid  = torch.linspace(cfg.L_MIN,  cfg.L_MAX,  cfg.N_L,  dtype=dtype)

    b_lo, b_hi, N_B = region['b']
    M_lo, M_hi, N_M = region['M']
    b_grid = torch.linspace(b_lo, b_hi, N_B, dtype=dtype)
    M_grid = torch.linspace(M_lo, M_hi, N_M, dtype=dtype)

    print(f"\nСетки:")
    print(f"  mu : [{cfg.MU_MIN}, {cfg.MU_MAX}]  N={cfg.N_MU}")
    print(f"  L  : [{cfg.L_MIN},  {cfg.L_MAX}]   N={cfg.N_L}")
    print(f"  b  : [{b_lo}, {b_hi}]  N={N_B}")
    print(f"  M  : [{M_lo}, {M_hi}]  N={N_M}")
    print(f"  Всего точек (mu,L,b,M): {cfg.N_MU}×{cfg.N_L}×{N_B}×{N_M} = "
          f"{cfg.N_MU * cfg.N_L * N_B * N_M:,}")

    # --- Расчёт Omega_ren ---
    print(f"\n[1/3] Вычисление Omega_ren ...")
    t0 = time.time()

    omega_ren = compute_omega_ren(
        mu_grid, L_grid, b_grid, M_grid,
        g=cfg.G,
        N_p=cfg.N_P,
        N_phi=cfg.N_P,
        n_modes_max=cfg.N_MODES_MAX,
        chunk_size=cfg.CHUNK_SIZE,
        device=device,
        dtype=dtype,
    )

    t_compute = time.time() - t0
    print(f"    Время: {_fmt(t_compute)}")
    print(f"    Форма тензора: {tuple(omega_ren.shape)}")
    print(f"    Память: {omega_ren.nbytes / 1024**2:.1f} MB")

    # --- Поиск минимума ---
    print(f"\n[2/3] Поиск минимума по (b, M) ...")
    t1 = time.time()

    b_min, M_min, omega_min = find_minimum(
        omega_ren, b_grid, M_grid, window=5,
    )

    t_minimize = time.time() - t1
    print(f"    Время: {_fmt(t_minimize)}")
    print(f"    M_min : [{M_min.min():.4f}, {M_min.max():.4f}]")
    print(f"    b_min : [{b_min.min():.4f}, {b_min.max():.4f}]")

    # --- Сохранение ---
    print(f"\n[3/3] Сохранение {cfg.OUTPUT_JSON} ...")

    meta = {
        "g":          cfg.G,
        "N_p":        cfg.N_P,
        "N_MU":       cfg.N_MU,
        "N_L":        cfg.N_L,
        "N_B":        N_B,
        "N_M":        N_M,
        "b_range":    [b_lo, b_hi],
        "M_range":    [M_lo, M_hi],
        "device":     str(device),
        "dtype":      str(dtype),
        "time_s":     round(t_compute + t_minimize, 1),
    }

    save_phase_diagram(
        mu_grid, L_grid, b_min, M_min, omega_min,
        output_path=cfg.OUTPUT_JSON,
        meta=meta,
    )

    print(f"\nГотово. Полное время: {_fmt(t_compute + t_minimize)}")


def benchmark() -> None:
    """
    Таблица времени GPU vs CPU для каждой функции.
    Запускается отдельно: python gpu/run_phase_diagram.py --bench
    """
    import pandas as pd

    device_cpu  = torch.device('cpu')
    device_gpu  = torch.device('cuda') if torch.cuda.is_available() else None
    dtype       = cfg.DTYPE

    # Маленькие сетки для сравнения
    mu_s  = torch.linspace(0.5, 5.0, 5, dtype=dtype)
    L_s   = torch.linspace(0.5, 3.0, 5, dtype=dtype)
    b_s   = torch.linspace(0.0, 3.0, 20, dtype=dtype)
    M_s   = torch.linspace(0.0, 3.0, 20, dtype=dtype)
    b0_s  = torch.tensor([0.0], dtype=dtype)
    M0_s  = torch.tensor([0.0], dtype=dtype)

    from gpu.functions.dU_torch import compute as dU_fn
    from gpu.functions.Omega_L_torch import compute as OL_fn
    from gpu.functions.Omega_muL_torch import compute as OmuL_fn

    rows = []

    for dev_name, dev in [('CPU', device_cpu)] + ([('GPU', device_gpu)] if device_gpu else []):
        def _t(fn, *args, **kw):
            if dev_name == 'GPU':
                torch.cuda.synchronize()
            t = time.perf_counter()
            fn(*args, **kw)
            if dev_name == 'GPU':
                torch.cuda.synchronize()
            return time.perf_counter() - t

        rows.append({
            'device':   dev_name,
            'function': 'dU',
            'time_s':   _t(dU_fn, b_s, M_s, N_p=cfg.N_P, N_phi=cfg.N_P, device=dev, dtype=dtype),
        })
        rows.append({
            'device':   dev_name,
            'function': 'Omega_L',
            'time_s':   _t(OL_fn, L_s, b_s, M_s, N_p=cfg.N_P, device=dev, dtype=dtype),
        })
        rows.append({
            'device':   dev_name,
            'function': 'Omega_muL',
            'time_s':   _t(OmuL_fn, mu_s, L_s, b_s, M_s,
                           N_p=cfg.N_P, chunk_size=cfg.CHUNK_SIZE,
                           device=dev, dtype=dtype),
        })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='function', columns='device', values='time_s')
    if 'GPU' in pivot.columns:
        pivot['Speedup'] = pivot['CPU'] / pivot['GPU']
    print("\nБенчмарк (секунды):")
    print(pivot.to_string(float_format='{:.3f}'.format))


if __name__ == '__main__':
    if '--bench' in sys.argv:
        benchmark()
    else:
        run()
