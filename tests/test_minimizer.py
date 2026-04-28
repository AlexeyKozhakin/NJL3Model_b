"""
Верификация gpu/minimizer.py.

Этап 1.7: минимум на аналитической параболе + smoke-тест на реальном Omega_ren.
"""
import sys
import os
import json
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpu.minimizer import find_minimum, save_phase_diagram

DTYPE = torch.float64


def _make_parabola(b_grid, M_grid, b0, M0, val0=0.0):
    """Параболоид с минимумом в (b0, M0)."""
    b = b_grid.view(-1, 1)
    M = M_grid.view(1, -1)
    return (b - b0) ** 2 + (M - M0) ** 2 + val0


def test_find_minimum_parabola_exact():
    """Минимум параболоида: argmin должен попасть в (b0, M0)."""
    b_grid = torch.linspace(0.0, 4.0, 41, dtype=DTYPE)
    M_grid = torch.linspace(0.0, 4.0, 41, dtype=DTYPE)
    b0, M0 = 2.5, 1.5

    omega = _make_parabola(b_grid, M_grid, b0, M0).view(1, 1, 41, 41)
    b_min, M_min, omega_min = find_minimum(omega, b_grid, M_grid)

    assert abs(b_min[0, 0] - b0) < 0.05, f"b_min={b_min[0,0]:.4f} != {b0}"
    assert abs(M_min[0, 0] - M0) < 0.05, f"M_min={M_min[0,0]:.4f} != {M0}"
    assert abs(omega_min[0, 0]) < 1e-3,  f"omega_min={omega_min[0,0]:.4g} != 0"


def test_find_minimum_batch_parabolas():
    """Разные минимумы для нескольких (mu, L) пар."""
    b_grid = torch.linspace(0.0, 5.0, 51, dtype=DTYPE)
    M_grid = torch.linspace(0.0, 5.0, 51, dtype=DTYPE)

    centers = [(1.0, 1.0), (3.0, 2.0), (0.5, 4.0), (4.5, 0.5)]
    N = len(centers)

    omega = torch.stack([
        _make_parabola(b_grid, M_grid, b0, M0)
        for b0, M0 in centers
    ]).view(N, 1, 51, 51)

    b_min, M_min, _ = find_minimum(omega, b_grid, M_grid)

    for i, (b0, M0) in enumerate(centers):
        assert abs(b_min[i, 0] - b0) < 0.1, f"b_min[{i}]={b_min[i,0]:.3f} != {b0}"
        assert abs(M_min[i, 0] - M0) < 0.1, f"M_min[{i}]={M_min[i,0]:.3f} != {M0}"


def test_find_minimum_boundary():
    """Минимум на границе сетки — argmin не должен падать."""
    b_grid = torch.linspace(0.0, 4.0, 20, dtype=DTYPE)
    M_grid = torch.linspace(0.0, 4.0, 20, dtype=DTYPE)

    # Минимум в (0, 0) — угол сетки
    omega = _make_parabola(b_grid, M_grid, 0.0, 0.0).view(1, 1, 20, 20)
    b_min, M_min, _ = find_minimum(omega, b_grid, M_grid, window=5)

    assert b_min[0, 0] >= 0.0
    assert M_min[0, 0] >= 0.0


def test_save_and_load_json():
    """JSON сохраняется и читается корректно."""
    mu_grid = torch.linspace(0.0, 3.0, 4, dtype=DTYPE)
    L_grid  = torch.linspace(0.5, 2.0, 3, dtype=DTYPE)
    b_min   = np.random.rand(4, 3)
    M_min   = np.random.rand(4, 3)
    omega_min = np.random.rand(4, 3)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        path = f.name

    try:
        save_phase_diagram(mu_grid, L_grid, b_min, M_min, omega_min,
                           output_path=path, meta={"g": -1.0, "N_p": 100})

        with open(path) as f:
            data = json.load(f)

        assert "mu" in data and "L" in data
        assert "M_min" in data and "b_min" in data and "Omega_min" in data
        assert data["meta"]["g"] == -1.0
        assert len(data["mu"]) == 4
        assert len(data["L"])  == 3
        assert len(data["M_min"]) == 4
        assert len(data["M_min"][0]) == 3
    finally:
        os.unlink(path)
