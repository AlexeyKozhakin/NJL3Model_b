"""
Совместимость: линии переходов при b = 0 переехали в пакет njl3.b0
(njl3/b0/lines.py). Модуль сохраняет прежние имена для notebooks/46–47.
"""

from njl3.b0.formulas import G2  # noqa: F401
from njl3.b0.lines import (  # noqa: F401
    mu_stability, mu_VT, peak, trace_L, tricritical_points,
)
from njl3.b0.minimize import M_vacuum, phase_at  # noqa: F401
