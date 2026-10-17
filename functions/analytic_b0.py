"""
Совместимость: формулы и минимизатор при b = 0 переехали в пакет njl3.b0
(njl3/b0/formulas.py, njl3/b0/minimize.py). Модуль сохраняет прежние имена
для скриптов notebooks/40–47.
"""

from njl3.b0.formulas import (  # noqa: F401
    G2, _I, _inv_expm1_minus_inv, dU, gap_G, gap_G0, gap_G_vec, gap_dG_vec,
    omega_L, omega_muL, thresholds,
)
from njl3.b0.formulas import modes as _modes  # noqa: F401
from njl3.b0.formulas import omega_rel as omega_rel_stable  # noqa: F401
from njl3.b0.formulas import omega_rel_closed as omega_ren_rel  # noqa: F401
from njl3.b0.formulas import part_A as _A, part_B as _B  # noqa: F401
from njl3.b0.minimize import find_minimum as find_minimum_b0_certified  # noqa: F401
from njl3.b0.minimize import find_minimum_scan as find_minimum_b0  # noqa: F401
from njl3.b0.minimize import isolate_roots as isolate_roots_b0  # noqa: F401
