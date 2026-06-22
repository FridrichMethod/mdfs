"""Physical constants in MD-standard units.

Unit system used throughout ``mdfs`` (the OpenMM/Amber convention):

- length: nanometers (nm)
- time: picoseconds (ps)
- mass: atomic mass units (amu / Da)
- energy: kilojoules per mole (kJ/mol)
- charge: elementary charge (e)

In this system velocities are nm/ps and ``E = 0.5 * m * v**2`` yields kJ/mol.
"""

from __future__ import annotations

from typing import Final

# Boltzmann constant in kJ/mol/K (CODATA, MD units). Matches OpenMM's
# ``unit.MOLAR_GAS_CONSTANT_R`` (= kB * N_A) expressed per mole.
BOLTZMANN_KJ_PER_MOL_K: Final[float] = 0.00831446261815324

# Coulomb constant 1 / (4 * pi * eps0) in kJ * nm / (mol * e**2).
# This is OpenMM's ``ONE_4PI_EPS0`` and converts q_i q_j / r (e**2 / nm) to kJ/mol.
ONE_4PI_EPS0: Final[float] = 138.935456

# Numerical softening floor (nm) used to keep norms, divisions, and atan2 gradients
# finite at degenerate geometries. The single source of truth for the per-function
# ``eps`` defaults; not a physical quantity.
EPS: Final[float] = 1e-12
