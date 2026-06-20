"""Sanity checks on physical constants."""

from __future__ import annotations

import pytest

from mdfs.constants import BOLTZMANN_KJ_PER_MOL_K, ONE_4PI_EPS0


def test_boltzmann_value():
    # kB * N_A in kJ/mol/K
    assert pytest.approx(0.00831446, rel=1e-5) == BOLTZMANN_KJ_PER_MOL_K


def test_coulomb_constant_value():
    # OpenMM ONE_4PI_EPS0 in kJ*nm/(mol*e^2)
    assert pytest.approx(138.935456, rel=1e-6) == ONE_4PI_EPS0
