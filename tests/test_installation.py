"""Smoke tests: core dependencies and the package import correctly."""

from __future__ import annotations

import importlib

import pytest

# Core runtime dependencies (notebook extras are optional and not tested here).
CORE_DEPENDENCIES = ["jax", "jaxlib", "mdtraj", "numpy", "openmm", "scipy"]


@pytest.mark.parametrize("dep", CORE_DEPENDENCIES)
def test_core_dependency_importable(dep):
    importlib.import_module(dep)


def test_import_mdfs():
    import mdfs

    assert mdfs.__author__ == "Zhaoyang Li"
    assert mdfs.__email__ == "zhaoyangli@stanford.edu"
    assert "simulate_langevin" in mdfs.__all__


def test_jax_basic_op():
    import jax.numpy as jnp

    assert jnp.allclose(jnp.array([1.0, 2.0]) + jnp.array([3.0, 4.0]), jnp.array([4.0, 6.0]))


def test_openmm_units():
    from openmm import unit

    assert (1.0 * unit.nanometer).value_in_unit(unit.nanometer) == pytest.approx(1.0)
