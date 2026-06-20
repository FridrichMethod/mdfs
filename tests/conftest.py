"""Shared pytest fixtures and configuration.

float64 is enabled globally so energy-conservation and OpenMM-comparison tests
are not limited by float32 round-off.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402
from openmm import app  # noqa: E402

import mdfs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
POLY_A_PDB = REPO_ROOT / "assets" / "poly_A.pdb"


@pytest.fixture(scope="session")
def poly_a_bundle():
    """A shared protonated poly_A: ``(SystemParams, openmm System, topology, positions)``.

    The same OpenMM ``System``/positions back both mdfs extraction and any OpenMM
    reference, so comparisons are not confounded by nondeterministic H placement.
    """
    topology, positions, forcefield = mdfs.prepare_topology(POLY_A_PDB)
    system = forcefield.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False
    )
    sp = mdfs.extract_system_params(system, topology, positions)
    return sp, system, topology, positions


@pytest.fixture(scope="session")
def poly_a_params(poly_a_bundle):
    """Just the :class:`~mdfs.params.SystemParams` for poly_A."""
    return poly_a_bundle[0]
