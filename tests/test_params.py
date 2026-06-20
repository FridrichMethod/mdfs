"""Tests for OpenMM parameter extraction and JAX set builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mdfs

POLY_A_PDB = Path(__file__).resolve().parent.parent / "assets" / "poly_A.pdb"


def test_poly_a_counts(poly_a_params):
    sp = poly_a_params
    # ff19SB + addHydrogens(pH 7) on the 10-residue polyalanine (OpenMM 8.5).
    assert sp.n_atoms == 103
    assert sp.bonds.shape == (102, 2)
    assert sp.angles.shape == (183, 3)
    assert sp.torsions.shape == (171, 4)
    assert sp.exclude_pairs.shape[0] == 534


def test_masses_and_charges(poly_a_params):
    sp = poly_a_params
    assert np.all(sp.masses > 0)
    assert sp.masses.shape == (sp.n_atoms,)
    # hydrogens ~1 amu present, heavy atoms heavier
    assert sp.masses.min() < 2.0
    assert sp.masses.max() > 10.0
    # nonbonded arrays are per-particle
    assert sp.charges.shape == (sp.n_atoms,)
    assert sp.sigma.shape == (sp.n_atoms,)
    assert sp.epsilon.shape == (sp.n_atoms,)


def test_exclude_mask_symmetric(poly_a_params):
    mask = poly_a_params.exclude_mask()
    assert mask.shape == (poly_a_params.n_atoms,) * 2
    assert np.array_equal(mask, mask.T)
    assert not np.any(np.diag(mask))  # no self-exclusions on the diagonal


def test_builders_shapes(poly_a_params):
    sp = poly_a_params
    bonded = mdfs.to_bonded_set(sp)
    assert bonded.bonds.shape == sp.bonds.shape
    assert bonded.k_r.shape == (sp.bonds.shape[0],)
    nb = mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms))
    assert nb.types.shape == (sp.n_atoms,)
    assert nb.q.shape == (sp.n_atoms,)
    assert nb.dsf is None and nb.r_cut_lj is None  # plain vacuum defaults


def test_system_params_from_pdb_roundtrip():
    sp, topology = mdfs.system_params_from_pdb(POLY_A_PDB)
    assert sp.n_atoms == topology.getNumAtoms() == 103
