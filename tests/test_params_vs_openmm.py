"""Validate mdfs energies and forces against OpenMM per force group.

mdfs should reproduce an OpenMM ``NoCutoff`` system to machine precision, minus
the (intentionally omitted) CMAP term.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openmm import Context, Platform, VerletIntegrator, unit

import mdfs


@pytest.fixture(scope="module")
def reference(poly_a_bundle):
    _sp, system, _topology, positions = poly_a_bundle
    for idx, force in enumerate(system.getForces()):
        force.setForceGroup(idx)
    names = {idx: f.__class__.__name__ for idx, f in enumerate(system.getForces())}
    ctx = Context(
        system, VerletIntegrator(1.0 * unit.femtosecond), Platform.getPlatformByName("Reference")
    )
    ctx.setPositions(positions)

    def group_energy(force_name: str) -> float:
        groups = {i for i, n in names.items() if n == force_name}
        if not groups:
            return 0.0
        state = ctx.getState(getEnergy=True, groups=groups)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    non_cmap = {i for i, n in names.items() if n != "CMAPTorsionForce"}
    forces = np.array(
        ctx
        .getState(getForces=True, groups=non_cmap)
        .getForces(asNumpy=True)
        .value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    )
    return group_energy, forces


@pytest.fixture(scope="module")
def mdfs_pieces(poly_a_params):
    sp = poly_a_params
    R = jnp.asarray(sp.positions)
    disp, _ = mdfs.free()
    bonded = mdfs.to_bonded_set(sp)
    nb = mdfs.to_nonbonded_set(sp)  # dense default
    return sp, R, disp, bonded, nb


def test_dense_and_pairlist_agree(poly_a_params):
    """The dense (N, N) and pair-list paths give identical energies and forces."""
    sp = poly_a_params
    R = jnp.asarray(sp.positions)
    disp, _ = mdfs.free()
    bonded = mdfs.to_bonded_set(sp)
    e_dense = mdfs.total_energy_fn(disp, bonded, mdfs.to_nonbonded_set(sp))
    e_pairs = mdfs.total_energy_fn(
        disp, bonded, mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms))
    )
    assert float(e_dense(R)) == pytest.approx(float(e_pairs(R)), rel=1e-9)
    f_dense = np.array(-jax.grad(e_dense)(R))
    f_pairs = np.array(-jax.grad(e_pairs)(R))
    assert np.max(np.abs(f_dense - f_pairs)) < 1e-6


def test_dense_and_pairlist_agree_periodic_cutoff(poly_a_params):
    """Dense == pair-list with PBC + LJ cutoff + DSF + real (1-2/1-3/1-4) exclusions."""
    from mdfs.energy import DSFParams

    sp = poly_a_params
    side = float(np.ptp(sp.positions, axis=0).max()) + 3.0
    box = jnp.array([side, side, side])
    r_cut = side / 2.0 - 0.1
    R = jnp.asarray(sp.positions - sp.positions.mean(axis=0) + side / 2.0)
    disp, _ = mdfs.periodic(box)
    bonded = mdfs.to_bonded_set(sp)
    dsf = DSFParams(alpha=2.0, r_cut=r_cut)
    e_dense = mdfs.total_energy_fn(disp, bonded, mdfs.to_nonbonded_set(sp, r_cut_lj=r_cut, dsf=dsf))
    e_pairs = mdfs.total_energy_fn(
        disp, bonded, mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms), r_cut_lj=r_cut, dsf=dsf)
    )
    assert float(e_dense(R)) == pytest.approx(float(e_pairs(R)), rel=1e-9)
    f_dense = np.array(-jax.grad(e_dense)(R))
    f_pairs = np.array(-jax.grad(e_pairs)(R))
    assert np.max(np.abs(f_dense - f_pairs)) < 1e-6


def test_bonded_terms_match_openmm(reference, mdfs_pieces):
    from mdfs import energy

    group_energy, _ = reference
    _sp, R, _disp, bonded, _nb = mdfs_pieces
    e_bond = float(energy.bond_energy(R, bonded.bonds, bonded.k_r, bonded.r0))
    e_ang = float(energy.angle_energy(R, bonded.angles, bonded.k_theta, bonded.theta0))
    e_tor = float(
        energy.torsion_energy(
            R, bonded.torsions, bonded.periodicity, bonded.torsion_k, bonded.phase
        )
    )
    assert e_bond == pytest.approx(group_energy("HarmonicBondForce"), rel=1e-6)
    assert e_ang == pytest.approx(group_energy("HarmonicAngleForce"), rel=1e-6)
    assert e_tor == pytest.approx(group_energy("PeriodicTorsionForce"), rel=1e-6)


def test_nonbonded_matches_openmm(reference, mdfs_pieces):
    from mdfs import energy

    group_energy, _ = reference
    _sp, R, disp, _bonded, nb = mdfs_pieces
    e_nb = float(energy.nonbonded_energy(R, nb, disp))
    assert e_nb == pytest.approx(group_energy("NonbondedForce"), rel=1e-6)


def test_total_energy_matches_openmm_minus_cmap(reference, mdfs_pieces):
    group_energy, _ = reference
    _sp, R, disp, bonded, nb = mdfs_pieces
    e_mdfs = float(mdfs.total_energy_fn(disp, bonded, nb)(R))
    e_ref = (
        group_energy("HarmonicBondForce")
        + group_energy("HarmonicAngleForce")
        + group_energy("PeriodicTorsionForce")
        + group_energy("NonbondedForce")
    )
    assert e_mdfs == pytest.approx(e_ref, rel=1e-6)


def test_forces_match_openmm(reference, mdfs_pieces):
    _group_energy, f_ref = reference
    _sp, R, disp, bonded, nb = mdfs_pieces
    energy_fn = mdfs.total_energy_fn(disp, bonded, nb)
    f_mdfs = np.array(-jax.grad(energy_fn)(R))
    max_abs = np.max(np.abs(f_ref))
    assert np.max(np.abs(f_mdfs - f_ref)) < 1e-3 * max_abs
