"""Tests for LINCS bond constraints and constrained integrators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mdfs
from mdfs import constraints as C


@pytest.fixture(scope="module")
def constrained(poly_a_params):
    """poly_A with H-bond constraints + minimized, constraint-satisfying positions."""
    sp = poly_a_params
    nb = mdfs.to_nonbonded_set(sp)
    # Minimize with the full bonded energy so X-H bonds sit at r0 (unrestrained
    # hydrogens would otherwise drift and break the LINCS projection).
    bonded_full = mdfs.to_bonded_set(sp)
    full_energy_fn, _, _ = mdfs.make_energy_fn(None, bonded_full, nb)
    R0 = mdfs.minimize_energy(full_energy_fn, jnp.asarray(sp.positions), max_iter=300).positions
    cset, bonded = C.setup_hbond_constraints(
        sp.bonds, sp.bond_r0, sp.masses, bonded_full, sp.n_atoms
    )
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nb)
    R0 = C.apply_position_constraint(R0, R0, cset)
    return sp, cset, bonded, nb, energy_fn, R0


def _bond_lengths(R, pairs):
    R = np.asarray(R)
    p = np.asarray(pairs)
    return np.linalg.norm(R[p[:, 0]] - R[p[:, 1]], axis=1)


def test_selection_and_dof(poly_a_params):
    sp = poly_a_params
    pairs, lengths = C.select_hbond_constraints(sp.bonds, sp.bond_r0, sp.masses)
    assert pairs.shape[0] == 52  # H-bonds in protonated poly_A
    assert lengths.shape == (52,)
    cset = C.make_constraint_set(pairs, lengths, sp.masses, sp.n_atoms)
    assert C.constrained_dof(cset) == 3 * sp.n_atoms - 52


def test_remove_constrained_bonds(poly_a_params):
    sp = poly_a_params
    bonded = mdfs.to_bonded_set(sp)
    pairs, _ = C.select_hbond_constraints(sp.bonds, sp.bond_r0, sp.masses)
    reduced = C.remove_constrained_bonds(bonded, pairs)
    assert reduced.bonds.shape[0] == bonded.bonds.shape[0] - pairs.shape[0]


def test_position_projection_restores_bonds(constrained):
    _sp, cset, _bonded, _nb, _energy_fn, R0 = constrained
    target = np.asarray(cset.lengths)
    perturbed = R0 + jax.random.normal(jax.random.PRNGKey(0), R0.shape) * 0.01
    unc = np.max(np.abs(_bond_lengths(perturbed, cset.pairs) - target))
    R_c = C.apply_position_constraint(R0, perturbed, cset)
    dev = np.max(np.abs(_bond_lengths(R_c, cset.pairs) - target))
    # a single LINCS projection of a large (~0.1 A) perturbation restores bonds ~100x
    assert dev < 5e-4 < unc  # nm


def test_velocity_projection_removes_along_bond(constrained):
    _sp, cset, _bonded, _nb, _efn, R0 = constrained
    V = jax.random.normal(jax.random.PRNGKey(1), R0.shape)
    Vc = np.asarray(C.apply_velocity_constraint(R0, V, cset))
    pairs = np.asarray(cset.pairs)
    u = np.asarray(R0)[pairs[:, 0]] - np.asarray(R0)[pairs[:, 1]]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    along = np.sum(u * (Vc[pairs[:, 0]] - Vc[pairs[:, 1]]), axis=1)
    assert np.max(np.abs(along)) < 5e-5


def test_constrained_nve_holds_bonds_at_2fs(constrained):
    sp, cset, bonded, nb, energy_fn, R0 = constrained
    mass = jnp.asarray(sp.masses)
    V0 = C.apply_velocity_constraint(
        R0, mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms), cset
    )
    state, step = mdfs.simulate_nve(R0, V0, None, bonded, nb, dt=0.002, mass=mass, constraints=cset)
    total = []
    for _ in range(1000):
        state = step(state)
        total.append(float(energy_fn(state.R)) + float(mdfs.kinetic_energy(state.V, mass)))
    total = np.array(total)
    dev = np.max(np.abs(_bond_lengths(state.R, cset.pairs) - np.asarray(cset.lengths)))
    assert dev < 1e-3  # nm; bonds stay near target throughout
    assert jnp.all(jnp.isfinite(state.R))
    # RATTLE conserves total energy at 2 fs (catches velocity-correction sign/factor bugs)
    assert (total.max() - total.min()) / abs(total.mean()) < 0.05


def test_constrained_baoab_temperature(constrained):
    sp, cset, bonded, nb, _energy_fn, R0 = constrained
    mass = jnp.asarray(sp.masses)
    dof = C.constrained_dof(cset)
    V0 = C.apply_velocity_constraint(
        R0, mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms), cset
    )
    state, step = mdfs.simulate_langevin(
        R0,
        V0,
        None,
        bonded,
        nb,
        dt=0.002,
        mass=mass,
        gamma=10.0,
        temperature=300.0,
        constraints=cset,
    )
    temps: list[float] = []
    key = jax.random.PRNGKey(1)
    for i in range(4000):
        state, key = step(state, key)
        if i >= 2000 and i % 20 == 0:
            temps.append(float(mdfs.temperature(state.V, mass, n_dof=dof)))
    assert np.mean(temps) == pytest.approx(300.0, rel=0.1)
