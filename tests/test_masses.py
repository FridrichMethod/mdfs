"""Tests for hydrogen mass repartitioning (HMR)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mdfs
from mdfs.masses import repartition_hydrogen_masses


def test_conserves_total_mass(poly_a_params):
    sp = poly_a_params
    m = repartition_hydrogen_masses(sp.masses, sp.bonds)
    assert m.sum() == pytest.approx(sp.masses.sum(), rel=1e-12)


def test_hydrogens_repartitioned(poly_a_params):
    sp = poly_a_params
    m = repartition_hydrogen_masses(sp.masses, sp.bonds, hydrogen_mass=3.024)
    is_h = sp.masses < 1.5
    assert np.allclose(m[is_h], 3.024)
    assert np.all(m > 0)
    assert m[~is_h].sum() < sp.masses[~is_h].sum()  # heavy atoms lost mass


def test_no_bonds_unchanged():
    m = np.array([1.008, 12.0])
    out = repartition_hydrogen_masses(m, np.zeros((0, 2), dtype=np.int32))
    assert np.array_equal(out, m)


def test_raises_on_too_large_hydrogen_mass():
    masses = np.array([1.008, 1.008, 1.008, 1.008, 12.0])
    bonds = np.array([[0, 4], [1, 4], [2, 4], [3, 4]])  # CH4-like; 4*(10-1) > 12
    with pytest.raises(ValueError, match="non-positive"):
        repartition_hydrogen_masses(masses, bonds, hydrogen_mass=10.0)


def test_raises_on_bridging_hydrogen():
    masses = np.array([12.011, 1.008, 12.011])  # C-H-C: H bonded to two heavy atoms
    bonds = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="bridging"):
        repartition_hydrogen_masses(masses, bonds)


def test_raises_on_bad_shapes_and_params():
    masses = np.array([1.008, 12.0])
    with pytest.raises(ValueError, match="hydrogen_mass must be positive"):
        repartition_hydrogen_masses(masses, np.array([[0, 1]]), hydrogen_mass=0.0)
    with pytest.raises(ValueError, match="shape"):
        repartition_hydrogen_masses(masses, np.array([0, 1]))  # 1-D bonds


def test_hydrogen_heavier_than_target_moves_mass_to_heavy():
    masses = np.array([1.4, 12.011])  # "H" heavier than target -> heavy atom gains
    m = repartition_hydrogen_masses(masses, np.array([[0, 1]]), hydrogen_mass=1.0)
    assert m[0] == pytest.approx(1.0)
    assert m[1] == pytest.approx(12.411)
    assert m.sum() == pytest.approx(masses.sum(), rel=1e-12)


def test_hmr_stabilizes_2fs(poly_a_params):
    """At dt = 2 fs, HMR conserves energy markedly better than unmodified masses."""
    sp = poly_a_params
    bonded = mdfs.to_bonded_set(sp)
    nb = mdfs.to_nonbonded_set(sp)
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nb)
    R0 = mdfs.minimize_energy(energy_fn, jnp.asarray(sp.positions), max_iter=200).positions

    def drift(mass: np.ndarray) -> float:
        mass = jnp.asarray(mass)
        V0 = mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms)
        state, step = mdfs.simulate_nve(R0, V0, None, bonded, nb, dt=0.002, mass=mass)
        total: list[float] = []
        mdfs.run(
            step,
            state,
            1000,
            report_interval=50,
            callback=lambda _i, s: total.append(
                float(energy_fn(s.R)) + float(mdfs.kinetic_energy(s.V, mass))
            ),
        )
        e = np.array(total)
        return (e.max() - e.min()) / abs(e.mean())

    assert drift(repartition_hydrogen_masses(sp.masses, sp.bonds)) < drift(sp.masses)
