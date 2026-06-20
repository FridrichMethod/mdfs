"""Tests for trajectory and energy reporters."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import mdfs
from mdfs.integrators import State
from mdfs.io import (
    EnergyLogger,
    TrajectoryRecorder,
    combine_callbacks,
    mdtraj_topology_from_openmm,
)


def _state(sp, t):
    return State(R=jnp.asarray(sp.positions), V=jnp.zeros((sp.n_atoms, 3)), box=jnp.zeros(3), t=t)


def test_trajectory_recorder_roundtrip(poly_a_bundle, tmp_path):
    sp, _system, topology, _pos = poly_a_bundle
    rec = TrajectoryRecorder(mdtraj_topology_from_openmm(topology))
    for i in range(3):
        rec(i, _state(sp, 0.001 * i))
    traj = rec.to_trajectory()
    assert traj.n_frames == 3
    assert traj.n_atoms == sp.n_atoms
    out = tmp_path / "t.xtc"
    rec.save(out)
    assert out.exists()


def test_energy_logger_records_and_csv(poly_a_params, tmp_path):
    sp = poly_a_params
    bonded = mdfs.to_bonded_set(sp)
    nb = mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms))
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nb)
    logger = EnergyLogger(energy_fn, sp.masses, log_to_logger=False)
    logger(10, _state(sp, 0.01))
    assert len(logger.records) == 1
    rec = logger.records[0]
    assert set(rec) == {
        "step",
        "time_ps",
        "potential_kj_mol",
        "kinetic_kj_mol",
        "total_kj_mol",
        "temperature_K",
    }
    assert np.isfinite(rec["potential_kj_mol"])
    out = tmp_path / "e.csv"
    logger.save_csv(out)
    assert out.exists() and out.read_text().count("\n") == 2  # header + 1 row


def test_combine_callbacks_invokes_all(poly_a_params):
    calls = []
    cb = combine_callbacks(
        lambda s, st: calls.append(("a", s)), lambda s, st: calls.append(("b", s))
    )
    cb(5, _state(poly_a_params, 0.0))
    assert ("a", 5) in calls and ("b", 5) in calls
