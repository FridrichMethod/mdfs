"""Trajectory and energy output helpers (reporters).

These are designed as ``callback(step, state)`` objects compatible with
:func:`mdfs.simulate.run`. Trajectories are written through MDTraj (XTC/DCD/PDB);
energy logs are written as plain CSV (no pandas dependency).
"""

from __future__ import annotations

import csv
import logging
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import mdtraj as md
import numpy as np

from mdfs.constants import BOLTZMANN_KJ_PER_MOL_K
from mdfs.integrators import EnergyFn, State, kinetic_energy, temperature
from mdfs.types import StrPath

logger = logging.getLogger(__name__)

Callback = Callable[[int, State], None]


def mdtraj_topology_from_openmm(openmm_topology: object) -> md.Topology:
    """Build an MDTraj topology from an OpenMM ``Topology`` (for writing trajectories)."""
    return md.Topology.from_openmm(openmm_topology)


@dataclass
class TrajectoryRecorder:
    """Accumulate frames during a run and write them as an MDTraj trajectory."""

    topology: md.Topology
    positions: list[np.ndarray] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def __call__(self, step: int, state: State) -> None:  # noqa: ARG002
        """Record the current frame (callback form ``(step, state)``)."""
        self.positions.append(np.asarray(state.R, dtype=np.float64))
        self.times.append(float(state.t))

    def to_trajectory(self) -> md.Trajectory:
        """Return the recorded frames as an MDTraj ``Trajectory`` (nm)."""
        if not self.positions:
            raise ValueError("No frames recorded; nothing to write.")
        xyz = np.stack(self.positions)  # (T, N, 3) in nm
        return md.Trajectory(xyz, self.topology, time=np.asarray(self.times))

    def save(self, path: StrPath) -> None:
        """Save the recorded trajectory (format inferred from extension, e.g. .xtc/.dcd/.pdb)."""
        self.to_trajectory().save(str(path))
        logger.info("Wrote %d frames to %s", len(self.positions), path)


@dataclass
class EnergyLogger:
    """Record step, time, potential/kinetic energy, and temperature during a run."""

    energy_fn: EnergyFn
    mass: jax.Array | float  # scalar or (N,) array, amu
    kB: float = BOLTZMANN_KJ_PER_MOL_K
    n_dof: int | None = None
    log_to_logger: bool = True
    records: list[dict[str, float]] = field(default_factory=list)

    def __call__(self, step: int, state: State) -> None:
        """Record energies/temperature for the current frame (callback form)."""
        pe = float(self.energy_fn(state.R))
        ke = float(kinetic_energy(state.V, self.mass))
        temp = float(temperature(state.V, self.mass, self.kB, self.n_dof))
        rec = {
            "step": int(step),
            "time_ps": float(state.t),
            "potential_kj_mol": pe,
            "kinetic_kj_mol": ke,
            "total_kj_mol": pe + ke,
            "temperature_K": temp,
        }
        self.records.append(rec)
        if self.log_to_logger:
            logger.info(
                "step=%d t=%.4f ps  PE=%.3f  KE=%.3f  E=%.3f kJ/mol  T=%.1f K",
                step,
                state.t,
                pe,
                ke,
                pe + ke,
                temp,
            )

    def save_csv(self, path: StrPath) -> None:
        """Write the recorded energy/temperature series to a CSV file."""
        if not self.records:
            raise ValueError("No records logged; nothing to write.")
        with pathlib.Path(path).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)
        logger.info("Wrote %d energy records to %s", len(self.records), path)


def combine_callbacks(*callbacks: Callback) -> Callback:
    """Combine several reporters into a single ``callback(step, state)``."""

    def combined(step: int, state: State) -> None:
        for cb in callbacks:
            cb(step, state)

    return combined
