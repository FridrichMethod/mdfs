"""High-level simulation drivers (NVE and Langevin NVT).

These wire together a periodic/free :mod:`mdfs.space`, the total energy from
:mod:`mdfs.energy`, and an integrator from :mod:`mdfs.integrators`, returning an
initial :class:`~mdfs.integrators.State` and a JIT-compiled stepper.

The default pair list is static all-pairs (:func:`mdfs.partition.all_pairs`),
which is correct for the small/vacuum systems mdfs targets. Larger or periodic
systems can pass a cutoff pair list instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from mdfs.constants import BOLTZMANN_KJ_PER_MOL_K
from mdfs.constraints import ConstraintSet
from mdfs.energy import BondedSet, NonbondedSet, total_energy_fn
from mdfs.integrators import (
    LangevinParams,
    State,
    langevin_baoab,
    velocity_verlet,
)
from mdfs.space import free as space_free
from mdfs.space import periodic as space_periodic
from mdfs.types import DisplacementFn, ShiftFn


def _build_space(box: jax.Array | None) -> tuple[DisplacementFn, ShiftFn, jax.Array]:
    if box is None:
        displacement_fn, shift_fn = space_free()
        return displacement_fn, shift_fn, jnp.zeros((3,))
    sim_box = jnp.asarray(box)
    displacement_fn, shift_fn = space_periodic(sim_box)
    return displacement_fn, shift_fn, sim_box


def make_energy_fn(
    box: jax.Array | None,
    bonded: BondedSet,
    nonbonded: NonbondedSet,
) -> tuple[Callable[[jax.Array], jax.Array], ShiftFn, jax.Array]:
    """Return ``(energy_fn(R), shift_fn, sim_box)`` for the given parameters/box."""
    displacement_fn, shift_fn, sim_box = _build_space(box)
    return total_energy_fn(displacement_fn, bonded, nonbonded), shift_fn, sim_box


def simulate_nve(
    R0: jax.Array,
    V0: jax.Array,
    box: jax.Array | None,
    bonded: BondedSet,
    nonbonded: NonbondedSet,
    dt: float,
    mass: jax.Array | float,
    constraints: ConstraintSet | None = None,
) -> tuple[State, Callable[[State], State]]:
    """Build ``(state, step_fn)`` for an NVE (velocity-Verlet / RATTLE) simulation.

    Pass ``constraints`` (and a ``bonded`` set with those bonds removed, e.g. via
    :func:`mdfs.setup_hbond_constraints`) to constrain bonds for a larger timestep.
    """
    energy_fn, shift_fn, sim_box = make_energy_fn(box, bonded, nonbonded)
    step_fn = velocity_verlet(energy_fn, shift_fn, dt, mass, constraints=constraints)
    return State(R=R0, V=V0, box=sim_box, t=0.0), step_fn


def simulate_langevin(
    R0: jax.Array,
    V0: jax.Array,
    box: jax.Array | None,
    bonded: BondedSet,
    nonbonded: NonbondedSet,
    dt: float,
    mass: jax.Array | float,
    gamma: float,
    temperature: float,
    kB: float = BOLTZMANN_KJ_PER_MOL_K,
    constraints: ConstraintSet | None = None,
) -> tuple[State, Callable[[State, jax.Array], tuple[State, jax.Array]]]:
    """Build ``(state, step_fn)`` for a Langevin (BAOAB, NVT) simulation.

    Pass ``constraints`` (with a matching reduced ``bonded`` set) to constrain
    bonds for a larger timestep; see :func:`mdfs.setup_hbond_constraints`.
    """
    energy_fn, shift_fn, sim_box = make_energy_fn(box, bonded, nonbonded)
    params = LangevinParams(gamma=gamma, temperature=temperature, kB=kB)
    step_fn = langevin_baoab(energy_fn, shift_fn, dt, mass, params, constraints=constraints)
    return State(R=R0, V=V0, box=sim_box, t=0.0), step_fn


def run(
    step_fn: Callable[..., Any],
    state: State,
    n_steps: int,
    *,
    key: jax.Array | None = None,
    report_interval: int = 0,
    callback: Callable[[int, State], None] | None = None,
) -> State:
    """Step a simulation ``n_steps`` times.

    For NVE leave ``key=None`` (``step_fn(state) -> state``); for Langevin pass a
    PRNG ``key`` (``step_fn(state, key) -> (state, key)``). When ``report_interval``
    and ``callback`` are set, the callback is invoked as ``callback(step, state)``.
    """
    for step in range(n_steps):
        if key is None:
            state = step_fn(state)
        else:
            state, key = step_fn(state, key)
        if report_interval and callback and (step + 1) % report_interval == 0:
            callback(step + 1, state)
    return state
