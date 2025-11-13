from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit

from mdfs.energy import BondedSet, NonbondedSet, total_energy_fn
from mdfs.integrators import (
    LangevinParams,
    langevin_baoab_pairs,
    velocity_verlet_pairs,
)
from mdfs.integrators import (
    State as IntegratorState,
)
from mdfs.partition import NeighborList, neighbor_list
from mdfs.space import free as space_free
from mdfs.space import periodic as space_periodic


@dataclass(frozen=True)
class SimState:
    integ: IntegratorState
    nbrs: NeighborList


def _make_energy_with_neighbors(
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    bonded: BondedSet,
    nb_template: NonbondedSet,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    E(R, box, pairs): inject CURRENT neighbor pairs into a NonbondedSet
    and call your total_energy_fn(displacement_fn, bonded, nb).
    """

    def E_nb(R: jax.Array, box: jax.Array, pairs: jax.Array) -> jax.Array:
        nb = NonbondedSet(
            pairs=pairs,
            types=nb_template.types,
            q=nb_template.q,
            lj_params=nb_template.lj_params,
            r_cut_lj=nb_template.r_cut_lj,
            dsf=nb_template.dsf,
            scale14_vdw=nb_template.scale14_vdw,
            scale14_elec=nb_template.scale14_elec,
            exclude_mask=nb_template.exclude_mask,
            shift_lj=nb_template.shift_lj,
        )
        return total_energy_fn(displacement_fn, bonded, nb)(R)

    return E_nb


# -----------------------------
# NVE (Velocity-Verlet)
# -----------------------------
def simulate_nve(
    R0: jax.Array,
    V0: jax.Array,
    box: jax.Array | None,
    bonded: BondedSet,
    nonbonded: NonbondedSet,
    dt: float,
    mass: jax.Array | float,
    r_cut_neighbor: float,
    skin: float = 0.2,
):
    """
    Construct (init_fn, step_fn) for NVE with a Verlet neighbor list.
    This mirrors jax_md.simulate APIs: we build the stepper once and then step.
    """
    if box is None:
        displacement_fn, shift_fn = space_free()
        sim_box = jnp.zeros((3,))
    else:
        displacement_fn, shift_fn = space_periodic(box)
        sim_box = jnp.asarray(box)
        # MIC sanity for short-range cutoffs
        assert float(r_cut_neighbor) <= float(jnp.min(sim_box)) * 0.5, (
            "Neighbor cutoff must satisfy r_cut <= min(box)/2 for MIC."
        )

    # Neighbor list fns (allocate/update), à la jax_md.partition. :contentReference[oaicite:3]{index=3}
    nbr_fns = neighbor_list(displacement_fn, r_cut_neighbor, skin)

    # Build-once: neighbor-aware energy and pair-aware stepper.
    energy_pairs_fn = _make_energy_with_neighbors(displacement_fn, bonded, nonbonded)
    stepper = velocity_verlet_pairs(
        energy_pairs_fn=energy_pairs_fn,
        displacement_fn=displacement_fn,
        shift_fn=shift_fn,
        dt=dt,
        mass=mass,
    )

    def init_fn(_: jax.Array) -> SimState:
        integ0 = IntegratorState(R=R0, V=V0, box=sim_box, t=0.0)
        nbrs0 = nbr_fns.allocate(R0)
        return SimState(integ=integ0, nbrs=nbrs0)

    @jit
    def step_fn(state: SimState) -> SimState:
        nbrs = nbr_fns.update(state.integ.R, state.nbrs)
        integ_new = stepper(state.integ, nbrs.pairs)
        return SimState(integ=integ_new, nbrs=nbrs)

    return init_fn, step_fn


# -----------------------------
# NVT (Langevin BAOAB)
# -----------------------------
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
    r_cut_neighbor: float,
    skin: float = 0.2,
    kB: float = 1.0,
):
    """
    Construct (init_fn, step_fn) for Langevin (BAOAB) with a Verlet neighbor list.
    """
    if box is None:
        displacement_fn, shift_fn = space_free()
        sim_box = jnp.zeros((3,))
    else:
        displacement_fn, shift_fn = space_periodic(box)
        sim_box = jnp.asarray(box)
        assert float(r_cut_neighbor) <= float(jnp.min(sim_box)) * 0.5, (
            "Neighbor cutoff must satisfy r_cut <= min(box)/2 for MIC."
        )

    nbr_fns = neighbor_list(displacement_fn, r_cut_neighbor, skin)

    energy_pairs_fn = _make_energy_with_neighbors(displacement_fn, bonded, nonbonded)
    stepper = langevin_baoab_pairs(
        energy_pairs_fn=energy_pairs_fn,
        displacement_fn=displacement_fn,
        shift_fn=shift_fn,
        dt=dt,
        mass=mass,
        params=LangevinParams(gamma=gamma, temperature=temperature, kB=kB),
    )

    def init_fn(_: jax.Array) -> SimState:
        integ0 = IntegratorState(R=R0, V=V0, box=sim_box, t=0.0)
        nbrs0 = nbr_fns.allocate(R0)
        return SimState(integ=integ0, nbrs=nbrs0)

    @jit
    def step_fn(state: SimState, key: jax.Array) -> tuple[SimState, jax.Array]:
        nbrs = nbr_fns.update(state.integ.R, state.nbrs)
        integ_new, key = stepper(state.integ, key, nbrs.pairs)
        return SimState(integ=integ_new, nbrs=nbrs), key

    return init_fn, step_fn


def run(
    init_fn,
    step_fn,
    n_steps: int,
    key: jax.Array | None = None,
    state: SimState | None = None,
    report_interval: int = 0,
    callback: Callable[[int, SimState], None] | None = None,
):
    """
    Simple loop for demos/tests. For Langevin, provide a PRNG key and a (state, key) step_fn.
    """
    if state is None:
        if key is None:
            key = jax.random.PRNGKey(0)
        state = init_fn(key)

    for step in range(n_steps):
        if key is None:
            state = step_fn(state)
        else:
            state, key = step_fn(state, key)

        if report_interval and ((step + 1) % report_interval == 0) and callback:
            callback(step + 1, state)

    return state
