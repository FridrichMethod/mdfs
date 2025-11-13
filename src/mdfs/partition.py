from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class NeighborList:
    """Neighbor pairs and bookkeeping for a Verlet neighbor list."""

    pairs: jax.Array  # (Np, 2) int32, i<j
    ref_R: jax.Array  # (N, 3) positions at last rebuild
    r_list: float  # cutoff used to build list (r_cut + skin)
    skin: float  # buffer distance


@dataclass(frozen=True)
class NeighborListFns:
    """Bundle of neighbor-list functions like jax_md.partition.neighbor_list."""

    allocate: Callable[[jax.Array], NeighborList]
    update: Callable[[jax.Array, NeighborList], NeighborList]


def _pair_indices(n: int) -> jax.Array:
    """All i<j pairs as (Np,2), integer dtype."""
    i = jnp.arange(n)
    ii = jnp.repeat(i, n)
    jj = jnp.tile(i, n)
    mask = ii < jj
    return jnp.stack([ii[mask], jj[mask]], axis=1).astype(jnp.int32)


def _build_pairs(
    R: jax.Array, displacement_fn: Callable[[jax.Array, jax.Array], jax.Array], r_list: float
) -> jax.Array:
    """Prune all i<j pairs by distance < r_list using the provided displacement_fn."""
    pairs = _pair_indices(R.shape[0])
    dR = displacement_fn(R[pairs[:, 0]], R[pairs[:, 1]])  # MIC if periodic
    r = jnp.linalg.norm(dR, axis=1)
    return pairs[r < r_list]


@jax.jit
def _max_displacement(R: jax.Array, ref_R: jax.Array) -> jax.Array:
    """Largest per-particle displacement since the last neighbor rebuild."""
    return jnp.max(jnp.linalg.norm(R - ref_R, axis=1))


def neighbor_list(
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array], r_cut: float, skin: float
) -> NeighborListFns:
    """
    Construct simple neighbor-list fns (allocate, update), à la jax_md.partition.
    Uses a Verlet buffer of 'skin' and rebuilds when max disp > 0.5 * skin.
    """
    r_list = r_cut + skin

    def allocate(R: jax.Array) -> NeighborList:
        return NeighborList(
            pairs=_build_pairs(R, displacement_fn, r_list),
            ref_R=R,
            r_list=r_list,
            skin=skin,
        )

    @jax.jit
    def update(R: jax.Array, nbrs: NeighborList) -> NeighborList:
        md = _max_displacement(R, nbrs.ref_R)

        def _rebuild(_):
            return NeighborList(
                pairs=_build_pairs(R, displacement_fn, nbrs.r_list),
                ref_R=R,
                r_list=nbrs.r_list,
                skin=nbrs.skin,
            )

        def _keep(_):
            return nbrs

        # Standard criterion (also used in JAX-MD examples): rebuild when
        # max displacement exceeds half the skin.  :contentReference[oaicite:1]{index=1}
        return jax.lax.cond(md > (0.5 * nbrs.skin), _rebuild, _keep, operand=None)

    return NeighborListFns(allocate=allocate, update=update)
