"""Neighbor-pair providers: static all-pairs and a Verlet neighbor list."""

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


def all_pairs(n: int) -> jax.Array:
    """All ``i < j`` index pairs as a static ``(n*(n-1)/2, 2)`` int32 array.

    For small systems (and any vacuum simulation) this fixed pair list is the
    simplest correct choice: it is jit-friendly and never needs rebuilding.
    """
    i, j = jnp.triu_indices(n, k=1)
    return jnp.stack([i, j], axis=1).astype(jnp.int32)


def _build_pairs(
    R: jax.Array, displacement_fn: Callable[[jax.Array, jax.Array], jax.Array], r_list: float
) -> jax.Array:
    """Prune all i<j pairs by distance < r_list using the provided displacement_fn."""
    pairs = all_pairs(R.shape[0])
    dR = displacement_fn(R[pairs[:, 0]], R[pairs[:, 1]])  # MIC if periodic
    r = jnp.linalg.norm(dR, axis=1)
    return pairs[r < r_list]


def _max_displacement(
    R: jax.Array,
    ref_R: jax.Array,
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Largest per-particle displacement since the last rebuild (minimum-image).

    Uses ``displacement_fn`` so a periodic box-face crossing (which ``shift_fn``
    wraps by ~L in raw coordinates) is not mistaken for a large displacement.
    """
    return jnp.max(jnp.linalg.norm(displacement_fn(ref_R, R), axis=1))


def neighbor_list(
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array], r_cut: float, skin: float
) -> NeighborListFns:
    """Construct simple neighbor-list fns (allocate, update), a la jax_md.partition.

    Uses a Verlet buffer of ``skin`` and rebuilds when max displacement exceeds
    half the skin.
    """
    r_list = r_cut + skin

    def allocate(R: jax.Array) -> NeighborList:
        return NeighborList(
            pairs=_build_pairs(R, displacement_fn, r_list),
            ref_R=R,
            r_list=r_list,
            skin=skin,
        )

    def update(R: jax.Array, nbrs: NeighborList) -> NeighborList:
        # The pruned pair list has a data-dependent length, so rebuilding is done
        # eagerly (outside jit). Drive jitted steppers with the returned pairs;
        # rebuild between steps in Python when atoms drift past half the skin.
        if float(_max_displacement(R, nbrs.ref_R, displacement_fn)) > 0.5 * nbrs.skin:
            return NeighborList(
                pairs=_build_pairs(R, displacement_fn, nbrs.r_list),
                ref_R=R,
                r_list=nbrs.r_list,
                skin=nbrs.skin,
            )
        return nbrs

    return NeighborListFns(allocate=allocate, update=update)
