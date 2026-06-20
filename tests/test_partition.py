"""Tests for the neighbor-pair providers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mdfs.partition import all_pairs, neighbor_list
from mdfs.space import free


def test_all_pairs_count_and_content():
    n = 6
    pairs = np.array(all_pairs(n))
    assert pairs.shape == (n * (n - 1) // 2, 2)
    assert np.all(pairs[:, 0] < pairs[:, 1])
    # all unique unordered pairs present
    s = {tuple(p) for p in pairs}
    assert len(s) == n * (n - 1) // 2


def test_neighbor_list_prunes_by_cutoff():
    disp, _ = free()
    R = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
    fns = neighbor_list(disp, r_cut=1.0, skin=0.2)
    nbrs = fns.allocate(R)
    pairs = {tuple(p) for p in np.array(nbrs.pairs)}
    assert (0, 1) in pairs  # within cutoff+skin
    assert (0, 2) not in pairs  # far apart


def test_neighbor_list_update_keeps_when_static():
    disp, _ = free()
    R = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    fns = neighbor_list(disp, r_cut=1.0, skin=0.2)
    nbrs = fns.allocate(R)
    nbrs2 = fns.update(R, nbrs)  # no movement -> unchanged
    assert np.array_equal(np.array(nbrs.pairs), np.array(nbrs2.pairs))
