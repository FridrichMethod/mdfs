"""Tests for free and periodic (MIC) space."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mdfs.space import free, periodic, wrap


def test_free_displacement_and_shift():
    disp, shift = free()
    Ra = jnp.array([[0.0, 0.0, 0.0]])
    Rb = jnp.array([[1.0, 2.0, 3.0]])
    assert np.allclose(np.array(disp(Ra, Rb)), [[1.0, 2.0, 3.0]])
    assert np.allclose(np.array(shift(Ra, Rb)), [[1.0, 2.0, 3.0]])


def test_periodic_minimum_image():
    box = jnp.array([10.0, 10.0, 10.0])
    disp, _ = periodic(box)
    Ra = jnp.array([[1.0, 1.0, 1.0]])
    Rb = jnp.array([[9.0, 1.0, 1.0]])  # nearest image is -2, not +8
    d = np.array(disp(Ra, Rb))
    assert np.allclose(d, [[-2.0, 0.0, 0.0]])


def test_periodic_shift_wraps():
    box = jnp.array([5.0, 5.0, 5.0])
    _, shift = periodic(box)
    R = jnp.array([[4.5, 0.0, 0.0]])
    dR = jnp.array([[1.0, 0.0, 0.0]])  # 5.5 -> wraps to 0.5
    assert np.allclose(np.array(shift(R, dR)), [[0.5, 0.0, 0.0]])


def test_wrap_into_box():
    box = jnp.array([3.0, 3.0, 3.0])
    R = jnp.array([[3.5, -0.5, 6.1]])
    w = np.array(wrap(R, box))
    assert np.all(w >= 0) and np.all(w < 3.0)
