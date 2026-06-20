"""Energy minimization to relax a structure before dynamics.

A freshly-protonated (e.g. AlphaFold) structure typically has small steric
clashes; running dynamics directly can be unstable. Minimizing first removes
those clashes. Two methods are provided: a robust capped steepest descent and
BFGS via :func:`jax.scipy.optimize.minimize`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.optimize import minimize as jsp_minimize

from mdfs.types import EnergyFn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinimizationResult:
    """Outcome of an energy minimization."""

    positions: jax.Array  # (N, 3) minimized positions, nm
    energy: float  # final potential energy, kJ/mol
    initial_energy: float  # potential energy before minimization, kJ/mol
    n_steps: int  # iterations actually performed
    converged: bool = True  # whether the optimizer reported convergence


def steepest_descent(
    energy_fn: EnergyFn,
    R0: jax.Array,
    n_steps: int = 500,
    learning_rate: float = 2e-5,
    max_step: float = 0.002,
) -> MinimizationResult:
    """Steepest descent with a per-atom displacement cap (``max_step`` nm).

    Robust from clashy starts: the cap prevents a single large force from
    exploding the geometry. The conservative defaults decrease energy
    monotonically on stiff (bonded) systems; for deeper minimization prefer
    :func:`minimize_energy` (BFGS). Returns the minimized positions and energies.
    """
    force_fn = grad(energy_fn)
    e0 = float(energy_fn(R0))

    @jit
    def step(R: jax.Array) -> jax.Array:
        g = force_fn(R)
        disp = -learning_rate * g
        norms = jnp.linalg.norm(disp, axis=1, keepdims=True)
        scale = jnp.minimum(1.0, max_step / (norms + 1e-12))
        return R + disp * scale

    R = R0
    for _ in range(n_steps):
        R = step(R)
    return MinimizationResult(
        positions=R, energy=float(energy_fn(R)), initial_energy=e0, n_steps=n_steps
    )


def minimize_energy(
    energy_fn: EnergyFn,
    R0: jax.Array,
    max_iter: int = 1000,
) -> MinimizationResult:
    """Minimize with BFGS (:func:`jax.scipy.optimize.minimize`).

    Operates on the flattened coordinate vector. Suitable for the small systems
    mdfs targets; for very large systems prefer :func:`steepest_descent`.
    """
    shape = R0.shape
    e0 = float(energy_fn(R0))

    def flat_energy(x: jax.Array) -> jax.Array:
        return energy_fn(x.reshape(shape))

    res = jsp_minimize(flat_energy, R0.reshape(-1), method="BFGS", options={"maxiter": max_iter})
    converged = bool(res.success)
    if not converged:
        logger.warning(
            "BFGS minimization did not converge (status=%s); returning the last iterate.",
            int(res.status),
        )
    return MinimizationResult(
        positions=res.x.reshape(shape),
        energy=float(res.fun),
        initial_energy=e0,
        n_steps=int(res.nit),
        converged=converged,
    )
