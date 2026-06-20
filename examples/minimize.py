"""Energy minimization of poly_A: steepest descent vs. BFGS.

Demonstrates both minimizers on the freshly-protonated structure before dynamics.

Run:
    python examples/minimize.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import mdfs
from mdfs.paths import REPO_ROOT


def main() -> None:
    sp, _ = mdfs.system_params_from_pdb(REPO_ROOT / "assets" / "poly_A.pdb")
    bonded = mdfs.to_bonded_set(sp)
    nonbonded = mdfs.to_nonbonded_set(sp)  # dense (N, N) path
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nonbonded)
    R0 = jnp.asarray(sp.positions)

    sd = mdfs.steepest_descent(energy_fn, R0, n_steps=500)
    bfgs = mdfs.minimize_energy(energy_fn, R0, max_iter=500)

    print(f"initial energy:        {float(energy_fn(R0)):10.2f} kJ/mol")
    print(f"steepest descent:      {sd.energy:10.2f} kJ/mol  ({sd.n_steps} steps)")
    print(
        f"BFGS:                  {bfgs.energy:10.2f} kJ/mol  ({bfgs.n_steps} iters, converged={bfgs.converged})"
    )


if __name__ == "__main__":
    main()
