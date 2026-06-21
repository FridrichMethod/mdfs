# Examples

Runnable demos covering the MD configurations mdfs implements. Run from the repo
root with the project's `.venv`:

```bash
.venv/bin/python examples/<name>.py
```

| Example | Ensemble | PBC | Nonbonded | Shows |
| --- | --- | --- | --- | --- |
| [`minimize.py`](minimize.py) | — | — | plain | steepest-descent vs BFGS minimization |
| [`nve_vacuum.py`](nve_vacuum.py) | NVE | no | plain LJ+Coulomb, all-pairs | velocity-Verlet; energy conservation |
| [`nvt_vacuum.py`](nvt_vacuum.py) | NVT | no | plain LJ+Coulomb, all-pairs | Langevin BAOAB; reporters (XTC + energy CSV) |
| [`nvt_periodic.py`](nvt_periodic.py) | NVT | yes | DSF + LJ cutoff, MIC | periodic box, damped-shifted-force electrostatics |
| [`nvt_hmr.py`](nvt_hmr.py) | NVT | no | plain LJ+Coulomb, dense | hydrogen mass repartitioning -> dt = 2 fs (~4× faster) |
| [`nvt_constraints.py`](nvt_constraints.py) | NVT | no | plain LJ+Coulomb, dense | LINCS H-bond constraints (RATTLE/BAOAB) -> dt = 2 fs |

All examples use the bundled `assets/poly_A.pdb` (10-residue polyalanine, 103 atoms
after adding hydrogens) and run on whatever device JAX selects (`JAX_PLATFORMS=cpu`
to force CPU). `float64` is enabled in the examples for accurate energetics.

Expected output (indicative): NVE relative energy drift ~1e-3 over 2 ps; NVT mean
temperature within a few percent of 300 K.

See [`../benchmarks/`](../benchmarks) for throughput and size-scaling numbers.
