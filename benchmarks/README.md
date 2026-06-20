# Benchmarks

Throughput and size-scaling for the implemented MD matrix (NVE / NVT, vacuum /
periodic). Larger systems are built by replicating poly_A into spatially-separated
copies (`benchmark.py --replicas ...`).

```bash
.venv/bin/python benchmarks/benchmark.py                 # GPU, float32
.venv/bin/python benchmarks/benchmark.py --x64           # float64
JAX_PLATFORMS=cpu .venv/bin/python benchmarks/benchmark.py   # force CPU
```

`steps/s` and `ns/day` use `dt = 0.5 fs`. Numbers below are indicative, measured on
an **NVIDIA RTX PRO 6000 (Blackwell)**, JAX 0.10, OpenMM 8.5 — your mileage varies.

## Size scaling (GPU, float32, all-pairs)

| config | atoms | pairs | steps/s | ns/day | ms/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVE vacuum | 103 | 5,253 | 2,881 | 124 | 0.35 |
| NVT vacuum | 103 | 5,253 | 2,743 | 119 | 0.36 |
| NVE vacuum | 515 | 132,355 | 156 | 6.7 | 6.4 |
| NVT vacuum | 515 | 132,355 | 154 | 6.7 | 6.5 |
| NVE vacuum | 1,030 | 529,935 | 21 | 0.9 | 48 |
| NVT vacuum | 1,030 | 529,935 | 22 | 0.9 | 46 |
| NVE vacuum | 2,060 | 2,120,770 | 2 | 0.1 | 474 |
| NVT periodic | 103 | 5,253 | 4,526 | 196 | 0.22 |

## Base system across device / precision (poly_A, 103 atoms, NVE)

| backend | ns/day | us/step |
| --- | ---: | ---: |
| GPU float32 | 124 | 332 |
| GPU float64 | 329 | 131 |
| CPU float32 | 178 | 243 |

## How to read this

- **At ~100 atoms it is overhead-bound, not compute-bound.** Each step launches
  dozens of tiny GPU kernels (bond/angle/dihedral/LJ/Coulomb/gather/...), so the
  ~0.1-0.35 ms/step is kernel-launch latency, not arithmetic. Evidence: CPU beats
  GPU, float64 beats float32, and `lax.scan` does not speed it up. The GPU is idle.
- **All-pairs is O(N^2).** Beyond a few hundred atoms throughput falls off as ~1/N^2
  (515 -> 1,030 -> 2,060 atoms: 156 -> 21 -> 2 steps/s). The practical ceiling for
  the default all-pairs path is ~hundreds to ~1,000 atoms.
- **Scope.** mdfs is a *small-system, differentiable* MD engine for teaching and
  research (energies/forces validated against OpenMM to machine precision). It is
  not a large-scale production code: there is no on-device neighbor list, the step
  loop is in Python, and electrostatics use DSF rather than PME. For large or
  solvated systems use OpenMM/GROMACS.
- **Speedups available** (not yet implemented): wrap the loop in `lax.scan` to
  remove per-step dispatch; an on-device cell/neighbor list to replace O(N^2);
  hydrogen-mass repartitioning or constraints for a larger timestep.
