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

## Dense (default) vs pair-list, and size scaling (GPU, float32)

mdfs computes forces by `jax.grad` of the energy. The **dense (N, N)** nonbonded
path (the default) makes that gradient a *reduction*; the **pair-list** path makes
it a *scatter*, which is far slower on GPU (see "Why the dense path" below).

| config | atoms | pairs | steps/s | ns/day | ms/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVE dense | 103 | 5,253 | 9,680 | 418 | 0.10 |
| NVE dense | 515 | 132,355 | 8,314 | 359 | 0.12 |
| NVE dense | 2,060 | 2,120,770 | 6,143 | 265 | 0.16 |
| NVE dense | 5,150 | 13,258,675 | 1,725 | 74 | 0.58 |
| NVT periodic dense | 103 | 5,253 | 9,411 | 407 | 0.11 |
| NVT dense + HMR (2 fs) | 103 | 5,253 | 9,287 | 1,605 | 0.11 |
| NVT dense + HMR (2 fs) | 2,060 | 2,120,770 | 6,068 | 1,049 | 0.16 |
| NVT constrained (LINCS, 2 fs) | 103 | 5,253 | 6,467 | 1,118 | 0.15 |
| NVT constrained + HMR (4 fs) | 103 | 5,253 | 9,458 | 3,269 | 0.11 |
| NVE pair-list | 103 | 5,253 | 2,846 | 123 | 0.35 |
| NVE pair-list | 515 | 132,355 | 151 | 6.5 | 6.6 |

Dense vs pair-list speedup: **3.4× at 103 atoms, 55× at 515, ~2,600× at 2,060**
(the pair-list path falls off as O(N²); at 2,060 atoms it is ~0.1 ns/day).

**Larger timesteps.** Two ways to take a bigger step:

- **HMR** (`mdfs.repartition_hydrogen_masses`) slows X-H stretches so dt = 2 fs is
  stable without constraints (no per-step solver cost): poly_A **1,605 ns/day**,
  2,060 atoms **1,049 ns/day**.
- **LINCS H-bond constraints** (`mdfs.setup_hbond_constraints`) remove the X-H
  stretch entirely → a robust **2 fs** (1,118 ns/day; the constraint solve adds
  per-step cost, so at tiny N this trails HMR-2fs) and, combined with HMR, **4 fs**
  → **3,269 ns/day** (the fastest; 4 fs is best with a thermostat — NVE drift is
  elevated by the H-angle limit).

End to end, dense + constraints + HMR at 4 fs is **~26×** faster at 103 atoms
than the original pair-list engine at 0.5 fs (124 ns/day), and dense + HMR is
**~10,000×** at 2,060 atoms (where the original was ~0.1 ns/day).

## Dense base system across device / precision (poly_A, 103 atoms, NVE)

| backend | ns/day | ms/step |
| --- | ---: | ---: |
| GPU float32 | 418 | 0.10 |
| GPU float64 | 314 | 0.14 |
| CPU float32 | 121 | 0.36 |

With the dense path the GPU is finally the fastest backend (it was slower than CPU
with the pair-list/scatter path).

## Why the dense path

Forces come from reverse-mode autodiff of the nonbonded energy. With a **pair
list**, the energy gathers `R[i], R[j]` over O(N²) pairs, so the gradient must
**scatter-add** O(N²) per-pair contributions back onto the atoms — and XLA's GPU
scatter-add is atomic/serialized, hundreds of times slower than the forward pass
(measured: forward ~20 µs, grad ~18 ms at 1,030 atoms). The **dense (N, N)**
formulation builds pairwise quantities by broadcasting (`R[:,None]-R[None,:]`) and
reduces to forces along an axis — a fast GPU reduction, no scatter. Cost: O(N²)
memory (~50 MB at 2,060 atoms, ~300 MB at 5,150), so it is ideal up to a few
thousand atoms. For larger or dilute/solvated systems, pass an O(N) neighbor list
(`mdfs.partition.neighbor_list`) to use the pair-list path.

## Acceleration status

- **Dense (N, N) nonbonded path — done (default).** Forces reduce instead of
  scatter; the big win above.
- **Hydrogen mass repartitioning — done.** `mdfs.repartition_hydrogen_masses`
  enables dt = 2 fs (~4×). See `examples/nvt_hmr.py`.
- **LINCS bond constraints — done.** `mdfs.setup_hbond_constraints` + RATTLE
  velocity-Verlet / constrained BAOAB give a robust 2 fs and (with HMR) 4 fs. See
  `examples/nvt_constraints.py`.
- **`lax.scan` loop — measured, not adopted.** Replacing the Python step loop with
  an on-device scan gave ~0.9× (no gain) on the dense path: async dispatch already
  hides per-step latency and the step is GPU-compute-bound, so scan only adds
  complexity and breaks per-step callbacks.
- **On-device cell list — future.** Would give O(N) neighbor search for large
  systems without the dense path's O(N²) memory; significant effort, and of
  limited benefit for the dense single-molecule regime mdfs targets.
- mdfs remains a small/medium-system differentiable engine (energies/forces
  validated against OpenMM to machine precision); for large solvated production
  runs use OpenMM/GROMACS.
