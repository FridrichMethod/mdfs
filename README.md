# mdfs

MD from scratch

> **Author:** [Zhaoyang Li](mailto:zhaoyangli@stanford.edu)  
> **Published:** October 30, 2025

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/) [![CI Status](https://github.com/fridrichmethod/mdfs/workflows/CI/badge.svg)](https://github.com/zhaoyangli/mdfs/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

1. Install `miniconda` following the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install).
2. Then run the following commands to setup the environment in the repo root directory:

```bash
# Create a conda environment
conda create -n mdfs python=3.12 --yes
conda activate mdfs

# Install uv
pip install uv

# Install required packages with correct versions
uv pip install -e .[dev]
```

For systems equipped with NVIDIA GPUs, CUDA-enabled versions of JAX and OpenMM (compatible with CUDA 12) can be installed using:

```bash
uv pip install -e .[dev,cuda12]
```

---

To validate your installation is successful, you may simply run `pytest`.

***Let's start now!***

## See Also

- [JAX, M.D.](https://github.com/google/jax-md): Differentiable, Hardware Accelerated, Molecular Dynamics
- [OpenMM](https://github.com/openmm/openmm): OpenMM is a toolkit for molecular simulation using high performance GPU code.
- [MDTraj](https://github.com/mdtraj/mdtraj): An open library for the analysis of molecular dynamics trajectories
