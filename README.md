# mdfs

MD from scratch

> **Author:** [Zhaoyang Li](mailto:zhaoyangli@stanford.edu)  
> **Published:** October 30, 2025

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
uv pip install -e .[dev,test]
```

If you are on a machine with a NVIDIA GPU, a CUDA-enabled version of jax (with CUDA 12 support) can be installed with

```bash
uv pip install jax[cuda12]
```

---

To validate your installation is successful, you may simply run `pytest`.

***Let's start now!***
