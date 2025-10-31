#!/bin/bash

# This script sets up the environment for BioCGM
set -euo pipefail

# Create a conda environment
# conda create -n mdfs python=3.12 --yes
# conda activate mdfs

# Install uv
pip install uv

# Install required packages with correct versions
uv pip install -e .[dev,test]
