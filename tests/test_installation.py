"""Test that the installation is correct and all dependencies work."""

import importlib


def test_all_dependencies_importable() -> None:
    """Test that all dependencies listed in pyproject.toml can be imported."""
    dependencies = [
        "jax",
        "jupyter",
        "matplotlib",
        "mdtraj",
        "numpy",
        "pandas",
        "py3Dmol",
        "scipy",
        "seaborn",
        "tqdm",
    ]

    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError as e:
            raise AssertionError(f"Failed to import {dep}: {e}") from e


def test_import_mdfs() -> None:
    """Test that the mdfs package can be imported."""
    import mdfs

    assert mdfs.__author__ == "Zhaoyang Li"
    assert mdfs.__email__ == "zhaoyangli@stanford.edu"


def test_import_jax() -> None:
    """Test that JAX can be imported and basic operations work."""
    import jax.numpy as jnp

    # Test basic JAX operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    result = x + y
    assert jnp.allclose(result, jnp.array([5.0, 7.0, 9.0]))


def test_import_numpy() -> None:
    """Test that NumPy can be imported and basic operations work."""
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5])
    assert arr.sum() == 15
    assert arr.mean() == 3.0


def test_import_scipy() -> None:
    """Test that SciPy can be imported and basic operations work."""
    import numpy as np
    from scipy import stats

    data = np.array([1, 2, 3, 4, 5])
    assert np.allclose(stats.gmean(data), 2.6051710846973517)


def test_import_pandas() -> None:
    """Test that Pandas can be imported and basic operations work."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.shape == (3, 2)
    assert df["a"].sum() == 6


def test_import_matplotlib() -> None:
    """Test that Matplotlib can be imported."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    plt.close(fig)


def test_import_seaborn() -> None:
    """Test that Seaborn can be imported."""
    import seaborn as sns

    # Just check it can be imported and accessed
    assert hasattr(sns, "set_style")


def test_import_mdtraj() -> None:
    """Test that MDTraj can be imported."""
    import mdtraj

    # MDTraj requires actual trajectory files to do useful things,
    # so we just verify it can be imported
    assert hasattr(mdtraj, "load")


def test_import_py3dmol() -> None:
    """Test that py3Dmol can be imported."""
    import py3Dmol

    # py3Dmol is mainly for Jupyter notebooks, so we just verify it can be imported
    assert hasattr(py3Dmol, "view")


def test_import_tqdm() -> None:
    """Test that tqdm can be imported and basic usage works."""
    from tqdm.auto import tqdm

    # Test that tqdm can iterate over a range
    result = list(tqdm(range(5), desc="test", disable=True))
    assert result == [0, 1, 2, 3, 4]
