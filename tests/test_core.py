import numpy as np
import pytest

from cantuccio.core import get_credible_interval, cornerplot


def test_credible_interval_unweighted():
    data = np.linspace(0, 100, 101)
    lo, med, hi = get_credible_interval(data, level=0.90)
    # The 90% HDI for uniform 0-100 is 5.0, 50.0, 95.0
    assert np.isclose(lo, 5.0)
    assert np.isclose(med, 50.0)
    assert np.isclose(hi, 95.0)


def test_credible_interval_weighted():
    data = np.array([1.0, 2.0, 3.0])
    # Give all weight to the middle point
    weights = np.array([0.0, 1.0, 0.0])
    lo, med, hi = get_credible_interval(data, level=0.90, weights=weights)
    assert np.isclose(lo, 1.1)
    assert np.isclose(med, 2.0)
    assert np.isclose(hi, 2.9)

    # Give all weight to the outer points
    weights = np.array([0.5, 0.0, 0.5])
    lo, med, hi = get_credible_interval(data, level=0.50, weights=weights)
    # The cdf will be 0.25 at 1.0, 0.5 at 1.0/3.0 transition, 0.75 at 3.0
    # Weighted percentiles with interpolation:
    # CDF array: [0.25, 0.5, 0.75]
    # For median (0.5), it will interpolate exactly to 2.0
    assert np.isclose(med, 2.0)


def test_cornerplot_execution_without_weights():
    samples = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
    }
    # Test that it executes without raising an exception
    fig, axes = cornerplot(samples)
    assert fig is not None
    assert axes.shape == (2, 2)


def test_cornerplot_execution_with_weights():
    samples = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
    }
    weights = np.random.uniform(0.1, 1.0, 100)
    
    # Test that it executes without raising an exception
    fig, axes = cornerplot(samples, weights=weights)
    assert fig is not None
    assert axes.shape == (2, 2)


def test_cornerplot_multiple_chains_with_weights():
    chain1 = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
    }
    chain2 = {
        "x": np.random.randn(100) + 1.0,
        "y": np.random.randn(100) + 1.0,
    }
    w1 = np.ones(100)
    w2 = np.ones(100) * 0.5
    
    fig, axes = cornerplot([chain1, chain2], weights=[w1, w2])
    assert fig is not None
    assert axes.shape == (2, 2)

def test_cornerplot_multiple_chains_not_full():
    chain1 = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
        "z": np.random.randn(100),
    }
    chain2 = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
    }
    
    fig, axes = cornerplot([chain1, chain2])
    assert fig is not None
    assert axes.shape == (3, 3)